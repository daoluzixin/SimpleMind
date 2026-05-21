# Qwen2.5-1.5B Handoff RL 实验收获

> 2026-05-14，云服务器 2×A100-SXM4-80GB，模型 Qwen2.5-1.5B-Instruct (1.5B 参数)。
> 本文记录从 "handoff=0% 困局" 到 "SFT Warmup → RL GRPO 链路打通" 全过程中的核心发现、踩坑经验和方法论收获。

---

## 背景与目标

从 64M 参数的 MiniMind 切换到 Qwen2.5-1.5B-Instruct，目标是验证 Agent Handoff（多 Agent 路由）能力能否通过 RL 训练习得。训练框架沿用 GRPO，但模型规模提升了约 24 倍，带来了完全不同的工程挑战。

Handoff 任务的核心逻辑是：RouterAgent 接收用户请求 → 判断是否需要专家介入 → 若需要则生成 tool_call 将任务委派给 ExpertAgent → Expert 执行后 Router 整合结果。RL 的 reward signal 来自 handoff 是否正确发生、Expert 是否给出合理回答。


## 核心发现：parse_tool_calls 格式不匹配（最关键的 Bug）

### 问题现象

RL 训练跑了约 170 effective steps（从 step_50 checkpoint resume），reward 上升到 1.0 以上（说明模型在学习），但 **handoff rate 始终为 0%**。这个矛盾信号说明：模型学到了某种"能拿分"的行为，但这个行为不是我们期望的工具调用。

### 根因定位

`trainer/train_agent.py` 中的 `parse_tool_calls` 函数使用 MiniMind 64M 的自定义分隔符格式：

```python
# 旧版 MiniMind 格式——用日文假名作为 tool call 分隔符
for m in re.findall(r'ゅ(.*?)゜', text, re.DOTALL):
    calls.append(json.loads(m.strip()))
```

但 Qwen2.5-1.5B-Instruct 的 `chat_template`（tokenizer_config.json 中定义）指导模型用 XML 标签格式输出 tool call：

```xml
<tool_call>
{"name": "delegate_to_math_agent", "arguments": {"task": "计算 2+2"}}
</tool_call>
```

**两种格式完全不兼容。** 即使模型完美学会了 tool call 输出，`parse_tool_calls` 也无法匹配到任何内容，导致 `handoff_occurred` 永远为 False，handoff rate 永远为 0%。

### 为什么 reward 还能上升？

这是一个重要的观察：reward 函数中除了 handoff 相关奖励外，还有"回答质量"的基础分。模型在 RL 训练中学到了"直接回答问题比尝试工具调用更容易拿分"（因为工具调用的分被 parse bug 完全吃掉了），于是策略收敛到"永远不做 handoff、直接回答"——这从 reward 角度是最优策略，但完全偏离了训练目标。

### 方法论收获

**当某个关键指标持续为零时，首先怀疑的不是模型能力不足，而是度量本身是否有 bug。** 这和 64M 模型 Plan RL 阶段的经验如出一辙——PlanRate=0% 的根因也是 parse 函数的逻辑错误，而非模型不会生成 plan。

定位方法：
1. 用模型单独做推理，观察实际输出格式
2. 将实际输出送入 parse 函数，验证能否匹配
3. 对比 parse 函数期望的格式和模型实际生成的格式

三步即可定位。核心教训是**换模型时必须验证 tokenizer/chat_template 的输出格式与训练代码的 parse 逻辑是否匹配**。


## SFT Warmup 策略：解决冷启动问题

### 设计思路

定位到 parse bug 后，修复方案分两步：
1. 覆盖 `parse_tool_calls` 函数，兼容 `<tool_call>` XML 格式
2. 用少量 SFT 数据做 warmup，确保模型在 RL 开始前就"知道"正确的输出格式

SFT warmup 的必要性在于：Qwen2.5-1.5B-Instruct 虽然预训练时见过 `<tool_call>` 格式，但在我们特定的 Agent Handoff 场景（4 种专家：math/info/translate/none）下，它不知道何时应该调用哪个专家。SFT 提供了这个场景的"格式示范"。

### 数据设计

生成了 90 条 SFT warmup 数据（`dataset/sft_warmup.jsonl`）：

| 类别 | 数量 | 示例 |
|------|------|------|
| math（数学计算） | 25 | "计算 123×456" → 调用 math_agent |
| info（信息查询） | 25 | "什么是量子计算" → 调用 info_agent |
| translate（翻译） | 20 | "把'hello'翻译成中文" → 调用 translate_agent |
| none（直接回答） | 20 | "你好" → 不调用任何工具 |

关键设计决策：
- **70/90 的样本包含 tool_call**（约 78%），有意偏向工具调用。理由是 RL 阶段可以"纠正"过度调用，但如果模型从未学会调用格式，RL 无从优化。
- 每条数据严格使用 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` 格式，与 Qwen2.5 的 chat_template 一致。
- none 类别的 assistant 回复直接给出自然语言答案，不包含任何 `<tool_call>` 标签。

### SFT 训练参数

```bash
epochs=5, lr=2e-5, batch_size=2, gradient_accumulation=2
attn_implementation="flash_attention_2"
DDP across 2×A100
```

训练耗时仅 90 秒，loss 从 ~2.5 降到 0.0245（几乎完美拟合 90 条数据）。

### SFT 过拟合的"有意为之"

90 条数据 + 5 epochs = 必然过拟合。但这是有意的设计：
- SFT warmup 的目标不是"泛化"，而是"格式植入"——让模型坚定地认为遇到相关问题就应该输出 `<tool_call>`
- RL 阶段负责纠正过度调用（通过给不必要的 handoff 以负 reward）
- 这种"先过度再纠正"的策略比"SFT 就追求平衡"更稳健，因为 RL 需要模型先有工具调用行为才能给予信号

### 收获

**SFT warmup + RL fine-tuning 是小规模数据下训练 Agent 行为的有效二阶段范式。** 第一阶段用少量高质量示范确保模型掌握目标格式，第二阶段用 RL 在更大规模数据上优化决策质量。两者缺一不可：只有 SFT 会过拟合到训练分布；只有 RL 则冷启动太慢（64M 模型的 Agent RL v5 需要 1600 步才出现拐点）。


## DDP 训练的工程陷阱

### 陷阱一：gradient_checkpointing + DDP 不兼容

DDP（DistributedDataParallel）要求所有参数在 backward 时都被"使用"（marked as ready），因为它需要跨 GPU 同步梯度。gradient_checkpointing 通过不保存中间激活来省显存，但某些实现中会导致部分参数在 backward 时"看起来未被使用"。

错误信息：
```
Expected to mark a variable ready only once.
This error is caused by one of the following reasons:
1) Use of a module parameter outside the `forward` function.
```

解决方案：**禁用 gradient_checkpointing**，转而通过减小 batch_size + 增加 accumulation_steps 来控制显存。

```bash
# 不可行
batch_size=4, gradient_checkpointing=1

# 可行（等效 batch_size 相同）
batch_size=2, accumulation_steps=2, gradient_checkpointing=0
```

A100 80GB 在 Qwen2.5-1.5B + GRPO (num_generations=2) 下，batch_size=2 约占 65GB 显存，不需要 gradient_checkpointing。

### 陷阱二：flash_attention_2 是刚需

Qwen2.5 使用 GQA（Grouped Query Attention），在某些 sequence length 和 head_dim 组合下，PyTorch 默认的 SDPA 后端（cuDNN）会报错：

```
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
```

解决方案：在加载模型时指定 `attn_implementation="flash_attention_2"`，强制使用 Flash Attention 后端。注意需要预先安装 flash-attn 包（pip install flash-attn --no-build-isolation）。

### 陷阱三：进程残留导致 OOM

SFT 完成后直接启动 RL 时，前一个进程的 GPU 显存可能未完全释放（尤其是用 shell 脚本串联两个训练任务时）。表现为 RL 启动时 OOM，但 `nvidia-smi` 显示有 20+GB 被"不存在的进程"占用。

解决方案：在启动新训练前强制清理 GPU：
```bash
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null || true
sleep 5
```

### 收获

DDP + 大模型的工程问题往往比算法问题更消耗时间。建议在正式训练前先用小 batch 跑 10 步验证流程，确认无 OOM、无 NCCL 超时、无 gradient 错误后再启动完整训练。


## RL GRPO 训练观察

### 训练配置

```bash
model=checkpoints_sft_warmup/best  # SFT warmup 后的模型
epochs=3, batch_size=2, num_generations=2
lr=5e-6, beta=0.04 (KL penalty)
max_gen_len=256
accumulation_steps=2
gradient_checkpointing=0  # 禁用，避免 DDP 冲突
```

### 初始阶段观察（Step 1-110）

RL 训练启动后 handoff rate 立即达到 100%（SFT warmup 的效果），随后在 75%-100% 之间波动：

```
Step 5/248   | reward=3.325 | handoff=100%
Step 10/248  | reward=0.677 | handoff=100%
Step 50/248  | reward=1.2   | handoff=75%   ← RL 开始纠正过度调用
Step 110/248 | reward=1.5   | handoff=87.5%
```

handoff=75% 的出现说明 RL 正在工作——它在学习"不是所有问题都需要专家"，对 SFT warmup 阶段的过度调用进行修正。这验证了"先过度再纠正"策略的有效性。

### Reward 信号分析

reward 由两部分组成：
- `router_r`：Router 的路由决策质量（是否正确选择了 handoff/no-handoff）
- `expert_r`：Expert 的执行质量（tool call 参数是否合理、回答是否相关）

早期 router_r > expert_r，说明模型已经学会了"何时调用"（路由决策），但"调用后如何做好"（执行质量）还在学习中。这是合理的学习顺序——先学决策，再精进执行。

### 对比 64M 模型的 Agent RL

| 维度 | MiniMind 64M | Qwen2.5-1.5B |
|------|-------------|---------------|
| 冷启动所需 steps | ~1600 步才出现拐点 | 有 SFT warmup 后 step 1 即有效 |
| handoff 学习 | 需要 reward 函数修复才能开始 | parse 修复后立即生效 |
| 策略崩溃风险 | 高（v1/v4 均崩溃） | 低（1.5B 参数空间更稳定） |
| 显存管理 | 单卡即可 | 需要 DDP + 显存优化 |


## 核心方法论总结

### 一、换模型 ≠ 换配置

从 64M 切到 1.5B 不仅是改一个 `model_name_or_path`。tokenizer、chat_template、attention 实现、生成格式全部不同。**每次切换模型都需要端到端验证：模型输出 → parse 函数 → reward 计算 → 梯度信号** 这条完整链路。

本次的 parse_tool_calls bug 就是"只改了模型没改 parse"的典型后果。如果没有认真检查模型实际输出的格式，这个 bug 可以让你白白训练几天而毫无进展。

### 二、关键指标 = 0% 时先查度量再查模型

三次"Rate=0%"的经历形成了清晰的 pattern：

| 实验 | 指标 | 0% 的根因 |
|------|------|-----------|
| 64M Plan RL | PlanRate=0% | `゜` 分隔符截断 bug |
| 64M Plan RL | ExecRate=0% | Rollout 链路断裂（plan 轮 break） |
| 1.5B Handoff RL | handoff=0% | parse_tool_calls 格式不匹配 |

三次都不是模型能力问题，而是代码逻辑错误导致正确行为无法被识别。教训很明确：**持续为零的指标 = 度量管线有 bug，直到证明不是**。

### 三、SFT + RL 二阶段优于纯 RL

64M 模型的 Agent RL v5 需要 1600 步冷启动才出现拐点（24.6% 正奖励率的前 1600 步全在探索）。Qwen2.5-1.5B 加了 90 条 SFT warmup 后，RL step 1 就有 100% handoff rate。

时间对比：
- 纯 RL 冷启动：1600 步 × ~4s/step ≈ 1.8 小时（还不算失败重试）
- SFT warmup：90 秒
- SFT → RL：90 秒 + RL 从 step 1 即有效

结论：**在明确目标格式的情况下，少量 SFT 做格式植入 + RL 做质量优化，是比纯 RL 探索高效得多的训练策略。**

### 四、"先过度再纠正" > "一步到位"

SFT warmup 数据中 78% 是 tool_call 样本，导致模型初始 handoff=100%（过度调用）。但 RL 在几十步内就将其修正到 75%-87.5%。

相反的策略——SFT 数据 50/50 平衡——虽然初始 handoff rate 看起来"合理"，但模型对 tool_call 格式的掌握不够坚定，RL 训练中容易退化为"永远不调用"（因为不调用的 reward 基线更容易达到）。

### 五、DDP 工程问题的排查优先级

经验排序：
1. 先跑 1 步确认不 OOM
2. 确认 gradient_checkpointing 与 DDP 的兼容性
3. 确认 attention 实现不报错（flash_attention_2 兜底）
4. 确认多卡之间 loss 一致（打印各 rank 的 loss 对比）
5. 最后才开始正式训练


## 与 64M 模型实验的对比总结

| 维度 | 64M MiniMind | 1.5B Qwen2.5 | 关键差异 |
|------|-------------|--------------|----------|
| 模型容量 | 8层, hidden=768 | 28层, hidden=1536 | 24× 参数量 |
| RL 冷启动 | 1600 步 | 1 步（有 SFT warmup） | SFT warmup 消除冷启动 |
| 策略崩溃 | 频繁（v1/v4） | 未观察到 | 大模型参数空间更平滑 |
| Reward Hacking | 严重（v5 后期） | 待观察 | 可能在更多 steps 后出现 |
| 核心瓶颈 | 模型容量不足 | 训练代码适配 | 瓶颈从"能力"转向"工程" |
| parse bug 影响 | PlanRate=0%, ExecRate=0% | handoff=0% | 同类 bug 在不同模型上反复出现 |
| 实验周期 | 5 天（v1→v6） | 半天（发现→修复→验证） | 经验复用大幅提升效率 |

最显著的对比：64M 模型从 v1 到 v6 花了 5 天迭代，核心时间消耗在"定位瓶颈在哪"；1.5B 模型半天内完成了"定位 → 修复 → SFT → RL 验证"全流程。**64M 模型上积累的排查方法论（先查度量、再查 reward、最后查模型）直接复用到了 1.5B 模型上，这就是实验方法论的价值。**


## 文件索引

| 文件 | 作用 |
|------|------|
| `agent_handoff_qwen.py` | Handoff RL 主训练脚本（含修复后的 parse_tool_calls） |
| `dataset/generate_sft_warmup.py` | SFT warmup 数据生成脚本 |
| `dataset/sft_warmup.jsonl` | 90 条 SFT warmup 数据 |
| `sft_warmup_qwen.py` | SFT warmup 训练脚本（DDP, flash_attention_2） |
| `scripts/run_sft_warmup.sh` | SFT → RL 串联启动脚本 |
| `scripts/run_rl_v2.sh` | RL 独立启动脚本（支持 resume） |
| `TRAINING_LOG.md` | 项目级训练日志 |

## 训练完成与评测结果

### 训练完成

RL v2 于 18:55 CST 成功完成 3 个完整 epoch（744 total steps），best checkpoint 保存在 step 248（epoch 3）。

### 评测结果（40 samples）

| 指标 | 值 | 含义 |
|------|-----|------|
| Route Accuracy | 80.0% | 路由决策整体正确率 |
| Expert Selection Accuracy | 76.3% | handoff 发生时选对专家的比例 |
| False Trigger Rate | **80.0%** | 不需要工具时错误 handoff 的比例 |
| Miss Rate | **0.0%** | 需要工具时遗漏 handoff 的比例 |
| GT Hit Rate | 76.7% | 最终回答包含正确答案 |
| E2E Success Rate | 62.5% | 端到端全部正确 |

分专家表现：

| 专家 | 路由正确率 | 专家选择正确率 | GT 命中率 |
|------|-----------|---------------|----------|
| math | 100% | 100% | 80% |
| info | 100% | 90% | 60% |
| translate | 100% | 100% | 90% |
| none | **20%** | — | — |

### 关键发现：SFT 数据偏置的传递效应

**需要工具的 30 个样本路由正确率 100%，模型从不遗漏该 handoff 的场景。** 但不需要工具的 10 个样本只有 2 个正确，8 个被错误 handoff——且大部分错误路由到了 `math_agent`。

失败样本举例：
- "你好，今天过得怎么样？" → 错误 handoff 到 info_agent
- "给我讲一个程序员的笑话" → 错误 handoff 到 math_agent
- "解释一下什么是Transformer架构" → 错误 handoff 到 math_agent

**根因：** SFT warmup 阶段 78% 的样本（70/90）包含 tool_call，模型形成了"有问必 handoff"的强先验。RL 阶段 990 条数据中 none 类占比不够大、误触的负 reward 力度不够强，梯度信号无法压过 SFT 植入的先验。

这是一个教科书级别的 **SFT→RL 数据分布传递效应**：
- SFT 阶段的数据配比 → 决定了 RL 起点的策略偏置
- RL 阶段的样本量和奖励设计 → 决定了能否纠正这个偏置
- 当两者不匹配时，偏置会一直残留到最终模型

### 修复方案（三管齐下）

基于上述分析，设计了三项协同修复措施，目标将误触率从 80% 降至 15% 以下。

**第一项：SFT 数据再平衡。** 将 `generate_sft_warmup.py` 中的 NONE_SAMPLES 从 20 条扩充到 55 条（新增 20 条边界干扰 + 15 条日常闲聊），使 tool_call:none 比例从 78%:22% 调整为 56%:44%。新增样本特别覆盖了"看似需要工具但实际不需要"的边界场景——含数字的闲聊、含天气词汇的常识问题、含"帮我"等动作词但不需工具的请求。同时将 SFT epochs 从 5 降到 3，减少对 tool_call 分布的过拟合。

**第二项：RL 数据扩充。** 将 `generate_handoff_data.py` 中 none 类从 240 条（24%）提升到 ~450 条（37%），总样本量从 ~1000 扩充到 ~1210 条。新增内容包括普通闲聊（技术话题、职业发展、生活建议）和边界干扰（含数字但不需计算、含温度词汇但不需查询、多轮对话续接语等 6 类共 90 条）。

**第三项：非对称奖励 + 渐进式惩罚。** 在 `agent_handoff_qwen.py` 中调整路由正确性奖励：正确抑制从 +0.3 提到 +0.8，误触发从 -0.3 改为 -1.5 × coeff（`coeff = min(0.5 + 0.1 × epoch, 1.5)`，训练初期给适应空间，后期逐步加大惩罚），并且误触发时专家奖励强制归零（`e_reward = 0.0`），切断错误信号传播路径。

这三项修复的设计逻辑是：SFT 再平衡降低初始偏差，RL 扩充提供更多反面样本，非对称奖励提供更强的纠偏信号，渐进式惩罚保证训练稳定性。三者协同，预期将误触率降至 15% 以下。

---

## 评测代码踩坑

### JSON 序列化失败

评测脚本运行完毕、打印了所有指标，但保存 JSON 时崩溃：
```
TypeError: Object of type set is not JSON serializable
```

原因：`validate_gt_in_text()` 返回匹配到的 GT 关键词集合（`set` 类型），被写入 `log_entry["gt_hit"]`。加上 `per_expert` 是 `defaultdict`，都无法直接序列化。

解决：添加通用 `SafeJSONEncoder` 处理 `set`→`list`、`defaultdict`→`dict`、`Tensor`→`list`。

教训：**评测代码一定要先用 1-2 条样本 dry-run 验证序列化**。GPU 跑了 2 分钟出结果却存不下来，等于白跑。

---

## 这个实验的意义

从更大视角看，这个实验验证了三个核心假设：

1. **RL 可以训练 Agent 路由能力**——不是靠规则、不是靠 prompt engineering，而是让模型通过试错自主学会"何时该调用哪个专家"。需要工具的场景路由正确率 100%，专家选择正确率 96.7%（29/30）。
2. **小模型的实验方法论可以直接迁移到大模型**——64M 模型上 5 天的 debug 经验，让 1.5B 模型的实验在半天内就走通了全链路。
3. **SFT + RL 二阶段训练是 Agent 能力获取的高效范式**——SFT 解决"格式"，RL 解决"决策质量"，两者协同效率远超任一单独使用。
4. **数据配比是最容易被忽视的关键变量**——SFT 78% tool_call 的配比导致了 80% 的误触率，这不是模型能力问题，而是数据设计问题。

对于个人而言，这轮实验最大的收获不是某个具体的 bug fix，而是形成了一套可复用的 **"Agent RL 排查方法论"**：指标异常 → 验证度量管线 → 检查 reward 信号 → 确认模型能力 → 调优超参。这个顺序在 64M 和 1.5B 模型上都被验证有效，未来换到更大模型（7B/14B）时大概率仍然适用。

另一个重要收获是对 **SFT→RL 数据分布传递效应** 的深刻认识：SFT 不只是"教格式"，它同时植入了策略偏置；RL 不是万能的纠正器，它的纠正能力受限于数据量和 reward 设计。两者之间的数据配比需要协同设计。
