# Handoff RL 训练踩坑录

> 记录具有泛用性的失误和解决方案，未来换模型/换任务时可直接参考。

---

## 一、换模型后度量管线失效 → 关键指标恒为零

**场景：** 从 64M MiniMind 切换到 Qwen2.5-1.5B，RL 训练中 reward 正常上升，但 handoff rate 始终为 0%。

**根因：** 训练代码中的 `parse_tool_calls` 用的是旧模型的输出格式（自定义分隔符 `ゅ...゜`），而 Qwen2.5 的 chat_template 指导模型输出 `<tool_call>...</tool_call>` XML 格式。**模型已经学会了正确行为，但度量管线无法识别，导致正确行为得不到奖励。**

**泛化规律：**

- **关键指标恒为零时，先怀疑度量代码，再怀疑模型能力。** 本项目三次遇到 Rate=0% 的问题（64M PlanRate、64M ExecRate、1.5B handoff），根因全部是 parse/度量逻辑的 bug，而非模型不会生成目标输出。
- **换模型 ≠ 改配置文件。** 每次切换 base model 都必须端到端验证：模型实际输出格式 → parse 函数能否匹配 → reward 信号是否正确传导。少一环都可能白训几天。

**排查三步法：**
1. 让模型单独做推理，直接看原始输出文本
2. 把实际输出喂给 parse 函数，确认能否匹配
3. 确认匹配结果能正确触发 reward 计算

---

## 二、SFT 数据配比偏置 → RL 无法纠正

**场景：** SFT warmup 阶段用 78% 的 tool_call 样本教模型学格式，RL 训练后模型对需要工具的场景路由 100% 正确，但对不需要工具的场景误触率高达 80%。

**根因：** SFT 植入的不只是"格式"，还有"策略偏置"——模型从第一天就学到"大部分输入都需要 handoff"。RL 阶段数据量（990 条）和 none 类占比不足以翻转这个先验。

**泛化规律：**

- **SFT→RL 存在数据分布传递效应。** SFT 的类别配比直接决定了 RL 的策略起点。如果 SFT 给了 80% 的正例，RL 需要极大的负反馈才能把模型从"默认做"修正为"该做才做"。
- **"先过度再纠正"策略有前提条件**——RL 阶段必须有足够的反面样本和足够强的惩罚力度。否则过度的部分会一直残留。
- **非对称 miss/false-trigger 成本时，reward 也应非对称设计。** 如果漏触的代价和误触的代价不同，reward 函数需要反映这种差异，否则模型会向"安全侧"（本例中是过度 handoff）偏移。

**解决框架：**
1. SFT 数据配比尽量接近目标分布（而非极端偏斜）
2. RL 阶段确保少数类有足够样本量（≥30%）
3. 对高频错误类型加大惩罚权重

---

## 三、gradient_checkpointing 与 DDP 不兼容

**场景：** 开启 gradient_checkpointing 后 DDP 训练报错 `Expected to mark a variable ready only once`。

**根因：** DDP 要求所有参数在 backward 时都被标记为 ready（用于跨卡梯度同步），而 gradient_checkpointing 的某些实现会导致部分参数"看起来未被使用"，触发 DDP 的检查逻辑。

**泛化规律：**

- **省显存的技巧不能无脑叠加。** gradient_checkpointing、DDP、mixed precision、activation offloading 各自独立正确，但组合使用时可能互相冲突。
- **替代方案：** 减小 batch_size + 增大 gradient_accumulation_steps（等效 batch size 不变但峰值显存更低）。

---

## 四、评测代码未做序列化预检

**场景：** 评测跑了 2 分钟生成了所有结果，但保存 JSON 时崩溃（中间函数返回了 `set` 类型，不可 JSON 序列化）。GPU 时间白费。

**泛化规律：**

- **任何需要 GPU 的流程，先用 1-2 条样本 dry-run 整个 pipeline（包括保存环节）。** 确认端到端无报错后再跑全量。成本是几秒，收益是避免浪费几分钟到几小时的 GPU 时间。
- **通用防御：** 写一个 `SafeJSONEncoder`（处理 set→list、defaultdict→dict、Tensor→tolist），在所有实验代码的 json.dump 中默认使用。

---

## 五、reward 正常上升 ≠ 训练目标达成

**场景：** parse bug 存在时，reward 仍然从负值涨到 1.0+。看起来"训练在收敛"，实际上模型收敛到了完全错误的策略——"永不调用工具，直接回答拿基础分"。

**泛化规律：**

- **Reward 是 RL 训练中最危险的信号——它"看起来在涨"不代表目标在达成。** 必须同时监控任务层面的 behavioral metrics（本例中是 handoff rate）。
- **如果 reward 在涨但目标行为未出现，几乎一定是 reward function 有 bug 或 reward hacking。** 模型找到了一条"绕过目标行为也能拿分"的捷径。
- **设计 reward 时确保目标行为是获得高 reward 的必要条件，而不只是充分条件。**

---

## 评测最终结果

| 指标 | 值 |
|------|-----|
| Route Accuracy | 80.0% |
| Expert Selection Accuracy | 76.3% |
| False Trigger Rate | 80.0% |
| Miss Rate | 0.0% |
| GT Hit Rate | 76.7% |
| E2E Success Rate | 62.5% |

核心结论：模型完美学会了"何时需要工具"（miss=0%，需工具时专家选择准确率 96.7%），但未学会"何时不需要工具"——这是 SFT 数据偏置的残留效应，而非模型能力不足。

---

## Plan-then-Execute RL v2 训练结果

**训练配置：**
- 基座: Qwen2.5-1.5B-Instruct，SFT warmup → RL GRPO
- 数据: `plan_execute_500.jsonl`（500 条多步任务），覆盖 5 种依赖类型
- 步数: 500 steps（从 SFT warmup checkpoint 继续）
- GPU: 2x A100-SXM4-80GB（双卡）

**训练过程关键观察：**
- handoff=100%、plan=100% 从 step 1 起即稳定（SFT warmup 的效果），全程仅有约 6 次单样本 plan=0% 抖动
- reward 均值 ~2.0，高分样本可达 3.8-4.5
- loss 快速下降至 <0.05，策略在 ~200 步后基本收敛
- KL 始终接近 0，模型未偏离参考策略

**数据分布（part2 测试集，500 条）：**
- dependency_type: sequential(238), parallel(144), conditional(71), mixed(16), single(31)
- experts: info(430), math(354), translate(204)
- num_steps: 0~4 步

**Checkpoint 保存：** 每 50 步存一次，保留最近 5 个 + best

**评测结果（200 条 = 160 工具调用 + 40 无工具，part2 测试集，模型未见过）：**

| 指标 | 结果 | 说明 |
|------|------|------|
| Handoff Accuracy | 97.5%（195/200） | 是否正确判断要不要用工具 |
| Plan Accuracy | 97.5%（195/200） | 多步任务是否正确使用 execute_plan |
| Expert Match | 91.0%（182/200） | 调用的专家是否与 GT 一致 |
| GT Hit | 72.0%（144/200） | 最终输出是否包含正确答案 |
| E2E Success | 69.0%（138/200） | 以上全部正确 |
| False Trigger Rate | 12.5%（5/40） | 不需要工具时错误发起调用 |
| Miss Rate | 0%（0/160） | 需要工具时未发起调用 |

按依赖类型细分：

| 依赖类型 | n | Handoff | Expert | GT Hit | E2E |
|----------|---|---------|--------|--------|-----|
| sequential | 77 | 100% | 100% | 68.8% | 68.8% |
| parallel | 46 | 100% | 97.8% | 87.0% | 87.0% |
| none（无工具） | 40 | 87.5% | 87.5% | 87.5% | 87.5% |
| single | 9 | 100% | 100% | 55.6% | 55.6% |
| mixed | 5 | 100% | 100% | 80.0% | 80.0% |
| conditional | 23 | 100% | 47.8% | 30.4% | 4.3% |

**数据集场景分布（训练集 500 条）：**

| 维度 | 分布 |
|------|------|
| 依赖类型 | sequential ~48%, parallel ~29%, conditional ~14%, single ~6%, mixed ~3% |
| 专家类型 | info（信息查询）、math（数学计算）、translate（翻译），大部分任务需要 2-3 种专家协作 |
| 步骤数 | 1步 ~5%, 2步 ~57%, 3步 ~36%, 4步 ~2% |
| 总数据量 | 训练 500 条（part1），测试 500 条（part2），共 1000 条 |

**测试集采样策略：** 从 part2 按 dependency_type 分层采样 160 条工具调用样本 + 生成 40 条不需要工具的纯问答样本（覆盖深度学习、NLP、编程等常识问题），共 200 条。

**结论：**
1. **少量数据高效训练** — 仅 500 条训练样本 + SFT warmup + RL GRPO，1.5B 模型即获得稳定的多步规划能力
2. **路由判断近乎完美** — 97.5% 准确率，漏触 0%，误触 12.5%（vs v1 的 80%，大幅改善）
3. **顺序/并行场景表现优秀** — E2E 69-87%，主流任务类型可靠
4. **条件分支是明确瓶颈** — conditional E2E 仅 4.3%，根因是模型缺乏根据中间结果动态调整后续步骤的能力，需要更强的 in-context reasoning
5. **GT Hit 仍有提升空间** — 部分 case 规划和专家都对了但最终答案丢信息，瓶颈在 Synthesize 整合阶段

**相比 Handoff v1 的改进：**

1. **SFT warmup 数据配比修正** — v1 的 SFT 数据 78% 是 tool_call 样本导致误触率 80%；v2 使用 `sft_plan_warmup.jsonl`，100% 多步任务（因为 plan 数据集本身全是需要规划的场景），避免了分布偏置问题
2. **从单步 Handoff 升级到多步 Plan-then-Execute** — Router 不再只是选专家，而是输出完整的多步执行计划（step/delegate/task/depends_on），按拓扑序执行
3. **分步 Reward 设计** — 奖励拆解为 router_reward（规划结构正确性）和 expert_reward（分步执行质量），分别计算后加权组合，避免 v1 中"reward 涨但行为错"的问题
4. **SFT warmup 必要性验证** — v2 Phase 1 实验（无 SFT 直接 RL）证明 plan=0% 持续 1000 步无改善；Phase 2（加 SFT warmup）从 step 1 起 plan=100%，确认了 SFT warmup 对格式学习的必要性
5. **评测脚本工程化** — 新增 `eval_plan.py`，支持分层采样（覆盖所有依赖类型）、按场景细分统计、SafeJSONEncoder 防序列化崩溃

**评测环境注意：** 服务器需使用 `/home/vipuser/miniconda3/bin/python`（系统 python3 缺少 typing_extensions）

---

## 服务器信息
- 地址: `root@js3.blockelite.cn -p 10212`（密码: Eek4feiC）
- GPU: 2x A100-SXM4-80GB
- 项目路径: `/root/minimind`
- Python: `/home/vipuser/miniconda3/bin/python`（PyTorch 2.5.0+cu124）
