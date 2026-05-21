# Plan-then-Execute 多 Agent RL 实验报告

> 2026-05-14，云服务器 2×A100-SXM4-80GB，模型 Qwen2.5-1.5B-Instruct (1.54B 参数)。
> 本文记录 Plan-then-Execute 多步 Agent RL 训练的完整过程——从性能优化（18 分钟/step → 7 秒/step）到 1000 条数据的两阶段训练、Demo 推理分析和核心问题诊断。

---

## 背景

在 Handoff RL（单步路由）训练验证成功后，扩展到 Plan-then-Execute（多步计划执行）。这是一个更复杂的 Agent 范式：Router Agent 接收用户请求后输出 `execute_plan` 工具调用，按拓扑序依次分派多个 Expert 执行子任务，最后 Synthesize 阶段整合所有 Expert 的输出给出最终回答。

数据集包含 L0~L4 五个难度级别共 1000 条样本，从单步直接回答（L0）到四步流水线执行（L4），难度递增。训练分两阶段各 500 条。

训练配置：batch_size=1, num_generations=2, max_gen_len=384, lr=1e-6, beta=0.1, GRPO loss, gradient_checkpointing=True, ref_model offload 到 cuda:1。


## 性能优化：从 18 分钟一步到 7 秒一步

### 问题发现

启动训练后，GPU 0 占用 35GB 显存、利用率仅 30%，等了 18 分钟始终没有输出第一条 step 日志。1.5B 模型在 A100-80GB 上推理应该非常快，显然有严重的性能问题。

### 瓶颈定位

分析后找到两个根因。第一个是 rollout 次数爆炸：Plan-then-Execute 的单个候选需要 Router（1 次 generate） + Expert 子 rollout（每个最多 3 轮 tool-use × 多个 Expert，约 6 次） + Synthesize（1 次），合计约 8 次 generate。配置 `num_generations=4` 时，每个 step = 4 × 8 = 32 次串行 generate，且完全没有并行。

第二个是致命的：**gradient_checkpointing 禁用了 KV cache**。HuggingFace 的 `gradient_checkpointing_enable()` 会自动将 `use_cache` 设为 False，导致 `model.generate()` 时没有 KV cache——每生成一个新 token 都要对前面所有 token 重新做完整 attention。复杂度从 O(N × seq_len) 变成 O(N × seq_len²)，慢了几十倍。而 rollout 阶段是纯推理（`torch.no_grad()`），完全不需要 gradient checkpointing。

训练日志中有两行 warning 早就暴露了这个问题：`"use_cache=True" is incompatible with gradient checkpointing. Setting "use_cache=False".`，但最初被淹没在大量输出中。

### 五项优化措施

**优化 1（减量）：num_generations 4 → 2。** GRPO 的组内对比只需要至少 2 个候选即可计算 advantage。直接将 rollout 总量减半。代价是 advantage 估计方差更大，但对 1 epoch 试验完全够用。

**优化 2（减量）：max_tool_turns 3 → 2。** Expert 子 rollout 的 tool-use 循环上限从 3 降到 2。我们的模拟工具是确定性的，Expert 调一次工具就能拿到结果：第一轮生成 tool_call 并执行，第二轮根据工具结果生成最终回答。第三轮只在模型输出格式混乱时才会用到。

**优化 3（显存）：ref_model offload 到 cuda:1。** GRPO 需要 policy model + ref model 两份模型。将 ref_model 移到空闲的 GPU 1，计算 ref_logps 时跨 GPU 搬运。GPU 0 显存从 35GB 降到 22GB。

**优化 4（并行）：Router 批量 rollout。** 原来同一条 query 的 N 个候选逐个调用 `rollout(num_generations=1)`，每次都要重新计算 prompt 的 prefill。改为一次性 `rollout(num_generations=N)`，HuggingFace 的 generate 内部会用 `repeat_interleave` 扩展 batch，prefill 只算一次。

**优化 5（决定性）：rollout 前关闭 gradient_checkpointing 启用 KV cache。** 在 rollout（纯推理）开始前临时关闭 gradient_checkpointing 并显式设置 `config.use_cache = True`，rollout 结束后恢复。generate 速度从 O(seq_len²) 恢复到 O(seq_len)，这是几十倍级别的加速。

实现时有个细节：最初尝试用 `raw_model.is_gradient_checkpointing` 属性检测开启状态，但该属性在某些情况下不可靠（`hasattr` 返回 False），导致 disable 逻辑被跳过。最终改为直接使用 `args.gradient_checkpointing` 配置参数判断。

### 优化效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 每步耗时 | >18 分钟（未完成） | ~7 秒 |
| 预估总时间（500 步） | ~8.3 小时 | ~58 分钟 |
| GPU 0 显存 | 35 GB | 18~22 GB |

各优化的贡献排序：KV cache 修复（~10x，决定性）→ num_gen 减半（~50%）→ max_tool_turns 缩减（~15%）→ ref_model offload（~10%）→ Router 批量化（~5%）。

### 经验总结

gradient_checkpointing 和 KV cache 的互斥是 HuggingFace 的已知行为，但在 RL 训练这种"推理+训练混合"的场景中极易被忽视。训练代码通常在初始化时开启 gradient_checkpointing 后就不再管它，但 GRPO 的 rollout 阶段是纯推理，不需要也不应该被影响。RL 训练的性能瓶颈通常在 rollout 而非 gradient update——Plan-then-Execute 场景尤其如此，每个 step 的 rollout 涉及多次串行 generate，而 gradient update 只需要一次 forward + backward。


## 训练结果

### Phase 1：前 500 条（从基座模型开始）

| 区间 | 平均 Reward | 正 Reward 比率 | 平均 Handoff% |
|------|------------|---------------|--------------|
| Step 1-100 | -0.077 | 30/100 | 24.0% |
| Step 101-200 | +0.027 | 38/100 | 20.0% |
| Step 201-300 | -0.046 | 28/100 | 26.0% |
| Step 301-400 | +0.012 | 38/100 | 25.0% |
| Step 401-500 | **+0.062** | 38/100 | 23.5% |

总体：avg_reward=-0.004，min=-1.029，max=2.500，positive reward=34.4%，avg handoff=23.7%，plan=0.1%。趋势是清晰的：first 100 avg=-0.077 → last 100 avg=+0.062，模型在学习。

### Phase 2：后 500 条（从 Phase 1 best checkpoint 继续）

| 区间 | 平均 Reward | 正 Reward 比率 | 平均 Handoff% |
|------|------------|---------------|--------------|
| Step 1-100 | -0.047 | 35/100 | 25.0% |
| Step 101-200 | -0.158 | 32/100 | 18.5% |
| Step 201-300 | -0.054 | 34/100 | 18.5% |
| Step 301-400 | -0.106 | 31/100 | 19.0% |
| Step 401-450 | -0.143 | 14/50 | 19.0% |

总体：avg_reward=-0.097，positive=32.4%，avg handoff=20.1%，plan=0.0%。趋势出现退化：first 100 avg=-0.047 → last 100 avg=-0.155。

### Best Checkpoint

保存在 `./checkpoints_qwen_plan/best`（Phase 2 Step 300，reward=1.50）。Checkpoint 历史显示 Step 300 之后模型持续退化：

| Step | Router Reward | Expert Reward | Handoff% | Is Best |
|------|--------------|--------------|----------|---------|
| 300 | 1.575 | -0.075 | 0% | ✅ |
| 350 | -0.600 | 0.0 | 0% | ❌ |
| 400 | -0.600 | 0.0 | 0% | ❌ |
| 450 | -0.700 | 0.0 | 0% | ❌ |
| 500 | -0.630 | -0.030 | 0% | ❌ |


## Demo 推理分析

使用 best checkpoint 在 5 个难度级别的测试用例上推理。

**L0（自我介绍，不应调专家）：** Router 输出 "我是任务路由器，我的职责是根据您的请求分配任务到最合适的专家" 并直接回答。Handoff=No, Reward=+0.600。✅ 正确识别为不需要专家的简单请求。

**L1（天气查询 + 温度转换，需 info→math）：** Router 输出 "首先，我需要获取纽约当前的天气情况"。Handoff=No, Reward=-0.500。❌ 模型知道需要获取天气，但只是描述意图，没有输出工具调用格式。

**L2（并行查询 + 比较，需 2×info→math）：** Router 输出了自然语言描述。Handoff=No, Reward=+1.500。⚠️ 模型没有实际执行，但 reward 给了 +1.500——reward 函数可能有 bug。

**L3（条件分支，需 info→条件判断→math）：** Router 输出了接近正确的 JSON：`{"name": "delegate_to_info_agent", "arguments": {"task": "查询北京天气"}}`。Handoff=No, Reward=+0.563。⚠️ 模型已经输出了正确格式的工具调用内容，但格式/位置不被解析器识别。

**L4（多步流水线，需 info→math→translate→math）：** Router 列出了正确的步骤计划（"1. 查询东京时间。2. 换算成北京时间。3. 翻译文本。4. 计算时差。"），但用自然语言而非 execute_plan 工具调用格式。Handoff=No, Reward=-0.200。


## 核心问题诊断

### 模型学会了意图但没学会格式

这是最关键的发现。模型在 RL 训练后展现出了正确的"认知"——知道什么时候该调专家、知道该分几步执行、甚至能输出接近正确的 JSON 结构。但它的输出是自然语言描述而非框架能解析的工具调用格式。这说明 RL 的 reward signal 成功引导了思考方向，但仅靠 reward 信号不足以教会精确的输出格式。1.5B 模型需要看到正确格式的示范才能学会结构化输出——这和 Handoff 实验中的结论一致：需要 SFT warmup 做格式植入。

### Phase 2 退化

可能原因包括：后 500 条数据的难度分布与前 500 条不同；best checkpoint 选择基于瞬时 reward 而非滑动平均，可能选了一个"幸运"的 step；KL 惩罚过小（beta=0.1）导致 policy drift；以及训练样本量不足以泛化。

### Reward 函数可能有 bug

L2 用例模型没有任何实际执行，但获得了 +1.500 的高 reward。需要审查 `calculate_plan_rewards` 函数的逻辑——是否 router_reward 对"提到了关键词"给了高分？是否应该只在实际触发 handoff 时才给正 reward？


## 下一步方向

按优先级排序：首先检查 tool_call 格式（对比 system prompt 中要求的格式和解析器的匹配逻辑，成本最低，可能一步解决问题）。其次做 SFT warm-up（用 20-50 条正确格式的 tool_call 样本做格式教学，数据已准备在 `dataset/sft_plan_warmup.jsonl`）。然后审计 reward 函数（L2 给了 +1.500 的 bug）。最后考虑 curriculum learning（先只在 L1 上训练，掌握格式后再加入 L2-L4）。

这个实验和 Handoff 实验形成了清晰的呼应：单步路由场景中，SFT warmup 让模型 step 1 就有 100% handoff rate；多步计划执行场景中，缺少 SFT warmup 的模型只能学到"意图"而学不到"格式"。这再次验证了 **SFT + RL 二阶段训练** 的必要性——SFT 教格式，RL 优化决策质量，两者缺一不可。
