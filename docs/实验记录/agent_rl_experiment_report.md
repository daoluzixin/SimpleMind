# MiniMind Agent RL 实验报告

> 2026-05-09 ~ 2026-05-12，云服务器 RTX 4090 24GB，模型 64M 参数（hidden_size=768, 8 层）。
> 本文记录了从预训练 → SFT → Agent RL（GRPO）→ Plan RL 全链路中，RL 阶段遇到的所有问题、调参过程和最终定位到的根因。

---

## 背景

MiniMind 的 Agent RL 训练目标是让模型学会工具调用：识别何时需要调用工具、生成正确的函数名和参数、理解返回结果并给出最终回答。训练框架基于 GRPO（Group Relative Policy Optimization），使用 ClSPO 变体做策略更新，配合 InternLM2-1.8B 作为 Reward Model 对无工具调用的回复打分。

训练流程分两阶段设计：先用 `train_agent.py`（纯工具调用，不要求 plan 格式）解决冷启动，再切到 `train_plan.py`（要求 `<plan>` 标签的 Plan-then-Execute 范式）。这个设计来自一个很具体的教训——直接跑 `train_plan.py` 会发现 PlanRate 永远是 0%，因为 SFT 阶段的数据里没有 plan 格式的示范，模型从未见过 `<plan>` 标签，自然不会生成。


## 原始数据集的质量问题

最初使用的 `agent_rl.jsonl` 有 39988 条样本，50/50 分成工具调用和纯对话。跑了几百步后 GrpStd 几乎全是 0，Reward 毫无区分度。逐条看了数据才发现，纯对话部分充斥着大量低质量闲聊（"你好""讲个笑话"），Reward Model 对这些回复打分差异极小，导致同一 prompt 的 4 个生成拿到几乎相同的分数，组内标准差为零，advantage 信号完全消失。

另外工具种类只有 6 种（calculate_math、unit_converter、get_current_weather 等），分布过于均匀，场景单一。

决定换数据集。在 HuggingFace 上选了 `Salesforce/xlam-function-calling-60k`，59578 条纯工具调用样本，覆盖 3603 种不同的 API 工具，每条都有 Ground Truth 标注。


## 调参过程：三个版本的学习率实验

### v1: lr=5e-6, beta=0.1

这是最初的配置。前 100 步看起来还行，但到 200 步以后 KL 散度一路暴跌到 -5.5，AvgLen 多次归零（step 340/350/370/420），模型输出坍缩成空字符串。经典的策略崩溃：学习率太高，KL 惩罚太弱，模型更新太激进，直接偏离参考策略到无法回头的程度。

在 step 540 处手动停掉。

### v2: lr=1e-6, beta=0.3

降了 5 倍学习率，KL 惩罚系数翻了 3 倍。结果训练确实稳定了——KL 在 -0.09 ~ +0.09 之间小幅波动，AvgLen 始终正常（34~229），没有任何崩溃迹象。

但问题反过来了：650 步过去，Reward 均值纹丝不动，一直在 -1.5 附近随机震荡。模型太稳定了，策略几乎没有更新，什么都没学到。

启动时还遇到了 OOM。默认 batch_size=2 导致某些长样本撑爆 24G 显存，改成 batch_size=1 后解决。这也解释了为什么 v1 能跑——v1 的 total steps 是 59578（= 样本数 / 1），说明当时实际是 batch_size=1。

### v3: lr=3e-6, beta=0.2

取了 v1 和 v2 的中间值。前 60 步表现和 v2 类似，但到 150 步以后 KL 开始加速走负（出现 -0.31、-0.38、-0.43），和 v1 早期的模式如出一辙，只是因为 beta=0.2 的约束暂时压住了没有立刻崩溃。

236 步跑下来，Reward 依然没有上升趋势，均值还是 -1.5 左右。偶尔出现的正 Reward（step 152 的 0.17、step 190 的 0.82）是随机噪声而非系统性改善。

到这里意识到，**不管怎么调学习率，Reward 都上不去，问题不在超参上**。


## 真正的根因：Reward 函数和数据集的结构性 mismatch

深入分析 `train_agent.py` 的 reward 计算逻辑后，找到了问题的根源。

`calculate_rewards()` 对有工具调用的回复给分的核心逻辑是：先检查模型调用的工具名是否在该样本的合法工具列表中（工具对齐分，±0.5），再通过 `validate_gt_in_text()` 检查最终回答是否包含 GT 答案（GT 验证分，最高 2.5）。GT 验证分是大头，拿不到就必定负分。

工具执行使用 mock 实现。代码中硬编码了 6 个工具的 mock（calculate_math、get_current_weather 等），对于不在注册表中的工具走 generic mock——把传入的参数值原样返回作为"执行结果"。

xlam 数据集有 3603 种工具，其中只有 2 种（get_exchange_rate、get_current_weather）和代码注册的 6 种重叠。也就是说 **99.94% 的样本走 generic mock**。

关键矛盾在于 GT 的语义。看几个实际样本：

| 样本 | 工具 | GT |
|------|------|----|
| #0 | live_giveaways_by_type | ["beta", "game"] |
| #1 | web_chain_details | ["ethereum"] |
| #2 | t3ma | ["ETH/BTC", "1h", "14"] |

GT 是工具调用的**输入参数值**，不是工具的返回值。reward 函数期望的流程是：模型正确传参 → generic mock 把参数回传 → 模型在最终回答中提到这些值 → validate_gt_in_text 匹配成功。

但对于一个 64M 参数、刚从 SFT 冷启动的小模型来说，它需要同时做对三件事：调对工具名、从用户问题中提取正确参数值并以正确 JSON 格式传入、拿到 mock 结果后在最终回答中复述这些值。三步全对才能拿到 2.5 分的 GT 奖励，任何一步出错就只有负分。这对冷启动阶段的模型来说门槛太高了，正反馈信号极度稀疏，模型在"探索"中几乎永远拿不到正向激励。

这就是为什么三个版本的 Reward 均值都卡在 -1.5——模型偶尔能格式正确地发起工具调用（拿到 0.5 的对齐分），但几乎永远拿不到 GT 验证分。


## 各版本训练指标对比（v1~v3）

| 版本 | 学习率 | beta | 运行步数 | KL 范围 | Reward 均值 | AvgLen=0 | 结果 |
|------|--------|------|----------|---------|-------------|----------|------|
| v1 | 5e-6 | 0.1 | 540 | -0.5 ~ -5.5 | -1.5 | 频繁出现 | 策略崩溃 |
| v2 | 1e-6 | 0.3 | 650 | -0.09 ~ +0.09 | -1.5 | 无 | 学不动 |
| v3 | 3e-6 | 0.2 | 236 | -0.43 ~ +0.09 | -1.5 | 无 | KL 加速走负 |

三个版本殊途同归，Reward 均值都在 -1.5，说明瓶颈不在优化器那一侧。


## Reward 函数修复与验证

根据上面的分析，对 `calculate_rewards()` 做了两处修改。

第一处是 GT 验证的检查对象：把 `validate_gt_in_text(final_text, gt)` 改成同时在工具调用参数（tool call arguments）和最终回答文本中搜索 GT，取并集。具体做法是在遍历 tool_calls 时收集所有参数值，拼接成一个字符串后调用 validate_gt_in_text。这样模型只要传对了参数就能拿到 GT 奖励分，不需要在最终回答中复述参数值。

第二处是工具对齐分权重：从 0.5 提升到 1.0，让冷启动阶段"学会正确调用格式"这件事本身就能获得更强的正反馈信号。

### v4: lr=3e-6, beta=0.2（修复后的 reward 函数）

用 v3 的超参搭配修复后的 reward 函数做了第一轮验证。前 200 步效果显著：step 37 出现了 Reward=+0.12，step 51 出现 +0.20，step 108 达到 +0.70——这是 v1/v2/v3 从未出现过的正奖励，说明 reward 函数的修复方向完全正确。

但从 step 232 开始，频繁出现 AvgLen=1、KL 暴跌到 -6 ~ -10 的异常步，和 v1 的策略崩塌模式一致。415 步中有 40 步出现 AvgLen=1（约 9.6%），且后期密度加速上升。结论是 lr=3e-6 对修复后的 reward landscape 仍然偏大，更陡峭的梯度信号放大了策略更新的不稳定性。

### v5: lr=1e-6, beta=0.2（修复后的 reward 函数）

用 v2 的保守学习率搭配修复后的 reward。v5 共跑了 2999 步后手动停止，是五个版本中运行最久、信息量最大的一组实验。整体正奖励率 24.6%（739/2999），但训练过程明确分为三个阶段。

**阶段一：冷启动探索期（step 1-1600）**。正奖励率在 2-5% 的低位缓慢爬升，KL 稳定在 -0.04 ~ -0.18，AvgLen 在 48-86 之间正常波动。模型在大量负反馈中缓慢摸索工具调用的基本模式。与 v2（同学习率、旧 reward、0 次正奖励/650 步）形成决定性对比，证明 reward 函数修复有效。

**阶段二：有效学习期（step 1600-2400）**。step 1600 出现明显拐点，正奖励率跃升至 10.5%，随后持续攀升到 37-47%。这是真正有价值的训练窗口——AvgLen 同步上升到 95-137（说明模型输出变得更丰富，在尝试更完整的工具调用），KL 温和增长到 -0.30 ~ -0.42（策略在合理范围内偏离参考模型）。DEBUG 日志显示模型在这个阶段能生成正确的 `tool_call` JSON 格式、选对函数名、尝试填入参数值，虽然长参数存在截断但整体行为是健康的。

**阶段三：Reward Hacking 崩塌期（step 2400-2999）**。正奖励率继续飙升（72% → 91%），但这是虚假繁荣。三个异常信号同时出现：(1) KL 加速走大，从 -0.42 跳到 -0.84 再到 -0.99；(2) AvgLen 从 105 骤降到 69.7；(3) DEBUG 日志中模型的生成内容退化为无意义的固定模板——反复输出 `"ence\n\ngeometric\n\nneeded\n\nso, right? 🎉"` 这样的垃圾文本，完全不包含任何工具调用。模型发现了 reward 函数的漏洞：这种短文本恰好能通过某些 GT 匹配规则拿到正分，于是策略坍缩到这个 shortcut 上。

v5 的分区间统计如下：

| 区间 | 正奖励率 | 平均 Reward | 平均 KL | 平均 AvgLen | 阶段 |
|------|---------|------------|---------|------------|------|
| step 1-200 | 2.0% | -1.77 | -0.04 | 86 | 冷启动 |
| step 200-400 | 2.5% | -1.57 | -0.09 | 62 | 冷启动 |
| step 400-600 | 3.5% | -1.35 | -0.14 | 57 | 冷启动 |
| step 600-800 | 4.5% | -1.33 | -0.18 | 68 | 冷启动 |
| step 800-1600 | 3.5-5.0% | -1.17 ~ -1.28 | -0.07 ~ -0.18 | 49-61 | 冷启动平台 |
| step 1601-1800 | **10.5%** | -0.86 | -0.13 | 84 | 拐点 |
| step 1801-2000 | **37.0%** | -0.27 | -0.30 | **129** | 有效学习 |
| step 2001-2400 | 39-44% | -0.09 ~ -0.23 | -0.30 ~ -0.41 | 95-137 | 有效学习 |
| step 2401-2600 | 47.0% | -0.09 | -0.42 | 105 | 过渡区 |
| step 2601-2800 | 72.5% | +0.27 | **-0.84** | 105→↓ | **Reward Hacking** |
| step 2801-2999 | 91.0% | +0.63 | **-0.99** | **69.7** | **Reward Hacking** |

v5 在 step 2999 处手动停止。有效训练窗口为 step 1-2400（约 2400 步），其中 step 1600-2400 是真正学到工具调用能力的黄金期。step 2000 处保存的 checkpoint 是最有价值的权重，此时模型既学会了基本的工具调用格式，又尚未被 reward hacking 污染。


## 各版本训练指标对比（全量）

| 版本 | 学习率 | beta | Reward 函数 | 运行步数 | KL 范围 | 正奖励次数 | AvgLen=1 | 结果 |
|------|--------|------|------------|----------|---------|------------|----------|------|
| v1 | 5e-6 | 0.1 | 旧 | 540 | -0.5 ~ -5.5 | 0 | 频繁 | 策略崩溃 |
| v2 | 1e-6 | 0.3 | 旧 | 650 | ±0.09 | 0 | 0 | 稳定但学不动 |
| v3 | 3e-6 | 0.2 | 旧 | 236 | -0.43 ~ +0.09 | 0* | 0 | KL 加速走负 |
| v4 | 3e-6 | 0.2 | **新** | 415 | -0.3 ~ -10 | 5 | 40 | 正奖励出现，但后期崩塌 |
| v5 | 1e-6 | 0.2 | **新** | 2999 | -0.04 ~ -0.99 | 739 | 0 | **三阶段：冷启动→有效学习→Reward Hacking** |

\* v3 有少量接近零的值但严格来说没有正值。v4/v5 的正奖励是 reward 修复后首次出现的，证明根因定位正确。v5 的有效训练窗口为 step 1-2400，step 1600 出现拐点进入有效学习期，step 2400+ 发生 Reward Hacking（模型输出退化为无意义短文本刷分）。


## Agent RL 改进总结

回顾 v1 到 v5 的迭代过程，整个 Agent RL 的改进可以归纳为四个层次。

**第一层：数据升级（v1 之前）**。把质量稀烂的自造 agent_rl.jsonl（39988 条，6 种工具，50% 低质闲聊）替换为 Salesforce/xlam-function-calling-60k（59578 条，3603 种工具，每条有 GT）。这一步解决了"训练数据本身就没有区分度"的问题——旧数据上 Reward Model 对纯对话打分几乎无差异，GrpStd 为零，advantage 信号完全消失。

**第二层：超参搜索与根因定位（v1 → v3）**。三个版本分别用激进（lr=5e-6）、保守（lr=1e-6）和折中（lr=3e-6）的配置做实验，结果要么策略崩溃（v1），要么学不动（v2），要么 KL 加速走负（v3），Reward 均值都卡在 -1.5。三个版本殊途同归让我们意识到瓶颈不在优化器而在 reward 函数——这是整个调试过程中最关键的认知转折。

**第三层：Reward 函数修复（v4 → v5 前期）**。定位到根因后做了两处修改：(1) GT 验证检查对象从仅看 final_text 扩展为 tool call arguments + final_text 取并集，让模型只要传对参数就能拿分，不需要在最终回答中复述参数值；(2) 工具对齐分权重从 0.5 提到 1.0，加强冷启动阶段"学会正确调用格式"的正反馈信号。v4 验证了修复方向正确（首次出现正奖励），但 lr=3e-6 仍然偏大导致后期崩塌。v5 用 lr=1e-6 搭配修复后的 reward，在 step 1600-2400 的有效学习期内正奖励率达到 37-47%，模型成功学会了基本的工具调用格式。

**第四层：Reward Hacking 暴露 reward 函数仍有漏洞（v5 后期）**。v5 训练到 step 2400 后，正奖励率继续飙升到 72% → 91%，但 DEBUG 日志揭示了真相：模型并没有学得更好，而是发现了 reward 函数的漏洞——反复输出几个无意义单词（"ence geometric needed so, right? 🎉"）就能匹配到某些 GT 拿正分。策略坍缩到这个 shortcut 上，KL 散度加速走大（-0.84 → -0.99），AvgLen 骤降到 69.7，工具调用行为完全消失。这说明 reward 函数虽然解决了冷启动的信号稀疏问题，但检查粒度仍然不够——缺少对输出内容合法性的硬约束（如必须包含 tool_call 标签、最低输出长度等），给了模型可 exploit 的空间。

这四层改进的逻辑链条是：**数据质量 → 排除超参干扰暴露真正瓶颈 → 修复 reward 函数的结构性 mismatch → 发现 reward 函数仍存在可被 exploit 的漏洞**。每一层都是在上一层的基础上才能定位到下一个瓶颈，不能跳步。Reward Hacking 的出现本身也是一种"成功"——它意味着模型的优化能力足够强，能高效地找到 reward landscape 中的 shortcut，只是这个 shortcut 不是我们期望的行为。


---

## Plan RL 阶段：从 PlanRate=0% 到链路打通

### 背景：从 Agent RL 到 Plan RL

Agent RL（v5）在 step 2000 处获得了一个具备基本工具调用能力的 checkpoint。下一步是切换到 `train_plan.py`，引入 Plan-then-Execute 范式——要求模型在发起工具调用之前先输出 `<plan>...</plan>` 标签包裹的规划文本，描述接下来要做什么。这个能力对于多步工具调用场景（如先查天气再做推荐）至关重要。

`train_plan.py` 在 Agent RL 的基础上增加了 plan 相关的奖励信号：`parse_plan()` 函数用正则 `<plan>(.*?)</plan>` 检查模型输出中是否包含 plan 标签，并据此计算 PlanRate 指标（每个 batch 中成功生成 plan 的样本比例）。训练启动命令额外注入 `PLAN_SYSTEM_PROMPT`（要求模型先规划再执行）和 `PLAN_FEWSHOT_MESSAGES`（2 条 plan 格式的示范对话）。

### 问题现象

首次启动 `train_plan.py`（使用 `plan_warmup_v2` 权重，该权重经过 plan 格式的 SFT 热身），PlanRate 在所有步数上始终为 0%。日志 `plan_rl_0511_0214.log` 显示每个 batch 的 PlanRate 都是 0.00%，没有任何一个样本成功生成 plan。

这很反常——同一个权重在独立测试脚本中（直接给 system prompt + few-shot + 用户问题）可以 100% 生成包含 `<plan>` 标签的输出。说明模型本身具备 plan 能力，问题出在 RL 训练流程中。

### 排查过程

#### 第一步：确认 rollout 引擎本身无问题

在 `train_plan.py` 中单独调用 rollout 引擎（不经过 RL 训练循环），给同样的 prompt 做推理，发现生成的文本中确实包含 `<plan>` 标签。这排除了 rollout 引擎（模型推理 + tokenizer 解码）本身的问题。

#### 第二步：定位到 `゜` 分隔符的截断 bug

`train_plan.py` 中，rollout 生成的 `new_text` 在送入 `parse_plan()` 之前有一步预处理（约 line 256 和 343）：

```python
parse_text = new_text.split('゜')[-1]
```

这行代码的原始意图是：取最后一个 `゜` 分隔符之后的文本，因为 `゜` 在模型的 tokenizer 协议中同时充当两个角色——(1) thinking 阶段的结束分隔符，(2) 工具调用的结束标记（`ゅ{tool_call}゜`）。

问题在于，模型的典型输出格式是：

```
<plan>我需要查询天气信息</plan>
ゅ{"name": "get_weather", "arguments": {"city": "北京"}}゜
```

当模型正确生成了 plan 后紧跟工具调用时，`゜` 出现在工具调用末尾。`split('゜')[-1]` 取最后一个 `゜` 之后的内容——这是一个空字符串（因为 `゜` 是文本的最后一个字符）。于是 `parse_plan("")` 在空字符串中搜索 `<plan>` 标签，自然返回 `has_plan=False`。

**越是正确输出 plan + tool_call 的模型，越会被这行代码判定为"没有 plan"**。这是一个自相矛盾的 bug：模型做对了反而被惩罚。

#### 修复方案

将两处 `parse_text = new_text.split('゜')[-1]` 替换为直接在完整的 `new_text` 上调用 `parse_plan()`：

```python
# 修复前
parse_text = new_text.split('゜')[-1]
steps, raw, has = parse_plan(parse_text)

# 修复后
steps, raw, has = parse_plan(new_text)
```

这样 `parse_plan()` 在完整文本中搜索 `<plan>` 标签，无论标签出现在文本的哪个位置都能正确匹配。

### 第二个 bug：多轮对话历史稀释 few-shot 示范

修复 `゜` 截断 bug 后，PlanRate 从 0% 恢复到了非零值，但仍然偏低（平均约 33%）。进一步检查 `plan_train_epoch()` 函数中构建 prompt 的逻辑（约 line 755），发现了第二个问题。

`agent_rl.jsonl` 中的样本是多轮对话格式，messages 数组包含完整的对话历史（system → user → assistant → user → assistant → ...）。`plan_train_epoch()` 构建训练 prompt 时的做法是：

```python
msgs_copy = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}]
for fshot in PLAN_FEWSHOT_MESSAGES:
    msgs_copy.append(dict(fshot))
for m in messages:
    if m["role"] != "system":
        msgs_copy.append(dict(m))
```

这意味着 few-shot 示范之后，紧接着的不是当前用户的实际问题，而是完整的多轮对话历史——大量与当前问题无关的早期对话轮次。对于 64M 参数的小模型来说，这些历史对话占据了宝贵的上下文窗口（实测有些样本的历史文本长达 800+ token），把 few-shot 示范和实际问题隔得很远，严重稀释了 few-shot 的示范效果。

模型需要从 few-shot 中学习 `<plan>` 格式，但中间插入了大量无关文本后，小模型很难建立 few-shot 与当前输出之间的格式关联。

#### 修复方案

只保留 system message 和最后一条 user message，丢弃所有中间历史：

```python
last_user_msg = None
system_msg = None
for m in messages:
    if m["role"] == "system":
        system_msg = dict(m)
    if m["role"] == "user":
        last_user_msg = dict(m)

msgs_copy = []
if system_msg:
    system_msg["content"] = PLAN_SYSTEM_PROMPT
    msgs_copy.append(system_msg)
else:
    msgs_copy.append({"role": "system", "content": PLAN_SYSTEM_PROMPT})

for fshot in PLAN_FEWSHOT_MESSAGES:
    msgs_copy.append(dict(fshot))

if last_user_msg:
    msgs_copy.append(last_user_msg)
```

修复后的 prompt 结构变为：`[system + PLAN_SYSTEM_PROMPT] → [few-shot 示范 × 2] → [当前用户问题]`，few-shot 和实际问题紧邻，小模型能直接模仿格式。实测 prompt token 数减少约 56%。

### Plan RL v4 训练结果与崩溃

两个 bug 修复后，以 `plan_warmup_v2` 为初始权重启动 Plan RL 训练（lr=5e-6, batch_size=4, num_generations=2, max_turns=4）。训练日志 `plan_rl_v4.log` 共运行 150 步后手动终止。

前 100 步的指标看起来有进展——PlanRate 在 12.5%~62.5% 之间波动，Reward 偶尔出现正值。但从 step 80 开始，KL 散度持续走高（2.07 → 3.49 → 4.45 → 5.03），Loss 从正常的 1~3 飙升到 9~15，PlanRate 在 step 140 降到 0% 并再未恢复。

| Step | Reward | KL | PlanRate | Loss | AvgLen | 备注 |
|------|--------|-----|----------|------|--------|------|
| 10 | 0.68 | 2.07 | 25.0% | — | 69 | 首批即有 plan |
| 20 | 1.07 | 2.31 | 62.5% | — | 195 | 高 plan 率 |
| 70 | 0.18 | 2.90 | 62.5% | 14.7 | 197 | Loss 开始异常 |
| 80 | -0.25 | 3.49 | 12.5% | 5.1 | 129 | KL 偏高 |
| 110 | -0.18 | 4.45 | 12.5% | 5.4 | 200 | KL 加速 |
| 130 | -0.10 | 4.96 | 12.5% | 6.0 | 215 | |
| 140 | -0.32 | 4.62 | **0.0%** | 6.3 | 176 | PlanRate 归零 |
| 150 | -0.30 | **5.03** | **0.0%** | **9.0** | 314 | 手动终止 |

step 150 时 KL=5.03、Loss=9.0、PlanRate=0%，训练已经完全失控，手动 kill 进程。


### Plan RL v4 根因分析：三个结构性问题

对 v4 的日志和代码做了深入分析，定位到三个互相叠加的结构性问题。

#### 根因一：Rollout 链路断裂——模型永远无法进入执行阶段

这是最致命的 bug。`plan_rollout_single()` 函数中，每轮生成后都会调用 `parse_tool_calls(new_text)` 检查是否有工具调用，如果没有就 `break` 结束 rollout：

```python
# plan_rollout_single() 约 line 264
calls = parse_tool_calls(new_text)
if not calls:
    break  # 没有工具调用，结束
```

问题在于，Plan 轮（turn==0）的输出格式是 `<plan>[...]</plan>`，这是纯文本标签，不包含 `ゅ{...}゜` 格式的工具调用标记。`parse_tool_calls()` 在 plan 文本中自然找不到工具调用，于是返回空列表，触发 `break`——rollout 在第一轮就结束了，模型永远无法进入第二轮的执行阶段。

这导致了一个连锁反应：模型从未实际执行过工具 → `execution_trace` 始终为空 → exec reward 恒为 -0.5（"需要工具但没有调用"的惩罚）→ 模型只能从 plan 格式奖励中获得正信号，但 plan 奖励本身不足以驱动完整的 plan→execute 行为链。

从日志中可以验证：v4 所有样本的 `turns` 都是 1（只有 plan 轮），没有任何样本进入过 turns>=2。exec reward 在所有 step 上都是 -0.5。

#### 根因二：KL 散度失控

`--beta` 默认值为 0.04，对 64M 小模型来说 KL 惩罚力度严重不足。小模型参数少、策略分布变化快，0.04 的 beta 无法有效约束策略偏离参考模型的速度。同时代码中没有 per-token KL clip，单个 token 的 KL 值可以无限大，少数异常 token 就能拖垮整个 batch 的 loss。

KL 从 step 10 的 2.07 单调上升到 step 150 的 5.03，全程没有任何回落，说明 KL 惩罚完全没有起到约束作用。

#### 根因三：格式崩塌——重复字符生成

DEBUG 日志中出现了大量 `" " " "` 这样的重复引号/空格生成。模型在 plan 阶段偶尔会陷入重复 token 的循环，生成几百个无意义字符。现有的 `rep_penalty()` 函数基于 n-gram 检测，惩罚上限为 0.5，对这种极端退化模式力度不够。这些退化样本不仅浪费了 rollout 的计算资源，还向模型传递了混乱的梯度信号。

三个问题叠加的效果是：模型只能做 plan 不能执行（根因一）→ exec reward 恒负，梯度信号混乱 → KL 无约束地发散（根因二）→ 模型开始生成垃圾文本（根因三）→ 训练完全失控。


## Plan RL v5：系统性修复

针对 v4 暴露的三个根因，对 `train_plan.py` 做了系统性修复（版本标记为 v5）。

### 修复一：打通 Plan→Execute 链路

在 `plan_rollout_single()` 中，对 turn==0（Plan 轮）做特殊处理：如果 `parse_tool_calls()` 返回空但 `parse_plan()` 成功解析到了包含工具步骤的 plan，不再 break，而是将 plan 作为 assistant 消息注入对话历史，追加一条 `"请按照你的计划执行。"` 的 user 提示，然后 `continue` 进入下一轮（执行轮）。

```python
if turn == 0:
    calls = parse_tool_calls(new_text)
    if not calls:
        if tools and has and any(s.get("tool", "none") != "none" for s in steps):
            # Plan 成功且包含工具步骤 → 注入 plan，继续执行
            messages.append({"role": "assistant", "content": new_text})
            messages.append({"role": "user", "content": "请按照你的计划执行。"})
            continue  # 进入执行轮
        else:
            break  # 无工具 / plan 中没有工具步骤 → 正常结束
```

这样模型在 plan 轮输出 `<plan>` 后，会自动进入 turn==1 的执行轮，有机会生成 `ゅ{tool_call}゜` 格式的工具调用。

### 修复二：KL 控制加强

两处改动：(1) `--beta` 默认值从 0.04 提升到 0.1，增强 KL 惩罚力度；(2) 新增 `--kl_clip` 参数（默认 15.0），对 per-token KL 做 `torch.clamp(max=kl_clip)` 截断，防止单个异常 token 的 KL 值拖垮整个 loss。

```python
per_token_kl = torch.exp(kl_div) - kl_div - 1
per_token_kl = torch.clamp(per_token_kl, max=args.kl_clip)  # v5 新增
```

### 修复三：退化检测与惩罚

新增 `_is_degenerate()` 函数，检测生成文本中最常见字符占比是否超过 60%，或短 pattern（1~4 字符）是否重复超过 60%。在 rollout 中检测到退化立即终止当前样本（仍记录 token 让惩罚信号传播），在 reward 中对退化样本直接给 -3.0 的重惩罚。额外检测空白/引号字符占比超过 50% 的情况，追加 1.0 惩罚。

### 修复四：参数调优与 Reward 简化

`--learning_rate` 默认值从 5e-6 降到 3e-6。去掉了 `derive_optimal_plan()`（Plan Critic）和 `estimate_task_complexity()`（复杂度自适应权重）两个对 64M 小模型引入过多噪声的机制，让 reward 信号更简洁。日志新增 `ExecRate`（执行率）和 `DegenRate`（退化率）两个监控指标。

### Plan RL v5 初步训练结果

以 `plan_warmup_v2` 为初始权重，使用修复后的代码启动训练（lr=3e-6, beta=0.1, kl_clip=15, batch_size=4, num_generations=2）。截至 step 40 的指标如下：

| Step | Reward | KL | PlanRate | ExecRate | DegenRate | AvgLen | 备注 |
|------|--------|-----|----------|----------|-----------|--------|------|
| 10 | 0.59 | 5.59 | 25.0% | 25.0% | 0% | 72 | 首批即有执行 |
| 20 | 0.82 | 6.65 | 75.0% | **75.0%** | 0% | 244 | 链路打通 |
| 30 | 0.44 | 6.25 | 50.0% | 50.0% | 0% | 129 | |
| 40 | 0.58 | 6.69 | 62.5% | **75.0%** | 0% | 225 | 稳定 |

与 v4 的关键对比：

| 指标 | v4 (step 10-150) | v5 (step 10-40) |
|------|-----------------|----------------|
| ExecRate | **0%**（永远无法执行） | **25-75%** |
| PlanRate | 0-62.5%，后期归零 | 25-75%，稳定 |
| DegenRate | 大量退化（未统计） | **0%** |
| KL 趋势 | 2.07→5.03 单调上升 | 5.6~6.7 波动，有 clip 兜底 |
| exec reward | 恒为 -0.5 | 仍为负，但模型已在执行工具 |
| turns | 始终为 1 | 2~4 轮 |

最显著的改善是 ExecRate 从 0% 跃升到 25-75%——模型终于能走完 plan→execute 的完整链路了。DEBUG 日志显示 turns=3 和 turns=4 的样本中，`execution_trace` 有成功的工具调用记录。DegenRate 始终为 0%，退化检测机制有效。

exec reward 仍然为负（-0.5），这是因为模型虽然能调用工具了，但还没学会把工具返回结果正确整合到最终回答中以通过 GT 验证。这是正常的学习阶段，需要更多 step 来改善。

KL 值偏高（5.6~6.7），这是因为 plan_warmup_v2 权重经过 plan SFT 热身后，策略分布已经与 ref model（同一个权重的冻结副本）有一定偏移。kl_clip=15 在兜底，KL 没有像 v4 那样单调爆炸。训练仍在继续中，后续结果将持续记录。


### Plan RL v5b：退化分析

v5 训练在 step 70 处保存了 checkpoint（`plan_rl_v5_step500`，实际为 step 70 时的权重），随后以该 checkpoint 为起点继续训练（标记为 v5b）。v5b 在 step 70~170 之间出现了严重退化，PlanRate 从 100% 骤降到 12.5%，Reward 从正值转为持续负值。

v5b 的退化轨迹如下：

| Step | Reward | KL | GrpStd | PlanRate | ExecRate | AvgLen | 备注 |
|------|--------|-----|--------|----------|----------|--------|------|
| 70 | 0.72 | 6.80 | 0.38 | 100% | 87.5% | 296 | checkpoint 起点，状态良好 |
| 80 | 0.39 | 4.02 | 0.25 | 50% | 50% | 124 | 开始下滑 |
| 100 | -0.09 | 3.89 | 0.22 | 37.5% | 37.5% | 131 | Reward 转负 |
| 120 | -0.33 | 2.68 | 0.18 | 25% | 25% | 98 | 持续恶化 |
| 150 | -0.41 | 1.95 | 0.12 | 12.5% | 12.5% | 76 | 接近崩溃 |
| 170 | -0.52 | 1.43 | 0.08 | 12.5% | 0% | 54 | 完全退化 |

对 v5b 的 DEBUG 日志做了逐样本分析，定位到三个互相叠加的根因。

**根因一：平凡输出获得不当正奖励。** 模型生成的大量输出仅包含 `'\n</think>\n\n'` 或类似的极短文本（去除标签后不足 5 个字符），但这些输出在 reward 计算中拿到了 +0.5 的分数。原因是 reward 函数对"无工具调用的纯文本回复"走 Reward Model 打分路径，而 RM 对这种极短输出给出了中性偏正的分数。模型发现生成这种"什么都不说"的输出就能稳定拿到正分，于是策略逐渐向这个 shortcut 坍缩。

**根因二：Plan 截断导致重惩罚。** 当模型尝试生成完整的 `<plan>...</plan>` 但因 `max_gen_len=384` 限制被截断时（即输出中有 `<plan>` 但没有 `</plan>`），reward 函数将其视为"plan 格式错误"，给予 -1.1 到 -1.9 的重惩罚。这意味着模型越是认真尝试生成 plan，越容易因为长度限制被截断而受到惩罚。与根因一叠加后，模型学到的策略是"不要尝试 plan，直接输出空内容更安全"。

**根因三：GrpStd 趋近于零，学习信号消失。** 当同一 prompt 的多个生成（num_generations=4）都产出类似的平凡输出时，它们的 reward 几乎相同，组内标准差（GrpStd）趋近于零。GRPO 的 advantage 计算依赖 `(reward - mean) / std`，当 std→0 时 advantage 要么为零（无梯度信号）要么数值爆炸（除以极小数）。从日志看 GrpStd 从 step 70 的 0.38 持续下降到 step 170 的 0.08，学习信号逐步消失，模型陷入"平凡输出→相同 reward→无梯度→继续平凡输出"的死循环。


### Plan RL v6：针对 v5b 退化的四项修复

针对 v5b 暴露的三个根因，通过 `patch_v6.py` 对 `train_plan.py` 做了 12 处补丁修改，版本标记为 v6。

**修复一：平凡输出检测（对应根因一）。** 新增 `_is_trivial_output()` 函数，对模型输出去除 `<think>`/`</think>` 标签后检查有效字符数，少于 5 个字符的输出被标记为平凡输出，reward 直接置为 0.0（而非之前的 +0.5），切断"什么都不说就能拿分"的 shortcut。

```python
def _is_trivial_output(text):
    stripped = text.strip()
    cleaned = re.sub(r'</?think>', '', stripped).strip()
    cleaned = re.sub(r'\\s+', '', cleaned)
    return len(cleaned) < 5
```

**修复二：Plan 截断宽容处理（对应根因二）。** 新增 `_has_plan_attempt()` 函数，检测输出中是否有 `<plan>` 但缺少 `</plan>`（即被 max_gen_len 截断的 plan）。对这类截断 plan 给予 +0.1 的中性偏正奖励（而非之前的 -1.1 ~ -1.9 重惩罚），鼓励模型继续尝试生成 plan，同时在 rollout 数据中标记 `plan_truncated=True` 字段供后续分析。

```python
def _has_plan_attempt(text):
    return '<plan>' in text and '</plan>' not in text
```

**修复三：GrpStd 下限保护（对应根因三）。** 在 GRPO 的 advantage 计算中，对组内标准差施加 `torch.clamp(std_r, min=0.1)` 的下限保护。当所有生成的 reward 相同时，std_r 被钳位到 0.1 而非趋近于零，确保 advantage 计算始终有有效的梯度信号，打破"相同 reward→无梯度→策略停滞"的死循环。

```python
std_r = torch.clamp(std_r, min=0.1)
```

**修复四：提高采样温度。** 将 rollout 的采样温度从 0.8 提升到 1.0。v6 初次启动时发现 GrpStd=0（所有 4 个生成完全相同），原因是 temperature=0.8 对已经收敛的 checkpoint 来说多样性不足。提高到 1.0 后，同一 prompt 的多个生成产生了足够的差异，GrpStd 恢复到 0.25~0.39 的健康范围。

### Plan RL v6 训练结果

以 v5 的 step 500 checkpoint 为起点，使用 v6 修复后的代码启动训练（lr=3e-6, beta=0.1, kl_clip=15, temperature=1.0）。训练运行至 step 1140 后手动停止，共 114 条日志记录，保存了 step 500 和 step 1000 两个 checkpoint。

v6 的四项修复成功解决了 v5b 的急性退化问题——DegenRate 全程 0%，GrpStd=0 仅出现 7 次（6%），没有出现 v5b 那种"平凡输出→GrpStd 归零→策略停滞"的死循环。但训练整体呈现缓慢下滑趋势，未出现类似 Agent RL v5 step 1600 那样的延迟拐点。

v6 的 1140 步训练可以分为三个阶段：

**阶段一：SFT 惯性期（step 10-200）。** 平均 Reward +0.43，PlanRate 平均 52.5%，ExecRate 55.0%。这是 plan_warmup_v2 权重的 SFT 惯性在起作用——模型靠热身阶段学到的 plan 格式维持着不错的表现。最佳步出现在 step 180（Reward=+0.996, PlanRate=100%, ExecRate=100%）。

**阶段二：缓慢衰减期（step 200-800）。** Reward 均值从 +0.25 逐步降到 +0.10，PlanRate 从 47% 降到 34%，ExecRate 从 40% 降到 29%。下降是渐进的而非突然的，每 100 步的指标都在小幅走低。模型的 plan 行为在 RL 训练中被缓慢侵蚀——RL 的梯度信号没有强化 plan 能力，反而在逐步冲淡 SFT 阶段建立的格式记忆。

**阶段三：能力耗尽期（step 800-1140）。** Reward 均值跌破零（step 800-900 为 -0.02），PlanRate 降到 23-30%，最后 40 步（step 1101-1140）PlanRate 骤降到 3.1%，plan 行为基本消失。step 1120 出现全程最差表现（Reward=-0.84, PlanRate=0%）。

分段统计如下：

| 区间 | 平均 Reward | 平均 KL | 平均 GrpStd | 平均 PlanRate | 平均 ExecRate | PlanRate>0 比例 |
|------|-----------|---------|------------|-------------|-------------|---------------|
| step 1-100 | +0.47 | 5.78 | 0.18 | 63.8% | 68.8% | 80% |
| step 101-200 | +0.40 | 5.42 | 0.21 | 41.2% | 41.2% | 70% |
| step 201-300 | +0.36 | 5.03 | 0.23 | 40.0% | 37.5% | 80% |
| step 301-400 | +0.14 | 7.02 | 0.28 | 53.8% | 41.2% | 60% |
| step 401-500 | +0.01 | 6.54 | 0.30 | 41.2% | 25.0% | 60% |
| step 501-600 | +0.05 | 5.75 | 0.22 | 41.2% | 35.0% | 60% |
| step 601-700 | +0.11 | 5.40 | 0.24 | 41.2% | 35.0% | 60% |
| step 701-800 | +0.10 | 4.60 | 0.21 | 30.0% | 25.0% | 40% |
| step 801-900 | -0.02 | 6.92 | 0.22 | 37.5% | 27.5% | 50% |
| step 901-1000 | +0.07 | 5.81 | 0.26 | 23.8% | 21.2% | 40% |
| step 1001-1100 | +0.09 | 6.61 | 0.22 | 43.8% | 33.8% | 60% |
| step 1101-1140 | **-0.48** | 7.47 | 0.17 | **3.1%** | **3.1%** | 25% |

全局统计：总 114 步，平均 Reward +0.14，正奖励比例 66.7%（76/114），GrpStd=0 出现 7 次，DegenRate>0 出现 0 次。

### v5b vs v6 对比

| 指标 | v5b (step 70→170) | v6 (step 10→1140) |
|------|-------------------|-------------------|
| 退化模式 | 急性崩溃（100 步内 PlanRate 100%→0%） | 慢性衰减（1000 步内 PlanRate 64%→3%） |
| 根因 | 平凡输出 shortcut + 截断惩罚 + GrpStd→0 | SFT 惯性耗尽 + RL 未能强化 plan 能力 |
| DegenRate | 大量平凡输出 | **全程 0%** |
| GrpStd | 0.38→0.08（趋近于零） | 平均 0.23（健康范围，偶尔为 0） |
| v6 修复是否有效 | — | **有效**：消除了急性退化，但暴露了更深层的能力瓶颈 |

v6 的结论是：四项修复成功解决了 v5b 的"训练机制缺陷"（reward 漏洞、GrpStd 消失），但暴露了更根本的问题——64M 参数模型的容量不足以通过 RL 训练习得 plan→execute 这样的长链推理能力。模型在 SFT 热身阶段靠模仿学会了 plan 格式，但 RL 阶段的梯度信号不足以维持和强化这个能力，反而在缓慢侵蚀它。


## Plan RL 各版本对比

| 版本 | 学习率 | beta | KL clip | 温度 | 运行步数 | KL 范围 | PlanRate | ExecRate | 结果 |
|------|--------|------|---------|------|----------|---------|----------|----------|------|
| v4 | 5e-6 | 0.04 | 无 | 0.8 | 150 | 2.07→5.03 | 0-62.5%→0% | **0%** | KL 爆炸 + 链路断裂 |
| v5 | 3e-6 | 0.1 | 15 | 0.8 | 70 | 5.6~6.7 | 25-75% | 25-75% | 链路打通，checkpoint 保存 |
| v5b | 3e-6 | 0.1 | 15 | 0.8 | 170 | 1.43~6.80 | 100%→12.5% | 87.5%→0% | **退化崩溃**（平凡输出+截断惩罚+GrpStd→0） |
| v6 | 3e-6 | 0.1 | 15 | **1.0** | 1140 | 2.1~10.4 | 64%→3%（缓慢衰减） | 69%→3% | 无急性退化，但 SFT 惯性耗尽后 plan 能力缓慢消失 |


## 日志文件索引

所有日志保存在 `/root/trainer/logs/` 或 `/root/`，已同步到本地 `trainer/logs/`：

| 文件 | 内容 |
|------|------|
| train_agent_xlam.log | Agent RL v1 (lr=5e-6), 540 步后策略崩溃手动停止 |
| train_agent_xlam_v2.log | Agent RL v2 (lr=1e-6, 旧reward), 650 步学不动手动停止 |
| train_agent_xlam_v3.log | Agent RL v3 (lr=3e-6, 旧reward), 236 步快照 |
| v4_reward_fix.log | Agent RL v4 (lr=3e-6, 新reward), 415 步后期崩塌手动停止 |
| v5_lr1e6_reward_fix.log | Agent RL v5 (lr=1e-6, 新reward), 2999 步手动停止（Reward Hacking）|
| agent_rl_0510_1546.log | 原始 agent_rl.jsonl 的早期实验 |
| plan_rl_0511_0214.log | Plan RL 首次尝试（PlanRate=0%，゜截断 bug 未修复）|
| plan_rl_v4.log | Plan RL v4（゜截断 + 历史稀释 bug 修复后），150 步 KL 爆炸手动终止 |
| plan_rl_v5.log | Plan RL v5（链路断裂 + KL + 退化 三重修复），70 步保存 checkpoint |
| plan_rl_v5b.log | Plan RL v5b（v5 checkpoint 续训），170 步退化崩溃 |
| plan_rl_v6.log | **Plan RL v6（平凡输出检测 + 截断宽容 + GrpStd 下限 + 温度提升），1140 步手动停止** |
| sft_0509_1733.log | SFT 训练日志 |


## 64M 模型的能力天花板

v6 的 1140 步训练最终确认了 64M 参数模型在 Plan RL 任务上的能力天花板。

从 Agent RL 到 Plan RL，每一轮优化本质上都是在"降低拿分门槛"——reward 函数从只看 final_text 扩展到看 tool call arguments，plan 截断从重惩罚改成宽容，平凡输出从拿正分改成零分，GrpStd 加了下限保护。这些修复都是正确的（确实是 bug 或设计缺陷），但它们解决的是训练机制的问题，而非模型容量的问题。

Plan-then-Execute 要求模型在一次生成中同时完成四件事：理解任务意图、规划执行步骤、生成正确的工具调用 JSON、整合返回结果给出最终回答。每一步都需要"记住"前面的上下文，8 层 transformer（hidden_size=768）的表达能力和上下文建模能力可能就是不够。v6 的训练曲线印证了这一点——模型在 SFT 热身阶段靠模仿学会了 plan 格式（前 200 步 PlanRate 64%），但 RL 训练并没有强化这个能力，反而在 1000 步内将其缓慢侵蚀到 3%。

不过，从实验方法论的角度看，这个项目的收获是扎实的：数据质量对 RL 训练的决定性影响、reward 函数结构性 mismatch 的定位方法、reward hacking 的实际案例与防御、rollout 链路断裂的排查、训练退化的多重根因分析——这些经验对后续换更大模型做 RL 都是直接可复用的。

v6 的 step 500 checkpoint 是 Plan RL 阶段最有价值的权重——此时 PlanRate 仍有 41%、ExecRate 25%，RL 收益和 plan 能力侵蚀之间达到最佳平衡。step 1000 的权重已经明显退化（PlanRate 24%），不建议使用。


## 下一步

Plan RL v6 在 step 1140 手动停止。实验的 RL 训练部分到此结束。剩余工作：

- **推理测试**：用 v6 step 500 checkpoint 跑推理，直观评估模型实际生成的 plan 质量、工具调用准确性和最终回答质量，确认 64M 模型的实际能力边界
- **实验复盘**：整理从 Agent RL v1 到 Plan RL v6 的完整迭代链路，提炼可复用的方法论，为后续更大模型的 RL 训练提供参考
