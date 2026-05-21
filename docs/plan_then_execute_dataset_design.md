# Plan-then-Execute 数据集设计方案

> 涉及文件: `dataset/plan_execute.jsonl`, `dataset/sft_plan_warmup.jsonl`, `dataset/generate_plan_data.py`
> 基于: 现有 3 Expert（Math/Info/Translate）+ 6 底层工具 + Mock 执行环境，不做任何扩展

## 为什么需要 Plan-then-Execute

当前 Handoff 架构是**单步路由**——用户问一个问题，Router 做一次决策：委托给谁（或直接回答）。这对简单问题够用，但真实场景中大量问题需要**多步协作**：

「帮我查杭州天气，如果气温低于 15 度就换算成华氏度告诉我」——这需要先 Info 查天气，再根据结果决定是否 Math 换算，是一个有**条件依赖**的两步链。

「100 美元换成人民币，再算这些钱能买几个 8 块钱的面包」——先 Info 查汇率，再 Math 做除法，两步之间有**数据依赖**。

Plan-then-Execute 让 Router 先输出一个结构化的**执行计划**，再按计划逐步调度 Expert，最后综合所有子结果生成最终回答。这对 RL 训练来说增加了两层新的学习信号：**规划质量**（计划是否合理）和**跨步信用分配**（哪一步对最终结果贡献最大）。

## 任务分类体系

按照**依赖关系复杂度**，把多步任务分为四个层级。训练时从低到高逐步引入，让模型先学会简单链式推理再处理复杂分支。

### Level 1: 顺序链（Sequential Chain）

两个子任务之间有线性数据依赖，前一步的输出是后一步的输入。

典型模式：`Expert_A → Expert_B`

示例：
- 「100 美元等于多少人民币？再帮我把结果翻译成日文」→ Info(汇率) → Translate(翻译结果)
- 「帮我把 'twenty-five' 翻译成中文，然后计算这个数字的平方」→ Translate → Math
- 「查一下东京现在几点，然后帮我算 UTC+9 和 UTC+8 差几个小时」→ Info(时间) → Math(时差)

### Level 2: 并行聚合（Parallel-then-Aggregate）

多个子任务彼此独立，可以并行执行，最后做一次聚合计算或对比。

典型模式：`[Expert_A, Expert_B] → Expert_C(aggregate)`

示例：
- 「分别查上海和东京的天气，告诉我哪个更热，温差多少度」→ [Info(上海), Info(东京)] → Math(温差)
- 「100 美元换人民币和 100 欧元换人民币，哪个更多，多多少？」→ [Info(美元), Info(欧元)] → Math(差值)
- 「把 'hello' 翻译成中文和日文，然后帮我数两个翻译加起来几个字」→ [Translate(中), Translate(日)] → Math(字数)

### Level 3: 条件分支（Conditional Branch）

前一步结果决定后续走哪条路径。Router 需要在中间做一次判断。

典型模式：`Expert_A → if condition → Expert_B else Expert_C`

示例：
- 「查杭州天气，如果气温超过 30 度帮我换算成华氏度，否则直接翻译成英文告诉我」→ Info → if >30 → Math else Translate
- 「100 美元换人民币，如果超过 700 就算能买多少杯 15 块的咖啡，否则翻译'太少了'成英文」→ Info → if >700 → Math else Translate
- 「查伦敦现在几点，如果是白天（6-18点）帮我查天气，否则帮我算距离明早 6 点还有几小时」→ Info(时间) → if 白天 → Info(天气) else Math(时差)

### Level 4: 多步链+聚合（Multi-step Pipeline）

三步或以上的长链，混合了顺序、并行和条件依赖。这是最复杂的层级，训练后期才引入。

典型模式：`Expert_A → Expert_B → Expert_C` 或 `[A, B] → C → D`

示例：
- 「查上海天气的气温，换算成华氏度，再把结果翻译成英文」→ Info → Math → Translate
- 「分别查北京和伦敦天气，算两地温差，把结果翻译成英文」→ [Info, Info] → Math → Translate
- 「查美元兑人民币汇率，把 250 美元换成人民币，然后算这些钱能买几个 56 元的东西，最后把结果翻译成英文」→ Info → Math → Math → Translate

## 数据格式

### RL 训练数据（`plan_execute.jsonl`）

每条样本是一个需要多步执行的用户查询，附带**执行计划 ground truth** 和**各步骤的验证信息**。

```json
{
  "query": "100美元等于多少人民币？再帮我把结果翻译成日文",
  "level": 1,
  "plan_gt": [
    {
      "step": 1,
      "expert": "info",
      "tool": "get_exchange_rate",
      "task": "查询美元兑人民币汇率",
      "depends_on": [],
      "expected_output_key": "rate"
    },
    {
      "step": 2,
      "expert": "math",
      "tool": "calculate_math",
      "task": "用汇率计算100美元等于多少人民币",
      "depends_on": [1],
      "expected_output_key": "amount_cny"
    },
    {
      "step": 3,
      "expert": "translate",
      "tool": "translate_text",
      "task": "将金额结果翻译成日文",
      "depends_on": [2],
      "expected_output_key": "translation"
    }
  ],
  "gt": ["723", "日本語"],
  "step_gts": [
    {"step": 1, "gt": ["7.23"]},
    {"step": 2, "gt": ["723"]},
    {"step": 3, "gt": ["日本語"]}
  ],
  "num_steps": 3,
  "experts_needed": ["info", "math", "translate"],
  "dependency_type": "sequential"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 用户的自然语言查询 |
| `level` | int | 难度层级 1-4 |
| `plan_gt` | list[dict] | 理想执行计划的 ground truth |
| `plan_gt[].step` | int | 步骤编号（从 1 开始）|
| `plan_gt[].expert` | string | 需要调度的专家（math/info/translate）|
| `plan_gt[].tool` | string | 预期使用的底层工具名 |
| `plan_gt[].task` | string | 传给专家的子任务描述 |
| `plan_gt[].depends_on` | list[int] | 依赖的前序步骤编号（空=无依赖）|
| `plan_gt[].expected_output_key` | string | 该步输出的语义标识（用于后续步骤引用）|
| `gt` | list[string] | 最终答案的验证片段（同现有格式）|
| `step_gts` | list[dict] | 每个步骤的中间结果验证 |
| `num_steps` | int | 总步骤数 |
| `experts_needed` | list[string] | 涉及的专家列表（去重）|
| `dependency_type` | string | sequential / parallel / conditional / mixed |

### Router 输出的计划格式

Router 在 Plan 阶段需要输出结构化计划。为了和现有 `<tool_call>` 格式保持一致性，采用单次 tool_call 携带完整 plan 的方式：

```xml
<tool_call>
{"name": "execute_plan", "arguments": {"plan": [
  {"step": 1, "delegate": "delegate_to_info_agent", "task": "查询美元兑人民币汇率", "depends_on": []},
  {"step": 2, "delegate": "delegate_to_math_agent", "task": "用步骤1的汇率结果计算100*汇率", "depends_on": [1]},
  {"step": 3, "delegate": "delegate_to_translate_agent", "task": "将步骤2的金额翻译成日文", "depends_on": [2]}
]}}
</tool_call>
```

这个设计的好处：Router 的 TOOLS 定义只需加一个 `execute_plan` 工具，不破坏原有的单步 `delegate_to_*` 路由能力。模型可以根据问题复杂度自行选择——简单问题用单步 delegate，复杂问题用 execute_plan。

### SFT Warmup 数据（`sft_plan_warmup.jsonl`）

和现有 SFT warmup 格式一致，只是 assistant 输出变成了 plan 格式：

```json
{
  "messages": [
    {"role": "system", "content": "<PLAN_ROUTER_SYSTEM_PROMPT>"},
    {"role": "user", "content": "100美元等于多少人民币？再帮我把结果翻译成日文"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"execute_plan\", \"arguments\": {\"plan\": [{\"step\": 1, \"delegate\": \"delegate_to_info_agent\", \"task\": \"查询美元兑人民币汇率\", \"depends_on\": []}, {\"step\": 2, \"delegate\": \"delegate_to_math_agent\", \"task\": \"用步骤1的汇率结果计算100*汇率\", \"depends_on\": [1]}, {\"step\": 3, \"delegate\": \"delegate_to_translate_agent\", \"task\": \"将步骤2的金额翻译成日文\", \"depends_on\": [2]}]}}\n</tool_call>"}
  ],
  "level": 1,
  "dependency_type": "sequential"
}
```

## 数据分布设计

总量目标：**500 条**（RL 数据）+ **60 条**（SFT Warmup）

### RL 数据分布

| Level | 类型 | 数量 | 占比 | Expert 组合 |
|-------|------|------|------|-------------|
| 1 | 顺序链(2步) | 180 | 36% | 6 种两两组合 × 30 |
| 2 | 并行聚合(2+1) | 120 | 24% | [Info,Info]→Math, [Info,Math]→Translate 等 |
| 3 | 条件分支(1+1) | 100 | 20% | Info→Math/Translate, Math→Info/Translate 等 |
| 4 | 多步链(3+步) | 60 | 12% | 3-4步长链 |
| — | 单步兼容 | 40 | 8% | 简单问题（保持单步 delegate 能力不退化）|

### Expert 组合覆盖

Level 1 的 6 种两两顺序组合，每种 30 条：

| 链 | 示例场景 |
|----|----------|
| Math → Info | 算出结果后查相关信息 |
| Math → Translate | 算出结果后翻译 |
| Info → Math | 查到数据后计算 |
| Info → Translate | 查到信息后翻译 |
| Translate → Math | 翻译后做数字处理 |
| Translate → Info | 翻译后查相关信息 |

### SFT Warmup 分布

60 条，每个 Level 15 条，覆盖主要 Expert 组合。其中：
- 40 条多步计划（execute_plan 格式）
- 20 条单步保留（原有 delegate_to_* 格式，防止能力退化）

## 与现有架构的关系

Plan-then-Execute 是对现有 Handoff 架构的**向上兼容扩展**，而非替换。

### 需要新增的部分

1. **ROUTER_TOOLS 新增 `execute_plan`** —— 一个新的 tool 定义，参数是 plan 列表
2. **ROUTER_SYSTEM_PROMPT 扩展** —— 告诉 Router 对于复杂问题可以先制定计划
3. **rollout 逻辑扩展** —— Plan 阶段解析 plan → 按 depends_on 拓扑排序执行 → 逐步调度 Expert → 汇总结果
4. **reward 函数扩展** —— 在现有路由奖励基础上增加：计划合理性、步骤依赖正确性、中间结果验证

### 不需要改动的部分

- Expert Agent 的 system prompt、tools、执行逻辑——完全复用
- 底层 6 个工具的定义和 Mock 实现——完全复用
- GT 验证逻辑（validate_gt_in_text）——复用
- GRPO/CISPO 训练框架——复用

### Rollout 流程变化

现有流程（3 阶段）：
```
Router决策 → Expert执行(1次) → Router整合
```

Plan-then-Execute 流程（N+2 阶段）：
```
Router规划(Plan) → Expert_1执行 → Expert_2执行 → ... → Expert_N执行 → Router整合(Synthesize)
```

序列拼接变为：
```
[plan_prompt(mask=0) | plan_response(mask=1) | expert_1_prompt(mask=0) | expert_1_response(mask=0/1) | ... | expert_N_response(mask=0/1) | synth_prompt(mask=0) | synth_response(mask=1)]
```

### Reward 扩展设计

在现有 7 项奖励基础上新增：

| 奖励项 | 分值 | 说明 |
|--------|------|------|
| 计划步骤数匹配 | ±0.2 | 实际步骤数 vs plan_gt 步骤数 |
| 专家调度顺序正确 | +0.3 | 计划中的 expert 顺序是否匹配 gt |
| 依赖关系正确 | +0.2 | depends_on 字段是否合理 |
| 中间步骤 GT 验证 | +0.5×N | 每个步骤的中间结果是否正确（step_gts）|
| 计划格式合法 | +0.1 | JSON 可解析、字段齐全 |

Credit Assignment 拆分：
- plan_reward（Router Plan 阶段）：计划质量相关的所有奖励
- expert_rewards[i]（每个 Expert 步骤）：该步骤的中间 GT 验证 + 工具调用质量
- synth_reward（Router Synthesize 阶段）：最终 GT 验证 + 回答质量

## 数据生成策略

### 模板化生成

每个 Expert 组合定义一组**查询模板**和**参数池**，通过组合采样生成多样化的数据。

以 `Info → Math` 为例：

模板池：
```python
templates_info_math = [
    {
        "query": "{amount}{currency_a}换成{currency_b}是多少？然后算能买几个{price}元的{item}",
        "params": {
            "amount": [50, 100, 200, 500, 1000],
            "currency_a": ["美元", "欧元", "英镑", "日元"],
            "currency_b": ["人民币"],
            "price": [5, 8, 10, 15, 25, 50],
            "item": ["面包", "咖啡", "笔", "本子", "矿泉水", "苹果"],
        },
        "plan": [
            {"expert": "info", "tool": "get_exchange_rate", "task_template": "查询{currency_a}兑{currency_b}汇率"},
            {"expert": "math", "tool": "calculate_math", "task_template": "计算{amount}×汇率÷{price}"},
        ],
    },
    {
        "query": "查一下{city}的气温，然后帮我算比{target_temp}度高多少",
        "params": {
            "city": ["上海", "北京", "杭州", "深圳", "东京", "伦敦", "纽约"],
            "target_temp": [0, 10, 15, 20, 25],
        },
        "plan": [
            {"expert": "info", "tool": "get_current_weather", "task_template": "查询{city}的天气"},
            {"expert": "math", "tool": "calculate_math", "task_template": "计算气温与{target_temp}的差值"},
        ],
    },
]
```

### GT 计算

因为所有工具都是 Mock 实现（返回固定数据），所以 GT 可以在生成时**预计算**：

1. 根据 plan 中的 tool + 参数，调用 `execute_tool()` 得到每步 mock 结果
2. 按依赖关系链式传递中间结果
3. 最终结果和各步中间结果作为 gt/step_gts 写入

这保证了 GT 与 RL 训练时的 Mock 环境完全一致。

### 条件分支数据的特殊处理

Level 3（条件分支）的 plan_gt 需要包含分支信息：

```json
{
  "query": "查杭州天气，如果超过30度换算华氏度，否则翻译成英文",
  "level": 3,
  "plan_gt": [
    {"step": 1, "expert": "info", "tool": "get_current_weather", "task": "查杭州天气", "depends_on": []},
    {"step": 2, "expert": "math", "tool": "calculate_math", "task": "将摄氏度换算为华氏度", "depends_on": [1], "condition": "step1.temperature > 30"},
    {"step": 2, "expert": "translate", "tool": "translate_text", "task": "将天气描述翻译成英文", "depends_on": [1], "condition": "step1.temperature <= 30"}
  ],
  "gt_branch": {
    "branch_key": "step1.temperature > 30",
    "true_gt": ["86", "华氏"],
    "false_gt": ["cloudy", "22"]
  },
  "dependency_type": "conditional"
}
```

由于 Mock 数据是确定的（杭州=22°C），我们可以预知走哪条分支，将对应的 gt 填入 `gt` 字段。但 `gt_branch` 保留完整信息，方便后续扩展随机化 Mock 数据。

## 训练策略建议

### 阶段一: SFT Warmup（~2 分钟）

用 60 条 SFT 数据教模型输出 `execute_plan` 格式。和现有 SFT warmup 串联：先用原有 90 条教单步路由，再追加 60 条教多步计划。

### 阶段二: RL 课程学习（Curriculum）

不直接上全部 500 条，而是按 Level 分阶段：

1. **Phase 1**（Level 1 only, 180 条）：先学会两步顺序链，这是最基础的多步能力。预计 1-2 epoch 收敛。
2. **Phase 2**（Level 1+2, 300 条）：加入并行聚合，学会同时调度多个 Expert。
3. **Phase 3**（Level 1+2+3, 400 条）：加入条件分支，学会根据中间结果做动态决策。
4. **Phase 4**（全部 500 条）：加入长链任务，挑战 3-4 步的复杂规划。

每个 Phase 跑 1-2 epoch，总训练量大约 1500-2000 steps。

### 与现有 Handoff 数据的关系

Plan-then-Execute 训练可以在现有 Handoff RL 训练完成后开始（从 Handoff checkpoint 继续），也可以**混合训练**——将 40 条单步兼容数据 + 原有 990 条 handoff 数据的一个子集混入，防止单步路由能力退化。

推荐混合比例：70% Plan-Execute + 30% 单步 Handoff。
