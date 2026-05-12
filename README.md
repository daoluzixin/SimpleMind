# SimpleMind

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 Agent RL 实验项目。在原版 64M 参数超小语言模型训练框架基础上，探索超小模型在 Agentic 场景下通过强化学习获得工具调用和推理规划能力的可行性与边界。

---

## 实验概览

本项目在 MiniMind 的 64M Dense 模型上进行了三条实验线：

**Agent RL（工具调用强化学习）** — 训练模型学会识别何时调用工具、生成正确的 function call 并理解返回结果。基于 GRPO/CISPO 框架，多轮 rollout + GT 验证 + 工具调用正确性综合评分。

**Plan-then-Execute RL（规划+执行）** — 在工具调用之前强制模型先输出结构化执行计划（`<plan>` 标签），再按计划执行工具，最后总结回答。Reward 包含格式分、计划对齐分、质量分和执行分四个维度。经历 v4→v6 共 6 个版本迭代。

**Agent Handoff（多 Agent 协同）** — RouterAgent 负责意图识别和任务分发，Expert Agents 负责实际工具调用。所有 Agent 共享权重但使用不同 system prompt 区分角色。代码框架已实现（1185 行），未进行训练实验。

---

## 实验结论

Plan RL 实验跑满全线（v4~v6），核心发现：**64M 参数存在明确的容量天花板**。

- v6 跑完 1140 步，PlanRate 从 64%（SFT 惯性）缓慢衰减至 3%，DegenRate 始终为 0%（无急性崩塌）
- 推理测试（v6 step500）显示模型学会了 plan 格式壳子（PlanRate 100%），但语义质量极差：参数照抄 few-shot、步骤重复、工具选择错误
- 结论：64M 参数不足以支撑 plan→execute 的语义推理，RL 只能强化表面模式而非真正的规划能力

Agent RL 基线实验验证了工具调用训练流程的可行性，模型能学会 function call 的格式但泛化能力有限。

---

## 项目结构

```
model/                  模型定义（Dense + MoE，对齐 Qwen3 生态）
trainer/
  train_agent.py        Agent RL 训练（多轮工具调用 GRPO/CISPO）
  train_plan.py         Plan-then-Execute RL 训练（v5 修复版）
  agent_handoff.py      多 Agent 协同框架（RouterAgent + Expert Agents）
  rollout_engine.py     Rollout 引擎（解耦生成后端）
  trainer_utils.py      训练工具函数
  train_*.py            其他训练脚本（pretrain/sft/lora/dpo/ppo/grpo/distillation）
  logs/                 实验日志（sft/agent_rl/plan_warmup/plan_rl 全版本）
scripts/                推理服务与工具（OpenAI API / WebUI / 模型转换）
test_plan_gen.py        Plan 模型推理测试脚本
```

---

## 训练日志

`trainer/logs/` 保留了全部有实验价值的完整日志：

| 日志 | 实验内容 |
|------|----------|
| sft_0509_1733 | SFT 基线训练 |
| ppl_score / ppl_filter | PPL 数据质量评估与过滤 |
| bucket_exp | 数据桶策略实验 |
| agent_rl_0510 | Agent RL 工具调用训练 |
| plan_warmup / v2 | Plan SFT 预热（教模型 plan 格式） |
| plan_rl_v4~v6 | Plan RL 正式训练（核心实验） |
| v4_reward_fix / v5_lr1e6_reward_fix | Reward 修复对照实验 |
| inference_v6_step500 | v6 模型推理质量评估 |

---

## 技术栈

- PyTorch 原生实现，不依赖 transformers/trl/peft 的高层封装
- Tokenizer: BPE + ByteLevel，支持 `<tool_call>` / `<tool_response>` / `<think>` / `<plan>` 标记
- 训练框架: DDP 分布式 + wandb 可视化 + 断点续训
- 推理: 兼容 OpenAI API 协议，支持 llama.cpp / vllm / ollama

---

## 快速开始

```bash
pip install -r requirements.txt

# Agent RL 训练
python trainer/train_agent.py --mode train --data_path dataset/agent_rl.jsonl

# Plan RL 训练
python trainer/train_plan.py --mode train --data_path dataset/agent_rl.jsonl

# 推理测试
python test_plan_gen.py

# 完整训练链路: Pretrain → SFT → LoRA → DPO/PPO/GRPO → Agentic RL → Plan-then-Execute
python trainer/train_pretrain.py
python trainer/train_full_sft.py
python trainer/train_grpo.py
```

---

## 致谢

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开源项目开发，感谢原作者提供的完整训练代码与数据集。

## License

Apache 2.0
