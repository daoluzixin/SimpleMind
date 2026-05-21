# SimpleMind

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 LLM Agent 训练与推理实验项目。从 64M 参数超小模型的原型验证出发，逐步扩展到 Qwen2.5-1.5B 的工程化 RL 训练和 Qwen2.5-7B 的推理引擎实现，覆盖 **预训练 → SFT → 数据质量工程 → Agent 强化学习 → 推理优化** 完整链路。

---

## 核心实验

### 1. Plan-then-Execute 多步 Agent RL（Qwen2.5-1.5B）

在 2×A100-80GB 上训练 Router/Expert 多 Agent 架构。Router 接收用户请求后输出 `execute_plan` 工具调用，按拓扑序分派 Expert 执行子任务，最终 Synthesize 整合回答。

训练范式为 SFT Warmup（格式植入）→ GRPO 策略优化 → Eval 三阶段。引入 Router/Expert 信用分离机制，规划阶段和执行阶段的梯度独立更新，互不干扰。

测试集结果：多步规划准确率 98%（49/50）、端到端成功率 58%（29/50）。端到端损耗主要来自 Expert 的 tool_call 格式遵循问题——模型学会了正确决策但未完全学会输出格式，验证了 SFT+RL 二阶段训练的必要性。

### 2. SFT→RL 数据分布传递效应（Qwen2.5-1.5B）

在 Handoff 单步路由实验中发现：SFT warmup 78% 的 tool_call 样本配比直接决定了 RL 的策略起点，导致误触率（不需要工具时仍发起委托）收敛于 80%，RL 阶段的 -0.3 惩罚无法翻转这个先验。

根因诊断后设计三方修复：SFT 配比对齐目标分布（78%→56%）、RL 数据扩充 none 类样本（24%→37%）、非对称 reward 设计（误触惩罚系数 1.5x + 渐进式惩罚）。

### 3. 数据质量自动过滤（MiniMind 64M）

基于模型 PPL 信号实现自动数据过滤管线：PPL 批量计算 → KDE 谷点检测自动定阈值 → 长度归一化消除系统偏差。分桶对比实验（8 桶各自独立训练）验证倒 U 型贡献曲线——极端段 val loss 较中间段高 24%。裁剪 35% 低贡献数据后 val loss 无损。

### 4. Continuous Batching 推理引擎（Qwen2.5-7B, RTX 4090）

从零实现教学级 CB 引擎，包含三种模式：Serial（baseline）、CB Sequential Decode（调度优化）、CB Batched Decode（批量 forward）。适配 HuggingFace transformers 5.9.0 DynamicCache API，实现左 padding KV Cache 对齐。

32 请求压测下 P95 延迟从 36s 降至 2.4s（15x 改善）。Batch Scaling 实验（bs=1/2/4/8 吞吐无差异）定位 Python 层面 KV Cache pad/cat/slice 操作为吞吐瓶颈，实证了 PagedAttention 必须 CUDA 实现的根因。

### 5. 64M 模型 RL 能力边界探索（MiniMind 64M）

在 64M 参数模型上完成了 Agent RL 的完整方法论验证：v1~v3 排除超参干扰定位 reward 函数结构性 mismatch → v4/v5 修复后首次获得正奖励 → v5 后期捕获 Reward Hacking（输出退化为无意义短文本刷分）→ Plan RL v4~v6 逐步打通 plan→execute 链路。最终确认 64M 参数存在容量天花板（PlanRate 从 64% 缓慢衰减至 3%），但训练机制和方法论可直接迁移到更大模型。

---

## 项目结构

```
model/                          模型定义（Dense + MoE）
trainer/
  train_agent.py                Agent RL 训练（多轮工具调用 GRPO）
  train_plan.py                 Plan-then-Execute RL（64M, v6）
  agent_handoff.py              多 Agent Handoff 框架（64M）
  rollout_engine.py             Rollout 引擎
  train_pretrain.py             预训练
  train_full_sft.py             全参 SFT
  train_grpo.py                 GRPO 基础实现
  train_dpo.py / train_ppo.py   DPO / PPO
  train_lora.py                 LoRA 微调
  train_distillation.py         知识蒸馏
  logs/                         全量实验日志（64M + 1.5B）
agent_handoff_qwen.py           Qwen2.5-1.5B Handoff RL 主训练脚本
agent_plan_qwen.py              Qwen2.5-1.5B Plan-then-Execute RL
sft_warmup_qwen.py              Qwen2.5-1.5B SFT Warmup
sft_plan_warmup.py              Plan 格式 SFT 数据准备
eval_handoff.py                 Handoff 评测脚本
eval_plan.py                    Plan-then-Execute 评测脚本
scripts/
  continuous_batching_engine.py CB 推理引擎核心实现
  benchmark_continuous_batching.py  CB 性能压测
  serve_openai_api.py           OpenAI 兼容 API 服务
  data_quality_scorer.py        PPL 数据质量评分与过滤
  web_demo.py                   WebUI 演示
  run_*.sh                      各实验启动脚本
benchmark_logs/                 CB 压测原始日志
docs/
  实验记录/                     完整实验报告（5 篇）
  操作指南/                     DDP 训练等操作文档
```

---

## 实验报告

`docs/实验记录/` 目录保存了每个实验方向的完整报告：

| 报告 | 内容 |
|------|------|
| plan_execute_experiment_report.md | Plan-then-Execute 全流程：性能优化（18min→7s/step）+ 两阶段训练 + Demo 诊断 |
| handoff_qwen_experiment_report.md | Handoff 单步路由：SFT→RL 传递效应发现 + 三方修复方案 |
| agent_rl_experiment_report.md | 64M Agent RL 完整迭代：v1~v5 + Plan RL v4~v6 + 能力边界确认 |
| continuous_batching_benchmark_report.md | CB 推理引擎：4 组压测 + Python overhead 根因分析 |

---

## 技术栈

- PyTorch 原生实现，不依赖 trl/peft 高层封装
- 训练：GRPO/CISPO 策略优化、DDP 分布式、gradient checkpointing + KV cache 推理分离
- 推理：Continuous Batching 调度、DynamicCache API、OpenAI 兼容 API
- 数据：PPL 自动过滤、KDE 分布分析、分桶消融实验

---

## 快速开始

```bash
pip install -r requirements.txt

# Qwen2.5-1.5B Handoff RL 训练（需要 2×A100）
bash scripts/run_handoff_qwen_single.sh

# Qwen2.5-1.5B Plan-then-Execute RL
bash scripts/run_plan_qwen.sh

# 64M Agent RL 训练（单卡 4090 即可）
python trainer/train_agent.py --mode train --data_path dataset/xlam_agent_rl.jsonl

# CB 推理引擎压测（Qwen2.5-7B, 单卡 4090）
bash scripts/run_benchmark.sh

# 评测
python eval_handoff.py
python eval_plan.py
```

---

## 致谢

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开源项目开发，感谢原作者提供的完整训练代码和预训练数据。

## License

Apache 2.0
