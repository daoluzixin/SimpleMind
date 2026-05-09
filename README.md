# SimpleMind

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的个人学习与扩展项目。在原版 MiniMind（64M 参数超小语言模型训练框架）基础上，深入研究 LLM 从结构到训练的完整链路，并扩展了 Agent RL、多 Agent 协同、Plan-then-Execute 等前沿训练范式。

---

## 项目定位

MiniMind 是一个"从零训练小型语言模型"的开源教程项目——所有核心算法均用 PyTorch 原生实现，不依赖 transformers/trl/peft 等框架的高层封装。SimpleMind 在此基础上作为个人的 LLM 底层学习教材，同时扩展了以下研究方向：

- **Agentic RL 训练**：多轮 Tool-Use 场景下的 GRPO/CISPO 强化学习，模型学会识别何时调用工具、正确生成调用请求、理解返回结果并生成最终回答
- **Agent Handoff 多 Agent 协同**：RouterAgent 意图路由 + 专家 Agent 执行的 A2A 协作架构，所有 Agent 共享权重但通过 system prompt 区分角色
- **Plan-then-Execute 训练范式**：强制模型先输出结构化执行计划（`<plan>` 标签），再按计划执行工具调用，reward 包含 plan 格式/对齐/质量/执行四维评估
- **完整 RLHF/RLAIF 链路**：PPO、GRPO、CISPO、DPO 等算法从零实现，含 Rollout Engine 解耦设计

---

## 模型架构

主线结构对齐 Qwen3/Qwen3-MoE 生态，核心组件包括：

| 组件 | 实现 |
|------|------|
| 归一化 | RMSNorm（替代 LayerNorm） |
| 位置编码 | RoPE + YaRN 长文本外推 |
| 注意力 | GQA + Flash Attention + KV Cache |
| 前馈网络 | SwiGLU 激活 |
| MoE | 门控路由 + 辅助负载均衡损失 |
| 词表 | BPE + ByteLevel，含 `<tool_call>`/`<think>` 等特殊标记 |

默认配置：Dense 约 64M 参数，MoE 约 198M-A64M 参数。

---

## 项目结构

```
.
├── model/
│   ├── model_minimind.py      # 模型核心定义（Config/Attention/FFN/MoE/CausalLM）
│   ├── model_lora.py          # LoRA 从零实现
│   └── tokenizer.json         # BPE 分词器
├── trainer/
│   ├── train_pretrain.py      # 预训练
│   ├── train_full_sft.py      # 全量 SFT
│   ├── train_lora.py          # LoRA 微调
│   ├── train_dpo.py           # DPO 对齐
│   ├── train_grpo.py          # GRPO 强化学习
│   ├── train_ppo.py           # PPO 强化学习
│   ├── train_agent.py         # Agentic RL（多轮工具调用训练）
│   ├── train_plan.py          # Plan-then-Execute 训练
│   ├── agent_handoff.py       # 多 Agent 协同训练
│   ├── train_distillation.py  # 知识蒸馏
│   ├── rollout_engine.py      # Rollout 推理引擎
│   └── trainer_utils.py       # 训练工具函数
├── scripts/
│   ├── serve_openai_api.py    # OpenAI 兼容 API 服务
│   ├── web_demo.py            # Streamlit 聊天 WebUI
│   ├── chat_api.py            # 工具调用推理示例
│   └── convert_model.py       # LoRA 权重合并导出
├── dataset/
│   ├── lm_dataset.py          # 数据加载器
│   ├── pretrain_t2t_mini.jsonl
│   ├── sft_t2t_mini.jsonl
│   ├── rlaif.jsonl
│   └── agent_rl.jsonl
├── eval_llm.py                # 模型评测入口
└── requirements.txt
```

---

## 训练流程

完整的训练链路覆盖从预训练到 Agent RL 的全过程：

**Pretrain → SFT → LoRA → DPO/PPO/GRPO → Agentic RL → Plan-then-Execute → Agent Handoff**

每个阶段均可独立运行，支持单机单卡和单机多卡（DDP/DeepSpeed），支持 wandb/swanlab 可视化和断点续训。

### 快速复现

```bash
# 环境准备
pip install -r requirements.txt

# 预训练（需要 pretrain_t2t_mini.jsonl）
python trainer/train_pretrain.py

# SFT（需要 sft_t2t_mini.jsonl）
python trainer/train_full_sft.py

# GRPO 强化学习
python trainer/train_grpo.py

# Agentic RL 工具调用训练
python trainer/train_agent.py

# Plan-then-Execute
python trainer/train_plan.py

# 多 Agent 协同
python trainer/agent_handoff.py
```

### 推理与部署

```bash
# CLI 推理
python eval_llm.py --load_from ./minimind-3

# OpenAI 兼容 API
python scripts/serve_openai_api.py

# WebUI（支持思考展示、工具选择、多轮 Tool Call）
cd scripts && streamlit run web_demo.py

# 第三方推理框架
ollama run jingyaogong/minimind-3
```

---

## 技术亮点

**全链路从零实现**：所有核心算法（GRPO/PPO/DPO/LoRA/蒸馏/MoE 路由/RoPE/YaRN）均用 PyTorch 原生代码实现，不依赖 trl/peft 等库的高层封装，每一行代码都可追溯其数学原理。

**Agentic RL 多轮工具调用**：模型在 Rollout 中进行多轮交互（生成 → 调用工具 → 观察结果 → 继续生成），reward 综合评估工具调用正确性、GT 匹配、格式规范，训练模型从"会说话"进化到"会做事"。

**Plan-then-Execute**：引入显式规划阶段，reward 由 plan_format + plan_adherence + plan_quality + execution 四部分组成，训练模型同时具备规划与执行能力。

**Agent Handoff A2A 协作**：RouterAgent 负责意图路由，专家 Agent 负责执行，所有角色共享模型权重，通过 Rollout 四阶段流程实现端到端训练。

---

## 致谢

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开源项目，感谢原作者提供的优秀教程框架。

## License

Apache 2.0
