# Qwen2.5-1.5B Agent Handoff — DDP 分布式训练操作指南

> 双卡 4090 (24GB × 2) 环境，基于 `agent_handoff_qwen.py` 的完整操作流程。

## 一、背景与目标

本文档描述如何在双卡 RTX 4090 上，使用 PyTorch DDP（DistributedDataParallel）对 Qwen2.5-1.5B-Instruct 进行 Agent Handoff 强化学习训练。训练目标是让模型学会多 Agent 协作：RouterAgent 负责意图路由，Expert Agent 执行工具调用，最终由 Router 整合结果。

训练算法为 GRPO（Group Relative Policy Optimization），每个 query 生成 4 个候选回答，组内归一化计算 advantage，分离 Router/Expert 的 credit assignment。

## 二、环境准备

### 2.1 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | 2 × NVIDIA RTX 4090 (24GB) |
| CPU 内存 | ≥ 32GB（模型加载时需要 CPU 暂存） |
| 磁盘 | ≥ 20GB（模型 ~3GB + checkpoint ~3GB/份 × 5） |
| CUDA | ≥ 12.1 |
| NCCL | 随 PyTorch 自带即可 |

### 2.2 软件依赖

```bash
# 核心依赖
pip install torch>=2.1.0 transformers>=4.37.0 accelerate modelscope

# 可选（推荐）
pip install flash-attn --no-build-isolation   # Flash Attention 2，显著加速
pip install swanlab                             # 训练可视化（wandb 替代）
pip install numpy
```

### 2.3 下载模型

国内推荐使用 ModelScope 下载（速度快）：

```bash
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
print(f'模型已下载到: {model_dir}')
"
```

默认下载路径为 `~/.cache/modelscope/hub/Qwen/Qwen2.5-1.5B-Instruct`。如果要自定义路径：

```bash
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
"
```

## 三、DDP 分布式训练原理（简要）

### 3.1 为什么用 DDP

单卡 4090 (24GB) 跑 1.5B 模型的 RL 训练很紧张：模型参数 ~3GB (bf16)，ref model 又 ~3GB，加上 optimizer states (~6GB for AdamW)、梯度 (~3GB)、激活值——单卡峰值显存超过 24GB。DDP 双卡分摊后，每卡只负责一半 batch 的计算，峰值显存可控。

### 3.2 DDP 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  torchrun --nproc_per_node=2 agent_handoff_qwen.py --mode train │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
       ┌──────┴──────┐                ┌──────┴──────┐
       │   rank 0    │                │   rank 1    │
       │  (cuda:0)   │                │  (cuda:1)   │
       └──────┬──────┘                └──────┬──────┘
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │ 加载完整模型+ref   │          │ 加载完整模型+ref   │
    │ DDP 包裹 model    │          │ DDP 包裹 model    │
    └─────────┬─────────┘          └─────────┬─────────┘
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │ DistributedSampler │          │ DistributedSampler │
    │ → 数据前半部分     │          │ → 数据后半部分     │
    └─────────┬─────────┘          └─────────┬─────────┘
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │ Rollout (no_grad)  │          │ Rollout (no_grad)  │
    │ → 生成候选回答     │          │ → 生成候选回答     │
    └─────────┬─────────┘          └─────────┬─────────┘
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │ Forward (DDP)      │          │ Forward (DDP)      │
    │ → 计算策略 loss    │          │ → 计算策略 loss    │
    └─────────┬─────────┘          └─────────┬─────────┘
              │                               │
              │         AllReduce 梯度         │
              │◄──────────────────────────────►│
              │                               │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │ optimizer.step()   │          │ optimizer.step()   │
    │ (同步后的梯度)     │          │ (同步后的梯度)     │
    └────────────────────┘          └────────────────────┘
```

### 3.3 关键设计决策

| 决策 | 原因 |
|------|------|
| Rollout 用 unwrapped model | 推理阶段不需要梯度同步，避免死锁 |
| Train forward 用 DDP wrapper | 必须经过 DDP 的 forward hook 才能触发 allreduce |
| `find_unused_parameters=True` | gradient_checkpointing 导致部分参数延迟参与计算图 |
| `gradient_checkpointing=True` | 1.5B 在 24GB 卡上不开会 OOM |
| ref_model 不用 DDP | 冻结参数，仅用于计算 KL 基准，不需要梯度同步 |
| checkpoint 后 barrier | 等待 rank 0 完成保存，防止进程不同步 |

## 四、操作步骤

### 4.1 准备训练数据

训练数据格式为 JSONL，每行一个样本：

```json
{"query": "北京今天天气怎么样？", "gt": ["28°C", "晴"], "needs_tool": true, "expert": "info"}
{"query": "帮我算 (15+27)*3", "gt": ["126"], "needs_tool": true, "expert": "math"}
{"query": "你觉得AI未来怎样？", "gt": [], "needs_tool": false, "expert": "none"}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| query | string | 用户问题 |
| gt | list[string] | Ground Truth 答案（用于奖励验证） |
| needs_tool | bool | 是否需要工具调用 |
| expert | string | 期望的专家类型：math/info/translate/none |

将数据放到 `./dataset/agent_handoff.jsonl`。

### 4.2 启动训练

**方式一：使用启动脚本（推荐）**

```bash
bash scripts/run_handoff_qwen_ddp.sh
```

脚本内部会执行 `torchrun --nproc_per_node=2`。可编辑脚本修改超参。

**方式二：直接命令行**

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
    agent_handoff_qwen.py \
    --mode train \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --data_path ./dataset/agent_handoff.jsonl \
    --save_dir ./checkpoints_qwen_handoff \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 1e-6 \
    --num_generations 4 \
    --max_gen_len 384 \
    --max_total_len 4096 \
    --beta 0.1 \
    --loss_type grpo \
    --gradient_checkpointing 1 \
    --dtype bfloat16 \
    --save_interval 10 \
    --use_wandb
```

**方式三：后台训练（服务器推荐）**

```bash
mkdir -p logs
nohup bash scripts/run_handoff_qwen_ddp.sh > logs/handoff_qwen_$(date +%m%d_%H%M).log 2>&1 &
```

查看进度：`tail -f logs/handoff_qwen_*.log`

### 4.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | Qwen/Qwen2.5-1.5B-Instruct | 模型 HF id 或本地路径 |
| `--batch_size` | 1 | per-GPU batch size，全局 batch = 2 |
| `--learning_rate` | 1e-6 | 1.5B RL 推荐的小学习率 |
| `--num_generations` | 4 | GRPO group 内候选数 |
| `--max_gen_len` | 384 | 每阶段最大生成 token 数 |
| `--max_total_len` | 4096 | 三段拼接后最大总长 |
| `--beta` | 0.1 | KL 散度惩罚系数 |
| `--loss_type` | grpo | grpo 或 cispo |
| `--gradient_checkpointing` | 1 | 必须开，否则 OOM |
| `--epochs` | 3 | 训练轮数 |
| `--save_interval` | 10 | 每 N 步保存一次 checkpoint |

### 4.4 监控训练

日志输出格式：

```
[Epoch 1/3] Step 5/100 | loss=0.2341 | reward=1.234 | router_r=0.567 | expert_r=0.667 | handoff=75% | kl=0.0023
```

关键指标解读：

| 指标 | 健康范围 | 异常信号 |
|------|----------|----------|
| reward | 逐步上升 → 1.5~3.0 | 持续 < 0 或剧烈震荡 |
| router_r | > 0.3 | 持续为负 = 路由学不会 |
| expert_r | > 0.3 | 持续为负 = 工具调用错误 |
| handoff_rate | 60%~80% | 100% = 所有问题都委托（过拟合） |
| kl | < 0.1 | > 0.5 = 模型偏离 ref 太远 |
| loss | 逐步下降 | NaN = 学习率太大或数据问题 |

如果使用了 `--use_wandb`，可以在 SwanLab 面板实时查看曲线。

### 4.5 断点续训

训练中断后恢复：

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
    agent_handoff_qwen.py \
    --mode train \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --from_resume 1 \
    --resume_mode latest \
    ...（其他参数不变）
```

`resume_mode` 支持三种：
- `latest`：恢复到最近一次保存点
- `best`：恢复到 reward 最高的检查点
- 数字（如 `50`）：恢复到第 50 步的检查点

### 4.6 Demo 验证

训练完成后，用 demo 模式验证效果：

```bash
# 加载最佳 checkpoint
python agent_handoff_qwen.py \
    --mode demo \
    --model_path ./checkpoints_qwen_handoff/best \
    --max_gen_len 384
```

或加载特定步的 checkpoint：

```bash
python agent_handoff_qwen.py \
    --mode demo \
    --model_path ./checkpoints_qwen_handoff/step_30
```

## 五、显存分析

### 5.1 预估显存占用（单卡 per-GPU）

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型参数 (bf16) | ~3.0 GB | 1.5B × 2 bytes |
| Ref model (bf16) | ~3.0 GB | 冻结，eval 模式 |
| Optimizer states | ~6.0 GB | AdamW 2×fp32 |
| 梯度 (bf16) | ~3.0 GB | 与参数等大 |
| 激活值 (checkpointing) | ~4.0 GB | 大幅减少（不开则 >12GB） |
| Rollout KV Cache | ~2.0 GB | 生成阶段临时占用 |
| **总计** | **~21 GB** | 24GB 卡有 ~3GB 余量 |

### 5.2 如果 OOM

如果遇到显存不足，依次尝试：

1. 确认 `--gradient_checkpointing 1` 已开启
2. 减小 `--max_gen_len`（如 256）
3. 减小 `--max_total_len`（如 3072）
4. 确认 `--batch_size 1`（不能再小了）
5. 减小 `--num_generations`（如 2，但会影响 GRPO 效果）

## 六、常见问题

### Q1: `NCCL error: unhandled system error`

通常是两张卡之间的通信问题。排查：
```bash
# 检查 GPU 状态
nvidia-smi

# 确认两张卡可见
echo $CUDA_VISIBLE_DEVICES

# 尝试指定网络接口
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

### Q2: 训练速度很慢

Agent Handoff 的四阶段 rollout 天然较慢（每个样本需要多次 generate）。优化建议：
- 确认 Flash Attention 2 已安装（看启动日志有无 flash_attention_2）
- 减小 `--max_gen_len` 到能覆盖工具调用的最短长度
- 数据集中混入一定比例的 `needs_tool=false` 样本（直接回答，不走完整四阶段）

### Q3: `RuntimeError: Expected all tensors to be on the same device`

检查 `--device` 参数是否被正确覆盖为 `cuda:{local_rank}`。DDP 模式下不需要手动指定 `--device`，代码会自动根据 `LOCAL_RANK` 设置。

### Q4: reward 一直为负

- 检查 `apply_chat_template` 是否正确生成了工具调用格式
- 用 `--mode demo` 看模型是否能正确解析工具调用
- Qwen2.5 的工具调用格式可能需要在 TOOLS 定义中微调

### Q5: 两张卡利用率不均衡

RL 训练中 rollout 是顺序的（每个样本需要多轮交互），DDP 只同步梯度不同步 rollout。如果数据集中不同样本的 rollout 长度差异大，会导致短的那个进程等待长的。解决方法是让 `DistributedSampler` 的 shuffle 更均匀，或设置 `--max_gen_len` 限制最大生成长度。

## 七、Checkpoint 目录结构

```
checkpoints_qwen_handoff/
├── ckpt_history.json          # 历史记录（步数、指标、时间戳）
├── best/                      # 最佳 checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── train_state.pt         # optimizer + scheduler + epoch/step
├── step_10/                   # 第 10 步
│   ├── ...（同 best）
├── step_20/                   # 第 20 步
│   ├── ...
└── ...（最多保留 max_keep=5 份）
```

使用 HuggingFace 原生格式保存（`save_pretrained`），可直接用于后续推理或上传到 Model Hub。

## 八、训练完成后

训练完成后会输出 `Training complete.`。建议操作：

1. **验证效果**：用 demo 模式检查模型是否学会了正确的路由和工具调用
2. **保留 best checkpoint**：`checkpoints_qwen_handoff/best/` 是 reward 最高的版本
3. **记录指标**：最终 reward、handoff_rate、kl 等，用于简历撰写
4. **清理中间 checkpoint**：只保留 best 和最后一个 step 即可

```bash
# 查看训练历史
python -c "
import json
with open('./checkpoints_qwen_handoff/ckpt_history.json') as f:
    data = json.load(f)
print(f'Best step: {data[\"best_step\"]}, metric: {data[\"best_metric\"]:.4f}')
for h in data['history']:
    print(f'  step={h[\"step\"]:3d} | {h[\"metrics\"]} | {h[\"timestamp\"]}')
"
```
