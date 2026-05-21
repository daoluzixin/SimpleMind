#!/bin/bash
# ==============================================================================
# MiniMind-Agent Handoff — Qwen2.5-1.5B-Instruct 单机双卡 DDP 训练
#
# 硬件: 单机 2× NVIDIA A100-SXM4-80GB
# 用法:
#   前台运行: bash scripts/run_handoff_qwen_ddp.sh
#   后台运行: nohup bash scripts/run_handoff_qwen_ddp.sh &
#   查看日志: tail -f /root/minimind/logs/handoff_train_*.log
# ==============================================================================

set -e

# ---- 激活 conda 环境 ----
source /home/vipuser/miniconda3/bin/activate base

# ---- 项目目录 ----
cd /root/minimind

# ---- 时间戳 & 日志 ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/handoff_train_${TIMESTAMP}.log"

# ---- 强制 Python 无缓冲输出，确保日志实时刷盘 ----
export PYTHONUNBUFFERED=1

# ---- 单机双卡配置 ----
NNODES=1
NPROC_PER_NODE=2
MASTER_ADDR="localhost"
MASTER_PORT=29500

# ---- 模型与数据 ----
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH="./dataset/agent_handoff.jsonl"
SAVE_DIR="./checkpoints_qwen_handoff"

# ---- 训练超参 ----
EPOCHS=3
BATCH_SIZE=4             # per-GPU batch size，全局 batch = 1×2×4 = 8
LR=5e-6                  # 调大学习率，加快格式突破
NUM_GENERATIONS=4        # GRPO group 内候选数
MAX_GEN_LEN=384          # 每阶段最大生成长度
MAX_TOTAL_LEN=4096       # 三段拼接后最大总长
BETA=0.1                 # KL 惩罚系数
LOSS_TYPE="grpo"         # grpo 或 cispo
GRAD_CLIP=1.0
ACCUMULATION=1

# ---- 工程参数 ----
LOG_INTERVAL=5
SAVE_INTERVAL=50         # 每50步存一次checkpoint
MAX_KEEP=3
DTYPE="bfloat16"

# ---- NCCL 调优（单机 NVLink） ----
export NCCL_P2P_LEVEL=NVL          # A100-SXM4 走 NVLink
export NCCL_IB_DISABLE=0           # 单机不需要 IB 但不影响
export NCCL_DEBUG=WARN             # 减少日志噪音，出问题改 INFO

# ---- CUDA 性能优化 ----
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TOKENIZERS_PARALLELISM=false

# ---- 启动信息 ----
{
echo "================================================================"
echo "  MiniMind-Agent Handoff Training (Qwen2.5-1.5B-Instruct + DDP)"
echo "  Hardware: 1 node × 2 A100-80G (NVLink)"
echo "  Global batch: ${NPROC_PER_NODE} GPUs × ${BATCH_SIZE} = $((NPROC_PER_NODE * BATCH_SIZE))"
echo "  Epochs: ${EPOCHS}, LR: ${LR}, Beta: ${BETA}"
echo "  Gradient Checkpointing: OFF (80G充裕)"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Save: ${SAVE_DIR}"
echo "  Log: ${LOG_FILE}"
echo "  Start: $(date)"
echo "================================================================"
} | tee ${LOG_FILE}

# ---- 启动 DDP 训练（stdbuf 禁用管道缓冲，保证实时刷盘） ----
stdbuf -oL -eL torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    agent_handoff_qwen.py \
    --mode train \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --num_generations ${NUM_GENERATIONS} \
    --max_gen_len ${MAX_GEN_LEN} \
    --max_total_len ${MAX_TOTAL_LEN} \
    --beta ${BETA} \
    --loss_type ${LOSS_TYPE} \
    --grad_clip ${GRAD_CLIP} \
    --accumulation_steps ${ACCUMULATION} \
    --gradient_checkpointing 0 \
    --dtype ${DTYPE} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --max_keep ${MAX_KEEP} \
    --wandb_project "MiniMind-Agent-Handoff-Qwen" \
    --from_resume 1 \
    --resume_mode 50 \
    2>&1 | stdbuf -oL tee -a ${LOG_FILE}

{
echo "================================================================"
echo "  Training complete! $(date)"
echo "  Log saved: ${LOG_FILE}"
echo "================================================================"
} | tee -a ${LOG_FILE}
