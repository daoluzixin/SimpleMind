#!/bin/bash
# ==============================================================================
# MiniMind-Agent Handoff — Qwen2.5-1.5B-Instruct 单卡训练启动脚本
#
# 单张 4090 (24GB) 即可运行，无需跨节点通信
# 用法:
#   bash scripts/run_handoff_qwen_single.sh
# ==============================================================================

set -e

# ---- 激活 conda 环境 ----
source /usr/local/miniconda3/bin/activate py312

# ---- 模型与数据 ----
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH="./dataset/agent_handoff.jsonl"
SAVE_DIR="./checkpoints_qwen_handoff"

# ---- 训练超参 ----
EPOCHS=3
BATCH_SIZE=1            # 单卡 batch size
LR=1e-6                 # 1.5B RL 推荐学习率
NUM_GENERATIONS=4       # GRPO group 内候选数
MAX_GEN_LEN=384         # 每阶段最大生成长度
MAX_TOTAL_LEN=4096      # 三段拼接后最大总长
BETA=0.1                # KL 惩罚系数
LOSS_TYPE="grpo"        # grpo 或 cispo
GRAD_CLIP=1.0
ACCUMULATION=2          # 梯度累积补偿（等效 batch=2）

# ---- 工程参数 ----
LOG_INTERVAL=1
SAVE_INTERVAL=10
MAX_KEEP=5
DTYPE="bfloat16"

# ---- 启动单卡训练 ----
echo "================================================================"
echo "  MiniMind-Agent Handoff Training (Qwen2.5-1.5B-Instruct)"
echo "  Mode: Single GPU"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Save: ${SAVE_DIR}"
echo "  Effective batch: ${BATCH_SIZE} × ${ACCUMULATION} = $((BATCH_SIZE * ACCUMULATION))"
echo "================================================================"

python agent_handoff_qwen.py \
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
    --gradient_checkpointing 1 \
    --dtype ${DTYPE} \
    --log_interval ${LOG_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    --max_keep ${MAX_KEEP} \
    --wandb_project "MiniMind-Agent-Handoff-Qwen"

echo "Training complete!"
