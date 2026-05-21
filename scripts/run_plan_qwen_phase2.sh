#!/bin/bash
# ==============================================================================
# Plan-then-Execute RL 训练 — 第二阶段（后 500 条）
#
# 从第一阶段的 best checkpoint 继续训练，使用后 500 条数据。
# ==============================================================================

set -e

source /home/vipuser/miniconda3/bin/activate

# ---- 模型与数据 ----
# 从第一阶段 best checkpoint 继续（run_plan_training 会自动找 best）
MODEL_PATH="./checkpoints_qwen_plan"
DATA_PATH="./dataset/plan_execute_500_part2.jsonl"
SAVE_DIR="./checkpoints_qwen_plan"

# ---- 训练超参（与第一阶段一致）----
EPOCHS=1
BATCH_SIZE=1
LR=1e-6
NUM_GENERATIONS=2
MAX_GEN_LEN=384
MAX_TOTAL_LEN=4096
BETA=0.1
LOSS_TYPE="grpo"
GRAD_CLIP=1.0
ACCUMULATION=2

# ---- 工程参数 ----
LOG_INTERVAL=1
SAVE_INTERVAL=50
MAX_KEEP=5
DTYPE="bfloat16"

echo "================================================================"
echo "  Plan-then-Execute RL Training — Phase 2 (后 500 条)"
echo "  Mode: Single GPU, resume from Phase 1 best checkpoint"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Save: ${SAVE_DIR}"
echo "  Effective batch: ${BATCH_SIZE} × ${ACCUMULATION} = $((BATCH_SIZE * ACCUMULATION))"
echo "================================================================"

python agent_plan_qwen.py \
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
    --wandb_project "MiniMind-Agent-Plan-Qwen" \
    --from_resume 1 \
    --resume_mode best

echo "Phase 2 Training complete!"
