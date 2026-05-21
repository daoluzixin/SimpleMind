#!/bin/bash
# ==============================================================================
# Plan-then-Execute SFT Warm-up — 教 Qwen2.5-1.5B-Instruct <tool_call> 格式
#
# 用途：在 RL 训练之前，用 120 条标注样本教会模型正确的 tool_call 输出格式。
# 训练仅 2 epoch，耗时约 1-2 分钟（A100-80GB）。
#
# 流程：
#   1. SFT Warm-up (本脚本)
#   2. RL 训练 (scripts/run_plan_qwen.sh，--model_path 改为 SFT checkpoint)
#
# 用法:
#   bash scripts/run_sft_warmup.sh
# ==============================================================================

set -e

# ---- 激活 conda 环境 ----
source /home/vipuser/miniconda3/bin/activate

# ---- 模型与数据 ----
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH="./dataset/sft_plan_warmup.jsonl"
SAVE_DIR="./checkpoints_qwen_plan_sft"

# ---- 训练超参 ----
EPOCHS=2                # 2 epoch 足够教会格式
BATCH_SIZE=4            # SFT 可用更大 batch（序列短，显存充裕）
LR=2e-5                 # SFT 学习率（比 RL 高 1 个数量级）
MAX_SEQ_LEN=1024        # 最大序列长度
GRAD_CLIP=1.0
DTYPE="bfloat16"

# ---- 日志 ----
LOG_INTERVAL=1

# ---- 启动训练 ----
echo "================================================================"
echo "  Plan-then-Execute SFT Warm-up (Qwen2.5-1.5B-Instruct)"
echo "  Purpose: Teach <tool_call> output format"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Save: ${SAVE_DIR}"
echo "  Config: ${EPOCHS} epochs, batch=${BATCH_SIZE}, lr=${LR}"
echo "================================================================"

# Step 1: 验证数据编码
echo ""
echo "[Step 1] Verifying data encoding..."
python sft_plan_warmup.py \
    --mode verify \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --max_seq_len ${MAX_SEQ_LEN}

echo ""
echo "[Step 2] Starting SFT training..."
python sft_plan_warmup.py \
    --mode train \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --grad_clip ${GRAD_CLIP} \
    --gradient_checkpointing 1 \
    --dtype ${DTYPE} \
    --log_interval ${LOG_INTERVAL}

echo ""
echo "================================================================"
echo "  SFT Warm-up Complete!"
echo "  Best checkpoint: ${SAVE_DIR}/best"
echo ""
echo "  Next step — RL training with SFT checkpoint:"
echo "    修改 scripts/run_plan_qwen.sh 中:"
echo "    MODEL_PATH=\"${SAVE_DIR}/best\""
echo "    然后运行: bash scripts/run_plan_qwen.sh"
echo "================================================================"
