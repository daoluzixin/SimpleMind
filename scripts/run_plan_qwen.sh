#!/bin/bash
# ==============================================================================
# Plan-then-Execute RL 训练 — Qwen2.5-1.5B-Instruct 单卡启动脚本
#
# 在 agent_handoff_qwen.py（单步 Handoff）基础上扩展多步计划执行能力。
# 单张 4090 (24GB) 或 A100 即可运行。
#
# 用法:
#   bash scripts/run_plan_qwen.sh
# ==============================================================================

set -e

# ---- 激活 conda 环境 ----
source /home/vipuser/miniconda3/bin/activate

# ---- 模型与数据 ----
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH="./dataset/plan_execute_500.jsonl"
SAVE_DIR="./checkpoints_qwen_plan"

# ---- 训练超参 ----
EPOCHS=1                # 先跑 1 epoch 验证
BATCH_SIZE=1            # 单卡 batch size（Plan rollout 显存开销更大）
LR=1e-6                 # 1.5B RL 推荐学习率
NUM_GENERATIONS=2       # GRPO group 内候选数（4→2 加速 50%）
MAX_GEN_LEN=384         # 每阶段最大生成长度
MAX_TOTAL_LEN=4096      # 三段拼接后最大总长
BETA=0.1                # KL 惩罚系数
LOSS_TYPE="grpo"        # grpo 或 cispo
GRAD_CLIP=1.0
ACCUMULATION=2          # 梯度累积（等效 batch=2）

# ---- 课程学习（可选）----
# 不设置 = 加载全部 L0~L4
# 0=[L0,L1], 1=[L0-L2], 2=[L0-L3], 3=[L0-L4]
# CURRICULUM_PHASE=0

# ---- 工程参数 ----
LOG_INTERVAL=1
SAVE_INTERVAL=50        # Plan 训练单步更慢，适当增大保存间隔
MAX_KEEP=5
DTYPE="bfloat16"

# ---- 启动训练 ----
echo "================================================================"
echo "  Plan-then-Execute RL Training (Qwen2.5-1.5B-Instruct)"
echo "  Mode: Single GPU"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Save: ${SAVE_DIR}"
echo "  Effective batch: ${BATCH_SIZE} × ${ACCUMULATION} = $((BATCH_SIZE * ACCUMULATION))"
echo "================================================================"

CURRICULUM_ARG=""
if [ -n "${CURRICULUM_PHASE}" ]; then
    CURRICULUM_ARG="--curriculum_phase ${CURRICULUM_PHASE}"
    echo "  Curriculum Phase: ${CURRICULUM_PHASE}"
fi

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
    ${CURRICULUM_ARG}

echo "Plan-then-Execute Training complete!"
