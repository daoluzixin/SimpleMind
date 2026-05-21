#!/bin/bash
# RL v2 训练 - 基于 SFT warmup checkpoint，从 step_50 resume
# 不开 gradient_checkpointing（与 DDP 冲突），batch=2, num_gen=2 省内存
set -e

source /home/vipuser/miniconda3/bin/activate base
cd /root/minimind

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"
mkdir -p $LOG_DIR

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start RL v2: $(date)" | tee ${LOG_DIR}/handoff_rl_v2_${TIMESTAMP}.log

stdbuf -oL torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    agent_handoff_qwen.py \
    --mode train \
    --model_path ./checkpoints_sft_warmup/best \
    --data_path ./dataset/agent_handoff.jsonl \
    --save_dir ./checkpoints_qwen_handoff_v2 \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --num_generations 2 \
    --max_gen_len 256 \
    --max_total_len 2048 \
    --beta 0.1 \
    --loss_type grpo \
    --grad_clip 1.0 \
    --accumulation_steps 2 \
    --gradient_checkpointing 0 \
    --dtype bfloat16 \
    --log_interval 5 \
    --save_interval 50 \
    --max_keep 3 \
    --wandb_project MiniMind-Agent-Handoff-Qwen-v2 \
    --from_resume 1 \
    --resume_mode 50 \
    2>&1 | tee -a ${LOG_DIR}/handoff_rl_v2_${TIMESTAMP}.log

echo "End RL v2: $(date)" | tee -a ${LOG_DIR}/handoff_rl_v2_${TIMESTAMP}.log
