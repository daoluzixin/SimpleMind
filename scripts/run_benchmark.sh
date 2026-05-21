#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Continuous Batching 推理引擎压测脚本
# 模型: Qwen2.5-7B-Instruct (本地已下载)
# 设备: 单卡 4090
# 输出: 日志保存到 ~/benchmark_logs/ 目录
# ═══════════════════════════════════════════════════════════════════════════════

set -e

PYTHON="/home/vipuser/miniconda3/bin/python"
MODEL_PATH="./Qwen2.5-7B-Instruct"
LOG_DIR="$HOME/benchmark_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Continuous Batching 压测  ${TIMESTAMP}"
echo "═══════════════════════════════════════════════════════════════"
echo "  模型: ${MODEL_PATH}"
echo "  日志目录: ${LOG_DIR}"
echo ""

# 检查模型是否下载完成
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: 模型未下载完成，请先运行:"
    echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ${MODEL_PATH}"
    exit 1
fi

SAFETENSOR_COUNT=$(ls ${MODEL_PATH}/*.safetensors 2>/dev/null | wc -l)
if [ "$SAFETENSOR_COUNT" -eq 0 ]; then
    echo "ERROR: 未找到 safetensors 文件，模型可能还在下载中"
    exit 1
fi

echo "  模型文件: ${SAFETENSOR_COUNT} 个 safetensors 分片"
echo ""

# 记录 GPU 信息
echo "─── GPU 信息 ───" | tee "${LOG_DIR}/gpu_info_${TIMESTAMP}.log"
nvidia-smi | tee -a "${LOG_DIR}/gpu_info_${TIMESTAMP}.log"
echo ""

# ─── 实验 1: 标准配置 (batch_size=4, 16请求, 64 tokens) ───
echo "▶ 实验 1: 标准配置 (batch=4, requests=16, tokens=64)"
LOG_FILE="${LOG_DIR}/bench_standard_${TIMESTAMP}.log"
$PYTHON ~/benchmark_continuous_batching.py \
    --model "$MODEL_PATH" \
    --device cuda \
    --dtype float16 \
    --num_requests 16 \
    --max_batch_size 4 \
    --max_new_tokens 64 \
    --prompt_len 64 \
    2>&1 | tee "$LOG_FILE"
echo "  → 日志: $LOG_FILE"
echo ""

# ─── 实验 2: 大 batch (batch_size=8, 32请求, 64 tokens) ───
echo "▶ 实验 2: 大 batch (batch=8, requests=32, tokens=64)"
LOG_FILE="${LOG_DIR}/bench_large_batch_${TIMESTAMP}.log"
$PYTHON ~/benchmark_continuous_batching.py \
    --model "$MODEL_PATH" \
    --device cuda \
    --dtype float16 \
    --num_requests 32 \
    --max_batch_size 8 \
    --max_new_tokens 64 \
    --prompt_len 64 \
    2>&1 | tee "$LOG_FILE"
echo "  → 日志: $LOG_FILE"
echo ""

# ─── 实验 3: 长生成 (batch_size=4, 16请求, 128 tokens) ───
echo "▶ 实验 3: 长生成 (batch=4, requests=16, tokens=128)"
LOG_FILE="${LOG_DIR}/bench_long_gen_${TIMESTAMP}.log"
$PYTHON ~/benchmark_continuous_batching.py \
    --model "$MODEL_PATH" \
    --device cuda \
    --dtype float16 \
    --num_requests 16 \
    --max_batch_size 4 \
    --max_new_tokens 128 \
    --prompt_len 64 \
    2>&1 | tee "$LOG_FILE"
echo "  → 日志: $LOG_FILE"
echo ""

# ─── 实验 4: Batch Size Scaling (只跑 batched decode，对比 batch 1/2/4/8) ───
echo "▶ 实验 4: Batch Size Scaling 对比"
LOG_FILE="${LOG_DIR}/bench_scaling_${TIMESTAMP}.log"
echo "═══ Batch Size Scaling Experiment ═══" > "$LOG_FILE"
for BS in 1 2 4 8; do
    echo "  batch_size=${BS}..." | tee -a "$LOG_FILE"
    $PYTHON ~/benchmark_continuous_batching.py \
        --model "$MODEL_PATH" \
        --device cuda \
        --dtype float16 \
        --num_requests 16 \
        --max_batch_size $BS \
        --max_new_tokens 64 \
        --prompt_len 64 \
        --mode batched \
        2>&1 | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
done
echo "  → 日志: $LOG_FILE"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  全部实验完成！日志目录: ${LOG_DIR}"
echo "═══════════════════════════════════════════════════════════════"
ls -la "${LOG_DIR}"/bench_*_${TIMESTAMP}.log
