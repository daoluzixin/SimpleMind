#!/bin/bash
# ==============================================================================
# 安装缺失依赖 → 下载模型 → 后台启动训练
# 用法: nohup bash scripts/start_after_install.sh > /root/minimind/logs/bootstrap.log 2>&1 &
# 查看日志: tail -f /root/minimind/logs/bootstrap.log
# ==============================================================================
set -e

# ---- 激活 conda 环境 ----
source /home/vipuser/miniconda3/bin/activate base
cd /root/minimind
mkdir -p logs

echo "[$(date)] Python=$(python --version), Torch=$(python -c 'import torch;print(torch.__version__)')"
echo "[$(date)] GPUs: $(python -c 'import torch;print(torch.cuda.device_count())')"

# ---- 安装项目依赖（transformers 等） ----
echo "[$(date)] 安装项目依赖..."
pip install -q transformers==4.57.6 datasets jsonlines rich wandb einops modelscope trl accelerate
echo "[$(date)] 依赖安装完成!"

# ---- 下载 Qwen2.5-1.5B-Instruct 模型 ----
MODEL_DIR="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct"
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[$(date)] 下载 Qwen2.5-1.5B-Instruct 模型..."
    python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
    echo "[$(date)] 模型下载完成!"
else
    echo "[$(date)] 模型已存在: ${MODEL_DIR}"
fi

# ---- 启动训练 ----
echo "[$(date)] 启动 Handoff 训练..."
bash scripts/run_handoff_qwen_ddp.sh
