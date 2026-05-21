#!/bin/bash
# 监控第一阶段训练，完成后自动启动第二阶段
LOG="/root/minimind/logs/plan_rl.log"
LOG2="/root/minimind/logs/plan_rl_phase2.log"

echo "[Monitor] Waiting for Phase 1 to complete..."

while true; do
    # 检查是否出现完成标志
    if grep -q "Plan-then-Execute Training complete" "$LOG" 2>/dev/null; then
        echo "[Monitor] Phase 1 completed! Starting Phase 2..."
        break
    fi
    # 检查进程是否还活着
    if ! pgrep -f agent_plan_qwen > /dev/null 2>&1; then
        if grep -q "Plan-then-Execute Training complete" "$LOG" 2>/dev/null; then
            echo "[Monitor] Phase 1 completed! Starting Phase 2..."
            break
        else
            echo "[Monitor] Phase 1 process died unexpectedly. Check logs."
            tail -5 "$LOG"
            exit 1
        fi
    fi
    sleep 30
done

echo "[Monitor] Launching Phase 2 training with remaining 500 samples..."
cd /root/minimind
nohup bash scripts/run_plan_qwen_phase2.sh > "$LOG2" 2>&1 &
echo "[Monitor] Phase 2 launched with PID $!"
echo "[Monitor] Log: $LOG2"
