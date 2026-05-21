# MiniMind 训练执行手册

> 云服务器 GPU (3090 24GB) 环境，所有命令在 `/root/trainer` 目录下执行。

## 通用注意事项

nohup 后台训练必须加 `python -u`（关闭输出缓冲），否则日志不会实时写入文件。日志文件名带时间戳，重跑不会覆盖旧日志。训练前先 `mkdir -p logs` 确保目录存在。

查看进度用 `tail -f logs/xxx.log`，确认进程存活用 `ps aux | grep train`。


## 第一步：预训练（Pretrain）

作用：让模型从大量文本中学习语言规律和基础知识（next token prediction）。

```bash
cd /root/trainer
nohup python -u train_pretrain.py \
  --epochs 2 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --max_seq_len 768 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_interval 1000 \
  --log_interval 100 \
  > logs/pretrain_$(date +%m%d_%H%M).log 2>&1 &
```

产出：`/root/out/pretrain_768.pth`，约 1.2 小时。

验证产出存在：`ls -lh /root/out/pretrain_768.pth`

日志关键指标：`grep "loss:" logs/pretrain_*.log | tail -5`


## 第二步：有监督微调（Full SFT）

作用：在预训练权重基础上，用多轮对话数据让模型学会对话格式、指令跟随、Tool Call。学习率比预训练小 50 倍（1e-5 vs 5e-4）防止遗忘。

前置条件：`/root/out/pretrain_768.pth` 存在。

```bash
nohup python -u train_full_sft.py \
  --from_weight pretrain \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --save_interval 1000 \
  --log_interval 100 \
  > logs/sft_$(date +%m%d_%H%M).log 2>&1 &
```

产出：`/root/out/full_sft_768.pth`，约 1.1 小时。

验证：`python ../eval_llm.py --weight full_sft`


## 第三步：PPL 数据质量过滤

作用：用模型自身的困惑度对预训练数据打分，过滤极低 PPL（已记忆）和极高 PPL（噪声）数据，验证"保留 65% 数据达到全量等效 val loss"。

前置条件：`/root/out/pretrain_768.pth` 存在。

```bash
# 3.1 对数据打分（用预训练权重计算每条数据的 PPL）
nohup python -u ../scripts/data_quality_scorer.py score \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --model_path ../out/pretrain_768.pth \
  > logs/ppl_score_$(date +%m%d_%H%M).log 2>&1 &

# 3.2 过滤（保留 P25~P90 的中间段数据）
nohup python -u ../scripts/data_quality_scorer.py filter \
  --scored_path ../dataset/pretrain_scored.jsonl \
  --strategy percentile --low 25 --high 90 \
  > logs/ppl_filter_$(date +%m%d_%H%M).log 2>&1 &
```

日志关键数据：
```bash
grep "PPL 统计" logs/ppl_score_*.log        # mean/median/std
grep "retained" logs/ppl_filter_*.log        # 保留率（目标 ~65%）
grep "P25.*P90" logs/ppl_filter_*.log        # 过滤阈值
```


## 第四步：分桶贡献实验

作用：将数据按 PPL 等频分 8 桶，每桶独立训练 1 epoch，测量各桶对 val loss 的边际贡献，验证中间桶贡献最大、极端桶贡献为负的倒 U 型曲线。

前置条件：PPL 打分完成（scored 文件存在），需要准备 validation set。

```bash
nohup python -u ../scripts/data_quality_scorer.py bucket_exp \
  --scored_path ../dataset/pretrain_scored.jsonl \
  --val_path ../dataset/pretrain_val.jsonl \
  --n_buckets 8 \
  > logs/bucket_exp_$(date +%m%d_%H%M).log 2>&1 &
```

日志关键数据：
```bash
grep "Δ=" logs/bucket_exp_*.log                    # 每桶 val loss 变化
grep -A 10 "分桶贡献排名" logs/bucket_exp_*.log    # 贡献排名表
```

预期结论：PPL 中间桶贡献为极端桶的 N 倍（N 越大，过滤越激进越合理）。


## 第五步：Agent RL 训练

作用：在 SFT 权重基础上，用 GRPO/CISPO 算法训练模型的多轮工具调用能力。

前置条件：`/root/out/full_sft_768.pth` 存在。

```bash
nohup python -u train_agent.py \
  --from_weight full_sft \
  --save_interval 1000 \
  --log_interval 10 \
  > logs/agent_rl_$(date +%m%d_%H%M).log 2>&1 &
```

日志关键数据：
```bash
grep "Reward:" logs/agent_rl_*.log | head -5    # 初始 reward
grep "Reward:" logs/agent_rl_*.log | tail -5    # 最终 reward
grep "\[Checkpoint\]" logs/agent_rl_*.log       # 保存记录
```


## 第六步：Plan-then-Execute RL 训练

作用：三阶段信用分离训练，Plan 和 Execute 分别计算独立 advantage，验证 PlanRate 上升和 plan/exec reward 分离。

前置条件：`/root/out/full_sft_768.pth` 存在。

```bash
nohup python -u train_plan.py \
  --from_weight full_sft \
  --save_interval 1000 \
  --log_interval 10 \
  > logs/plan_rl_$(date +%m%d_%H%M).log 2>&1 &
```

日志关键数据：
```bash
grep "PlanRate:" logs/plan_rl_*.log             # Plan 成功率趋势
grep "Reward:" logs/plan_rl_*.log               # 总 reward 趋势
grep "plan=.*exec=" logs/plan_rl_*.log          # plan/exec 分离情况
```

预期结论：PlanRate 从初始值逐步上升，plan reward 和 exec reward 各自独立收敛。


## 第七步：Continuous Batching 推理测试

作用：验证推理引擎吞吐提升，测量 TTFT 和 tokens/sec。

前置条件：任意训好的权重（如 `full_sft_768.pth`）。

```bash
nohup python -u ../scripts/continuous_batching_engine.py \
  --benchmark \
  > logs/cb_bench_$(date +%m%d_%H%M).log 2>&1 &
```

日志关键数据：
```bash
grep "TTFT" logs/cb_bench_*.log                 # 首 token 延迟
grep "tokens/sec" logs/cb_bench_*.log           # 吞吐量
```


## 日志收集与分析

训练全部完成后，在服务器上提取关键数据：

```bash
# 一键查看所有实验的首尾 loss/reward
for f in logs/*.log; do
  echo "=== $(basename $f) ==="
  grep -E "(loss:|Reward:)" $f | head -1
  grep -E "(loss:|Reward:)" $f | tail -1
  echo ""
done
```

拉回本地：

```bash
scp -r gpu-server:/root/trainer/logs/ ~/PycharmProjects/minimind-master/docs/server_logs/
```
