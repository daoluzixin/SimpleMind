"""
MiniMind 预训练脚本

本脚本实现了 MiniMind 模型的预训练流程（Pretrain），
即在大规模无标注文本上进行自监督学习（下一个 token 预测）。

训练流程:
1. 初始化分布式环境和随机种子
2. 配置模型参数和检查断点续训
3. 设置混合精度训练（BF16/FP16）
4. 可选: 配置 SwanLab/wandb 实验追踪
5. 加载模型和预训练数据集
6. 从断点恢复训练状态（如有）
7. 可选: torch.compile 加速 + DDP 包装
8. 逐 epoch 训练
9. 清理分布式进程

关键特性:
- 梯度累积: 将多个 step 的梯度累加后再更新，等效增大 batch size
- 混合精度: BF16 或 FP16，减少显存占用并加速训练
- 余弦学习率调度: 带 warm-up 的余弦退火
- 断点续训: 自动保存/恢复完整的训练状态
- 分布式训练: 支持 PyTorch DDP 多卡训练
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, CheckpointManager

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """执行一个 epoch 的预训练

    训练循环的核心流程:
    1. 获取一个 batch 的 (input_ids, labels)
    2. 计算学习率并更新优化器
    3. 前向传播：计算 loss（交叉熵 + MoE 辅助损失）
    4. 反向传播：梯度累积
    5. 梯度累积步数达到后：梯度裁剪 → 参数更新 → 清零梯度
    6. 定期打印日志和保存模型

    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        start_step: 起始步数（用于断点续训）
        wandb: 实验追踪实例
    """
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将数据移到 GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ===== 学习率调度 =====
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 前向传播（混合精度） =====
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss  # 总损失 = 交叉熵损失 + MoE 辅助损失
            loss = loss / args.accumulation_steps  # 除以累积步数，保证梯度累积后等效于大 batch

        # ===== 反向传播（缩放梯度以防 FP16 下溢） =====
        scaler.scale(loss).backward()

        # ===== 梯度累积步数达到后更新参数 =====
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 将梯度缩放回原始尺度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸

            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放因子

            optimizer.zero_grad(set_to_none=True)  # 清零梯度（set_to_none=True 更高效）

        # ===== 日志打印 =====
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实 loss 值
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 纯交叉熵损失
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间: 已用时间/已走步数 * 剩余步数
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ===== 模型保存 =====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式（关闭 Dropout 等）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解包 DDP / torch.compile 包装，获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为 FP16，节省磁盘空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 通过 CheckpointManager 保存完整续训状态（含 top-k 管理和 best model 追踪）
            current_loss = loss.item() * args.accumulation_steps
            ckpt_mgr.save(model, optimizer, epoch, step,
                         metrics={'loss': current_loss, 'lr': optimizer.param_groups[-1]['lr']},
                         wandb=wandb, scaler=scaler)
            model.train()  # 切回训练模式
            del state_dict

        # 释放中间变量，减少显存占用
        del input_ids, labels, res, loss

    # 处理 epoch 末尾未达到累积步数的残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_keep", type=int, default=3, help="最多保留的checkpoint数量（0=不限制）")
    parser.add_argument("--resume_mode", type=str, default="latest", help="续训模式: latest/best/step编号")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))  # 不同 GPU 使用不同种子，增加数据多样性
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckpt_mgr = CheckpointManager(lm_config, weight=args.save_weight, save_dir='../checkpoints',
                                  max_keep=args.max_keep, track_metric='loss', metric_mode='min')
    resume_mode = args.resume_mode if args.resume_mode in ('latest', 'best') else int(args.resume_mode)
    ckp_data = ckpt_mgr.load(resume_mode) if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else ("mps" if "mps" in args.device else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    elif device_type == "mps":
        autocast_ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)  # MPS 仅支持 float16
    else:
        autocast_ctx = nullcontext()  # CPU 不支持 autocast
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # 使用 SwanLab 替代 wandb（国内友好的实验追踪工具）
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 有 id 时必须恢复，否则创建新实验
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # 分布式数据采样器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' and device_type == 'cuda'))  # GradScaler 仅 CUDA 有效
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # torch.compile 编译加速（需要 PyTorch 2.0+）
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])  # DDP 包装
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # DDP 每个 epoch 需要设置不同的种子
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 随机打乱数据
        # 断点续训时，跳过已训练的 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()