"""MiniMind LoRA（Low-Rank Adaptation）微调训练脚本

LoRA 是一种参数高效的微调方法，通过在预训练模型的权重矩阵上注入低秩分解矩阵，
仅训练少量新增参数即可实现全参数微调的效果，大幅降低训练成本。

LoRA 核心原理:
- 原始权重 W ∈ R^{d×k} 冻结不训练
- 注入低秩矩阵 ΔW = A × B，其中 A ∈ R^{d×r}, B ∈ R^{r×k}，r << min(d, k)
- 前向传播: y = W·x + (A·B)·x = W·x + ΔW·x
- 仅训练 A 和 B 的参数，参数量远小于原始权重

训练流程:
1. 加载预训练模型权重
2. 对指定层（q_proj, k_proj, v_proj, o_proj 等）注入 LoRA
3. 冻结原始参数，仅训练 LoRA 参数
4. 使用 SFT 数据进行监督微调
5. 保存时仅保存 LoRA 权重，推理时可合并回原始模型

适用场景:
- 领域适配（如医疗、法律等垂直领域）
- 风格迁移（如身份、语气等）
- 资源受限下的快速微调
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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, CheckpointManager

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """LoRA 微调训练一个 epoch 的主循环

    训练流程与 SFT 类似，但只更新 LoRA 参数:
    1. 前向传播: 计算交叉熵损失 + MoE 辅助损失
    2. 梯度累积: 将损失除以累积步数后反向传播
    3. 梯度裁剪: 仅对 LoRA 参数进行梯度裁剪
    4. 优化器更新: 仅更新 LoRA 参数
    5. 学习率调度: 使用余弦退火策略
    6. 定期保存 LoRA 权重

    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        lora_params: LoRA 参数列表（仅对这些参数进行梯度裁剪）
        start_step: 起始步数（用于断点续训）
        wandb: wandb 日志记录器
    """
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)  # 输入 token IDs: [B, S]
        labels = labels.to(args.device)        # 标签 token IDs: [B, S]
        last_step = step

        # ========== 学习率调度: 余弦退火 ==========
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 前向传播 + 混合精度 ==========
        with autocast_ctx:
            res = model(input_ids, labels=labels)  # 前向传播，自动计算交叉熵损失
            loss = res.loss + res.aux_loss  # 总损失 = 语言模型损失 + MoE 辅助损失
            loss = loss / args.accumulation_steps  # 除以梯度累积步数

        # ========== 反向传播（使用 GradScaler 处理 FP16 混合精度） ==========
        scaler.scale(loss).backward()

        # ========== 梯度累积 + 优化器更新 ==========
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度，以便进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)  # 仅对 LoRA 参数裁剪梯度
            scaler.step(optimizer)   # 更新 LoRA 参数
            scaler.update()          # 更新 GradScaler 的缩放因子
            optimizer.zero_grad(set_to_none=True)  # 清零梯度（set_to_none=True 更高效）

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 还原累积前的实际损失
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0  # MoE 辅助损失
            current_logits_loss = current_loss - current_aux_loss  # 纯语言模型损失
            current_lr = optimizer.param_groups[-1]['lr']  # 当前学习率
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 预估剩余时间（分钟）
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ========== 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA 只保存 LoRA 权重（不保存原始模型参数，节省空间）
            save_lora(model, lora_save_path)
            current_loss = loss.item() * args.accumulation_steps
            ckpt_mgr.save(model, optimizer, epoch, step,
                         metrics={'loss': current_loss, 'lr': optimizer.param_groups[-1]['lr']},
                         wandb=wandb, scaler=scaler)
            model.train()

        del input_ids, labels, res, loss

    # 处理 epoch 末尾未完成的梯度累积步
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_keep", type=int, default=3, help="最多保留的checkpoint数量（0=不限制）")
    parser.add_argument("--resume_mode", type=str, default="latest", help="续训模式: latest/best/step编号")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckpt_mgr = CheckpointManager(lm_config, weight=args.lora_name, save_dir='../checkpoints',
                                  max_keep=args.max_keep, track_metric='loss', metric_mode='min')
    resume_mode = args.resume_mode if args.resume_mode in ('latest', 'best') else int(args.resume_mode)
    ckp_data = ckpt_mgr.load(resume_mode) if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 加载预训练模型权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 对模型的注意力层注入 LoRA 低秩矩阵
    apply_lora(model)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非 LoRA 参数，收集 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True   # LoRA 参数可训练
            lora_params.append(param)
        else:
            param.requires_grad = False  # 原始参数冻结
    
    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler: 用于 FP16 混合精度训练的梯度缩放（BF16 不需要）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器仅训练 LoRA 参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)  # strict=False 允许 LoRA 参数缺失
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # LoRA 的 monkey-patch forward 与 torch.compile 不兼容，自动关闭
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()