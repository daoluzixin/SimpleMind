"""MiniMind 知识蒸馏训练脚本

知识蒸馏（Knowledge Distillation）是一种模型压缩技术，通过让小模型（学生）
学习大模型（教师）的输出分布，在保持性能的同时大幅减少模型参数量。

蒸馏核心原理:
- 教师模型: 大模型（如 MoE 版本），输出 soft label（软标签/概率分布）
- 学生模型: 小模型（如 dense 版本），学习教师的输出分布
- 蒸馏损失: KL 散度，衡量学生与教师输出分布的差异
- 温度参数: 放大 logits 的温度，使软标签包含更多"暗知识"（dark knowledge）

总损失 = alpha * CE_Loss + (1 - alpha) * Distillation_Loss
- CE_Loss: 学生模型与真实标签的交叉熵损失
- Distillation_Loss: 学生与教师输出分布的 KL 散度

典型使用场景:
- MoE → Dense: 用 MoE 大模型蒸馏 Dense 小模型
- 大 → 小: 用更大 hidden_size 的模型蒸馏更小的模型
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, CheckpointManager

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """计算知识蒸馏的 KL 散度损失

    蒸馏损失 = T^2 * KL(softmax(teacher_logits / T) || log_softmax(student_logits / T))

    温度 T 的作用:
    - T=1: 退化为标准 KL 散度
    - T>1: 放大 logits 差异，使软标签更"软"，包含更多暗知识
    - 乘以 T^2: 保持梯度量级与 CE 损失一致

    Args:
        student_logits: 学生模型的 logits [B*S, V]
        teacher_logits: 教师模型的 logits [B*S, V]（已 detach）
        temperature: 蒸馏温度，推荐范围 1.0~2.0
        reduction: 归约方式，默认 batchmean
    Returns:
        蒸馏损失标量
    """
    with torch.no_grad():
        # 教师模型的软标签概率分布（温度缩放后的 softmax）
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型的 log 概率分布（温度缩放后的 log_softmax）
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度: 衡量学生分布与教师分布的差异
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 乘以 T^2 保持梯度量级一致
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """蒸馏训练一个 epoch 的主循环

    训练流程:
    1. 学生模型前向传播获取 logits
    2. 教师模型前向传播获取 logits（无梯度）
    3. 计算 CE 损失（学生 vs 真实标签）
    4. 计算蒸馏损失（学生 vs 教师 KL 散度）
    5. 总损失 = alpha * CE + (1-alpha) * Distill
    6. 反向传播 + 梯度累积 + 优化器更新

    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        teacher_model: 教师模型（冻结）
        lm_config_student: 学生模型配置
        start_step: 起始步数（用于断点续训）
        wandb: wandb 日志记录器
        alpha: CE 损失权重，总损失 = alpha * CE + (1-alpha) * Distill
        temperature: 蒸馏温度
    """
    start_time = time.time()
    last_step = start_step
    
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)  # 冻结教师模型参数

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)  # 输入 token IDs: [B, S]
        labels = labels.to(args.device)        # 标签 token IDs: [B, S]
        loss_mask = (labels[..., 1:] != -100).float()  # 有效 token 的 mask（-100 为忽略位置）

        # ========== 学习率调度: 余弦退火 ==========
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 学生模型前向传播 ==========
        with autocast_ctx:
            res = model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()  # 预测下一个 token 的 logits

        # ========== 教师模型前向传播（无梯度） ==========
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果教师模型词表更大，截断到学生模型的词表大小
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss: 学生模型 vs 真实标签
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # 加上 MoE 辅助损失（如果使用 MoE）
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss: 学生模型 vs 教师模型的 KL 散度
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # ========== 反向传播 + 梯度累积 ==========
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ========== 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 处理 torch.compile 的包装
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 保存为半精度
            current_loss = loss.item() * args.accumulation_steps
            ckpt_mgr.save(model, optimizer, epoch, step,
                         metrics={'loss': current_loss, 'ce_loss': ce_loss_raw.item(), 'distill_loss': distill_loss.item(), 'lr': optimizer.param_groups[-1]['lr']},
                         wandb=wandb, scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # 处理 epoch 末尾未完成的梯度累积步
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 模拟用 MoE 模型蒸馏 Dense 模型，也可以用更大 teacher_hidden_size 模型蒸馏更小 student_hidden_size 的
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
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
    # 学生模型配置（待训练的小模型）
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    # 教师模型配置（冻结的大模型）
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    ckpt_mgr = CheckpointManager(lm_config_student, weight=args.save_weight, save_dir='../checkpoints',
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
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)  # 冻结教师模型
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
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
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()