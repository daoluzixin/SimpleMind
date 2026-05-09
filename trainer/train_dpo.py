"""
MiniMind DPO（Direct Preference Optimization）训练脚本

本脚本实现了 DPO 算法，通过人类偏好数据对模型进行对齐训练。
DPO 是一种无需单独训练奖励模型的 RLHF 替代方案，直接利用偏好数据
优化策略模型，使其更倾向于生成 chosen（优质）回复，而非 rejected（劣质）回复。

DPO 核心思想:
    传统 RLHF: 训练奖励模型 → 用 PPO 优化策略
    DPO: 直接通过偏好数据优化策略，无需奖励模型

    DPO 损失: L = -log σ(β * (log π(y_w|x) / π(y_l|x) - log π_ref(y_w|x) / π_ref(y_l|x)))
    其中:
    - y_w: chosen（优质）回复, y_l: rejected（劣质）回复
    - π: 当前策略模型, π_ref: 参考模型（冻结的 SFT 模型）
    - β: 控制对齐强度，越大则偏离参考模型的惩罚越强

训练流程:
1. 将 chosen 和 rejected 数据拼接为一个 batch
2. 用冻结的参考模型计算 ref_log_probs
3. 用策略模型计算 policy_log_probs
4. 计算 DPO 损失 + MoE 辅助损失
5. 反向传播更新策略模型
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
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, CheckpointManager

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """将模型输出的 logits 转换为每个 token 的对数概率

    用于 DPO 损失计算中，获取策略模型和参考模型对 chosen/rejected 回复的对数概率。

    执行流程:
    1. 对 logits 做 log_softmax 得到对数概率分布
    2. 用 labels 作为索引，从对数概率中 gather 出每个位置真实 token 的对数概率

    Args:
        logits: 模型输出, shape (batch_size, seq_len, vocab_size)
        labels: 目标 token ID, shape (batch_size, seq_len)

    Returns:
        log_probs_per_token: 每个 token 的对数概率, shape (batch_size, seq_len)
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)  # 在词表维度做 log_softmax
    # gather: 从每个位置的词表概率分布中，取出真实 token 对应的对数概率
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """计算 DPO 损失

    DPO 损失公式:
        L = -log σ(β * (log_ratio_policy - log_ratio_ref))
        其中:
        - log_ratio_policy = log π(y_w|x) - log π(y_l|x)  (策略模型的偏好对数比)
        - log_ratio_ref = log π_ref(y_w|x) - log π_ref(y_l|x)  (参考模型的偏好对数比)

    直觉理解:
    - 当策略模型比参考模型更偏好 chosen（log_ratio_policy > log_ratio_ref）时，损失趋近 0
    - 当策略模型比参考模型更偏好 rejected 时，损失增大
    - β 控制对齐强度：β 越大，模型越倾向于对齐偏好数据

    数据排列约定:
    batch 的前半部分是 chosen 数据，后半部分是 rejected 数据

    Args:
        ref_log_probs: 参考模型的对数概率, shape (batch_size, seq_len)
        policy_log_probs: 策略模型的对数概率, shape (batch_size, seq_len)
        mask: loss mask，1 表示参与计算的位置，0 表示不参与
        beta: DPO 温度参数，控制对齐强度

    Returns:
        标量 DPO 损失值
    """
    # 对序列维度求和（只计算 mask=1 的位置）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开（前半 chosen，后半 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算策略模型和参考模型的偏好对数比
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs  # log(π(y_w)/π(y_l))
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs       # log(π_ref(y_w)/π_ref(y_l))
    # DPO 核心: logits = log(π/π_ref) 的对数比
    logits = pi_logratios - ref_logratios
    # 损失: -log σ(β * logits)，σ 为 sigmoid 函数
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """执行一个 epoch 的 DPO 训练

    每个训练步骤:
    1. 获取 chosen 和 rejected 数据
    2. 拼接为一个 batch 输入参考模型（冻结，no_grad）和策略模型
    3. 计算 DPO 损失 + MoE 辅助损失
    4. 反向传播更新策略模型

    Args:
        epoch: 当前 epoch
        loader: 数据加载器
        iters: 总迭代步数
        ref_model: 冻结的参考模型
        lm_config: 模型配置
        start_step: 起始步数
        wandb: 实验追踪实例
        beta: DPO 温度参数
    """
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # 从 batch 中获取 chosen 和 rejected 数据
        x_chosen = batch['x_chosen'].to(args.device)      # chosen 输入 token IDs
        x_rejected = batch['x_rejected'].to(args.device)   # rejected 输入 token IDs
        y_chosen = batch['y_chosen'].to(args.device)       # chosen 目标 token IDs
        y_rejected = batch['y_rejected'].to(args.device)   # rejected 目标 token IDs
        mask_chosen = batch['mask_chosen'].to(args.device)  # chosen loss mask
        mask_rejected = batch['mask_rejected'].to(args.device)  # rejected loss mask

        # 将 chosen 和 rejected 拼接为一个 batch（前半 chosen，后半 rejected）
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ===== 学习率调度 =====
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 前向传播 =====
        with autocast_ctx:
            # 参考模型推理（冻结，不计算梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 策略模型推理（需要梯度）
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 计算 DPO 损失 + MoE 辅助损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # ===== 反向传播 =====
        scaler.scale(loss).backward()

        # ===== 梯度累积步数达到后更新参数 =====
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ===== 日志打印 =====
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ===== 模型保存 =====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            current_loss = loss.item() * args.accumulation_steps
            ckpt_mgr.save(model, optimizer, epoch, step,
                         metrics={'loss': current_loss, 'dpo_loss': dpo_loss_val.item(), 'lr': optimizer.param_groups[-1]['lr']},
                         wandb=wandb, scaler=scaler)
            model.train()
            del state_dict

        # 释放中间变量
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # 处理 epoch 末尾未达到累积步数的残余梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数（控制对齐强度）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
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
    ckpt_mgr = CheckpointManager(lm_config, weight=args.save_weight, save_dir='../checkpoints',
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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义策略模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结，用于计算 DPO 损失中的基准偏好）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)  # 冻结参考模型参数
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()