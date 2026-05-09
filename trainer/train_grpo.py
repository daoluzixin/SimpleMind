"""MiniMind GRPO（Group Relative Policy Optimization）训练脚本

GRPO 是 PPO 的简化变体，无需训练 Critic（价值模型），通过同一 prompt 的多个生成样本
的组内相对排名来估计优势函数，从而大幅简化 RLHF 流程。

GRPO 核心流程:
1. Rollout: 对每个 prompt 生成 num_generations 个回复
2. 奖励计算: 结合奖励模型评分 + 规则奖励（长度、重复惩罚、思考格式等）
3. 优势估计: 对同一 prompt 的多个回复做组内归一化（mean/std），得到相对优势
4. 策略更新: 使用裁剪目标函数（GRPO/ClSPO）+ KL 惩罚更新 Actor

与 PPO 的关键区别:
- 无需 Critic 模型，优势由组内奖励的相对排名决定
- 支持 GRPO 和 ClSPO 两种 loss 类型
- ClSPO 只裁剪 ratio 上界，对负优势不裁剪，更稳定

关键组件:
- Actor: 策略模型（待优化的语言模型）
- Ref Model: 参考模型（冻结的 SFT 模型，用于计算 KL 惩罚）
- Reward Model: 奖励模型（对回复质量打分）
- Rollout Engine: 推理引擎（支持 PyTorch 原生或 SGLang）
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, CheckpointManager
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """重复惩罚：计算文本中 n-gram 的重复率，重复率越高惩罚越大（上限 cap）

    Args:
        text: 待检测的文本
        n: n-gram 的 n 值，默认 3（三元组）
        cap: 惩罚上限，默认 0.5
    Returns:
        重复惩罚值，范围 [0, cap]
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())  # 将文本拆分为词和标点
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]  # 生成所有 n-gram
    # 重复率 = (n-gram总数 - 去重后数量) * cap * 2 / n-gram总数，上限为 cap
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """计算每个回复的总奖励（规则奖励 + 奖励模型评分）

    规则奖励包含:
    - 长度奖励: 20~800 字符 +0.5，否则 -0.5
    - 思考奖励: 有思考内容且 20~300 字符 +1.0，否则 -0.5
    - 思考格式: 恰好1个思考分隔符 +0.25，否则 -0.25
    - 重复惩罚: n-gram 重复率惩罚（减去）

    Args:
        prompts: prompt 列表，长度 B
        responses: 回复列表，长度 B * num_generations
        reward_model: 奖励模型，用于对回复质量打分
    Returns:
        rewards: 每个回复的总奖励张量，形状 [B * num_generations]
    """
    rewards = torch.zeros(len(responses), device=args.device)  # 初始化奖励为 0

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j  # 当前回复在展平列表中的索引
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 prompt 中解析 ChatML 格式的消息（system/user/assistant 角色）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response

                # 规则奖励1: 长度奖励 —— 回复长度在 20~800 字符之间得 +0.5，否则 -0.5
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # 规则奖励2: 思考格式奖励 —— 如果包含思考分隔符
                if '```' + '\n' in response:
                    thinking_content, answer_content = response.split('```' + '\n', 1)
                    # 思考内容长度在 20~300 字符得 +1.0，否则 -0.5
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 恰好1个思考分隔符得 +0.25，多个则 -0.25
                    rewards[response_idx] += 0.25 if response.count('```' + '\n') == 1 else -0.25
                    answer = answer_content.strip()

                # 规则奖励3: 重复惩罚 —— n-gram 重复率越高，惩罚越大
                rewards[response_idx] -= rep_penalty(answer)

                # 奖励模型评分
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 将奖励模型评分转换为张量并累加到总奖励
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """GRPO 训练一个 epoch 的主循环

    每个训练步骤:
    1. 对 batch 中的每个 prompt 生成 num_generations 个回复（Rollout）
    2. 使用当前模型和参考模型计算每个 token 的对数概率
    3. 计算奖励（规则奖励 + 奖励模型评分）
    4. 对同一 prompt 的多个回复做组内优势归一化
    5. 计算策略损失（GRPO/ClSPO 裁剪目标 + KL 惩罚）
    6. 反向传播 + 梯度累积 + 优化器更新

    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        rollout_engine: 推理引擎（PyTorch 或 SGLang）
        ref_model: 参考模型（冻结），用于计算 KL 惩罚
        reward_model: 奖励模型
        start_step: 起始步数（用于断点续训）
        wandb: wandb 日志记录器
        use_sglang: 是否使用 SGLang 推理引擎
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], 长度 B

        # ========== 第1步: 编码 prompt 并截断到最大长度 ==========
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            # 截取 prompt 的最后 max_seq_len 个 token（保留最近的上下文）
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ========== 第2步: Rollout —— 用 Actor 对每个 prompt 生成 num_generations 个回复 ==========
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,  # 每个 prompt 生成多少个回复
            max_new_tokens=args.max_gen_len,  # 生成的最大 token 数
            temperature=0.8,  # 采样温度（越高越随机）
        )
        outputs = rollout_result.output_ids        # 完整输出: [B*num_gen, prompt_len + gen_len]
        completion_ids = rollout_result.completion_ids  # 仅生成部分: [B*num_gen, gen_len]
        completions = rollout_result.completions        # 解码后的文本列表
        old_per_token_logps = rollout_result.per_token_logps.to(args.device)  # 旧策略的逐 token 对数概率
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # 每个 prompt 的长度

        # 构建完整 attention mask（非 pad 位置为 1）
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # logp_pos: 每个生成 token 在完整序列中的位置索引（用于从 logits 中提取对应位置的对数概率）
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # ========== 第3步: 用当前模型和参考模型计算对数概率 ==========
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            # MoE 辅助损失（负载均衡），非 MoE 模型为 0
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 提取当前策略的逐 token 对数概率:
            # 1. logits[:, :-1, :] 取除最后一个位置外的所有 logits（预测下一个 token）
            # 2. log_softmax 归一化
            # 3. gather 提取每个位置实际 token 的对数概率
            # 4. 再用 logp_pos 索引只取生成部分的 token
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
        
        # 参考模型的对数概率（用于计算 KL 散度）
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ========== 第4步: 计算奖励（规则 + 奖励模型） ==========
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # ========== 调试模式: 打印采样结果 ==========
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # ========== 第5步: GRPO 优势估计 —— 组内相对排名归一化 ==========
        # GRPO 核心: 同一 prompt 的多个回复进行组内归一化，代替 PPO 中的 Critic
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen] 按组分组
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # 组内均值，展平回 [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # 组内标准差，展平回 [B*num_gen]
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # 组内 Z-score 归一化优势 [B*num_gen]

        # ========== 第6步: 构建生成部分的 mask（到 EOS 为止的有效 token） ==========
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()  # 非 pad 的位置
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R] EOS 位置
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)  # 默认 EOS 在最后
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]  # 有 EOS 的取第一个 EOS 位置
        # completion_mask: 从起始到第一个 EOS 的所有位置为 1，其余为 0
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # ========== 第7步: 计算 KL 散度和策略损失 ==========
        # KL 散度: KL(pi_ref || pi_current)，用于约束策略不要偏离参考模型太远
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R] 逐 token KL 散度

        # 重要性采样比率: pi_current / pi_old
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        if args.loss_type == "cispo":
            # ClSPO (Clipped IS Policy Optimization): 只裁剪 ratio 上界
            # 对正优势: 限制 ratio 不超过 epsilon_high，防止过大更新
            # 对负优势: 不裁剪，允许策略自由降低坏动作的概率
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO: 双侧裁剪，类似 PPO 的 clip 目标
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)       # 未裁剪的优势项
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)  # 裁剪后的优势项
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # 对每个序列取平均（除以有效 token 数），再对所有序列取平均
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # 除以梯度累积步数
        loss.backward()

        # ========== 第8步: 梯度累积 + 优化器更新 ==========
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
            optimizer.step()     # 更新参数
            scheduler.step()     # 更新学习率（余弦退火）
            optimizer.zero_grad()  # 清零梯度

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps  # 还原累积前的实际损失
            current_aux_loss = aux_loss.item()  # MoE 辅助损失
            avg_reward_val = rewards.mean().item()  # 平均奖励
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()  # 平均回复长度
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)  # 平均 KL 散度
            advantages_mean_val = advantages.mean().item()  # 优势均值
            advantages_std_val = advantages.std().item()  # 优势标准差
            current_lr = optimizer.param_groups[0]['lr']  # 当前学习率

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # ========== 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 处理 torch.compile 的包装
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 保存为半精度
            ckpt_mgr.save(model, optimizer, epoch, step,
                         metrics={'reward': avg_reward_val, 'policy_loss': policy_loss_val, 'kl_ref': kl_ref_val, 'lr': current_lr},
                         wandb=wandb, scheduler=scheduler)
            model.train()
            del state_dict

        # 同步更新 Rollout 引擎的策略模型权重
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 释放中间变量，节省显存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # 处理 epoch 末尾未完成的梯度累积步
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_keep", type=int, default=3, help="最多保留的checkpoint数量（0=不限制）")
    parser.add_argument("--resume_mode", type=str, default="latest", help="续训模式: latest/best/step编号")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # max_seq_len 包含 prompt + 生成的总长度
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckpt_mgr = CheckpointManager(lm_config, weight=args.save_weight, save_dir='../checkpoints',
                                  max_keep=args.max_keep, track_metric='reward', metric_mode='max')
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型（Actor）: 待优化的语言模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型: 冻结的 SFT 模型，用于计算 KL 惩罚，防止策略偏离太远
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)  # 冻结参数，不计算梯度
    # Reward模型: 对回复质量打分
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎: 可插拔替换，负责策略模型的推理（生成回复）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)  # 每个 epoch 的总迭代步数
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    # 余弦退火学习率调度器，eta_min 为最低学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])       # 恢复模型参数
        optimizer.load_state_dict(ckp_data['optimizer'])  # 恢复优化器状态
        scheduler.load_state_dict(ckp_data['scheduler'])  # 恢复学习率调度器状态
        start_epoch = ckp_data['epoch']   # 恢复 epoch
        start_step = ckp_data.get('step', 0)  # 恢复 step
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)  # torch.compile 加速
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])  # DDP 包装
    rollout_engine.update_policy(model)  # 同步策略模型权重到 Rollout 引擎
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 设置 epoch 以确保每轮数据打乱不同
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()  # 设置随机种子
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0  # 断点续训跳过的步数
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)  # 跳过已训练的 batch
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()