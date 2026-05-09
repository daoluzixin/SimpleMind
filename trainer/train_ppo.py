"""MiniMind PPO（Proximal Policy Optimization）训练脚本

PPO 是 RLHF 的经典方法，需要同时训练 Actor（策略模型）和 Critic（价值模型）。
与 GRPO 的关键区别: PPO 需要额外的 Critic 模型来估计状态价值函数 V(s)。

PPO 核心流程:
1. Rollout: 使用 Actor 模型生成回复
2. 奖励计算: 结合奖励模型评分和规则奖励（长度、重复惩罚等）
3. 优势估计: 使用 GAE（Generalized Advantage Estimation）计算优势函数
   - GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
   - δ_t = r_t + γV(s_{t+1}) - V(s_t)
4. PPO 更新: 使用裁剪目标函数 + KL 惩罚更新 Actor 和 Critic
   - Actor loss: max(-A_t * ratio, -A_t * clip(ratio, 1-ε, 1+ε)) + β * KL
   - Critic loss: max((V - R)^2, (clip(V, V_old±ε_v) - R)^2)

关键组件:
- Actor: 策略模型（待优化的语言模型）
- Critic: 价值模型（估计每个状态的价值函数 V(s)）
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
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, CheckpointManager
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """重复惩罚：计算文本中 n-gram 的重复率，重复率越高惩罚越大（上限 cap）"""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


class CriticModel(MiniMindForCausalLM):
    """Critic（价值）模型：估计每个状态的价值函数 V(s)

    继承自 MiniMindForCausalLM，将 lm_head 替换为 value_head，
    输出每个位置的状态价值估计（标量）。
    """
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)  # 价值头：输出单一标量价值

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """前向传播: Transformer主干 → RMSNorm → value_head → 价值估计 [B, S]"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """计算每个回复的总奖励（规则奖励 + 奖励模型评分）

    规则奖励包含:
    - 长度奖励: 20~800 字符 +0.5，否则 -0.5
    - 思考奖励: 有思考内容且 20~300 字符 +1.0，否则 -0.5
    - 思考格式: 恰好1个思考分隔符 +0.25，否则 -0.25
    - 重复惩罚: n-gram 重复率惩罚（减去）

    Args:
        prompts: prompt 列表，长度 B
        responses: 回复列表，长度 B
        reward_model: 奖励模型，用于对回复质量打分
    Returns:
        rewards: 每个回复的总奖励张量，形状 [B]
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从 prompt 中解析 ChatML 格式的消息
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response

            # 规则奖励1: 长度奖励
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

            # 规则奖励2: 思考格式奖励
            think_tag = '```' + '\n'
            if think_tag in response:
                thinking_content, answer_content = response.split(think_tag, 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count(think_tag) == 1 else -0.25
                answer = answer_content.strip()

            # 规则奖励3: 重复惩罚
            rewards[i] -= rep_penalty(answer)

            # 奖励模型评分
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # 累加奖励模型评分
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    """PPO 训练一个 epoch 的主循环

    每个训练步骤:
    1. Rollout: 使用 Actor 生成回复
    2. 计算奖励和参考模型对数概率
    3. 使用 Critic 估计状态价值
    4. 使用 GAE 计算优势函数
    5. 多轮 PPO 更新（ppo_update_iters 次）:
       - 计算 Actor loss（裁剪目标 + KL 惩罚）
       - 计算 Critic loss（裁剪价值函数）
       - 反向传播 + 梯度累积
    6. 定期保存模型和同步 Rollout 引擎

    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        rollout_engine: 推理引擎
        ref_model: 参考模型（冻结）
        actor_scheduler: Actor 学习率调度器
        critic_scheduler: Critic 学习率调度器
        reward_model: 奖励模型
        start_step: 起始步数
        wandb: wandb 日志记录器
        use_sglang: 是否使用 SGLang 推理引擎
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], 长度 B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_seq_len, padding_side="left").to(args.device)

        # ========== 第1步: Rollout —— 使用 Actor 生成回复 ==========
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,       # PPO 每个 prompt 只生成 1 个回复（与 GRPO 不同）
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids          # 完整输出: [B, P+R]
        completion_ids = rollout_result.completion_ids  # 仅生成部分: [B, R]
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # prompt 长度
        responses_text = rollout_result.completions       # 解码后的文本列表
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # 旧策略的逐 token 对数概率

        # ========== 第2步: 计算奖励 ==========
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        # ========== 调试模式: 打印采样结果 ==========
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # ========== 第3步: 准备 mask 和位置索引 ==========
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1] 用于 log_softmax gather
        B = len(prompts)
        resp_labels = completion_ids
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx  # 生成 token 在完整序列中的位置

        # 构建回复部分的 mask
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        resp_lengths = resp_pad_mask.sum(dim=1)
        valid_resp = resp_lengths > 0
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()  # 策略 loss 的 mask
        resp_value_mask = resp_policy_mask.clone()  # 价值 loss 的 mask

        # ========== 第4步: 计算 Critic 价值和参考模型对数概率（无梯度） ==========
        with torch.no_grad():
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask  # 提取生成部分的价值
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ========== 第5步: 构建逐 token 奖励 ==========
        token_rewards = torch.zeros_like(old_resp_logp)  # 默认为 0
        last_idx = resp_lengths - 1  # [B] 每个序列最后一个有效 token 的索引
        # 在每个序列的末尾加上外部奖励
        token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

        # ========== 第6步: GAE（Generalized Advantage Estimation）计算优势函数 ==========
        gen_len = old_resp_values.size(1)
        lastgaelam = torch.zeros(B, device=args.device)
        advs_rev = []
        # 从后往前计算 GAE
        for t in reversed(range(gen_len)):
            nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advs_rev.append(lastgaelam)
        advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R] 翻转回正序
        returns = advantages + old_resp_values  # [B, R] 回报 = 优势 + 价值

        # 优势归一化
        adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
        adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
        advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ========== 第7步: PPO 多轮更新 ==========
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        policy_loss_sum = 0.0; value_loss_sum = 0.0; kl_sum = 0.0; kl_ref_sum = 0.0
        clipfrac_sum = 0.0; aux_loss_sum = 0.0; log_count = 0

        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model

        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo: break
            b_inds = torch.randperm(B, device=args.device)  # 随机打乱 batch 索引
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]  # mini-batch 索引

                # Critic 前向传播
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                # Actor 前向传播
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                # 重要性采样比率
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                # 近似 KL 散度（用于早停判断）
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True  # KL 过大，提前停止 PPO 更新

                ratio = torch.exp(log_ratio)
                # 裁剪比例（用于监控）
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))

                # KL 参考惩罚: KL(pi_ref || pi_current)
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # Actor loss: 裁剪目标 + KL 惩罚
                policy_loss = (
                    (torch.max(-advantages[inds] * ratio,
                               -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                     * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                    + args.kl_coef * kl_ref_penalty
                )

                # Critic loss: 裁剪价值函数（防止价值估计变化过大）
                value_loss = 0.5 * (
                    torch.max((mb_resp_values - returns[inds]) ** 2,
                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0  # loss 置零
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                loss.backward()

                # 累计统计量
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1
                grad_accum_step += 1

                # 梯度累积 + 优化器更新
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 处理未完成的梯度累积步
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ========== 同步 Rollout 引擎 ==========
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(actor_model)

        # ========== 日志记录 ==========
        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val, "kl_ref": kl_ref_val, "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val, "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val, "actor_lr": actor_lr, "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ========== 保存模型检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)  # 处理 torch.compile 的包装
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)  # 保存为半精度
            # 通过 CheckpointManager 保存完整状态（包括 Critic）
            ckpt_mgr.save(actor_model, actor_optimizer, epoch, step,
                         metrics={'reward': rewards.mean().item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'lr': actor_optimizer.param_groups[0]['lr']},
                         wandb=wandb, scheduler=actor_scheduler,
                         critic_model=critic_model, critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # 释放中间变量
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos
        del resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
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
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--max_keep", type=int, default=3, help="最多保留的checkpoint数量（0=不限制）")
    parser.add_argument("--resume_mode", type=str, default="latest", help="续训模式: latest/best/step编号")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor 模型（策略模型）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Critic 模型（价值模型）: 继承自 MiniMindForCausalLM，将 lm_head 替换为 value_head
    critic_model = CriticModel(lm_config).to(args.device)
    critic_model.load_state_dict(
        {k: v for k, v in init_model(lm_config, base_weight, device=args.device)[0].state_dict().items()
         if k in critic_model.state_dict()}, strict=False
    )
    # 参考模型（冻结）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # 奖励模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout 引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # Actor 和 Critic 使用不同的学习率
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs * args.ppo_update_iters
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        if 'critic_model' in ckp_data:
            critic_model.load_state_dict(ckp_data['critic_model'])
            critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
            critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        critic_model = torch.compile(critic_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model,
                           actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang=(args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model,
                           actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang=(args.rollout_engine == "sglang"))

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
