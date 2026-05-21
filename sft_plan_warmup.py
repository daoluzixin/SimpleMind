"""Plan-then-Execute SFT Warm-up — Qwen2.5-1.5B-Instruct

目标：用少量标注样本（120条）教会模型 <tool_call> 格式，为后续 RL 训练提供格式先验。
训练仅覆盖 assistant 回复部分的 token（prompt 部分 mask=0 不参与 loss）。

关键设计：
    1. 使用 apply_chat_template(tools=PLAN_ROUTER_TOOLS) 编码，让模型在有工具定义的上下文中学习
    2. loss 仅计算 assistant 回复部分，避免 system/user 内容干扰
    3. 2 epoch 足够教会格式（120 × 2 / batch_size 步）
    4. lr=2e-5，比 RL 高一个数量级，SFT 需要快速拟合

用法:
    python sft_plan_warmup.py \\
        --model_path Qwen/Qwen2.5-1.5B-Instruct \\
        --data_path ./dataset/sft_plan_warmup.jsonl \\
        --save_dir ./checkpoints_qwen_plan_sft \\
        --epochs 2 --batch_size 4 --learning_rate 2e-5
"""
import os
import sys
import json
import math
import random
import argparse
import warnings
from contextlib import nullcontext
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_handoff_qwen import init_model_qwen, is_main_process, Logger, setup_seed
from agent_plan_qwen import PLAN_ROUTER_SYSTEM_PROMPT, PLAN_ROUTER_TOOLS


# ==============================================================================
#  SFT 数据集
# ==============================================================================

class SFTWarmupDataset(Dataset):
    """SFT Warm-up 数据集

    每条样本格式：
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<tool_call>\n{...}\n</tool_call>"}
        ],
        "level": 0-4,
        "dependency_type": "single|parallel|sequential|conditional|mixed"
    }
    """

    def __init__(self, data_path, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        if not os.path.exists(data_path):
            Logger(f"Warning: {data_path} not found")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

        Logger(f"Loaded {len(self.samples)} SFT samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]

        # 用 apply_chat_template 编码完整对话（含 tools 上下文）
        # 不加 generation prompt（因为我们有完整的 assistant 回复）
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            tools=PLAN_ROUTER_TOOLS,
        )

        # 编码 prompt 部分（不含 assistant 回复）来确定 loss mask 起始位置
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            tools=PLAN_ROUTER_TOOLS,
        )

        # Tokenize
        full_ids = self.tokenizer(
            full_text, add_special_tokens=False, truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]

        # Loss mask: 只对 assistant 部分计算 loss
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # 截断到 max_seq_len
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        return {
            "input_ids": full_ids,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id=0):
    """动态 padding collate 函数"""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ==============================================================================
#  训练循环
# ==============================================================================

def run_sft_warmup(args):
    """SFT Warm-up 训练主流程"""
    setup_seed(42)

    # 模型初始化
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = init_model_qwen(
        args.model_path, device=args.device, dtype=dtype,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    # 数据集
    dataset = SFTWarmupDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    if len(dataset) == 0:
        Logger("Error: No data loaded, exiting.")
        return

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(loader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.learning_rate / 10,
    )

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    autocast_ctx = (nullcontext() if device_type == "cpu"
                    else torch.cuda.amp.autocast(dtype=dtype))

    # 训练
    Logger("=" * 70)
    Logger("SFT Warm-up Training — Plan-then-Execute Format Teaching")
    Logger(f"  Model: {args.model_path}")
    Logger(f"  Data: {len(dataset)} samples, {args.epochs} epochs")
    Logger(f"  Batch size: {args.batch_size}, Total steps: {total_steps}")
    Logger(f"  LR: {args.learning_rate}, dtype: {args.dtype}")
    Logger(f"  Save to: {args.save_dir}")
    Logger("=" * 70)

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, batch in enumerate(loader):
            global_step += 1
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # Shift: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                # Cross-entropy loss (只在 labels != -100 的位置计算)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            loss.backward()

            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 统计
            valid_tokens = (shift_labels != -100).sum().item()
            epoch_loss += loss.item() * valid_tokens
            epoch_tokens += valid_tokens

            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                ppl = math.exp(min(avg_loss, 20))  # 防溢出
                Logger(
                    f"  [Epoch {epoch+1}/{args.epochs}] Step {global_step}/{total_steps} | "
                    f"loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | "
                    f"ppl={ppl:.2f} | lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        # Epoch 结束统计
        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        Logger(f"\n  Epoch {epoch+1} Complete: avg_loss={avg_epoch_loss:.4f}, ppl={ppl:.2f}")

        # 每个 epoch 保存一次
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(args.save_dir, "best")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            Logger(f"  ✓ Best model saved to {save_path} (loss={best_loss:.4f})")

    # 最终保存
    final_path = os.path.join(args.save_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    Logger(f"\n  Final model saved to {final_path}")
    Logger("\nSFT Warm-up Training complete.")
    Logger(f"  Next step: use '{args.save_dir}/best' as --model_path for RL training")


# ==============================================================================
#  验证模式：检查 SFT 数据编码是否正确
# ==============================================================================

def run_verify(args):
    """验证 SFT 数据的编码和 mask 是否正确"""
    _, tokenizer = init_model_qwen(
        args.model_path, device="cpu",
        gradient_checkpointing=False,
    )

    dataset = SFTWarmupDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    Logger(f"\nVerifying {len(dataset)} samples...\n")

    stats = {"total": 0, "has_tool_call": 0, "avg_response_len": 0}
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # 统计
        response_len = sum(1 for l in labels if l != -100)
        has_tc = "<tool_call>" in tokenizer.decode(input_ids)
        stats["total"] += 1
        stats["has_tool_call"] += int(has_tc)
        stats["avg_response_len"] += response_len

        # 显示
        Logger(f"Sample {i}:")
        Logger(f"  Total tokens: {len(input_ids)}")
        Logger(f"  Response tokens (loss computed): {response_len}")
        Logger(f"  Has <tool_call>: {has_tc}")

        # 解码 response 部分
        response_ids = [tid for tid, l in zip(input_ids, labels) if l != -100]
        response_text = tokenizer.decode(response_ids)
        Logger(f"  Response preview: {response_text[:150]}...")
        Logger("")

    if stats["total"] > 0:
        stats["avg_response_len"] /= stats["total"]
        Logger(f"Summary: {stats['has_tool_call']}/{stats['total']} samples have <tool_call>")
        Logger(f"  Avg response length: {stats['avg_response_len']:.0f} tokens")


# ==============================================================================
#  Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Warm-up for Plan-then-Execute")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "verify"])

    # 模型
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    # 数据
    parser.add_argument("--data_path", type=str, default="./dataset/sft_plan_warmup.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # 训练
    parser.add_argument("--save_dir", type=str, default="./checkpoints_qwen_plan_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")

    # 日志
    parser.add_argument("--log_interval", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "train":
        run_sft_warmup(args)
    else:
        run_verify(args)
