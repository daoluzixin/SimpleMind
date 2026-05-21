"""SFT Warmup 训练脚本 - Qwen2.5-1.5B-Instruct

对模型做短暂的 SFT，教会它输出正确的 <tool_call> 路由格式。
训练完成后，再用 RL (GRPO) 进一步优化路由准确率。

用法:
    # 单卡
    python sft_warmup_qwen.py --model_path Qwen/Qwen2.5-1.5B-Instruct --data_path dataset/sft_warmup.jsonl

    # 双卡 DDP
    torchrun --nproc_per_node=2 sft_warmup_qwen.py --model_path Qwen/Qwen2.5-1.5B-Instruct --data_path dataset/sft_warmup.jsonl
"""

import os
import sys
import json
import math
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
#  工具函数
# ============================================================

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content, flush=True)


# ============================================================
#  数据集
# ============================================================

ROUTER_TOOLS = [
    {"type": "function", "function": {
        "name": "delegate_to_math_agent",
        "description": "将数学计算、单位换算任务委托给数学专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要数学专家处理的任务描述"}
        }, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "delegate_to_info_agent",
        "description": "将信息查询任务（天气、时间、汇率）委托给信息查询专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要信息专家处理的任务描述"}
        }, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "delegate_to_translate_agent",
        "description": "将翻译任务委托给翻译专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要翻译专家处理的任务描述"}
        }, "required": ["task"]}}},
]


class SFTWarmupDataset(Dataset):
    """SFT Warmup 数据集

    将 messages 转换为 input_ids + labels，
    只在 assistant 回复部分计算 loss。
    """

    def __init__(self, data_path, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        Logger(f"Loaded {len(self.data)} SFT warmup examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # 使用 Qwen2.5 的 chat_template 构建完整对话
        # 对于需要 tool 的样本，传入 tools 让模板包含工具信息
        needs_tool = item.get("needs_tool", False)

        # 构建 prompt（不含 assistant 回复）
        prompt_messages = messages[:-1]  # system + user
        if needs_tool:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                tools=ROUTER_TOOLS,
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

        # 构建完整文本（含 assistant 回复）
        assistant_content = messages[-1]["content"]
        full_text = prompt_text + assistant_content + self.tokenizer.eos_token

        # tokenize
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                   truncation=True, max_length=self.max_len)["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                     truncation=True, max_length=self.max_len)["input_ids"]

        # labels: prompt 部分设为 -100（不计算 loss），只在 response 部分计算
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 确保长度一致
        assert len(full_ids) == len(labels), f"Length mismatch: {len(full_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """动态 padding"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(item["input_ids"]), dtype=torch.long),
                                          torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


# ============================================================
#  训练
# ============================================================

def train(args):
    # DDP 初始化
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logger("=" * 70)
    Logger("SFT Warmup Training - Qwen2.5-1.5B-Instruct")
    Logger(f"  Model: {args.model_path}")
    Logger(f"  Data: {args.data_path}")
    Logger(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    Logger(f"  Save to: {args.save_dir}")
    Logger(f"  DDP: {ddp}, Device: {device}")
    Logger("=" * 70)

    # 加载 tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"  Parameters: {total_params / 1e6:.1f}M")

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 数据集
    dataset = SFTWarmupDataset(args.data_path, tokenizer, max_len=args.max_len)
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine LR scheduler
    total_steps = args.epochs * len(dataloader)
    warmup_steps = min(10, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    model.train()
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_tokens = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            num_tokens = (labels != -100).sum().item()
            epoch_tokens += num_tokens

            if global_step % args.log_interval == 0 or global_step == 1:
                avg_loss = epoch_loss / (step + 1)
                elapsed = time.time() - start_time
                Logger(
                    f"[Epoch {epoch+1}/{args.epochs}] Step {step+1}/{len(dataloader)} | "
                    f"loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"tokens={num_tokens} | elapsed={elapsed:.0f}s"
                )

        # Epoch 结束
        avg_epoch_loss = epoch_loss / len(dataloader)
        Logger(f"\n[Epoch {epoch+1}/{args.epochs}] Completed | avg_loss={avg_epoch_loss:.4f} | tokens={epoch_tokens}")

        # 保存 checkpoint
        if is_main_process():
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
            raw_model = model.module if ddp else model
            raw_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            Logger(f"[Checkpoint] Saved epoch {epoch+1} -> {save_path}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(args.save_dir, "best")
                raw_model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                Logger(f"[Checkpoint] New best loss={best_loss:.4f} -> {best_path}")

    total_time = time.time() - start_time
    Logger(f"\nSFT Warmup Complete! Total time: {total_time:.0f}s, Best loss: {best_loss:.4f}")

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="SFT Warmup for Handoff Routing")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="dataset/sft_warmup.jsonl")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_sft_warmup")
    parser.add_argument("--epochs", type=int, default=3)  # v2: 5→3, 减少过拟合倾向
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--log_interval", type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
