"""MiniMind-Agent Handoff — Qwen2.5-1.5B-Instruct 适配版

基于 trainer/agent_handoff.py 的完整适配，将底层模型从 MiniMind 64M
替换为 Qwen2.5-1.5B-Instruct，支持 DDP 双卡 4090 分布式训练。

核心改动（相对于原版 agent_handoff.py）:
    1. 模型初始化: MiniMindForCausalLM → AutoModelForCausalLM.from_pretrained
    2. compute_per_token_logps: 移除 logits_to_keep 依赖，改用 HF 标准 forward
    3. TorchRolloutEngine: 使用 HF 标准 generate() 接口
    4. gradient_checkpointing: 1.5B 模型在 24GB 4090 上必须开启
    5. 移除 MoE aux_loss 相关逻辑（Qwen2.5 不使用 MoE）
    6. CheckpointManager: 使用 HF save_pretrained/from_pretrained 保存/加载

架构不变:
    RouterAgent → [MathAgent | InfoAgent | TranslateAgent] → RouterAgent (Synthesize)
    所有 Agent 共享同一模型权重，用不同 system prompt 区分角色。
    GRPO/CISPO + Credit Assignment 分离 Router/Expert reward。

使用方式:
    # 单卡训练
    python agent_handoff_qwen.py --mode train --model_path Qwen/Qwen2.5-1.5B-Instruct

    # DDP 双卡训练（推荐）
    torchrun --nproc_per_node=2 agent_handoff_qwen.py --mode train --model_path Qwen/Qwen2.5-1.5B-Instruct

    # Demo
    python agent_handoff_qwen.py --mode demo --model_path Qwen/Qwen2.5-1.5B-Instruct

依赖:
    pip install torch transformers modelscope accelerate
    复用 trainer/train_agent.py 中的工具定义、Mock 数据和 reward 计算逻辑。
"""
import os
import sys
import re
import gc
import json
import math
import random
import argparse
import warnings
import time as _time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---- 复用原项目中与模型无关的工具/数据 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer.train_agent import (
    TOOLS, MOCK_RESULTS, CHECK_ARGS, WEATHER_DATA, TIME_DATA,
    EXCHANGE_DATA, TRANSLATE_DATA, UNIT_DATA,
    execute_tool, rep_penalty, validate_gt_in_text,
)


def parse_tool_calls(text):
    """从模型输出文本中解析工具调用请求

    兼容三种格式（按优先级）:
    1. Qwen2.5 格式: <tool_call>{"name": "xxx", "arguments": {...}}</tool_call>
    2. MiniMind 格式: ゅ{"name": "xxx", "arguments": {...}}゜
    3. Fallback: 裸 JSON（含 "name" + "arguments" 字段，且 name 在白名单内）
    """
    # 合法工具名白名单
    VALID_TOOL_NAMES = {
        "delegate_to_math_agent", "delegate_to_info_agent",
        "delegate_to_translate_agent", "execute_plan",
    }

    calls = []
    # 1. Qwen2.5 <tool_call> XML 格式（优先）
    for m in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    if calls:
        return calls

    # 2. 兼容旧版 MiniMind 格式
    for m in re.findall(r'ゅ(.*?)゜', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    if calls:
        return calls

    # 3. Fallback: 从文本中提取裸 JSON tool_call（兼容模型未输出标签的情况）
    #    使用贪心策略：找到 {"name": 开头的 JSON 对象
    for m in re.finditer(r'\{\s*"name"\s*:', text):
        start = m.start()
        # 尝试从此位置解析完整 JSON 对象
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                obj = json.loads(text[start:end])
                if (isinstance(obj, dict) and "name" in obj
                        and "arguments" in obj
                        and obj["name"] in VALID_TOOL_NAMES):
                    calls.append(obj)
            except (json.JSONDecodeError, ValueError):
                pass
    return calls


# ==============================================================================
#  通用工具（从 trainer_utils.py 提取，去除 MiniMind 特定依赖）
# ==============================================================================

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    backend = os.environ.get("DIST_BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SkipBatchSampler(Sampler):
    """支持断点续训的 BatchSampler，跳过前 skip_batches 个 batch"""

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# ==============================================================================
#  Qwen 模型初始化
# ==============================================================================

def init_model_qwen(model_path: str, device: str = "cuda", dtype=torch.bfloat16,
                    gradient_checkpointing: bool = True):
    """加载 Qwen2.5-1.5B-Instruct（HuggingFace 标准接口）

    Args:
        model_path: HuggingFace model id 或本地路径
        device: 目标设备
        dtype: 模型精度（bfloat16 推荐）
        gradient_checkpointing: 是否开启梯度检查点（24GB 4090 必须开）

    Returns:
        (model, tokenizer)
    """
    Logger(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Qwen2.5 可能没有 pad_token，需要手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 如果安装了 flash-attn；否则会自动降级
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        Logger("  Gradient checkpointing: ENABLED")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"  Parameters: {param_count:.2f}B total, {trainable_count:.1f}M trainable")

    return model.to(device), tokenizer


# ==============================================================================
#  Rollout 引擎（适配 HuggingFace 标准接口）
# ==============================================================================

def compute_per_token_logps_hf(model, input_ids: Tensor, n_keep: int,
                               attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算生成部分每个 token 的对数概率（HuggingFace 标准版，显存优化）

    使用分块计算避免在整个序列上同时持有 [B, L, V] 的 log_softmax 结果。
    每次只处理 chunk_size 个 token 的 logits，显著降低峰值显存。

    Args:
        model: 策略模型（可能被 DDP 包裹）
        input_ids: 完整 token IDs [B, L]
        n_keep: 需要计算 logps 的 token 数（completion 长度）
        attention_mask: 注意力掩码 [B, L]

    Returns:
        per_token_logps: [B, n_keep]
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

    # HuggingFace 标准 forward
    outputs = unwrapped(input_ids, attention_mask=attention_mask)
    # 只取最后 n_keep 个位置的 logits（shift by 1）
    logits = outputs.logits[:, -(n_keep + 1):-1, :]  # [B, n_keep, V]
    target_ids = input_ids[:, -n_keep:]  # [B, n_keep]

    # 分块计算 log_softmax + gather，避免一次性在整个 [B, n_keep, V] 上分配显存
    chunk_size = 128  # 每次处理 128 个 token
    B = logits.size(0)
    T = logits.size(1)
    per_token_logps = torch.empty(B, T, device=logits.device, dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_logits = logits[:, start:end, :]  # [B, chunk, V]
        chunk_targets = target_ids[:, start:end]  # [B, chunk]
        chunk_log_probs = F.log_softmax(chunk_logits.float(), dim=-1)
        per_token_logps[:, start:end] = torch.gather(
            chunk_log_probs, 2, chunk_targets.unsqueeze(-1)
        ).squeeze(-1)
        del chunk_logits, chunk_log_probs  # 立即释放

    del logits  # 释放完整 logits
    return per_token_logps


@dataclass
class RolloutResult:
    """Rollout 结果"""
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


class TorchRolloutEngineHF:
    """HuggingFace 标准 Rollout 引擎

    使用 HF 的 model.generate() 接口（与 Qwen2.5 完全兼容）。
    """

    def __init__(self, policy_model, tokenizer, device="cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor,
                num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        model = self.policy_model.module if isinstance(
            self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()

        with torch.no_grad(), ctx:
            expanded_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
            expanded_mask = attention_mask.repeat_interleave(num_generations, dim=0)

            # HuggingFace 标准 generate
            output_ids = model.generate(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()

            # 计算 per-token logps
            per_token_logps = compute_per_token_logps_hf(
                self.policy_model, output_ids, completion_ids.size(1),
                attention_mask=full_mask
            )

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids, completion_ids, per_token_logps, completions,
            prompt_ids.new_full((output_ids.size(0),), prompt_len),
            attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
        )

    def update_policy(self, model):
        self.policy_model = model


# ==============================================================================
#  Checkpoint 管理（适配 HuggingFace save_pretrained）
# ==============================================================================

class QwenCheckpointManager:
    """Checkpoint 管理器（HuggingFace 原生格式）

    使用 save_pretrained / from_pretrained 保存和加载，
    兼容 HuggingFace 生态。
    """

    def __init__(self, save_dir="./checkpoints_qwen", max_keep=5,
                 track_metric="reward", metric_mode="max"):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.track_metric = track_metric
        self.metric_mode = metric_mode
        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.best_step = -1
        self._history = []
        os.makedirs(save_dir, exist_ok=True)
        self._load_history()

    # ---- 内部方法 ----

    def _history_path(self):
        return os.path.join(self.save_dir, "ckpt_history.json")

    def _load_history(self):
        path = self._history_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._history = data.get("history", [])
                self.best_metric = data.get("best_metric", self.best_metric)
                self.best_step = data.get("best_step", -1)
            except (json.JSONDecodeError, KeyError):
                self._history = []

    def _save_history(self):
        data = {
            "history": self._history,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "track_metric": self.track_metric,
            "metric_mode": self.metric_mode,
        }
        path = self._history_path()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def _step_dir(self, step):
        return os.path.join(self.save_dir, f"step_{step}")

    def _best_dir(self):
        return os.path.join(self.save_dir, "best")

    def _is_better(self, value):
        if self.metric_mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    # ---- 公开方法 ----

    def save(self, model, tokenizer, optimizer, scheduler, epoch, step,
             metric_value=None, extra_metrics=None, wandb=None):
        if not is_main_process():
            return

        step_dir = self._step_dir(step)
        os.makedirs(step_dir, exist_ok=True)

        # 保存模型和 tokenizer（HF 格式）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        raw_model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)

        # 保存训练状态
        train_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                train_state["wandb_id"] = getattr(run, "id", None) if run else None
            else:
                train_state["wandb_id"] = getattr(wandb, "id", None)
        torch.save(train_state, os.path.join(step_dir, "train_state.pt"))

        # 追踪最佳
        metrics = extra_metrics or {}
        is_best = False
        if metric_value is not None and self._is_better(metric_value):
            self.best_metric = metric_value
            self.best_step = step
            is_best = True
            best_dir = self._best_dir()
            if os.path.exists(best_dir):
                import shutil
                shutil.rmtree(best_dir)
            raw_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save(train_state, os.path.join(best_dir, "train_state.pt"))
            Logger(f"[Checkpoint] New best {self.track_metric}={metric_value:.4f} at step {step}")

        # 历史记录
        record = {
            "step": step, "epoch": epoch, "path": step_dir,
            "metrics": metrics, "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_best": is_best,
        }
        self._history.append(record)

        # 淘汰旧 checkpoint
        if self.max_keep > 0 and len(self._history) > self.max_keep:
            import shutil
            to_remove = self._history[:-self.max_keep]
            self._history = self._history[-self.max_keep:]
            for old in to_remove:
                old_path = old["path"]
                if os.path.isdir(old_path) and old_path != self._best_dir():
                    shutil.rmtree(old_path, ignore_errors=True)
                    Logger(f"[Checkpoint] Removed old: {os.path.basename(old_path)}")

        self._save_history()
        Logger(f"[Checkpoint] Saved step {step} (epoch {epoch + 1})")

    def load(self, resume_mode="latest"):
        """加载 checkpoint

        Args:
            resume_mode: "latest" / "best" / int(step)

        Returns:
            dict with keys: model_path, optimizer, scheduler, epoch, step, wandb_id
            或 None
        """
        if resume_mode == "best":
            best_dir = self._best_dir()
            if os.path.isdir(best_dir):
                state_path = os.path.join(best_dir, "train_state.pt")
                if os.path.exists(state_path):
                    Logger(f"[Checkpoint] Loading best (step {self.best_step})")
                    state = torch.load(state_path, map_location="cpu")
                    state["model_path"] = best_dir
                    return state
            Logger("[Checkpoint] No best found, falling back to latest")
            resume_mode = "latest"

        if isinstance(resume_mode, int):
            step_dir = self._step_dir(resume_mode)
            state_path = os.path.join(step_dir, "train_state.pt")
            if os.path.exists(state_path):
                Logger(f"[Checkpoint] Loading step {resume_mode}")
                state = torch.load(state_path, map_location="cpu")
                state["model_path"] = step_dir
                return state
            Logger(f"[Checkpoint] Step {resume_mode} not found, falling back to latest")
            resume_mode = "latest"

        # latest: 取历史中最后一个
        if self._history:
            last = self._history[-1]
            state_path = os.path.join(last["path"], "train_state.pt")
            if os.path.exists(state_path):
                Logger(f"[Checkpoint] Loading latest (step {last['step']})")
                state = torch.load(state_path, map_location="cpu")
                state["model_path"] = last["path"]
                return state

        return None


# ==============================================================================
#  Expert Agent 配置（与原版完全一致）
# ==============================================================================

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

ROUTER_SYSTEM_PROMPT = (
    "你是一个任务路由器（Router Agent）。你的职责是分析用户请求，"
    "决定是自己直接回答，还是委托给合适的专家 Agent。\n"
    "专家列表：\n"
    "- delegate_to_math_agent: 数学计算、公式求解、单位换算\n"
    "- delegate_to_info_agent: 天气查询、时间查询、汇率查询\n"
    "- delegate_to_translate_agent: 文本翻译\n"
    "如果问题不需要工具，直接回答即可。"
)

MATH_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("calculate_math", "unit_converter")]
INFO_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("get_current_weather", "get_current_time", "get_exchange_rate")]
TRANSLATE_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("translate_text",)]

EXPERT_CONFIG = {
    "delegate_to_math_agent": {
        "system_prompt": "你是一个数学计算专家。使用提供的工具来精确计算数学表达式和进行单位换算。务必调用工具获取精确结果。",
        "tools": MATH_TOOLS,
    },
    "delegate_to_info_agent": {
        "system_prompt": "你是一个信息查询专家。使用提供的工具查询天气、时间、汇率等实时信息。务必调用工具获取最新数据。",
        "tools": INFO_TOOLS,
    },
    "delegate_to_translate_agent": {
        "system_prompt": "你是一个翻译专家。使用提供的工具进行文本翻译。务必调用翻译工具以获取准确翻译。",
        "tools": TRANSLATE_TOOLS,
    },
}


# ==============================================================================
#  数据集
# ==============================================================================

class HandoffDataset(Dataset):
    """Agent Handoff 训练数据集"""

    def __init__(self, data_path):
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        Logger(f"Loaded {len(self.data)} examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item.get("query", item.get("question", "")),
            "gt": item.get("gt", item.get("ground_truth", [])),
            "needs_tool": item.get("needs_tool", True),
            "expert": item.get("expert", "none"),
        }


# ==============================================================================
#  四阶段 Rollout（与原版逻辑完全一致，仅去除 MiniMind 特有依赖）
# ==============================================================================

def handoff_rollout_single(rollout_engine, tokenizer, user_query, needs_tool=None,
                           max_new_tokens=384, thinking_ratio=0.0, device="cuda"):
    """单个样本的四阶段 Handoff Rollout

    Phase 1: RouterAgent 决策（路由 or 直接回答）
    Phase 2: 专家 Agent 执行工具调用（可多轮）
    Phase 3: RouterAgent 整合专家结果
    Phase 4: 返回完整结果

    与原版 agent_handoff.py 中 handoff_rollout_single 逻辑完全一致。
    """
    open_thinking = (random.random() < thinking_ratio) if thinking_ratio > 0 else False

    # ===== Phase 1: RouterAgent 决策 =====
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    router_context = tokenizer.apply_chat_template(
        router_messages, tokenize=False, add_generation_prompt=True,
        tools=ROUTER_TOOLS,
    )
    router_inputs = tokenizer(router_context, return_tensors="pt", add_special_tokens=False).to(device)
    router_prompt_ids = router_inputs["input_ids"][0].tolist()

    router_result = rollout_engine.rollout(
        prompt_ids=router_inputs["input_ids"],
        attention_mask=router_inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
    )
    router_gen_ids = router_result.completion_ids[0].tolist()
    router_gen_logps = router_result.per_token_logps[0].tolist()

    # 去 pad 和 eos
    pairs = [(t, lp) for t, lp in zip(router_gen_ids, router_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    router_gen_ids = [t for t, _ in pairs]
    router_gen_logps = [lp for _, lp in pairs]
    router_text = router_result.completions[0]

    all_outputs = [router_text]

    # ===== Phase 2: 判断是否需要 Handoff =====
    handoff_call = None
    for call in parse_tool_calls(router_text):
        name = call.get("name", "")
        if name.startswith("delegate_to_") and name in EXPERT_CONFIG:
            handoff_call = call
            break

    if handoff_call is None:
        return {
            "final_answer": router_text,
            "router_prompt_ids": router_prompt_ids,
            "router_response_ids": router_gen_ids,
            "router_response_mask": [1] * len(router_gen_ids),
            "router_old_logps": router_gen_logps,
            "expert_prompt_ids": [],
            "expert_response_ids": [],
            "expert_response_mask": [],
            "expert_old_logps": [],
            "synth_prompt_ids": [],
            "synth_response_ids": [],
            "synth_response_mask": [],
            "synth_old_logps": [],
            "handoff_occurred": False,
            "delegated_expert": None,
            "delegate_task": "",
            "all_outputs": all_outputs,
            "needs_tool": needs_tool,
        }

    delegated_expert = handoff_call["name"]
    task_args = handoff_call.get("arguments", {})
    if isinstance(task_args, str):
        try:
            task_args = json.loads(task_args)
        except (json.JSONDecodeError, ValueError):
            task_args = {}
    delegated_task = task_args.get("task", user_query)

    # ===== Phase 3: 专家 Agent 子 Rollout =====
    expert_config = EXPERT_CONFIG[delegated_expert]
    expert_messages = [
        {"role": "system", "content": expert_config["system_prompt"]},
        {"role": "user", "content": delegated_task},
    ]
    expert_tools = expert_config["tools"]

    expert_response_ids = []
    expert_response_mask = []
    expert_old_logps = []
    expert_prompt_ids = None
    expert_final_answer = ""
    max_tool_turns = 3

    for turn in range(max_tool_turns):
        expert_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False, add_generation_prompt=True,
            tools=expert_tools,
        )
        expert_inputs = tokenizer(expert_context, return_tensors="pt", add_special_tokens=False).to(device)
        if expert_prompt_ids is None:
            expert_prompt_ids = expert_inputs["input_ids"][0].tolist()

        expert_result = rollout_engine.rollout(
            prompt_ids=expert_inputs["input_ids"],
            attention_mask=expert_inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = expert_result.completion_ids[0].tolist()
        new_logps = expert_result.per_token_logps[0].tolist()

        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = expert_result.completions[0]

        all_outputs.append(new_text)
        expert_response_ids.extend(new_ids)
        expert_response_mask.extend([1] * len(new_ids))
        expert_old_logps.extend(new_logps)

        # 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            expert_final_answer = new_text
            break

        expert_messages.append({"role": "assistant", "content": new_text})

        # 执行工具
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
            expert_messages.append({"role": "tool", "content": result_str})

        # 工具模板 token（mask=0）
        is_last_turn = (turn == max_tool_turns - 1)
        observe_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False,
            add_generation_prompt=not is_last_turn,
            tools=expert_tools,
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(expert_prompt_ids) + len(expert_response_ids)
        obs_delta = observe_ids[current_len:]
        expert_response_ids.extend(obs_delta)
        expert_response_mask.extend([0] * len(obs_delta))
        expert_old_logps.extend([0.0] * len(obs_delta))

    if not expert_final_answer:
        expert_final_answer = all_outputs[-1] if len(all_outputs) > 1 else ""

    # ===== Phase 4: RouterAgent 整合回答 =====
    synth_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": router_text},
        {"role": "tool", "content": json.dumps({"expert_result": expert_final_answer}, ensure_ascii=False)},
    ]

    synth_context = tokenizer.apply_chat_template(
        synth_messages, tokenize=False, add_generation_prompt=True,
        tools=ROUTER_TOOLS,
    )
    synth_inputs = tokenizer(synth_context, return_tensors="pt", add_special_tokens=False).to(device)
    synth_prompt_ids = synth_inputs["input_ids"][0].tolist()

    synth_result = rollout_engine.rollout(
        prompt_ids=synth_inputs["input_ids"],
        attention_mask=synth_inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens // 2,
        temperature=0.7,
    )
    synth_gen_ids = synth_result.completion_ids[0].tolist()
    synth_gen_logps = synth_result.per_token_logps[0].tolist()

    pairs = [(t, lp) for t, lp in zip(synth_gen_ids, synth_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    synth_gen_ids = [t for t, _ in pairs]
    synth_gen_logps = [lp for _, lp in pairs]
    synth_text = synth_result.completions[0]

    all_outputs.append(synth_text)
    final_answer = synth_text

    return {
        "final_answer": final_answer,
        "router_prompt_ids": router_prompt_ids,
        "router_response_ids": router_gen_ids,
        "router_response_mask": [1] * len(router_gen_ids),
        "router_old_logps": router_gen_logps,
        "expert_prompt_ids": expert_prompt_ids or [],
        "expert_response_ids": expert_response_ids,
        "expert_response_mask": expert_response_mask,
        "expert_old_logps": expert_old_logps,
        "synth_prompt_ids": synth_prompt_ids,
        "synth_response_ids": synth_gen_ids,
        "synth_response_mask": [1] * len(synth_gen_ids),
        "synth_old_logps": synth_gen_logps,
        "handoff_occurred": True,
        "delegated_expert": delegated_expert,
        "delegate_task": delegated_task,
        "all_outputs": all_outputs,
        "needs_tool": needs_tool,
    }


# ==============================================================================
#  Reward 计算（与原版一致，移除 thinking 分隔符对 MiniMind 的依赖）
# ==============================================================================

def calculate_handoff_rewards(results_batch, gt_batch, num_gen, reward_model=None, device="cuda", **kwargs):
    """计算 Handoff 场景的奖励（分离 Credit Assignment）

    奖励组成:
    1. 路由正确性: 需要工具的问题是否触发了 handoff（+0.5/-0.5）
    2. 专家选择正确性: 是否路由到了正确的专家（+0.3/-0.3）
    3. 委托质量: delegate task 描述是否保留了关键信息（+0.3）
    4. Thinking 格式奖励（适配 Qwen <think> 标签）
    5. GT 验证: 最终回答是否包含正确答案（核心，+2.5）
    6. 工具调用质量
    7. 重复惩罚
    """
    total = len(results_batch)
    rewards = torch.zeros(total, device=device)
    router_rewards = torch.zeros(total, device=device)
    expert_rewards = torch.zeros(total, device=device)

    for idx, result in enumerate(results_batch):
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        r_reward = 0.0
        e_reward = 0.0

        needs_tool = result.get("needs_tool")
        if needs_tool is None:
            needs_tool = len(gt) > 0
        handoff_occurred = result["handoff_occurred"]
        final_answer = result["final_answer"]

        # 去思考部分：Qwen 使用 <think>...</think> 标签
        answer_text = final_answer
        # 兼容 Qwen 的 <think> 标签
        if "<think>" in answer_text and "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[-1].strip()
        # 兼容原版的 ゜ 分隔符
        elif "゜" in answer_text:
            answer_text = answer_text.split("゜")[-1].strip()

        # ---- 1. 路由正确性（v2: 非对称奖励） ----
        # 支持渐进式惩罚系数 false_trigger_penalty_coeff（默认1.0，由训练循环按 epoch 递增）
        ft_coeff = kwargs.get("false_trigger_penalty_coeff", 1.0)
        if needs_tool and handoff_occurred:
            r_reward += 0.5
        elif needs_tool and not handoff_occurred:
            r_reward -= 0.5
        elif not needs_tool and not handoff_occurred:
            r_reward += 0.8      # v2: 0.3 → 0.8, 增强正确抑制的正向激励
        elif not needs_tool and handoff_occurred:
            r_reward -= 1.5 * ft_coeff  # v2: 0.3 → 1.5*coeff, 非对称重惩误触发
            e_reward = 0.0       # v2: 误触发时专家奖励归零，切断错误信号传播

        # ---- 2. 专家选择正确性 ----
        if handoff_occurred and result.get("delegated_expert"):
            expected_expert_map = {
                "math": "delegate_to_math_agent",
                "info": "delegate_to_info_agent",
                "translate": "delegate_to_translate_agent",
            }
            expected = result.get("expected_expert")
            if expected and expected in expected_expert_map:
                if result["delegated_expert"] == expected_expert_map[expected]:
                    r_reward += 0.3
                else:
                    r_reward -= 0.3

        # ---- 3. 委托质量 ----
        if handoff_occurred:
            delegate_task = result.get("delegate_task", "")
            query = result.get("original_query", "")
            if delegate_task and query:
                query_nums = set(re.findall(r'\d+(?:\.\d+)?', query))
                task_nums = set(re.findall(r'\d+(?:\.\d+)?', delegate_task))
                if query_nums and query_nums.issubset(task_nums):
                    r_reward += 0.15
                query_entities = set(re.findall(r'[\u4e00-\u9fff]+', query))
                task_entities = set(re.findall(r'[\u4e00-\u9fff]+', delegate_task))
                if query_entities and len(query_entities & task_entities) >= len(query_entities) * 0.5:
                    r_reward += 0.15

        # ---- 4. Thinking 格式奖励（适配 Qwen <think> 标签）----
        for output in result["all_outputs"]:
            think_part = ""
            if "<think>" in output and "</think>" in output:
                think_part = output.split("<think>")[1].split("</think>")[0]
            elif "゜" in output:
                think_part = output.split("゜")[0]
            if think_part and 20 <= len(think_part.strip()) <= 300:
                r_reward += 0.2
                break

        # ---- 5. GT 验证（核心奖励）----
        if gt:
            verified = validate_gt_in_text(answer_text, gt)
            gt_score = 2.5 * len(verified) / len(gt)
            if handoff_occurred:
                e_reward += gt_score * 0.6
                r_reward += gt_score * 0.4
            else:
                r_reward += gt_score
        elif not handoff_occurred:
            if reward_model is not None:
                try:
                    messages = [{"role": "user", "content": result.get("original_query", "")},
                                {"role": "assistant", "content": answer_text}]
                    score = reward_model.get_score(messages, answer_text)
                    r_reward += score
                except Exception:
                    pass
            r_reward += 0.3 if 5 <= len(answer_text) <= 800 else -0.3

        # ---- 6. 工具调用质量 ----
        if handoff_occurred:
            tool_calls = []
            for output in result["all_outputs"][1:-1]:
                tool_calls.extend(parse_tool_calls(output))

            if tool_calls:
                expert_name = result.get("delegated_expert", "")
                expert_tools = EXPERT_CONFIG.get(expert_name, {}).get("tools", TOOLS)
                valid_names = {t["function"]["name"] for t in expert_tools}
                valid_count = 0
                for tc in tool_calls:
                    name = tc.get("name", "")
                    raw = tc.get("arguments", {})
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            raw = {}
                    check = CHECK_ARGS.get(name)
                    valid_count += int(bool(name in valid_names and check and check(raw)))

                tool_gap = abs(valid_count - max(len(gt), 1)) + max(0, len(tool_calls) - valid_count)
                e_reward += 0.5 if tool_gap == 0 else -0.3 * min(tool_gap, 2)

        # ---- 7. 重复惩罚 ----
        penalty = rep_penalty(answer_text)
        r_reward -= penalty * 0.5
        e_reward -= penalty * 0.5

        # ---- 汇总 ----
        total_reward = r_reward + e_reward
        rewards[idx] = max(min(total_reward, 4.0), -3.0)
        router_rewards[idx] = max(min(r_reward, 2.5), -2.0)
        expert_rewards[idx] = max(min(e_reward, 2.5), -2.0)

    return rewards, router_rewards, expert_rewards


# ==============================================================================
#  训练步（核心：移除 aux_loss，使用 HF 标准 forward）
# ==============================================================================

def handoff_train_step(model, ref_model, rollout_engine, tokenizer, batch, args,
                       optimizer, scheduler, autocast_ctx, step, reward_model=None,
                       false_trigger_penalty_coeff=1.0):
    """单个训练步

    与原版 agent_handoff.py 的 handoff_train_step 逻辑一致，
    但移除了 MoE aux_loss 和 MiniMind 特有接口。
    """
    queries = batch["queries"]
    gt_batch = batch["gt"]
    needs_tool_batch = batch.get("needs_tool", [None] * len(queries))
    expert_batch = batch.get("expert", ["none"] * len(queries))

    # ===== Rollout =====
    results_batch = []
    with torch.no_grad():
        for i, query in enumerate(queries):
            for _ in range(args.num_generations):
                result = handoff_rollout_single(
                    rollout_engine, tokenizer, query,
                    needs_tool=needs_tool_batch[i],
                    max_new_tokens=args.max_gen_len,
                    thinking_ratio=args.thinking_ratio,
                    device=args.device,
                )
                result["original_query"] = query
                result["expected_expert"] = expert_batch[i]
                results_batch.append(result)

    # 释放 rollout 阶段的 GPU 缓存，为 train forward 腾出显存
    torch.cuda.empty_cache()
    gc.collect()

    # ===== Reward =====
    rewards, router_rewards, expert_rewards = calculate_handoff_rewards(
        results_batch, gt_batch, args.num_generations,
        reward_model=reward_model, device=args.device,
        false_trigger_penalty_coeff=false_trigger_penalty_coeff,
    )

    # ===== 打包序列（三段拼接）=====
    packed_samples = []
    segment_info = []

    for result in results_batch:
        prompt_ids = result["router_prompt_ids"]
        response_ids = list(result["router_response_ids"])
        response_mask = list(result["router_response_mask"])
        old_logps = list(result["router_old_logps"])

        router_len = len(response_ids)

        expert_start = router_len
        if result["handoff_occurred"] and result["expert_response_ids"]:
            expert_prompt = result["expert_prompt_ids"]
            response_ids.extend(expert_prompt)
            response_mask.extend([0] * len(expert_prompt))
            old_logps.extend([0.0] * len(expert_prompt))

            response_ids.extend(result["expert_response_ids"])
            response_mask.extend(result["expert_response_mask"])
            old_logps.extend(result["expert_old_logps"])

        expert_end = len(response_ids)

        synth_start = expert_end
        if result["handoff_occurred"] and result["synth_response_ids"]:
            synth_prompt = result["synth_prompt_ids"]
            response_ids.extend(synth_prompt)
            response_mask.extend([0] * len(synth_prompt))
            old_logps.extend([0.0] * len(synth_prompt))

            response_ids.extend(result["synth_response_ids"])
            response_mask.extend(result["synth_response_mask"])
            old_logps.extend(result["synth_old_logps"])

        synth_end = len(response_ids)

        ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + response_mask
        full_old_logps = [0.0] * max(len(prompt_ids) - 1, 0) + old_logps

        if len(ids) > args.max_total_len:
            overflow = len(ids) - args.max_total_len
            ids = ids[overflow:]
            mask = mask[overflow:]
            full_old_logps = full_old_logps[-(len(ids) - 1):]

        prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
        packed_samples.append((ids, mask, prompt_len, full_old_logps))

        segment_info.append({
            "router_range": (0, router_len),
            "expert_range": (expert_start, expert_end),
            "synth_range": (synth_start, synth_end),
        })

    # ===== Padding =====
    seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
    max_len = seq_lens.max().item()
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples],
        device=args.device,
    )
    full_response_masks = torch.tensor(
        [mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
        device=args.device, dtype=torch.float32,
    )
    old_per_token_logps = torch.tensor(
        [lps + [0.0] * ((max_len - 1) - len(lps)) for _, _, _, lps in packed_samples],
        device=args.device, dtype=torch.float32,
    )
    full_mask = (input_ids != tokenizer.pad_token_id).long()

    # ===== 策略 Loss（无 aux_loss）=====
    # 注意：训练时必须通过 DDP wrapper 做 forward，否则梯度不会跨进程同步
    with autocast_ctx:
        res = model(input_ids, attention_mask=full_mask)
        logits = res.logits[:, :-1, :]  # [B, L-1, V]
        # 分块计算 per_token_logps，节省显存（保持梯度图完整）
        target_ids = input_ids[:, 1:]  # [B, L-1]
        T = logits.size(1)
        _chunk = 128
        _logps_chunks = []
        for _s in range(0, T, _chunk):
            _e = min(_s + _chunk, T)
            _chunk_logits = logits[:, _s:_e, :]
            _lp = F.log_softmax(_chunk_logits, dim=-1)
            _gathered = torch.gather(
                _lp, 2, target_ids[:, _s:_e].unsqueeze(-1)
            ).squeeze(-1)
            _logps_chunks.append(_gathered)
        per_token_logps = torch.cat(_logps_chunks, dim=1)
        del logits, _logps_chunks

    with torch.no_grad():
        ref_per_token_logps = compute_per_token_logps_hf(
            ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask
        )

    # Completion mask
    completion_mask = full_response_masks[:, 1:]
    is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
    eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1,
                         device=args.device, dtype=torch.long)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
    completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
    token_counts = completion_mask.sum(dim=1)
    valid_rows = token_counts > 0

    # ===== Credit Assignment =====
    grouped_rewards = rewards.view(-1, args.num_generations)
    grouped_router = router_rewards.view(-1, args.num_generations)
    grouped_expert = expert_rewards.view(-1, args.num_generations)

    def normalize_group(grouped):
        mean = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        return (grouped.view(-1) - mean) / (std + 1e-4)

    total_advantages = normalize_group(grouped_rewards)
    router_advantages = normalize_group(grouped_router)
    expert_advantages = normalize_group(grouped_expert)

    B = completion_mask.size(0)
    T = completion_mask.size(1)
    per_token_advantages = torch.zeros(B, T, device=args.device)

    for i in range(B):
        per_token_advantages[i] = total_advantages[i]
        if results_batch[i]["handoff_occurred"]:
            per_token_advantages[i] = (router_advantages[i] + expert_advantages[i]) / 2

    # KL + 策略 loss
    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps - old_per_token_logps)

    if args.loss_type == "cispo":
        clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
        per_token_loss = -(clamped_ratio * per_token_advantages * per_token_logps - args.beta * per_token_kl)
    else:
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss1 = ratio * per_token_advantages
        per_token_loss2 = clipped_ratio * per_token_advantages
        per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

    policy_loss = (
        ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
        if valid_rows.any() else per_token_loss.sum() * 0.0
    )

    loss = policy_loss / args.accumulation_steps
    loss.backward()

    if step % args.accumulation_steps == 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    del per_token_logps, ref_per_token_logps, per_token_advantages

    return {
        "loss": loss.item() * args.accumulation_steps,
        "reward_mean": rewards.mean().item(),
        "router_reward_mean": router_rewards.mean().item(),
        "expert_reward_mean": expert_rewards.mean().item(),
        "handoff_rate": sum(1 for r in results_batch if r["handoff_occurred"]) / len(results_batch),
        "kl": ((kl_div) * completion_mask).sum().item() / max(token_counts.sum().item(), 1),
    }


# ==============================================================================
#  完整训练流程
# ==============================================================================

def run_handoff_training(args):
    """Qwen2.5-1.5B-Instruct Agent Handoff 训练（支持 DDP）"""

    # 初始化分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # CheckpointManager
    ckpt_mgr = QwenCheckpointManager(
        save_dir=args.save_dir,
        max_keep=args.max_keep,
        track_metric="reward",
        metric_mode="max",
    )

    # 断点续训
    ckp_data = ckpt_mgr.load(resume_mode=args.resume_mode) if args.from_resume else None

    # 模型初始化
    model_path = ckp_data["model_path"] if ckp_data else args.model_path
    model, tokenizer = init_model_qwen(
        model_path, device=args.device, dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Ref model（不需要梯度，不需要 gradient_checkpointing）
    ref_model, _ = init_model_qwen(
        args.model_path, device=args.device, dtype=dtype,
        gradient_checkpointing=False,
    )
    ref_model = ref_model.eval().requires_grad_(False)

    # wandb
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_id = ckp_data.get("wandb_id") if ckp_data else None
            resume = "must" if wandb_id else None
            wandb.init(
                project=args.wandb_project,
                name=f"Qwen1.5B-Handoff-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
                id=wandb_id, resume=resume,
            )
        except ImportError:
            Logger("Warning: swanlab not installed, wandb disabled")
            wandb = None

    # Rollout 引擎
    rollout_engine = TorchRolloutEngineHF(
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    # 数据集
    train_ds = HandoffDataset(args.data_path)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    def collate_fn(batch):
        return {
            "queries": [b["query"] for b in batch],
            "gt": [b["gt"] for b in batch],
            "needs_tool": [b["needs_tool"] for b in batch],
            "expert": [b["expert"] for b in batch],
        }

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps,
                                  eta_min=args.learning_rate / 10)

    # 恢复训练状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # DDP
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=True,  # gradient_checkpointing 需要
        )
    rollout_engine.update_policy(model)

    # ===== 训练循环 =====
    Logger("=" * 80)
    Logger("MiniMind-Agent Handoff Training (Qwen2.5-1.5B-Instruct)")
    Logger(f"  Model: {args.model_path}")
    Logger(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    Logger(f"  Handoff: {len(train_ds)} examples, num_generations={args.num_generations}")
    Logger(f"  Experts: math_agent, info_agent, translate_agent")
    Logger(f"  Loss: {args.loss_type}, beta={args.beta}")
    Logger(f"  DDP: {'Yes' if dist.is_initialized() else 'No'}"
           f"{f', world_size={dist.get_world_size()}' if dist.is_initialized() else ''}")
    Logger(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    Logger("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        # v2: 渐进式误触发惩罚系数，随 epoch 递增
        ft_penalty_coeff = min(0.5 + 0.1 * epoch, 1.5)
        Logger(f"  [Epoch {epoch+1}] false_trigger_penalty_coeff = {ft_penalty_coeff:.2f}")

        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)

        last_step = skip
        for step_offset, batch in enumerate(loader, start=skip + 1):
            global_step = step_offset
            last_step = global_step

            metrics = handoff_train_step(
                model=model,
                ref_model=ref_model,
                rollout_engine=rollout_engine,
                tokenizer=tokenizer,
                batch=batch,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
                autocast_ctx=autocast_ctx,
                step=global_step,
                reward_model=None,  # 可选：加载外部 reward model
                false_trigger_penalty_coeff=ft_penalty_coeff,  # v2: 渐进式惩罚
            )

            # 日志
            if global_step % args.log_interval == 0 and is_main_process():
                Logger(
                    f"[Epoch {epoch + 1}/{args.epochs}] Step {global_step}/{iters} | "
                    f"loss={metrics['loss']:.4f} | reward={metrics['reward_mean']:.3f} | "
                    f"router_r={metrics['router_reward_mean']:.3f} | "
                    f"expert_r={metrics['expert_reward_mean']:.3f} | "
                    f"handoff={metrics['handoff_rate']:.0%} | kl={metrics['kl']:.4f}"
                )
                if wandb:
                    wandb.log({
                        "reward": metrics["reward_mean"],
                        "router_reward": metrics["router_reward_mean"],
                        "expert_reward": metrics["expert_reward_mean"],
                        "handoff_rate": metrics["handoff_rate"],
                        "policy_loss": metrics["loss"],
                        "kl_ref": metrics["kl"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })

            # 保存 checkpoint
            if (global_step % args.save_interval == 0 or global_step == iters) and is_main_process():
                model.eval()
                ckpt_mgr.save(
                    model=model, tokenizer=tokenizer,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, step=global_step,
                    metric_value=metrics["reward_mean"],
                    extra_metrics={
                        "router_reward": metrics["router_reward_mean"],
                        "expert_reward": metrics["expert_reward_mean"],
                        "handoff_rate": metrics["handoff_rate"],
                    },
                    wandb=wandb,
                )
                model.train()
            # 所有进程等待 rank 0 完成 checkpoint 保存
            if (global_step % args.save_interval == 0 or global_step == iters) and dist.is_initialized():
                dist.barrier()

            # 同步 rollout 引擎
            if global_step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        # Epoch 末尾梯度累积处理
        if last_step > skip and last_step % args.accumulation_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    Logger("\nTraining complete.")
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
#  Demo
# ==============================================================================

def run_handoff_demo(args):
    """Handoff 推理 Demo"""
    setup_seed(42)

    model, tokenizer = init_model_qwen(
        args.model_path, device=args.device,
        gradient_checkpointing=False,
    )
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if "cuda" in args.device else nullcontext()
    rollout_engine = TorchRolloutEngineHF(
        policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
    )

    test_queries = [
        ("北京今天天气怎么样？", ["28°C", "晴"], True, "info"),
        ("帮我算一下 (15 + 27) * 3", ["126"], True, "math"),
        ("100美元能换多少人民币？", ["7.21", "721"], True, "info"),
        ("你觉得人工智能的未来会怎样？", [], False, "none"),
        ("帮我把'机器学习很有趣'翻译成英文", ["Machine learning is interesting"], True, "translate"),
        ("5公里等于多少英里？", ["3.1069", "3.107"], True, "math"),
    ]

    Logger("=" * 80)
    Logger("Agent Handoff Demo — Qwen2.5-1.5B-Instruct")
    Logger("  RouterAgent → [MathAgent | InfoAgent | TranslateAgent] → RouterAgent (Synthesize)")
    Logger("=" * 80)

    for query, gt, needs_tool, expected_expert in test_queries:
        Logger(f"\n{'─' * 60}")
        Logger(f"[User] {query}")
        Logger(f"[Expected] expert={expected_expert}, gt={gt}")
        Logger(f"{'─' * 60}")

        with torch.no_grad():
            result = handoff_rollout_single(
                rollout_engine, tokenizer, query,
                needs_tool=needs_tool,
                max_new_tokens=args.max_gen_len,
                thinking_ratio=0.0,
                device=args.device,
            )
            result["original_query"] = query
            result["expected_expert"] = expected_expert

        Logger(f"  Handoff: {'Yes' if result['handoff_occurred'] else 'No'}")
        if result["handoff_occurred"]:
            Logger(f"  Expert: {result['delegated_expert']}")
            Logger(f"  Delegate Task: {result['delegate_task']}")
        Logger(f"  # Outputs: {len(result['all_outputs'])}")
        for i, out in enumerate(result["all_outputs"]):
            if i == 0:
                role = "RouterAgent (Route)"
            elif i == len(result["all_outputs"]) - 1 and result["handoff_occurred"]:
                role = "RouterAgent (Synthesize)"
            else:
                role = f"ExpertAgent (turn {i})"
            display = out[:200] + "..." if len(out) > 200 else out
            Logger(f"  [{role}] {display}")

        rewards, r_rewards, e_rewards = calculate_handoff_rewards(
            [result], [gt], num_gen=1, device=args.device,
        )
        Logger(f"  Reward: total={rewards[0].item():.3f}, "
               f"router={r_rewards[0].item():.3f}, expert={e_rewards[0].item():.3f}")

    Logger(f"\n{'=' * 80}")
    Logger("Demo Complete.")


# ==============================================================================
#  Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Agent Handoff (Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "demo"])

    # 模型参数
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model id 或本地路径")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1],
                        help="是否开启梯度检查点（24GB 4090 必须开）")

    # 训练参数
    parser.add_argument("--save_dir", type=str, default="./checkpoints_qwen_handoff")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="per-GPU batch size（双卡 DDP 全局 batch=2）")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="学习率（1.5B 模型 RL 推荐 1e-6）")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # RL 参数
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO group 内候选数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数")
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--thinking_ratio", type=float, default=0.0,
                        help="thinking 模式概率（0=关闭）")

    # 序列长度
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=384,
                        help="每阶段最大生成 token 数")
    parser.add_argument("--max_total_len", type=int, default=4096,
                        help="三段拼接后的最大总长度")

    # 数据
    parser.add_argument("--data_path", type=str, default="./dataset/agent_handoff.jsonl")

    # 断点续训
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--resume_mode", type=str, default="latest",
                        help="恢复模式: latest/best/步数")

    # 工程
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_keep", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-Handoff-Qwen")

    args = parser.parse_args()
    args.gradient_checkpointing = bool(args.gradient_checkpointing)

    if args.mode == "train":
        run_handoff_training(args)
    else:
        run_handoff_demo(args)
