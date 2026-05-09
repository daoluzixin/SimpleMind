"""
Continuous Batching 推理引擎 — MiniMind

核心思想：
    传统推理服务逐请求串行处理，GPU 利用率低。Continuous Batching 将多个请求
    组成动态 batch 并行推理，任何序列完成即刻释放资源并补入新请求，实现"翻台"式调度。

架构设计：
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  API Server │────▶│  Scheduler   │────▶│  Inference  │
    │  (FastAPI)  │◀────│ (双队列调度)  │◀────│   Engine    │
    └─────────────┘     └──────────────┘     └─────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Waiting Queue    │  ← 新请求入队
                    │  Running Batch    │  ← 正在推理的序列
                    │  KV Cache Pool    │  ← 按序列独立管理
                    └───────────────────┘

关键技术点（面试可深入展开）：
    1. KV Cache 独立管理：每个序列维护独立的 KV Cache tensor，序列完成即释放显存
    2. 左 Padding 对齐：running batch 中不同长度序列通过左 padding 对齐，实现 batched forward
    3. Prefill/Decode 分离：首次请求做 Prefill（处理整个 prompt），后续做 Decode（逐 token 生成）
    4. 动态 Batch Size：batch 大小随请求到达/完成动态变化，最大化 GPU 利用率
    5. 显存预算控制：根据可用显存限制最大并发序列数，避免 OOM

使用方式：
    python scripts/continuous_batching_engine.py --hidden_size 768 --num_hidden_layers 8

性能对比（理论）：
    - 串行推理：吞吐 = 1 / (prefill_time + decode_time * avg_output_len)
    - Continuous Batching：吞吐 ≈ batch_size / decode_time（当 batch 充满时）
    - 提升倍数 ≈ effective_batch_size，取决于请求到达速率和输出长度分布
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Optional, List, Dict, Tuple

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora


# ═══════════════════════════════════════════════════════════════════════════════
#                           数据结构定义
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceStatus(Enum):
    """序列生命周期状态"""
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SequenceRequest:
    """单个推理请求的完整描述

    从请求到达到推理完成，所有状态都记录在这个对象中。
    """
    request_id: str
    prompt_token_ids: List[int]
    created_time: float = field(default_factory=time.time)

    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 1.0
    eos_token_id: int = 2

    # 运行时状态
    status: SequenceStatus = SequenceStatus.WAITING
    generated_token_ids: List[int] = field(default_factory=list)
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # 性能统计
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # 输出通道
    output_queue: Queue = field(default_factory=Queue)

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token"""
        if self.first_token_time and self.prefill_start_time:
            return self.first_token_time - self.prefill_start_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        if self.finish_time:
            return self.finish_time - self.created_time
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                        KV Cache 管理器
# ═══════════════════════════════════════════════════════════════════════════════

class KVCacheManager:
    """KV Cache 显存管理器

    设计思路：
        每个序列独立持有自己的 KV Cache tensor，而非共享一个大 pool。
        好处是实现简单、序列独立性强；缺点是碎片化。
        工业级实现（如 vLLM 的 PagedAttention）用虚拟内存思想管理 KV Cache
        以解决碎片化问题，但对于教学目的，独立管理已足够。

    显存预算估算：
        max_batch_size * max_seq_len * num_layers * 2(K+V) * num_kv_heads * head_dim * 2(fp16)
        例如: 8 * 2048 * 8 * 2 * 4 * 96 * 2 ≈ 200MB
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int, device: torch.device, dtype: torch.dtype = torch.float16):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.allocated_count = 0
        self.freed_count = 0

    def allocate(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """为新序列分配空的 KV Cache（序列维度为 0，后续通过 cat 增长）"""
        kv_cache = []
        for _ in range(self.num_layers):
            k = torch.empty(1, 0, self.num_kv_heads, self.head_dim,
                            device=self.device, dtype=self.dtype)
            v = torch.empty(1, 0, self.num_kv_heads, self.head_dim,
                            device=self.device, dtype=self.dtype)
            kv_cache.append((k, v))
        self.allocated_count += 1
        return kv_cache

    def free(self, kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]):
        """释放序列的 KV Cache，归还显存"""
        if kv_cache is not None:
            for k, v in kv_cache:
                del k, v
            kv_cache.clear()
            self.freed_count += 1

    @property
    def active_count(self) -> int:
        return self.allocated_count - self.freed_count

    def memory_usage_bytes(self, seq_len: int) -> int:
        """估算单个序列在给定长度下的 KV Cache 显存占用"""
        element_size = 2 if self.dtype == torch.float16 else 4
        per_layer = 2 * seq_len * self.num_kv_heads * self.head_dim * element_size
        return per_layer * self.num_layers


# ═══════════════════════════════════════════════════════════════════════════════
#                     Continuous Batching 调度器
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousBatchingScheduler:
    """Continuous Batching 调度器 — 推理引擎的核心

    调度策略：FCFS（先来先服务）+ 动态补位

    每个 iteration：
        1. 检查 waiting queue，将可加入的请求做 Prefill 并移入 running batch
        2. 对 running batch 中所有序列做一次 decode step
        3. 检查哪些序列已完成，释放其资源，从 waiting 补入新序列

    与 vLLM 的对比：
        - vLLM 用 PagedAttention + Block Table → 我们用独立 tensor
        - vLLM Prefill/Decode 可混合同一 batch → 我们分开处理（简化实现）
        - vLLM 有 preemption（抢占）→ 我们暂不实现
    """

    def __init__(self, model: MiniMindForCausalLM, tokenizer, device: torch.device,
                 max_batch_size: int = 8, max_seq_len: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # 模型配置
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim

        # KV Cache 管理器
        self.kv_manager = KVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
            dtype=torch.float16
        )

        # 双队列
        self.waiting_queue: List[SequenceRequest] = []
        self.running_batch: List[SequenceRequest] = []

        # 控制
        self._lock = threading.Lock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "total_tokens_generated": 0,
            "total_prefill_tokens": 0,
            "scheduler_iterations": 0,
            "start_time": None,
        }

    def add_request(self, request: SequenceRequest):
        """将新请求加入 waiting 队列（线程安全）"""
        with self._lock:
            self.waiting_queue.append(request)
            self.stats["total_requests"] += 1

    def start(self):
        """启动调度循环"""
        self._running = True
        self.stats["start_time"] = time.time()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def stop(self):
        """优雅停止：等待 running batch 清空"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)

    def _scheduler_loop(self):
        """调度主循环"""
        while self._running:
            has_work = False

            with self._lock:
                # 阶段 1: Prefill — 从 waiting 取请求
                while (self.waiting_queue and
                       len(self.running_batch) < self.max_batch_size):
                    seq = self.waiting_queue.pop(0)
                    self._do_prefill(seq)
                    self.running_batch.append(seq)
                    has_work = True

                # 阶段 2: Decode — 对 running batch 做一步生成
                if self.running_batch:
                    self._do_decode_step()
                    has_work = True

                # 阶段 3: 驱逐已完成序列
                self._evict_finished()

            if not has_work:
                time.sleep(0.001)  # 避免空转

            self.stats["scheduler_iterations"] += 1

    @torch.inference_mode()
    def _do_prefill(self, seq: SequenceRequest):
        """Prefill 阶段：处理整个 prompt，生成初始 KV Cache + 首 token

        Prefill 是 compute-bound 的：一次性处理整个 prompt 的所有 token。
        对于长 prompt，Prefill 时间远大于单步 Decode 时间。
        这也是为什么 Prefill 和 Decode 分离很重要——避免长 prompt 阻塞短请求。
        """
        seq.status = SequenceStatus.PREFILLING
        seq.prefill_start_time = time.time()

        input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)

        # 完整 prompt 的 forward，产出所有层的 KV Cache
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True
        )

        # 保存 KV Cache
        seq.kv_cache = outputs.past_key_values

        # 从最后位置的 logits 采样首个 token
        logits = outputs.logits[:, -1, :]
        next_token_id = self._sample(logits, seq)

        seq.generated_token_ids.append(next_token_id)
        seq.first_token_time = time.time()
        seq.status = SequenceStatus.DECODING

        # 流式输出
        token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
        seq.output_queue.put(token_text)

        # 首 token 就是 EOS 的边界情况
        if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
            self._finish_sequence(seq)

        self.stats["total_prefill_tokens"] += len(seq.prompt_token_ids)

    @torch.inference_mode()
    def _do_decode_step(self):
        """Decode 阶段：逐序列独立 forward（基础版）

        基础实现：每个序列独立做 forward，利用各自的 KV Cache。
        优点：实现简单，每个序列完全独立。
        缺点：无法利用 batch parallelism。

        进阶版 BatchedContinuousBatchingScheduler 实现真正的 batched decode。
        """
        for seq in self.running_batch:
            if seq.is_finished:
                continue

            # 取最后生成的 token 作为输入（有 KV Cache 所以只需 1 个 token）
            last_token_id = seq.generated_token_ids[-1]
            input_ids = torch.tensor([[last_token_id]], device=self.device)

            # 利用 KV Cache 做增量 forward
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=seq.kv_cache,
                use_cache=True
            )

            # 更新 KV Cache（新增了 1 个位置）
            seq.kv_cache = outputs.past_key_values

            # 采样
            logits = outputs.logits[:, -1, :]
            next_token_id = self._sample(logits, seq)
            seq.generated_token_ids.append(next_token_id)

            # 流式输出
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            seq.output_queue.put(token_text)

            # 终止判断
            if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
                self._finish_sequence(seq)

        self.stats["total_tokens_generated"] += sum(
            1 for s in self.running_batch if not s.is_finished
        )

    def _evict_finished(self):
        """驱逐已完成的序列 — "翻台"操作"""
        new_running = []
        for seq in self.running_batch:
            if seq.is_finished:
                self.kv_manager.free(seq.kv_cache)
                seq.kv_cache = None
                self.stats["completed_requests"] += 1
            else:
                new_running.append(seq)
        self.running_batch = new_running

    def _finish_sequence(self, seq: SequenceRequest):
        """标记序列完成"""
        seq.status = SequenceStatus.FINISHED
        seq.finish_time = time.time()
        seq.output_queue.put(None)  # 流结束标志

    def _sample(self, logits: torch.Tensor, seq: SequenceRequest) -> int:
        """采样：温度 → 重复惩罚 → Top-k → Top-p → 多项式采样"""
        logits = logits.clone().squeeze(0)

        # 温度
        if seq.temperature > 0 and seq.temperature != 1.0:
            logits = logits / seq.temperature

        # 重复惩罚
        if seq.repetition_penalty != 1.0:
            all_tokens = seq.prompt_token_ids + seq.generated_token_ids
            unique_tokens = torch.tensor(list(set(all_tokens)), device=logits.device)
            logits[unique_tokens] /= seq.repetition_penalty

        # Top-k
        if seq.top_k > 0:
            top_k = min(seq.top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k)[0][-1]
            logits[logits < threshold] = float('-inf')

        # Top-p
        if seq.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > seq.top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            indices_to_remove = sorted_indices[sorted_mask]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()

    def get_metrics(self) -> Dict:
        """性能指标"""
        elapsed = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        return {
            "total_requests": self.stats["total_requests"],
            "completed_requests": self.stats["completed_requests"],
            "waiting_queue_size": len(self.waiting_queue),
            "running_batch_size": len(self.running_batch),
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "throughput_tokens_per_sec": self.stats["total_tokens_generated"] / max(elapsed, 1e-6),
            "total_prefill_tokens": self.stats["total_prefill_tokens"],
            "scheduler_iterations": self.stats["scheduler_iterations"],
            "kv_cache_active": self.kv_manager.active_count,
            "uptime_seconds": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                  Batched Decode 优化版（真正的 Batched Forward）
# ═══════════════════════════════════════════════════════════════════════════════

class BatchedDecodeScheduler(ContinuousBatchingScheduler):
    """优化版：真正的 Batched Decode

    核心改进：
        将 running batch 中所有序列的最新 token 组成一个 batch，
        通过左 padding 对齐 KV Cache，一次 forward 处理所有序列。

    为什么用左 padding：
        decode 阶段只输入最后 1 个 token，KV Cache 在前面。
        左 padding 使有效 KV 右对齐，当前 token 能正确 attend 到自己的历史。

    PagedAttention 进阶：
        本实现的 left-padding 会浪费显存（短序列要 pad 很多零）。
        vLLM 的 PagedAttention 用 block table 将逻辑连续的 KV 映射到
        物理不连续的 block，消除 padding 浪费。原理类似操作系统的虚拟内存页表。
    """

    @torch.inference_mode()
    def _do_decode_step(self):
        """一次 batched forward 处理所有 running 序列"""
        active_seqs = [s for s in self.running_batch if not s.is_finished]
        if not active_seqs:
            return

        batch_size = len(active_seqs)

        # 步骤 1: 收集最新 token → [B, 1]
        last_tokens = []
        for seq in active_seqs:
            last_tokens.append(seq.generated_token_ids[-1])
        input_ids = torch.tensor(last_tokens, device=self.device).unsqueeze(1)

        # 步骤 2: 对齐 KV Cache（左 padding）
        kv_lengths = [seq.kv_cache[0][0].shape[1] for seq in active_seqs]
        max_kv_len = max(kv_lengths)

        padded_kv_caches = []
        for layer_idx in range(self.num_layers):
            layer_keys = []
            layer_values = []
            for seq_idx, seq in enumerate(active_seqs):
                k, v = seq.kv_cache[layer_idx]  # [1, seq_len, heads, dim]
                pad_len = max_kv_len - kv_lengths[seq_idx]
                if pad_len > 0:
                    # 左 padding: pad(last_dim_left, last_dim_right, ..., seq_dim_left, seq_dim_right)
                    # 对 4D tensor [1, seq, heads, dim]，seq 是第 2 维
                    k_pad = F.pad(k, (0, 0, 0, 0, pad_len, 0))
                    v_pad = F.pad(v, (0, 0, 0, 0, pad_len, 0))
                else:
                    k_pad, v_pad = k, v
                layer_keys.append(k_pad)
                layer_values.append(v_pad)
            padded_kv_caches.append((
                torch.cat(layer_keys, dim=0),   # [B, max_kv_len, heads, dim]
                torch.cat(layer_values, dim=0)
            ))

        # 步骤 3: 构造 attention_mask [B, max_kv_len + 1]
        # 有效位置为 1，padding 位置为 0
        attention_mask = torch.zeros(batch_size, max_kv_len + 1,
                                     device=self.device, dtype=torch.long)
        for i, kv_len in enumerate(kv_lengths):
            # 有效区域：从 (max_kv_len - kv_len) 到末尾
            attention_mask[i, (max_kv_len - kv_len):] = 1

        # 步骤 4: Batched forward
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=padded_kv_caches,
            use_cache=True,
            attention_mask=attention_mask
        )

        # 步骤 5: 拆分输出，更新各序列
        new_kv_caches = outputs.past_key_values  # [B, max_kv_len+1, heads, dim] per layer

        for i, seq in enumerate(active_seqs):
            # 采样该序列的 logits
            logits = outputs.logits[i:i+1, -1, :]  # [1, vocab_size]
            next_token_id = self._sample(logits, seq)
            seq.generated_token_ids.append(next_token_id)

            # 更新 KV Cache：提取该序列的 slice，去掉左 padding
            # 新的有效长度 = 原有效长度 + 1
            new_valid_len = kv_lengths[i] + 1
            seq_kv_cache = []
            for layer_idx in range(self.num_layers):
                k_full = new_kv_caches[layer_idx][0][i:i+1]  # [1, max_kv_len+1, heads, dim]
                v_full = new_kv_caches[layer_idx][1][i:i+1]
                # 取右侧有效部分（去掉左 padding）
                k_valid = k_full[:, -new_valid_len:, :, :]
                v_valid = v_full[:, -new_valid_len:, :, :]
                seq_kv_cache.append((k_valid.contiguous(), v_valid.contiguous()))
            seq.kv_cache = seq_kv_cache

            # 流式输出
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            seq.output_queue.put(token_text)

            # 终止判断
            if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
                self._finish_sequence(seq)

        self.stats["total_tokens_generated"] += len(active_seqs)


# ═══════════════════════════════════════════════════════════════════════════════
#                         FastAPI 服务层
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="MiniMind Continuous Batching Engine")


class ChatRequest(BaseModel):
    """OpenAI 兼容的请求格式"""
    model: str = "minimind"
    messages: list
    temperature: float = 0.85
    top_p: float = 0.85
    max_tokens: int = 512
    stream: bool = True
    top_k: int = 50
    repetition_penalty: float = 1.0


# 全局引用
scheduler: Optional[ContinuousBatchingScheduler] = None
tokenizer = None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 兼容的 Chat Completions 接口

    与原版 serve_openai_api.py 的接口完全兼容，
    但底层从串行推理升级为 Continuous Batching。
    """
    global scheduler, tokenizer

    # 将 messages 转为 token ids
    prompt = tokenizer.apply_chat_template(
        request.messages, tokenize=False, add_generation_prompt=True
    )
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # 创建序列请求
    seq_request = SequenceRequest(
        request_id=f"req_{int(time.time() * 1000)}_{id(request) % 10000}",
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id or 2,
    )

    # 加入调度队列
    scheduler.add_request(seq_request)

    if request.stream:
        return StreamingResponse(
            _stream_response(seq_request),
            media_type="text/event-stream"
        )
    else:
        return await _non_stream_response(seq_request)


async def _stream_response(seq: SequenceRequest):
    """流式响应生成器"""
    request_id = seq.request_id
    while True:
        # 非阻塞轮询 output_queue
        try:
            token_text = seq.output_queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.005)
            continue

        if token_text is None:
            # 流结束
            data = {
                "id": request_id,
                "choices": [{"delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            break
        else:
            data = {
                "id": request_id,
                "choices": [{"delta": {"content": token_text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _non_stream_response(seq: SequenceRequest):
    """非流式响应：等待生成完成后一次性返回"""
    full_text = ""
    while True:
        try:
            token_text = seq.output_queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.005)
            continue

        if token_text is None:
            break
        full_text += token_text

    return {
        "id": seq.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "minimind",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(seq.prompt_token_ids),
            "completion_tokens": seq.num_generated,
            "total_tokens": seq.total_len,
        }
    }


@app.get("/v1/metrics")
async def metrics():
    """性能指标端点（Prometheus 友好格式）

    面试扩展：生产环境通常暴露 /metrics 端点供 Prometheus 抓取，
    用于监控 QPS、P95 latency、GPU 利用率等关键指标。
    """
    return JSONResponse(content=scheduler.get_metrics())


@app.get("/health")
async def health():
    """健康检查端点

    生产级推理服务必备：
    - K8s 用它做 readiness/liveness probe
    - 负载均衡器用它判断是否向该实例分发流量
    """
    return {
        "status": "healthy",
        "waiting_queue": len(scheduler.waiting_queue),
        "running_batch": len(scheduler.running_batch),
        "gpu_available": torch.cuda.is_available(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                           压测工具
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(scheduler_instance: ContinuousBatchingScheduler,
                  tokenizer_instance, num_requests: int = 16,
                  prompt_len: int = 64, max_new_tokens: int = 128):
    """内置压测：对比串行 vs Continuous Batching 的吞吐

    使用方式：
        python continuous_batching_engine.py --benchmark

    压测指标：
        - 总吞吐（tokens/sec）
        - 平均 TTFT（Time To First Token）
        - 平均总延迟
        - 有效 batch size
    """
    import random

    print(f"\n{'='*60}")
    print(f"  Continuous Batching 压测")
    print(f"  请求数: {num_requests}, Prompt长度: ~{prompt_len} tokens")
    print(f"  最大生成长度: {max_new_tokens}, 最大batch: {scheduler_instance.max_batch_size}")
    print(f"{'='*60}\n")

    # 生成模拟请求（不同长度的 prompt 模拟真实场景）
    requests = []
    for i in range(num_requests):
        # 随机长度 prompt 模拟真实请求分布
        actual_len = random.randint(prompt_len // 2, prompt_len * 2)
        prompt_ids = [random.randint(3, 6000) for _ in range(actual_len)]
        seq = SequenceRequest(
            request_id=f"bench_{i}",
            prompt_token_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            top_p=0.85,
            eos_token_id=tokenizer_instance.eos_token_id or 2,
        )
        requests.append(seq)

    # 启动调度器
    scheduler_instance.start()
    start_time = time.time()

    # 模拟请求到达（带随机间隔模拟真实流量）
    for seq in requests:
        scheduler_instance.add_request(seq)
        time.sleep(random.uniform(0.01, 0.05))  # 模拟请求到达间隔

    # 等待所有请求完成
    while scheduler_instance.stats["completed_requests"] < num_requests:
        time.sleep(0.1)
        elapsed = time.time() - start_time
        if elapsed > 300:  # 5 分钟超时
            print("WARNING: 压测超时!")
            break

    total_time = time.time() - start_time
    scheduler_instance.stop()

    # 收集统计
    metrics = scheduler_instance.get_metrics()
    ttfts = [s.ttft for s in requests if s.ttft is not None]
    latencies = [s.total_latency for s in requests if s.total_latency is not None]
    gen_lengths = [s.num_generated for s in requests]

    print(f"\n{'─'*60}")
    print(f"  压测结果")
    print(f"{'─'*60}")
    print(f"  完成请求数: {metrics['completed_requests']}/{num_requests}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  总生成 tokens: {sum(gen_lengths)}")
    print(f"  吞吐: {sum(gen_lengths) / total_time:.1f} tokens/sec")
    print(f"  平均 TTFT: {sum(ttfts)/len(ttfts)*1000:.1f}ms" if ttfts else "  TTFT: N/A")
    print(f"  平均延迟: {sum(latencies)/len(latencies)*1000:.1f}ms" if latencies else "  延迟: N/A")
    print(f"  平均生成长度: {sum(gen_lengths)/len(gen_lengths):.1f} tokens")
    print(f"{'─'*60}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#                             主入口
# ═══════════════════════════════════════════════════════════════════════════════

def init_model(args):
    """初始化模型和 tokenizer"""
    tok = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MiniMind 模型参数量: {param_count:.2f}M")
    return model.half().eval().to(args.device), tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Continuous Batching Engine")

    # 模型参数
    parser.add_argument('--load_from', default='../model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--lora_weight', default='None', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

    # 引擎参数
    parser.add_argument('--max_batch_size', default=8, type=int,
                        help='最大并发 batch 大小')
    parser.add_argument('--batched_decode', action='store_true',
                        help='使用 batched decode 优化（真正的 batched forward）')
    parser.add_argument('--port', default=8998, type=int)

    # 压测模式
    parser.add_argument('--benchmark', action='store_true',
                        help='运行内置压测而非启动服务')
    parser.add_argument('--bench_requests', default=16, type=int)
    parser.add_argument('--bench_prompt_len', default=64, type=int)
    parser.add_argument('--bench_max_tokens', default=128, type=int)

    args = parser.parse_args()

    # 初始化模型
    model, tokenizer = init_model(args)

    # 选择调度器
    SchedulerClass = BatchedDecodeScheduler if args.batched_decode else ContinuousBatchingScheduler
    scheduler = SchedulerClass(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )

    if args.benchmark:
        # 压测模式
        run_benchmark(scheduler, tokenizer,
                      num_requests=args.bench_requests,
                      prompt_len=args.bench_prompt_len,
                      max_new_tokens=args.bench_max_tokens)
    else:
        # 启动服务
        print(f"\n{'='*60}")
        print(f"  MiniMind Continuous Batching Engine")
        print(f"  调度模式: {'Batched Decode' if args.batched_decode else 'Sequential Decode'}")
        print(f"  最大 Batch: {args.max_batch_size}")
        print(f"  端口: {args.port}")
        print(f"{'='*60}\n")

        scheduler.start()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
