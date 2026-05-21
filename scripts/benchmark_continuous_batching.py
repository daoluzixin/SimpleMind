"""
Continuous Batching 推理引擎压测 — 兼容 HuggingFace 标准模型

本脚本独立于 MiniMind 自定义模型，直接加载 HuggingFace 上的开源模型（如 Qwen2.5-0.5B-Instruct）
进行推理性能压测。对比三种推理模式：

    1. Serial（串行逐条）：传统方式，每次只处理一个请求
    2. Continuous Batching - Sequential Decode：动态调度，但 decode 阶段逐序列 forward
    3. Continuous Batching - Batched Decode：动态调度 + 真正的 batched forward（左 padding 对齐）

核心验证目标：
    Continuous Batching 在不改变生成质量的前提下，吞吐提升 3-5 倍。

HuggingFace 标准 KV Cache 格式适配：
    HuggingFace 模型的 past_key_values 格式为 tuple of tuple:
        past_key_values[layer_idx] = (key, value)
        key/value shape: [batch_size, num_heads, seq_len, head_dim]
    注意 seq_len 在第 3 维（dim=2），而非 MiniMind 的第 2 维（dim=1）。
    本脚本的 KV Cache 管理全部按此格式实现。

使用方式：
    # 默认使用 Qwen2.5-0.5B-Instruct，自动下载
    python scripts/benchmark_continuous_batching.py

    # 指定模型和设备
    python scripts/benchmark_continuous_batching.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps

    # 调整压测参数
    python scripts/benchmark_continuous_batching.py --num_requests 32 --max_new_tokens 64 --max_batch_size 4

依赖：
    pip install torch transformers
"""

import argparse
import time
import random
import threading
import statistics
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


# ═══════════════════════════════════════════════════════════════════════════════
#                           数据结构
# ═══════════════════════════════════════════════════════════════════════════════

class SequenceStatus(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class SequenceRequest:
    """单个推理请求"""
    request_id: str
    prompt_token_ids: List[int]
    created_time: float = field(default_factory=time.time)

    # 生成参数
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    eos_token_id: int = 2

    # 运行时状态
    status: SequenceStatus = SequenceStatus.WAITING
    generated_token_ids: List[int] = field(default_factory=list)
    kv_cache: Optional[object] = None  # HF DynamicCache or tuple of (key, value) per layer

    # 性能统计
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # 输出通道
    output_queue: Queue = field(default_factory=Queue)

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token (ms)"""
        if self.first_token_time and self.prefill_start_time:
            return (self.first_token_time - self.prefill_start_time) * 1000
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """总延迟 (ms)"""
        if self.finish_time and self.created_time:
            return (self.finish_time - self.created_time) * 1000
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                     采样函数（共用）
# ═══════════════════════════════════════════════════════════════════════════════

def sample_token(logits: torch.Tensor, seq: SequenceRequest) -> int:
    """Top-k + Top-p + Temperature 采样"""
    logits = logits.clone().squeeze(0).float()

    # 温度
    if seq.temperature > 0 and seq.temperature != 1.0:
        logits = logits / seq.temperature

    # 重复惩罚
    if seq.repetition_penalty != 1.0 and seq.generated_token_ids:
        unique_tokens = list(set(seq.generated_token_ids[-64:]))  # 只看最近 64 tokens
        token_ids = torch.tensor(unique_tokens, device=logits.device)
        logits[token_ids] /= seq.repetition_penalty

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


# ═══════════════════════════════════════════════════════════════════════════════
#                  模式 1: 串行推理 Baseline
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_serial_inference(model, tokenizer, requests: List[SequenceRequest], device):
    """串行逐条推理 — 最朴素的方式，作为 baseline

    每个请求完整处理完才处理下一个，无并发无 batching。
    这就是大多数初学者写的推理代码。
    """
    for seq in requests:
        seq.prefill_start_time = time.time()

        input_ids = torch.tensor([seq.prompt_token_ids], device=device)

        # Prefill: 整个 prompt 一次性 forward
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values

        # 首 token
        logits = outputs.logits[:, -1, :]
        next_token_id = sample_token(logits, seq)
        seq.generated_token_ids.append(next_token_id)
        seq.first_token_time = time.time()

        # Decode: 逐 token 生成
        for _ in range(seq.max_new_tokens - 1):
            if next_token_id == seq.eos_token_id:
                break

            next_input = torch.tensor([[next_token_id]], device=device)
            outputs = model(input_ids=next_input, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(logits, seq)
            seq.generated_token_ids.append(next_token_id)

        seq.status = SequenceStatus.FINISHED
        seq.finish_time = time.time()

        # 释放 KV Cache
        del past_kv


# ═══════════════════════════════════════════════════════════════════════════════
#             模式 2: Continuous Batching - Sequential Decode
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousBatchingScheduler:
    """Continuous Batching 调度器（Sequential Decode 版本）

    HuggingFace KV Cache 格式：
        past_key_values = tuple of (key_tensor, value_tensor) per layer
        key/value shape: [batch_size, num_kv_heads, seq_len, head_dim]

    每个序列独立持有自己的 KV Cache（batch_size=1），
    decode 阶段逐序列做 forward。
    """

    def __init__(self, model, tokenizer, device, max_batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size

        self.waiting_queue: List[SequenceRequest] = []
        self.running_batch: List[SequenceRequest] = []

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.completed_count = 0

    def add_request(self, request: SequenceRequest):
        with self._lock:
            self.waiting_queue.append(request)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)

    def _loop(self):
        while self._running:
            has_work = False

            with self._lock:
                # 补位：从 waiting 取新请求做 Prefill
                while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
                    seq = self.waiting_queue.pop(0)
                    self._prefill(seq)
                    self.running_batch.append(seq)
                    has_work = True

                # Decode
                if self.running_batch:
                    self._decode_step()
                    has_work = True

                # 驱逐完成序列
                self._evict()

            if not has_work:
                time.sleep(0.0005)

    @torch.inference_mode()
    def _prefill(self, seq: SequenceRequest):
        """Prefill: 一次性处理整个 prompt"""
        seq.status = SequenceStatus.PREFILLING
        seq.prefill_start_time = time.time()

        input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
        outputs = self.model(input_ids=input_ids, use_cache=True)

        # 保存 KV Cache (HF 格式)
        seq.kv_cache = outputs.past_key_values

        # 采样首 token
        logits = outputs.logits[:, -1, :]
        next_token_id = sample_token(logits, seq)
        seq.generated_token_ids.append(next_token_id)
        seq.first_token_time = time.time()
        seq.status = SequenceStatus.DECODING

        # 检查终止
        if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
            self._finish(seq)

    @torch.inference_mode()
    def _decode_step(self):
        """Decode: 逐序列 forward（利用各自独立的 KV Cache）"""
        for seq in self.running_batch:
            if seq.is_finished:
                continue

            last_token_id = seq.generated_token_ids[-1]
            input_ids = torch.tensor([[last_token_id]], device=self.device)

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=seq.kv_cache,
                use_cache=True
            )
            seq.kv_cache = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token_id = sample_token(logits, seq)
            seq.generated_token_ids.append(next_token_id)

            if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
                self._finish(seq)

    def _finish(self, seq: SequenceRequest):
        seq.status = SequenceStatus.FINISHED
        seq.finish_time = time.time()
        seq.output_queue.put(None)

    def _evict(self):
        new_running = []
        for seq in self.running_batch:
            if seq.is_finished:
                seq.kv_cache = None  # 释放
                self.completed_count += 1
            else:
                new_running.append(seq)
        self.running_batch = new_running


# ═══════════════════════════════════════════════════════════════════════════════
#             模式 3: Continuous Batching - Batched Decode
# ═══════════════════════════════════════════════════════════════════════════════

class BatchedContinuousBatchingScheduler(ContinuousBatchingScheduler):
    """优化版：Batched Decode

    核心改进：
        将 running batch 中所有序列的最新 token 拼成一个 batch tensor，
        通过左 padding 对齐 KV Cache，一次 model forward 处理所有序列。

    HuggingFace KV Cache 对齐：
        标准格式 key shape: [1, num_heads, seq_len, head_dim]
        左 padding 需要在 dim=2 (seq_len 维度) 前面补零。

    为什么 Batched Decode 更快：
        GPU 的并行计算单元（CUDA cores / Tensor cores）在 batch_size > 1 时利用率更高。
        逐序列 forward 每次只用 batch=1 的计算量，无法填满 GPU 的计算流水线。
        Batched forward 将多个序列合并为一次矩阵乘法，吞吐线性提升。
    """

    @torch.inference_mode()
    def _decode_step(self):
        """Batched Decode: 一次 forward 处理所有 running 序列"""
        active_seqs = [s for s in self.running_batch if not s.is_finished]
        if not active_seqs:
            return

        batch_size = len(active_seqs)

        # 步骤 1: 收集最新 token → [B, 1]
        last_tokens = [seq.generated_token_ids[-1] for seq in active_seqs]
        input_ids = torch.tensor(last_tokens, device=self.device).unsqueeze(1)

        # 步骤 2: 对齐 KV Cache（左 padding）
        # DynamicCache (transformers 5.x): cache.layers[layer].keys shape: [1, num_heads, seq_len, head_dim]
        num_layers = len(active_seqs[0].kv_cache)
        kv_lengths = [seq.kv_cache.layers[0].keys.shape[2] for seq in active_seqs]  # dim=2 是 seq_len
        max_kv_len = max(kv_lengths)

        # 构造 batched DynamicCache
        batched_cache = DynamicCache()
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for seq_idx, seq in enumerate(active_seqs):
                k = seq.kv_cache.layers[layer_idx].keys    # [1, heads, seq_len, dim]
                v = seq.kv_cache.layers[layer_idx].values
                pad_len = max_kv_len - kv_lengths[seq_idx]
                if pad_len > 0:
                    # 左 padding on dim=2 (seq_len)
                    k = F.pad(k, (0, 0, pad_len, 0))
                    v = F.pad(v, (0, 0, pad_len, 0))
                layer_keys.append(k)
                layer_values.append(v)
            batched_cache.update(
                torch.cat(layer_keys, dim=0),   # [B, heads, max_kv_len, dim]
                torch.cat(layer_values, dim=0),
                layer_idx
            )

        # 步骤 3: 构造 attention_mask [B, max_kv_len + 1]
        attention_mask = torch.zeros(batch_size, max_kv_len + 1,
                                     device=self.device, dtype=torch.long)
        for i, kv_len in enumerate(kv_lengths):
            attention_mask[i, (max_kv_len - kv_len):] = 1

        # 步骤 4: Batched forward
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=batched_cache,
            attention_mask=attention_mask,
            use_cache=True
        )

        # 步骤 5: 拆分结果，更新各序列的 KV Cache
        new_kv = outputs.past_key_values
        num_layers_out = len(new_kv)

        for i, seq in enumerate(active_seqs):
            logits = outputs.logits[i:i+1, -1, :]
            next_token_id = sample_token(logits, seq)
            seq.generated_token_ids.append(next_token_id)

            # 提取该序列的 KV Cache slice，去掉左 padding
            new_valid_len = kv_lengths[i] + 1
            seq_cache = DynamicCache()
            for layer_idx in range(num_layers_out):
                k_full = new_kv.layers[layer_idx].keys[i:i+1]   # [1, heads, max_kv_len+1, dim]
                v_full = new_kv.layers[layer_idx].values[i:i+1]
                # 取右侧有效部分
                k_valid = k_full[:, :, -new_valid_len:, :].contiguous()
                v_valid = v_full[:, :, -new_valid_len:, :].contiguous()
                seq_cache.update(k_valid, v_valid, layer_idx)
            seq.kv_cache = seq_cache

            if next_token_id == seq.eos_token_id or seq.num_generated >= seq.max_new_tokens:
                self._finish(seq)


# ═══════════════════════════════════════════════════════════════════════════════
#                         压测主逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_requests(tokenizer, num_requests: int, prompt_len: int,
                           max_new_tokens: int, eos_token_id: int) -> List[dict]:
    """生成压测用的请求数据（固定 seed 保证可复现）

    使用真实的 chat template 构造 prompt，让 tokenizer 行为与线上一致。
    """
    random.seed(42)

    # 用一批固定的中文问题作为 prompt（模拟真实聊天场景）
    sample_prompts = [
        "请解释什么是机器学习中的过拟合问题？",
        "如何用Python实现快速排序算法？",
        "什么是Transformer架构的自注意力机制？",
        "请介绍一下强化学习的基本概念。",
        "如何优化深度学习模型的训练速度？",
        "什么是梯度消失问题？如何解决？",
        "请解释卷积神经网络的工作原理。",
        "什么是大语言模型的涌现能力？",
        "如何实现一个简单的推荐系统？",
        "请介绍分布式训练中的数据并行。",
        "什么是知识蒸馏技术？",
        "请解释注意力机制中的KV Cache。",
        "如何评估一个语言模型的性能？",
        "什么是RLHF训练方法？",
        "请介绍模型量化的几种方法。",
        "什么是Chain-of-Thought推理？",
    ]

    requests_data = []
    for i in range(num_requests):
        prompt_text = sample_prompts[i % len(sample_prompts)]

        # 使用 chat template 构造标准格式
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        except Exception:
            # Fallback: 直接 encode
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        # 截断或补齐到目标长度附近（模拟长度异构）
        target_len = random.randint(max(8, prompt_len // 2), prompt_len * 2)
        if len(prompt_ids) > target_len:
            prompt_ids = prompt_ids[:target_len]
        # 如果太短就重复填充（保证有一定长度）
        while len(prompt_ids) < max(8, prompt_len // 2):
            prompt_ids = prompt_ids + prompt_ids[:8]
            prompt_ids = prompt_ids[:target_len]

        requests_data.append({
            "request_id": f"bench_{i:03d}",
            "prompt_token_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_id,
        })

    return requests_data


def create_requests(requests_data: List[dict]) -> List[SequenceRequest]:
    """从数据创建 SequenceRequest 对象（每次压测前需要重新创建）"""
    requests = []
    for data in requests_data:
        seq = SequenceRequest(
            request_id=data["request_id"],
            prompt_token_ids=data["prompt_token_ids"][:],
            max_new_tokens=data["max_new_tokens"],
            eos_token_id=data["eos_token_id"],
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        requests.append(seq)
    return requests


def compute_percentile(values: List[float], p: int) -> float:
    """计算分位数"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def print_results(name: str, requests: List[SequenceRequest], total_time: float):
    """打印单次压测结果"""
    ttfts = [s.ttft for s in requests if s.ttft is not None]
    latencies = [s.total_latency for s in requests if s.total_latency is not None]
    gen_lengths = [s.num_generated for s in requests]
    total_tokens = sum(gen_lengths)
    completed = sum(1 for s in requests if s.is_finished)

    print(f"\n  {'─' * 56}")
    print(f"  [{name}]")
    print(f"  {'─' * 56}")
    print(f"  完成请求: {completed}/{len(requests)}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  总生成 tokens: {total_tokens}")
    print(f"  吞吐 (throughput): {total_tokens / max(total_time, 0.001):.1f} tokens/sec")
    print(f"  每请求平均生成: {total_tokens / max(completed, 1):.1f} tokens")

    if ttfts:
        print(f"  TTFT  — avg: {statistics.mean(ttfts):.1f}ms, "
              f"P50: {compute_percentile(ttfts, 50):.1f}ms, "
              f"P95: {compute_percentile(ttfts, 95):.1f}ms")

    if latencies:
        print(f"  延迟  — avg: {statistics.mean(latencies):.0f}ms, "
              f"P50: {compute_percentile(latencies, 50):.0f}ms, "
              f"P95: {compute_percentile(latencies, 95):.0f}ms, "
              f"P99: {compute_percentile(latencies, 99):.0f}ms")

    return {
        "name": name,
        "completed": completed,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "throughput": total_tokens / max(total_time, 0.001),
        "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0,
        "p95_latency_ms": compute_percentile(latencies, 95) if latencies else 0,
    }


def run_continuous_batching_benchmark(scheduler_class, model, tokenizer, device,
                                       requests_data, max_batch_size, arrival_interval):
    """运行 Continuous Batching 模式压测"""
    requests = create_requests(requests_data)

    sched = scheduler_class(
        model=model, tokenizer=tokenizer,
        device=device, max_batch_size=max_batch_size
    )
    sched.start()

    start_time = time.time()

    # 模拟请求陆续到达
    for seq in requests:
        seq.created_time = time.time()
        sched.add_request(seq)
        time.sleep(arrival_interval)

    # 等待全部完成
    total_requests = len(requests)
    while sched.completed_count < total_requests:
        time.sleep(0.05)
        if time.time() - start_time > 600:
            print("  WARNING: 压测超时 (>600s)，中止")
            break

    total_time = time.time() - start_time
    sched.stop()

    return requests, total_time


# ═══════════════════════════════════════════════════════════════════════════════
#                             主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Continuous Batching 推理引擎压测 (HuggingFace 兼容)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本压测（自动下载 Qwen2.5-0.5B）
  python scripts/benchmark_continuous_batching.py

  # 用 MPS 加速（Apple Silicon）
  python scripts/benchmark_continuous_batching.py --device mps

  # 更大压力
  python scripts/benchmark_continuous_batching.py --num_requests 32 --max_batch_size 8

  # 只跑 serial baseline
  python scripts/benchmark_continuous_batching.py --mode serial
        """
    )

    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct', type=str,
                        help='HuggingFace 模型名或本地路径')
    parser.add_argument('--device', default=None, type=str,
                        help='推理设备 (cuda/mps/cpu)，默认自动检测')
    parser.add_argument('--dtype', default='float16', choices=['float16', 'float32', 'bfloat16'],
                        help='模型精度')

    parser.add_argument('--num_requests', default=16, type=int,
                        help='压测请求数')
    parser.add_argument('--prompt_len', default=64, type=int,
                        help='平均 prompt 长度 (tokens)')
    parser.add_argument('--max_new_tokens', default=64, type=int,
                        help='每个请求最大生成 token 数')
    parser.add_argument('--max_batch_size', default=4, type=int,
                        help='Continuous Batching 最大并发 batch 大小')
    parser.add_argument('--arrival_interval', default=0.02, type=float,
                        help='请求到达间隔 (秒)，模拟真实流量')

    parser.add_argument('--mode', default='all', choices=['all', 'serial', 'sequential', 'batched'],
                        help='运行模式: all=三种都跑, serial/sequential/batched=只跑一种')

    args = parser.parse_args()

    # 自动检测设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # dtype 映射
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # MPS 不支持 float16 的某些操作，自动降级到 float32
    if args.device == 'mps' and dtype == torch.float16:
        print("  注意: MPS 设备不完全支持 float16，切换到 float32")
        dtype = torch.float32

    print(f"\n{'═' * 60}")
    print(f"  Continuous Batching 推理引擎压测")
    print(f"{'═' * 60}")
    print(f"  模型: {args.model}")
    print(f"  设备: {args.device}")
    print(f"  精度: {dtype}")
    print(f"  请求数: {args.num_requests}")
    print(f"  Prompt长度: ~{args.prompt_len} tokens")
    print(f"  最大生成: {args.max_new_tokens} tokens/request")
    print(f"  最大Batch: {args.max_batch_size}")
    print(f"  到达间隔: {args.arrival_interval}s")
    print(f"{'═' * 60}")

    # 加载模型
    print(f"\n  正在加载模型 {args.model} ...")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval().to(args.device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    load_time = time.time() - load_start
    print(f"  加载完成: {param_count:.0f}M 参数, 耗时 {load_time:.1f}s")

    # 获取 eos_token_id
    eos_token_id = tokenizer.eos_token_id or 2

    # 生成测试请求数据（固定 seed，三种模式用同样的数据）
    print(f"  生成测试请求数据...")
    requests_data = generate_test_requests(
        tokenizer, args.num_requests, args.prompt_len,
        args.max_new_tokens, eos_token_id
    )
    prompt_lengths = [len(d["prompt_token_ids"]) for d in requests_data]
    print(f"  Prompt 长度分布: min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
          f"avg={statistics.mean(prompt_lengths):.0f}")

    # Warmup: 跑一个短请求预热模型和 CUDA/MPS
    print(f"  Warmup...")
    with torch.inference_mode():
        warmup_ids = torch.tensor([requests_data[0]["prompt_token_ids"][:16]], device=args.device)
        _ = model(warmup_ids, use_cache=True)
        if args.device == 'cuda':
            torch.cuda.synchronize()

    all_results = []

    # ─── 模式 1: 串行推理 ───
    if args.mode in ('all', 'serial'):
        print(f"\n  ▶ 运行串行推理 (Serial Baseline)...")
        serial_requests = create_requests(requests_data)
        # 为串行模式设置 created_time
        for seq in serial_requests:
            seq.created_time = time.time()

        start = time.time()
        run_serial_inference(model, tokenizer, serial_requests, args.device)
        serial_time = time.time() - start

        result = print_results("Serial (逐条串行)", serial_requests, serial_time)
        all_results.append(result)

        # 清理
        del serial_requests
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # ─── 模式 2: Continuous Batching - Sequential Decode ───
    if args.mode in ('all', 'sequential'):
        print(f"\n  ▶ 运行 Continuous Batching (Sequential Decode)...")
        cb_requests, cb_time = run_continuous_batching_benchmark(
            ContinuousBatchingScheduler, model, tokenizer, args.device,
            requests_data, args.max_batch_size, args.arrival_interval
        )
        result = print_results("Continuous Batching (Sequential Decode)", cb_requests, cb_time)
        all_results.append(result)

        del cb_requests
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # ─── 模式 3: Continuous Batching - Batched Decode ───
    if args.mode in ('all', 'batched'):
        print(f"\n  ▶ 运行 Continuous Batching (Batched Decode)...")
        bd_requests, bd_time = run_continuous_batching_benchmark(
            BatchedContinuousBatchingScheduler, model, tokenizer, args.device,
            requests_data, args.max_batch_size, args.arrival_interval
        )
        result = print_results("Continuous Batching (Batched Decode)", bd_requests, bd_time)
        all_results.append(result)

        del bd_requests
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # ─── 汇总对比 ───
    if len(all_results) > 1:
        print(f"\n{'═' * 60}")
        print(f"  汇总对比")
        print(f"{'═' * 60}")
        print(f"  {'模式':<40} {'吞吐(tok/s)':>12} {'加速比':>8} {'P95延迟':>10}")
        print(f"  {'─' * 72}")

        baseline_throughput = all_results[0]["throughput"] if all_results else 1
        for r in all_results:
            speedup = r["throughput"] / max(baseline_throughput, 0.001)
            print(f"  {r['name']:<40} {r['throughput']:>10.1f} {speedup:>7.2f}x "
                  f"{r['p95_latency_ms']:>8.0f}ms")

        print(f"  {'─' * 72}")
        print(f"\n  结论: Continuous Batching 相对串行推理的吞吐提升 = "
              f"{all_results[-1]['throughput'] / max(baseline_throughput, 0.001):.2f}x")

    print(f"\n{'═' * 60}")
    print(f"  压测完成")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
