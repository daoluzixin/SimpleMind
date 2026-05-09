"""
Rollout 引擎模块

本文件为 PPO/GRPO 等 RL 训练提供可插拔的推理（Rollout）引擎，
负责使用策略模型生成回复并计算每个 token 的对数概率。

支持两种引擎:
1. TorchRolloutEngine: 使用 PyTorch 原生模型推理（默认）
2. SGLangRolloutEngine: 使用 SGLang HTTP API 推理（高性能，需独立部署 SGLang 服务）

所有引擎统一返回 RolloutResult 数据类，包含:
- output_ids: 完整输出 token IDs（prompt + completion）
- completion_ids: 仅生成部分的 token IDs
- per_token_logps: 每个 token 的对数概率
- completions: 解码后的文本
- prompt_lens: 每个 prompt 的长度
- completion_mask: 生成部分的掩码

使用方式:
    engine = create_rollout_engine(engine_type="torch", policy_model=model, tokenizer=tokenizer, ...)
    result = engine.rollout(prompt_ids, attention_mask, num_generations=1, max_new_tokens=1024)
    engine.update_policy(model)  # 训练后同步最新策略权重
"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
# python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """计算生成部分每个 token 的对数概率

    用于 Rollout 阶段，获取策略模型对每个生成 token 的对数概率，
    这些概率在 PPO/GRPO 中作为 old_logps 使用（重要性采样的基准）。

    执行流程:
    1. 使用 logits_to_keep 参数仅计算需要的 logits（节省内存）
    2. 对 logits 做 log_softmax
    3. 用 gather 取出每个位置实际 token 的对数概率

    Args:
        model: 策略模型
        input_ids: 完整 token IDs（prompt + completion）
        n_keep: 需要保留的 token 数（即 completion 的长度）
        attention_mask: 注意力掩码

    Returns:
        per_token_logps: shape (batch_size, n_keep)，每个 token 的对数概率
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # logits_to_keep=n_keep+1: 只计算最后 n_keep+1 个位置的 logits，[:-1] 去掉最后一个
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        # 从 log_softmax 分布中 gather 出真实 token 的对数概率
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    """Rollout 结果数据类

    统一封装 Rollout 引擎的输出，供 PPO/GRPO 训练使用。

    Attributes:
        output_ids: 完整输出 token IDs, shape [B*num_gen, P+R]
        completion_ids: 生成部分 token IDs, shape [B*num_gen, R]
        per_token_logps: 每个 token 的对数概率, shape [B*num_gen, R]
        completions: 解码后的文本列表
        prompt_lens: 每个 prompt 的长度, shape [B*num_gen]
        completion_mask: 生成部分的掩码, shape [B*num_gen, R]
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    """Rollout 引擎抽象基类

    定义了所有 Rollout 引擎必须实现的接口:
    - rollout: 生成回复并计算对数概率
    - update_policy: 同步最新策略模型权重
    """
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """执行 Rollout：生成回复并计算对数概率"""
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """更新策略模型权重"""
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    """PyTorch 原生 Rollout 引擎

    直接使用模型对象的 generate 方法进行推理，
    适用于单卡或 DDP 训练场景，无需额外部署服务。

    Args:
        policy_model: 策略模型（可能被 DDP 或 torch.compile 包装）
        tokenizer: 分词器
        device: 运行设备
        autocast_ctx: 混合精度上下文
    """
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """使用 PyTorch 模型执行 Rollout

        执行流程:
        1. 解包 DDP/compile 包装获取原始模型
        2. 使用 model.generate 生成回复
        3. 使用 compute_per_token_logps 计算每个 token 的对数概率
        4. 解码生成的文本

        Args:
            prompt_ids: prompt token IDs, shape [B, P]
            attention_mask: 注意力掩码, shape [B, P]
            num_generations: 每个 prompt 生成的样本数
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            RolloutResult 数据类
        """
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # 使用 model.generate 自回归生成
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),  # 复制 prompt 以生成多个样本
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )  # [B*num_gen, P+R]
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()  # 非 padding 位置为 1
            # 计算每个 token 的对数概率
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    def update_policy(self, model: torch.nn.Module):
        """更新策略模型引用（训练后同步最新权重）"""
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    """SGLang HTTP API Rollout 引擎

    通过 HTTP API 调用独立部署的 SGLang 推理服务，
    利用 SGLang 的高性能推理能力（RadixAttention、连续批处理等）加速 Rollout。

    需要先启动 SGLang 服务:
    python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998

    Args:
        base_url: SGLang 服务 URL
        model_path: 模型路径（用于加载 tokenizer）
        shared_ckpt_path: 共享存储路径（用于更新权重）
        timeout: HTTP 请求超时时间
    """
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """通过 SGLang HTTP API 执行 Rollout

        执行流程:
        1. 去除 prompt 的左侧 padding，提取有效 token
        2. 构造 HTTP 请求 payload
        3. 解析响应中的 output_ids 和 logprobs
        4. 对齐并填充为张量格式

        Args:
            prompt_ids: prompt token IDs
            attention_mask: 注意力掩码
            num_generations: 每个 prompt 的生成数量
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            RolloutResult 数据类
        """
        # 去除左侧 padding tokens，只保留有效 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 按 num_generations 复制 prompt
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,  # 请求返回 logprob
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            # 解析 logprobs（SGLang 可能返回不同格式）
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            # 对齐 logprobs 和 completion_ids 的长度
            if len(logprobs) < len(completion_ids):
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            """将变长序列填充为固定长度的张量"""
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        """将最新策略模型权重同步到 SGLang 服务

        执行流程:
        1. 仅在 rank 0 上执行权重保存和更新
        2. 保存模型到共享存储路径（transformers 格式）
        3. 调用 SGLang 的 update_weights_from_disk API 加载新权重
        4. 在所有进程间同步更新结果

        Args:
            model: 最新策略模型

        Raises:
            RuntimeError: 权重更新失败时
        """
        ok = True
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
                # 通知 SGLang 服务从磁盘加载新权重
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            # 广播更新结果，确保所有进程状态一致
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        """清空 SGLang 服务的 KV Cache"""
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        """检查 SGLang 服务健康状态"""
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


# ===== 工厂函数 =====
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """Rollout 引擎工厂函数

    根据引擎类型创建对应的 Rollout 引擎实例。

    Args:
        engine_type: 引擎类型，"torch" 或 "sglang"
        policy_model: 策略模型（torch 引擎需要）
        tokenizer: 分词器
        device: 运行设备
        autocast_ctx: 混合精度上下文
        sglang_base_url: SGLang 服务 URL
        sglang_model_path: SGLang 模型路径
        sglang_shared_path: SGLang 共享存储路径

    Returns:
        RolloutEngine 实例

    Raises:
        ValueError: 不支持的引擎类型
    """
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
