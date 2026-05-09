"""
MiniMind 模型核心定义文件

本文件实现了 MiniMind 语言模型的完整架构，包括：
- MiniMindConfig: 模型超参数配置类
- RMSNorm: 均方根归一化层（替代 LayerNorm，计算效率更高）
- precompute_freqs_cis: 预计算 RoPE 位置编码的 cos/sin 频率表（支持 YaRN 外推）
- apply_rotary_pos_emb: 将旋转位置编码应用到 Q/K 向量
- repeat_kv: GQA 中将 KV 头复制扩展以匹配 Q 头数量
- Attention: 多头注意力层（支持 GQA、Flash Attention、KV Cache）
- FeedForward: 前馈网络层（SwiGLU 激活）
- MOEFeedForward: 混合专家前馈层（含门控路由和辅助负载均衡损失）
- MiniMindBlock: 单层 Transformer Block（Pre-Norm 架构）
- MiniMindModel: 模型主体（Embedding + N 层 Block + 最终 RMSNorm）
- MiniMindForCausalLM: 因果语言模型（含 LM Head、损失计算、generate 推理）
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """MiniMind 模型配置类，继承自 HuggingFace PretrainedConfig

    管理模型的所有超参数，包括：
    - 基础维度参数（hidden_size, num_layers, vocab_size 等）
    - 注意力参数（num_attention_heads, num_key_value_heads, flash_attn 等）
    - RoPE 位置编码参数（rope_theta, rope_scaling, YaRN 配置等）
    - MoE 专家参数（num_experts, num_experts_per_tok 等）
    """
    model_type = "minimind"  # 模型类型标识，用于 HuggingFace AutoModel 自动注册
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # ===== 基础模型参数 =====
        self.hidden_size = hidden_size          # 隐藏层维度，控制模型表达能力
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数
        self.use_moe = use_moe                  # 是否使用混合专家（MoE）架构
        self.dropout = kwargs.get("dropout", 0.0)   # Dropout 概率
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句首 token ID
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 句尾 token ID

        # ===== 注意力参数 =====
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否使用 Flash Attention 加速
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # Q 的头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # KV 的头数（GQA：KV 头数 < Q 头数）
        # 每个注意力头的维度，默认为 hidden_size / num_attention_heads
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 前馈网络激活函数类型

        # 前馈网络中间层维度，使用 π 的倍数向上取整到 64 的整数倍（常见优化技巧）
        # 例如 hidden_size=768 → intermediate_size ≈ 768 * π / 64 ≈ 37.7 → 38*64 = 2432
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # ===== RoPE 位置编码参数 =====
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大支持的位置编码长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 的 epsilon，防止除零
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 基础频率，越大高频衰减越慢
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 是否共享 embedding 和 lm_head 权重

        # ===== YaRN 位置编码外推配置 =====
        # inference_rope_scaling: 推理时是否启用 RoPE 缩放（YaRN 方法），可将 2048 长度外推到 32768
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,        # 高频截止参数，控制哪些频率维度被视为"高频"
            "beta_slow": 1,         # 低频截止参数，控制哪些频率维度被视为"低频"
            "factor": 16,           # 外推缩放因子，16 表示长度扩展 16 倍
            "original_max_position_embeddings": 2048,  # 原始训练时的最大位置长度
            "attention_factor": 1.0,  # 注意力分数缩放因子
            "type": "yarn"          # 外推方法类型
        } if self.inference_rope_scaling else None

        # ===== MoE（混合专家）专用配置（use_moe=False 时忽略） =====
        self.num_experts = kwargs.get("num_experts", 4)            # 专家总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 激活的专家数
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # MoE 专家的中间层维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)   # 是否对 top-k 专家权重归一化
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡辅助损失系数

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """均方根归一化层（Root Mean Square Normalization）

    与 LayerNorm 相比，RMSNorm 不需要计算均值，仅除以均方根，计算更高效。
    公式: output = weight * (x / sqrt(mean(x^2) + eps))

    Args:
        dim: 归一化的维度大小
        eps: 防止除零的小常数
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def norm(self, x):
        """计算 RMS 归一化: x * rsqrt(mean(x^2) + eps)"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播：先转 float32 计算归一化（数值稳定），再乘以 weight，最后转回原数据类型"""
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE（旋转位置编码）的 cos 和 sin 频率表

    RoPE 通过将位置信息编码为旋转矩阵，使注意力分数天然包含相对位置信息。
    该函数支持 YaRN（Yet another RoPE extensioN method）外推方法，
    可将短序列训练的模型外推到更长的序列。

    核心公式:
        freqs[i] = 1 / (rope_base^(2i/d))   # 每个维度对应的基础频率
        cos_table[pos, i] = cos(pos * freqs[i])  # 位置 pos 在维度 i 的 cos 值
        sin_table[pos, i] = sin(pos * freqs[i])  # 位置 pos 在维度 i 的 sin 值

    YaRN 外推原理:
        对不同频率维度应用不同的缩放因子：
        - 高频维度：保持原频率（不缩放），保留精确的局部位置信息
        - 低频维度：除以 factor（缩放），使模型能感知更远的相对距离
        - 中间维度：线性过渡（ramp 函数）

    Args:
        dim: 每个注意力头的维度（head_dim）
        end: 需要预计算的最大序列长度
        rope_base: RoPE 的基础频率 θ
        rope_scaling: YaRN 缩放配置字典，None 表示不使用外推

    Returns:
        freqs_cos: shape [end, dim]，cos 值表
        freqs_sin: shape [end, dim]，sin 值表
    """
    # 计算每个维度对应的基础频率: freqs[i] = 1 / (θ^(2i/d))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 解析 YaRN 配置参数
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)  # 原始训练时的最大长度
        factor = rope_scaling.get("factor", 16)          # 缩放因子
        beta_fast = rope_scaling.get("beta_fast", 32.0)  # 高频截止波长参数
        beta_slow = rope_scaling.get("beta_slow", 1.0)   # 低频截止波长参数
        attn_factor = rope_scaling.get("attention_factor", 1.0)  # 注意力分数缩放因子

        # 仅在当前序列长度超过原始最大长度时才进行外推缩放
        if end / orig_max > 1.0:
            # inv_dim(b): 计算波长为 b 的维度索引
            # 波长 λ = 2π * θ^(2i/d)，因此 i = d * ln(λ/(2π)) / (2*ln(θ))
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # low: 高频区域的边界索引（高于此索引的维度被视为高频，不缩放）
            # high: 低频区域的边界索引（低于此索引的维度被视为低频，完全缩放）
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp: 线性过渡函数，从 0（高频不缩放）到 1（低频完全缩放）
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 应用 YaRN 缩放: freqs' = freqs * ((1-ramp) + ramp/factor)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 计算外积: freqs_table[pos, i] = pos * freqs[i]
    freqs = torch.outer(t, freqs).float()
    # 将 cos/sin 值复制拼接为 dim 维度（原始 freqs 只有 dim//2 个频率，需复制以匹配 head_dim）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """将旋转位置编码（RoPE）应用到 Query 和 Key 向量

    RoPE 的核心思想：将 Q/K 向量看作二维平面上的复数，通过旋转操作注入位置信息。
    旋转后，Q·K^T 的点积天然包含相对位置信息。

    数学原理:
        rotate_half(x) 将向量的后半部分取反后与前半部分拼接，实现 90° 旋转
        最终: q_embed = q * cos + rotate_half(q) * sin

    Args:
        q: Query 张量, shape [batch, num_heads, seq_len, head_dim]
        k: Key 张量, shape [batch, num_kv_heads, seq_len, head_dim]
        cos: cos 值, shape [seq_len, head_dim]
        sin: sin 值, shape [seq_len, head_dim]
        unsqueeze_dim: 广播维度，通常为 1（num_heads 维度）

    Returns:
        q_embed: 添加位置编码后的 Query
        k_embed: 添加位置编码后的 Key
    """
    def rotate_half(x):
        """将向量后半部分取反后与前半部分交换拼接，实现复数旋转的等价操作

        例如: [x1, x2, x3, x4] → [-x3, -x4, x1, x2]
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # q_embed = q * cos + rotate_half(q) * sin（旋转公式）
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """在 GQA（Grouped Query Attention）中，将 KV 头复制扩展以匹配 Q 头数量

    GQA 通过减少 KV 头数来降低 KV Cache 内存占用，但计算注意力时需要将 KV 头
    重复扩展到与 Q 头相同的数量。

    例如: num_attention_heads=8, num_key_value_heads=4 → n_rep=2
    将 4 个 KV 头每个复制 2 次，得到 8 个 KV 头

    Args:
        x: 输入张量, shape [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 每个 KV 头需要复制的次数（= num_attention_heads / num_key_value_heads）

    Returns:
        扩展后的张量, shape [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x  # MHA 情况下无需复制
    # 在第 4 维扩展 n_rep 倍，然后 reshape 合并
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    """多头自注意力层，支持 GQA（分组查询注意力）、Flash Attention 和 KV Cache

    架构说明:
        - Q/K/V 分别通过线性映射得到
        - 支持 Q 头数 > KV 头数的 GQA 模式（通过 repeat_kv 扩展 KV）
        - Q 和 K 在计算注意力前会经过 RMSNorm（QK-Norm，提升训练稳定性）
        - 优先使用 Flash Attention 加速，不支持时退回手动实现
        - KV Cache: 推理时缓存历史 KV，避免重复计算

    Args:
        config: MiniMindConfig 配置对象
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # ===== GQA 头数配置 =====
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads          # Q 的头数
        self.n_local_kv_heads = self.num_key_value_heads         # KV 的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个 KV 头需复制给多少个 Q 头
        self.head_dim = config.head_dim                           # 每个头的维度
        self.is_causal = True                                     # 因果注意力（下三角 mask）

        # ===== Q/K/V 投影层 =====
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # ===== QK-Norm: 对 Q 和 K 做归一化，防止注意力分数爆炸 =====
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ===== Dropout =====
        self.attn_dropout = nn.Dropout(config.dropout)  # 注意力权重 Dropout
        self.resid_dropout = nn.Dropout(config.dropout) # 残差连接 Dropout
        self.dropout = config.dropout

        # ===== Flash Attention 可用性检测 =====
        # 需要 PyTorch 支持 scaled_dot_product_attention 且配置开启
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """注意力层前向传播

        执行流程:
        1. 线性映射得到 Q/K/V
        2. reshape 为多头格式 [batch, seq, heads, head_dim]
        3. 对 Q/K 应用 QK-Norm
        4. 应用 RoPE 旋转位置编码
        5. 拼接 KV Cache（如有）
        6. 扩展 KV 头（GQA）
        7. 计算注意力（Flash Attention 或手动实现）
        8. 输出投影 + 残差 Dropout

        Args:
            x: 输入隐藏状态, shape [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) 位置编码元组
            past_key_value: 历史 KV Cache，shape (key, value)
            use_cache: 是否返回当前 KV 供后续使用
            attention_mask: 注意力掩码，shape [batch, seq_len]

        Returns:
            output: 注意力输出, shape [batch, seq_len, hidden_size]
            past_kv: 当前步的 KV 缓存（use_cache=True 时）
        """
        bsz, seq_len, _ = x.shape

        # 步骤 1-2: 线性映射 Q/K/V 并 reshape 为多头格式
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)        # [B, S, H_q, D]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)     # [B, S, H_kv, D]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)     # [B, S, H_kv, D]

        # 步骤 3: QK-Norm，防止注意力分数数值过大
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 步骤 4: 应用 RoPE 旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 步骤 5: 拼接 KV Cache（推理时将历史 KV 拼接到当前 KV 前面）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 在序列维度拼接历史 Key
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 在序列维度拼接历史 Value
        past_kv = (xk, xv) if use_cache else None  # 返回当前 KV 供后续使用

        # 步骤 6: transpose 为 [B, H, S, D] 格式，并扩展 KV 头（GQA）
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        # 步骤 7: 计算注意力
        # Flash Attention 路径：自动处理因果 mask，无需手动构建
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 手动注意力计算路径：需要显式构建因果 mask 和处理 attention_mask
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # QK^T / sqrt(d)
            if self.is_causal:
                # 因果 mask：当前位置只能看到之前位置，用 -inf 屏蔽未来位置
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                # 自定义 attention_mask：0 的位置被屏蔽（加 -1e9）
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv  # softmax → dropout → 加权求和

        # 步骤 8: reshape 回 [B, S, H*D] 并通过输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """标准前馈网络层（SwiGLU 变体）

    使用门控激活结构: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    相比传统 ReLU FFN，SwiGLU 在相同参数量下表现更好。

    结构: x → [gate_proj → SiLU] × [up_proj] → down_proj → output
                    \\_____________×____________/

    Args:
        config: MiniMindConfig 配置对象
        intermediate_size: 中间层维度（默认从 config 中获取）
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 门控投影
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 下投影（降维回 hidden_size）
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # 上投影
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数，默认 SiLU

    def forward(self, x):
        """SwiGLU 前向传播: down_proj(silu(gate_proj(x)) * up_proj(x))"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """混合专家（Mixture of Experts）前馈层

    将标准 FFN 替换为多个并行的专家网络，通过门控路由器（Router）
    为每个 token 选择 top-k 个专家进行计算，然后加权求和。

    关键设计:
    - 门控路由器: 线性层 + Softmax，输出每个专家的概率分布
    - Top-k 选择: 每个 token 只激活 k 个专家，大幅减少计算量
    - 归一化: 可选对 top-k 权重重新归一化，确保权重和为 1
    - 辅助损失: 负载均衡损失（aux_loss），防止所有 token 都路由到同一个专家
    - 训练时的特殊处理: 未被选中的专家也进行 0 * param 的计算，保证梯度流

    Args:
        config: MiniMindConfig 配置对象
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 门控路由器：输入 → 专家概率
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # N 个并行的专家 FFN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """MoE 前向传播

        执行流程:
        1. 展平输入为 [batch*seq, hidden_dim]
        2. 通过门控路由器计算每个 token 对每个专家的概率
        3. 选择 top-k 个专家及其权重
        4. 对每个专家：找到路由到该专家的 token，计算输出并加权累加
        5. 计算辅助负载均衡损失

        Args:
            x: 输入张量, shape [batch, seq_len, hidden_dim]

        Returns:
            输出张量, shape [batch, seq_len, hidden_dim]
            同时设置 self.aux_loss 为负载均衡辅助损失
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # 展平: [B*S, D]

        # 步骤 1: 计算门控路由概率
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [B*S, num_experts]

        # 步骤 2: 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # 可选：对 top-k 权重重新归一化
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # 步骤 3: 遍历每个专家，计算路由到该专家的 token 的输出
        y = torch.zeros_like(x_flat)  # 累加器
        for i, expert in enumerate(self.experts):
            # mask: 哪些 token 选择了当前专家 i
            mask = (topk_idx == i)
            if mask.any():
                # token_idx: 路由到当前专家的 token 的索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # weight: 这些 token 对应的专家权重
                weight = topk_weight[mask].view(-1, 1)
                # 计算专家输出并加权累加
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时：即使专家未被选中，也要进行 0*param 的计算
                # 这确保所有专家参数都有梯度流，避免参数"冻结"
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 步骤 4: 计算负载均衡辅助损失
        # 目的：鼓励各专家被均匀选择，防止"赢者通吃"
        # loss = N * Σ(f_i * P_i)，其中 f_i 是专家 i 被选中的频率，P_i 是平均路由概率
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # f_i: 每个专家被选中的频率
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()  # 推理时不需要辅助损失

        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """单层 Transformer Block（Pre-Norm 架构）

    结构:
        x → RMSNorm → Attention → +残差 → RMSNorm → FFN/MoE → +残差 → output
             (input_layernorm)           (post_attention_layernorm)

    特点:
    - Pre-Norm: 先归一化再计算，比 Post-Norm 训练更稳定
    - MoE 支持: 根据配置自动选择标准 FFN 或 MoE FFN

    Args:
        layer_id: 层编号（可用于层间差异化策略）
        config: MiniMindConfig 配置对象
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)                                    # 自注意力层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力前的 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # FFN 前的 RMSNorm
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # FFN 或 MoE

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """Transformer Block 前向传播

        Args:
            hidden_states: 输入隐藏状态
            position_embeddings: 位置编码 (cos, sin)
            past_key_value: KV Cache
            use_cache: 是否返回 KV Cache
            attention_mask: 注意力掩码

        Returns:
            hidden_states: 输出隐藏状态
            present_key_value: 当前层的 KV Cache
        """
        # 自注意力子层: RMSNorm → Attention → 残差连接
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # FFN/MoE 子层: RMSNorm → FFN/MoE → 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """MiniMind 模型主体

    由以下部分组成:
    1. Token Embedding: 将 token ID 映射为向量
    2. Dropout: 嵌入层后的 Dropout
    3. N 层 MiniMindBlock: Transformer 层堆叠
    4. 最终 RMSNorm: 输出归一化
    5. RoPE 频率表: 预计算的位置编码 cos/sin 值

    Args:
        config: MiniMindConfig 配置对象
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # ===== 模型组件 =====
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # Token 嵌入层
        self.dropout = nn.Dropout(config.dropout)                                 # 嵌入后 Dropout
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # Transformer 层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

        # ===== 预计算 RoPE 频率表 =====
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 不随模型保存
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """模型主体前向传播

        执行流程:
        1. Token 嵌入 + Dropout
        2. 获取当前位置的 RoPE 编码
        3. 逐层通过 Transformer Block
        4. 最终 RMSNorm
        5. 汇总 MoE 辅助损失

        Args:
            input_ids: 输入 token ID, shape [batch, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 历史 KV Cache 列表
            use_cache: 是否返回 KV Cache

        Returns:
            hidden_states: 最终隐藏状态
            presents: 各层的 KV Cache
            aux_loss: MoE 辅助损失之和
        """
        batch_size, seq_length = input_ids.shape

        # 处理 past_key_values：兼容 HuggingFace 的 DynamicCache 对象
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前输入的起始位置（有 KV Cache 时，从缓存长度开始）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 步骤 1: Token 嵌入
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 修复: 在 meta-device 初始化时（transformers>=5.x），buffer 可能丢失，需重新计算
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        # 步骤 2: 获取当前序列位置对应的 RoPE 编码
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        # 步骤 3: 逐层通过 Transformer
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 步骤 4: 最终 RMSNorm
        hidden_states = self.norm(hidden_states)

        # 步骤 5: 计算 MoE 辅助损失（负载均衡损失）
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """MiniMind 因果语言模型

    在 MiniMindModel 基础上添加:
    - LM Head: 将隐藏状态映射到词表维度的线性层
    - 损失计算: 交叉熵损失（用于训练）
    - generate: 自回归生成方法（支持 KV Cache、采样、重复惩罚等）

    继承 PreTrainedModel 和 GenerationMixin 以兼容 HuggingFace 生态。

    Args:
        config: MiniMindConfig 配置对象
    """
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 权重共享的键映射

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)  # 模型主体
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 语言模型头

        # 如果配置了权重共享，将 embedding 权重直接引用给 lm_head
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()  # HuggingFace 初始化后处理（权重初始化等）




    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """因果语言模型前向传播

        执行流程:
        1. 通过模型主体获取隐藏状态
        2. 通过 LM Head 计算 logits
        3. 如提供 labels，计算交叉熵损失

        Args:
            input_ids: 输入 token ID
            attention_mask: 注意力掩码
            past_key_values: KV Cache
            use_cache: 是否返回 KV Cache
            logits_to_keep: 保留最后多少个位置的 logits（节省内存）
            labels: 目标标签（用于计算损失）

        Returns:
            MoeCausalLMOutputWithPast: 包含 loss, aux_loss, logits, past_key_values, hidden_states
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # logits_to_keep: 只计算最后 N 个位置的 logits（推理优化，减少不必要的计算）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算交叉熵损失（仅在有 labels 时）
        loss = None
        if labels is not None:
            # 将 logits 和 labels 对齐：logits 预测下一个 token
            # x = logits[..., :-1, :] 对应位置 0 到 T-2 的预测
            # y = labels[..., 1:]  对应位置 1 到 T-1 的目标
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # -100 为忽略标签

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """自回归文本生成

        逐步生成 token，每次将前一步的输出拼接到输入中，直到达到最大长度或遇到 EOS。

        采样策略:
        1. 温度缩放: logits / temperature，控制分布的尖锐程度
        2. 重复惩罚: 对已生成的 token 降低概率
        3. Top-k 采样: 只保留概率最高的 k 个 token
        4. Top-p（Nucleus）采样: 保留累积概率达到 p 的最小 token 集合
        5. 多项式采样或贪心选择

        Args:
            inputs: 输入 token ID, shape [batch, seq_len]
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，越高越随机
            top_p: Nucleus 采样阈值
            top_k: Top-k 采样阈值
            eos_token_id: 结束 token ID
            streamer: 流式输出器（用于逐 token 输出）
            use_cache: 是否使用 KV Cache 加速
            num_return_sequences: 每个输入生成的序列数
            do_sample: 是否采样（False 则贪心选择）
            repetition_penalty: 重复惩罚系数

        Returns:
            生成的 token ID 序列, shape [batch, seq_len + generated_len]
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)  # 支持多条并行生成
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 跟踪每个序列是否已完成

        if streamer: streamer.put(input_ids.cpu())  # 流式输出初始输入

        for _ in range(max_new_tokens):
            # 计算 KV Cache 中已有的长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 前向传播：有 KV Cache 时只需传入最后一个 token
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 更新 attention_mask（新增 1 个位置）
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None

            # 取最后一个位置的 logits
            logits = outputs.logits[:, -1, :] / temperature  # 温度缩放

            # 重复惩罚：对已出现过的 token 的 logits 除以惩罚系数
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty

            # Top-k 采样：只保留概率最高的 k 个 token，其余设为 -inf
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-p（Nucleus）采样：保留累积概率达到 p 的最小 token 集合
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p  # 累积概率超过 p 的位置
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 至少保留概率最高的 1 个 token
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')  # 屏蔽低概率 token

            # 采样或贪心选择下一个 token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            # 已完成的序列：替换为 EOS token（保证所有序列长度一致）
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            # 拼接新 token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer: streamer.put(next_token.cpu())  # 流式输出

            # 检测 EOS：标记已完成的序列
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break  # 所有序列都已完成

        if streamer: streamer.end()  # 流式输出结束

        # 可选：返回 KV Cache（用于后续继续生成）
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids