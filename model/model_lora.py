"""
LoRA（Low-Rank Adaptation）低秩适配模块

本文件实现了 LoRA 微调的核心功能：
- LoRA: 低秩适配层，将权重更新分解为两个低秩矩阵 A 和 B 的乘积
- apply_lora: 将 LoRA 层注入模型的所有方阵线性层
- load_lora: 从文件加载 LoRA 权重到模型
- save_lora: 仅保存 LoRA 权重（不包含基础模型权重）
- merge_lora: 将 LoRA 权重合并回基础模型权重（推理时消除额外计算开销）

LoRA 原理:
    原始线性层: y = Wx
    LoRA 微调后: y = Wx + BAx
    其中 A ∈ R^(d×r), B ∈ R^(r×d), r << d（低秩约束）
    训练时冻结 W，只训练 A 和 B，参数量从 d² 降低到 2dr
"""
import torch
from torch import nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """LoRA 低秩适配层

    将权重增量 ΔW 分解为两个低秩矩阵 B @ A 的乘积：
    - A: 降维矩阵，将输入从 in_features 映射到 rank 维（低秩空间）
    - B: 升维矩阵，将 rank 维映射回 out_features 维

    初始化策略:
    - A: 高斯初始化（N(0, 0.02)），提供非零初始值
    - B: 全零初始化，确保训练开始时 LoRA 的输出为 0（ΔW = BA = 0）
      即初始时模型行为与原始模型完全一致

    Args:
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA 的秩（rank），控制低秩矩阵的大小，越大表达能力越强但参数越多
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A: d → r
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B: r → d
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化，保证训练初期LoRA输出为零
        self.B.weight.data.zero_()

    def forward(self, x):
        """LoRA 前向传播: output = B(A(x))，即 ΔWx"""
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """将 LoRA 层注入模型的所有方阵线性层

    遍历模型的所有 nn.Linear 模块，如果权重矩阵是方阵（in_features == out_features），
    则为该层添加 LoRA 适配器，并 monkey-patch 其 forward 方法。

    修改后的 forward: y = Wx + BAx（原始输出 + LoRA 增量输出）

    注意: 只对方阵线性层应用 LoRA，因为 MiniMind 的 Q/K/V/O 投影层
    在 tie_word_embeddings 模式下 input_dim == output_dim == hidden_size

    Args:
        model: 目标模型
        rank: LoRA 秩，默认 16
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 将 LoRA 层挂载为模块的属性
            original_forward = module.forward  # 保存原始 forward

            # 显式绑定：使用默认参数捕获 original_forward 和 lora
            # 避免闭包中的延迟绑定问题
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 原始输出 + LoRA 增量

            module.forward = forward_with_lora  # monkey-patch forward


def load_lora(model, path):
    """从文件加载 LoRA 权重到模型

    加载流程:
    1. 读取保存的 state_dict
    2. 去除键名中的 'module.' 前缀（兼容 DDP 模型保存格式）
    3. 遍历模型中有 lora 属性的模块，加载对应的 LoRA 权重

    Args:
        model: 目标模型（已通过 apply_lora 注入 LoRA 层）
        path: LoRA 权重文件路径
    """
    state_dict = torch.load(path, map_location=model.device)
    # 去除 DDP 保存时的 'module.' 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 提取当前模块对应的 LoRA 权重（键名格式: {module_name}.lora.{A/B}.weight）
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """仅保存 LoRA 权重（不包含基础模型权重）

    遍历模型中所有有 lora 属性的模块，收集 LoRA 层的参数。
    保存格式: {module_name}.lora.{A/B}.weight → 参数值

    Args:
        model: 目标模型
        path: 保存路径
    """
    raw_model = getattr(model, '_orig_mod', model)  # 处理 torch.compile 包装
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name  # 去除 DDP 前缀
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """将 LoRA 权重合并回基础模型权重

    合并原理: W_merged = W_base + B @ A
    将 LoRA 的增量权重直接加到基础模型的权重上，然后保存合并后的完整模型。
    合并后推理时不再需要 LoRA 层，消除了额外的矩阵乘法开销。

    执行流程:
    1. 加载 LoRA 权重到模型的 LoRA 层
    2. 遍历所有线性层，将 LoRA 权重 (B@A) 加到原始权重上
    3. 保存合并后的完整模型权重

    Args:
        model: 目标模型
        lora_path: LoRA 权重文件路径
        save_path: 合并后模型的保存路径
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    # 收集所有非 LoRA 参数
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    # 对有 LoRA 的线性层，将 LoRA 权重合并到原始权重
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # W_merged = W_base + B @ A（核心合并操作）
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
