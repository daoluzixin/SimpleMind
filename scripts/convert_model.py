"""MiniMind 模型格式转换工具

提供多种模型格式之间的转换功能:
1. PyTorch → Transformers-MiniMind 格式（保留原始模型结构）
2. PyTorch → Transformers-Qwen3 格式（兼容 HuggingFace 生态）
3. Transformers → PyTorch 格式（反向转换）
4. 合并 LoRA 权重到基础模型
5. Jinja ↔ JSON 聊天模板格式互转

支持的转换路径:
    - convert_torch2transformers_minimind: 保留 MiniMind 原生结构的转换
    - convert_torch2transformers: 转换为 Qwen3/Qwen3Moe 兼容结构（推荐，生态更好）
    - convert_transformers2torch: 从 Transformers 格式转回 PyTorch
    - convert_merge_base_lora: 将 LoRA 权重合并到基础模型并保存
    - convert_jinja_to_json / convert_json_to_jinja: 聊天模板格式互转
"""
import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    """将 PyTorch 格式的 MiniMind 模型转换为 Transformers-MiniMind 格式

    保留 MiniMind 原生模型结构，注册到 HuggingFace AutoModel 体系，
    使得可以通过 AutoModelForCausalLM.from_pretrained() 加载。

    Args:
        torch_path: PyTorch 权重文件路径（.pth）
        transformers_path: 输出的 Transformers 格式目录
        dtype: 保存的精度类型，默认 float16
    """
    # 注册 MiniMind 配置和模型类到 HuggingFace Auto 体系
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    # 同时保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0 的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        # 修复 tokenizer_config.json 中的特殊 token 配置
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        # 修复 config.json 中的 RoPE 参数（移除 rope_parameters，保留 rope_theta）
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    """将 PyTorch 格式的 MiniMind 模型转换为 Transformers-Qwen3 兼容格式

    将 MiniMind 模型结构映射到 Qwen3/Qwen3Moe 结构，
    使得可以直接使用 HuggingFace 生态中的所有工具（vLLM、SGLang 等）。

    非 MoE 模型 → Qwen3ForCausalLM
    MoE 模型   → Qwen3MoeForCausalLM

    Args:
        torch_path: PyTorch 权重文件路径（.pth）
        transformers_path: 输出的 Transformers 格式目录
        dtype: 保存的精度类型，默认 float16
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)

    # MiniMind 和 Qwen3 共享的配置参数
    common_config = {
        "vocab_size": lm_config.vocab_size,                    # 词表大小
        "hidden_size": lm_config.hidden_size,                  # 隐藏层维度
        "intermediate_size": lm_config.intermediate_size,      # FFN 中间层维度
        "num_hidden_layers": lm_config.num_hidden_layers,      # Transformer 层数
        "num_attention_heads": lm_config.num_attention_heads,  # 注意力头数
        "num_key_value_heads": lm_config.num_key_value_heads,  # KV 头数（GQA）
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,  # 每个头的维度
        "max_position_embeddings": lm_config.max_position_embeddings,  # 最大位置编码
        "rms_norm_eps": lm_config.rms_norm_eps,                # RMSNorm epsilon
        "rope_theta": lm_config.rope_theta,                    # RoPE 基础频率
        "tie_word_embeddings": lm_config.tie_word_embeddings   # 是否共享嵌入权重
    }

    if not lm_config.use_moe:
        # 非 MoE 模型 → Qwen3 结构
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False,  # 不使用滑动窗口注意力
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        # MoE 模型 → Qwen3Moe 结构
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,                  # 专家数量
            num_experts_per_tok=lm_config.num_experts_per_tok,  # 每个token激活的专家数
            moe_intermediate_size=lm_config.moe_intermediate_size,  # MoE FFN 中间层维度
            norm_topk_prob=lm_config.norm_topk_prob             # 是否归一化 top-k 概率
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)

        # ======= transformers-5.0 兼容: 合并 MoE 专家权重格式 =======
        # MiniMind 格式: experts.{e}.gate_proj.weight / up_proj.weight / down_proj.weight (独立)
        # Qwen3Moe 格式: experts.gate_up_proj (合并) / experts.down_proj (堆叠)
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                # 合并 gate_proj 和 up_proj → gate_up_proj
                new_sd[f'{p}.gate_up_proj'] = torch.cat([
                    torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]),
                    torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])
                ], dim=1)
                # 堆叠 down_proj
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0 的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """将 Transformers 格式的模型转换回 PyTorch 格式

    Args:
        transformers_path: Transformers 格式模型目录
        torch_path: 输出的 PyTorch 权重文件路径（.pth）
    """
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    # 保存为半精度 PyTorch 格式
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    """将 LoRA 权重合并到基础模型中，保存为完整的基础模型格式

    合并后可以无需 LoRA 模块直接推理，适合部署场景。

    Args:
        base_torch_path: 基础模型的 PyTorch 权重路径
        lora_path: LoRA 权重文件路径
        merged_torch_path: 合并后模型的保存路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    # 先注入 LoRA 模块，再将 LoRA 权重合并到基础权重中
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    """将 Jinja 格式的聊天模板转换为 JSON 格式（用于 tokenizer_config.json）

    Args:
        jinja_path: Jinja 模板文件路径
    """
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    """从 tokenizer_config.json 中提取聊天模板并保存为 Jinja 文件

    Args:
        json_file_path: 包含 chat_template 的 JSON 文件路径
        output_path: 输出的 Jinja 模板文件路径
    """
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    # 默认模型配置
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # 示例: 将 PyTorch 格式转换为 Transformers-Qwen3 兼容格式
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # # 合并 LoRA 权重
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
