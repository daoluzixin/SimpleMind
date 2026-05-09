#!/usr/bin/env python3
"""为 train_tokenizer.py 添加详细注释 - 处理剩余未注释的行"""

with open('train_tokenizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
for i, l in enumerate(lines):
    s = l.rstrip('\n')
    stripped = s.strip()

    # === special_tokens_list 内部行加注释 ===
    if '"<|im_start|>"' in l and '"<|im_end|>"' in l and '# ' not in l:
        s = s + '  # 未知 token、对话轮次起始/结束标记'
    elif '"<|object_ref_start|>"' in l and '# ' not in l:
        s = s + '  # 目标检测/框选相关的边界标记（用于多模态模型）'
    elif '"<|vision_start|>"' in l and '# ' not in l:
        s = s + '  # 视觉/图像/视频相关的边界与填充标记'
    elif '"<|audio_start|>"' in l and '# ' not in l:
        s = s + '  # 音频/TTS 语音合成相关的标记'

    # === additional_tokens_list 前加注释 ===
    if stripped == 'additional_tokens_list = [':
        out.append('    # 附加特殊 token，用于标记工具调用（Function Calling）的结构化边界\n')

    # === num_buffer / buffer_tokens / all_special_tokens ===
    if stripped.startswith('num_buffer = special_tokens_num'):
        s = '    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)  # 计算需要预留的缓冲 token 数量'
    if stripped.startswith('buffer_tokens = [f"<|buffer'):
        s = '    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]  # 生成缓冲 token，预留位置以便未来扩展'
    if stripped.startswith('all_special_tokens = special_tokens_list'):
        s = '    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens  # 合并所有特殊 token 为完整列表'

    # === trainer ===
    if stripped == 'trainer = trainers.BpeTrainer(':
        out.append('    trainer = trainers.BpeTrainer(  # 创建 BPE 训练器，配置训练参数\n')
        continue
    if stripped == 'vocab_size=vocab_size,':
        s = '        vocab_size=vocab_size,  # 目标词表大小'
    if stripped == 'show_progress=True,':
        s = '        show_progress=True,  # 训练时在终端显示进度条'
    if stripped.startswith('initial_alphabet=pre_tokenizers.ByteLevel.alphabet()'):
        s = '        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表使用字节级别的 256 个字符'
    if stripped.startswith('special_tokens=all_special_tokens'):
        s = '        special_tokens=all_special_tokens  # 注册所有特殊 token，使其在词表中占据固定位置'

    # === 训练过程 ===
    if stripped.startswith('texts = get_texts(data_path)'):
        s = '    texts = get_texts(data_path)  # 获取训练文本的迭代器'
    if stripped.startswith('tokenizer.train_from_iterator(texts'):
        s = '    tokenizer.train_from_iterator(texts, trainer=trainer)  # 从文本迭代器训练 BPE 模型，执行合并操作直到达到目标词表大小'
    if stripped == 'tokenizer.decoder = decoders.ByteLevel():':
        s = '    tokenizer.decoder = decoders.ByteLevel()  # 设置解码器为字节级别，确保编码后的 ID 能正确还原为原始文本'
    if stripped.startswith('tokenizer.add_special_tokens(special_tokens_list)'):
        s = '    tokenizer.add_special_tokens(special_tokens_list)  # 将预定义的特殊 token 添加到分词器中'

    # === 保存 ===
    if stripped.startswith('os.makedirs(tokenizer_dir'):
        s = '    os.makedirs(tokenizer_dir, exist_ok=True)  # 创建 tokenizer 保存目录（若已存在则不报错）'
    if stripped == 'tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))':
        s = '    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 将完整的 tokenizer 保存为 JSON 文件'
    if stripped == 'tokenizer.model.save(tokenizer_dir)':
        s = '    tokenizer.model.save(tokenizer_dir)  # 单独保存 BPE 模型文件（vocab.json 和 merges.txt）'

    # === 修正 added_tokens ===
    if stripped.startswith('tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")') and '# ' not in l:
        out.append('\n')
        out.append('    # ---- 修正 added_tokens 中非特殊 token 的 special 标志 ----\n')
        s = '    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")  # 重新读取刚保存的 tokenizer.json'
    if stripped == 'tokenizer_data = json.load(f)':
        s = '        tokenizer_data = json.load(f)  # 加载 tokenizer 的 JSON 数据'
    if stripped.startswith('for token_info in tokenizer_data.get('):
        s = "    for token_info in tokenizer_data.get('added_tokens', []):  # 遍历所有已添加的 token"
    if "token_info['content'] not in special_tokens_list" in stripped:
        s = "        if token_info['content'] not in special_tokens_list:  # 如果该 token 不在预定义的特殊 token 列表中"
    if stripped == "token_info['special'] = False":
        s = "            token_info['special'] = False  # 将其 special 标志设为 False（如附加 token 和缓冲 token 不应被当作特殊 token）"
    if "json.dump(tokenizer_data, f" in stripped and '# ' not in l:
        s = '        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)  # 将修正后的数据写回 tokenizer.json，保留 Unicode 字符、缩进 2 空格'

    # === added_tokens_decoder ===
    if stripped == 'added_tokens_decoder = {}':
        out.append('\n')
        out.append('    # ---- 构建 added_tokens_decoder 字典，用于 tokenizer_config.json ----\n')
        s = '    added_tokens_decoder = {}  # 键为 token ID（字符串），值为该 token 的详细属性'
    if stripped.startswith('for i, token in enumerate(all_special_tokens)'):
        s = '    for i, token in enumerate(all_special_tokens):  # 遍历所有特殊 token'
    if stripped == 'idx = tokenizer.token_to_id(token)':
        s = '        idx = tokenizer.token_to_id(token)  # 获取该 token 在词表中的 ID'
    if stripped.startswith('added_tokens_decoder[str(idx)]'):
        s = '        added_tokens_decoder[str(idx)] = {  # 以 ID 字符串为键，记录 token 属性'
    if stripped == '"content": token,':
        s = '            "content": token,  # token 的文本内容'
    if stripped == '"lstrip": False,':
        s = '            "lstrip": False,  # 解码时不从左侧剥离空格'
    if stripped == '"normalized": False,':
        s = '            "normalized": False,  # 该 token 不参与标准化处理'
    if stripped == '"rstrip": False,':
        s = '            "rstrip": False,  # 解码时不从右侧剥离空格'
    if stripped == '"single_word": False,':
        s = '            "single_word": False,  # 该 token 不被视为单个词（即可以和其他 token 组合）'
    if '"special": True if token in special_tokens_list' in stripped:
        s = '            "special": True if token in special_tokens_list else False  # 仅预定义特殊 token 标记为 True'

    # === config 字典 ===
    if stripped == 'config = {':
        out.append('\n')
        out.append('    # ---- 构建 tokenizer_config.json 配置，兼容 HuggingFace Transformers 的 PreTrainedTokenizerFast 接口 ----\n')
    if stripped == '"add_bos_token": False,':
        s = '        "add_bos_token": False,  # 编码时是否自动在开头添加 BOS token'
    if stripped == '"add_eos_token": False,':
        s = '        "add_eos_token": False,  # 编码时是否自动在结尾添加 EOS token'
    if stripped == '"add_prefix_space": False,':
        s = '        "add_prefix_space": False,  # 是否在文本前添加空格（ByteLevel 预分词器通常不需要）'
    if stripped.startswith('"added_tokens_decoder": added_tokens_decoder'):
        s = '        "added_tokens_decoder": added_tokens_decoder,  # 附加 token 的 ID -> 属性映射'
    if stripped.startswith('"additional_special_tokens"'):
        s = '        "additional_special_tokens": [t for t in special_tokens_list if t not in [""]],  # 除 UNK 外的额外特殊 token 列表'
    if stripped == '"bos_token": "<|im_start|>",':
        s = '        "bos_token": "<|im_start|>",  # 句首/对话起始标记'
    if stripped == '"clean_up_tokenization_spaces": False,':
        s = '        "clean_up_tokenization_spaces": False,  # 解码时是否清理多余空格'
    if stripped == '"eos_token": "<|im_end|>",':
        s = '        "eos_token": "<|im_end|>",  # 句尾/对话结束标记'
    if stripped == '"legacy": True,':
        s = '        "legacy": True,  # 是否使用旧版分词行为（兼容性）'
    if stripped == '"model_max_length": 131072,':
        s = '        "model_max_length": 131072,  # 模型支持的最大序列长度（128K）'
    if stripped.startswith('"pad_token"') and '# ' not in l:
        s = '        "pad_token": "",  # 填充标记，用于 batch 中不等长序列的对齐'
    if stripped == '"sp_model_kwargs": {},':
        s = '        "sp_model_kwargs": {},  # SentencePiece 模型的额外参数（本分词器不使用 SentencePiece）'
    if stripped == '"spaces_between_special_tokens": False,':
        s = '        "spaces_between_special_tokens": False,  # 解码时是否在特殊 token 之间插入空格'
    if stripped.startswith('"unk_token"') and '# ' not in l:
        s = '        "unk_token": "",  # 未知 token 标记，遇到词表外的字符时使用'
    if stripped == '"image_token": "<|image_pad|>",':
        s = '        "image_token": "<|image_pad|>",  # 图像填充 token（多模态使用）'
    if stripped == '"audio_token": "<|audio_pad|>",':
        s = '        "audio_token": "<|audio_pad|>",  # 音频填充 token（多模态使用）'
    if stripped == '"video_token": "<|video_pad|>",':
        s = '        "video_token": "<|video_pad|>",  # 视频填充 token（多模态使用）'
    if stripped == '"vision_bos_token": "<|vision_start|>",':
        s = '        "vision_bos_token": "<|vision_start|>",  # 视觉输入起始标记'
    if stripped == '"vision_eos_token": "<|vision_end|>",':
        s = '        "vision_eos_token": "<|vision_end|>",  # 视觉输入结束标记'
    if stripped == '"audio_bos_token": "<|audio_start|>",':
        s = '        "audio_bos_token": "<|audio_start|>",  # 音频输入起始标记'
    if stripped == '"audio_eos_token": "<|audio_end|>",':
        s = '        "audio_eos_token": "<|audio_end|>",  # 音频输入结束标记'
    if stripped.startswith('"chat_template"') and '# Jinja2' not in l:
        out.append('        # Jinja2 格式的对话模板，定义了多轮对话、工具调用、推理思考等场景下的 prompt 组装规则\n')
    if stripped == '"tokenizer_class": "PreTrainedTokenizerFast"':
        s = '        "tokenizer_class": "PreTrainedTokenizerFast"  # 指定使用 HuggingFace Transformers 的快速分词器类'

    # === 保存 config ===
    if stripped.startswith('with open(os.path.join(tokenizer_dir, "tokenizer_config.json")'):
        s = '    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:  # 打开配置文件用于写入'
    if stripped.startswith('json.dump(config') and '# ' not in l:
        s = '        json.dump(config, f, ensure_ascii=False, indent=4)  # 将配置字典序列化为 JSON 并写入文件，保留 Unicode 字符、缩进 4 空格'
    if stripped == 'print("Tokenizer training completed.")':
        s = '    print("Tokenizer training completed.")  # 训练完成提示'

    # === eval_tokenizer 函数 ===
    if stripped == 'def eval_tokenizer(tokenizer_dir):':
        out.append('\n\n')
        out.append('def eval_tokenizer(tokenizer_dir):\n')
        out.append('    """加载训练好的 tokenizer 并进行多项评估测试。\n')
        out.append('\n')
        out.append('    测试内容包括：\n')
        out.append('    1. 对话模板渲染（apply_chat_template）\n')
        out.append('    2. 编码-解码一致性验证\n')
        out.append('    3. 压缩率测试（字符数/Token数的比值）\n')
        out.append('    4. 流式解码（字节缓冲）测试\n')
        out.append('\n')
        out.append('    Args:\n')
        out.append('        tokenizer_dir: tokenizer 文件所在目录\n')
        out.append('    """\n')
        continue
    if stripped == 'from transformers import AutoTokenizer':
        s = '    from transformers import AutoTokenizer  # 从 HuggingFace Transformers 导入自动分词器类'
    if stripped.startswith('tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)'):
        s = '    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)  # 从本地目录加载训练好的 tokenizer'

    # === messages ===
    if stripped == 'messages = [':
        out.append('\n')
        out.append('    # ---- 1. 对话模板渲染测试 ----\n')
        s = '    messages = [  # 构造一组多轮对话消息，用于测试 chat_template 的渲染效果'
    if '"role": "system"' in stripped and '聊天机器人' in stripped and '# 系统提示' not in l:
        s = '        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},  # 系统提示'
    if stripped == '{"role": "user", "content": \'你来自哪里？\'}':
        s = '        {"role": "user", "content": \'你来自哪里？\'},  # 用户第一轮提问'
    if stripped == '{"role": "assistant", "content": \'我来自月球\'}':
        s = '        {"role": "assistant", "content": \'我来自月球\'},  # 助手第一轮回复'
    if stripped == '{"role": "user", "content": \'你到底来自哪里？\'}':
        s = '        {"role": "user", "content": \'你到底来自哪里？\'},  # 用户第二轮追问'
    if stripped == '{"role": "assistant", "content": \'我来自地球\'}':
        s = '        {"role": "assistant", "content": \'我来自地球\'},  # 助手第二轮回复'

    # === apply_chat_template ===
    if stripped == 'new_prompt = tokenizer.apply_chat_template(':
        s = '    new_prompt = tokenizer.apply_chat_template(  # 使用 tokenizer 的 chat_template 将消息列表渲染为完整的 prompt 字符串'
    if stripped == 'tokenize=False' and i > 100:
        s = '        tokenize=False  # 返回字符串而非 token ID 列表'

    # === 编码解码 ===
    if stripped.startswith("print('tokenizer词表长度"):
        out.append('\n')
        out.append('    # ---- 2. 编码-解码一致性验证 ----\n')
        s = "    print('tokenizer词表长度：', len(tokenizer))  # 输出词表的总大小"
    if stripped.startswith('model_inputs = tokenizer(new_prompt)'):
        s = '    model_inputs = tokenizer(new_prompt)  # 将 prompt 文本编码为 token ID'
    if stripped.startswith("print('encoder长度"):
        s = "    print('encoder长度：', len(model_inputs['input_ids']))  # 输出编码后的 token 数量"
    if stripped.startswith('response = tokenizer.decode'):
        s = "    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)  # 将 token ID 解码回文本（保留特殊 token）"
    if stripped.startswith("print('decoder一致性"):
        s = "    print('decoder一致性：', response == new_prompt, \"\\n\")  # 验证解码结果是否与原始文本完全一致"

    # === 压缩率 ===
    if stripped.startswith("print('压缩率测试"):
        out.append('\n')
        out.append('    # ---- 3. 压缩率测试（Chars/Tokens）：衡量分词器将文本压缩为 token 的效率 ----\n')
        s = "    print('压缩率测试（Chars/Tokens）：')  # 压缩率 = 字符数 / Token数，值越高表示每个 token 承载的信息越多"
    if stripped == 'test_texts = [':
        s = '    test_texts = [  # 测试样本：包含中文、英文和中英混合三种场景'

    # === 压缩率计算 ===
    if stripped == 'total_compression = 0':
        out.append('\n')
        s = '    total_compression = 0  # 累计所有样本的压缩率，用于计算平均值'
    if stripped.startswith('for i, text in enumerate(test_texts)'):
        s = '    for i, text in enumerate(test_texts):  # 逐个测试样本'
    if stripped == 'encoded = tokenizer.encode(text)':
        s = '        encoded = tokenizer.encode(text)  # 将文本编码为 token ID 列表'
    if stripped == 'token_count = len(encoded)':
        s = '        token_count = len(encoded)  # 编码后的 token 数量'
    if stripped == 'char_count = len(text)':
        s = '        char_count = len(text)  # 原始文本的字符数'
    if stripped == 'compression_ratio = char_count / token_count':
        s = '        compression_ratio = char_count / token_count  # 计算该样本的压缩率'
    if stripped.startswith('total_compression += compression_ratio'):
        s = '        total_compression += compression_ratio  # 累加压缩率'
    if stripped.startswith('print(f"样本'):
        s = '        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")  # 格式化输出每个样本的结果'
    if stripped.startswith('print(f"平均压缩率'):
        s = '    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")  # 输出所有样本的平均压缩率'

    # === 流式解码 ===
    if stripped.startswith("print('流式解码"):
        out.append('\n')
        out.append('    # ---- 4. 流式解码（字节缓冲）测试：模拟大模型逐 token 生成时的解码过程 ----\n')
        out.append('    # 字节级 BPE 分词器可能出现一个 UTF-8 字符被拆到多个 token 中的情况，\n')
        out.append('    # 需要缓冲这些 token 直到能完整解码出一个字符\n')
        s = "    print('流式解码（字节缓冲）测试：')"
    if stripped == "input_ids = model_inputs['input_ids']":
        s = "    input_ids = model_inputs['input_ids']  # 获取之前编码的 token ID 列表"
    if stripped == 'token_cache = []' and i > 150:
        s = '    token_cache = []  # 缓冲区：暂存尚未能完整解码的 token ID'
    if stripped.startswith('for tid in input_ids:'):
        s = '    for tid in input_ids:  # 逐个 token 模拟流式输入'
    if stripped == 'token_cache.append(tid)':
        s = '        token_cache.append(tid)  # 将当前 token ID 加入缓冲区'
    if stripped == 'current_decode = tokenizer.decode(token_cache)':
        s = '        current_decode = tokenizer.decode(token_cache)  # 尝试解码缓冲区中的所有 token'
    if stripped.startswith("if current_decode and '\\ufffd' not in current_decode"):
        s = "        if current_decode and '\\ufffd' not in current_decode:  # 如果解码结果非空且不包含 Unicode 替换字符，说明缓冲区中的 token 已能完整解码"
    if stripped.startswith('display_ids = token_cache[0]'):
        s = '            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache  # 格式化显示：单个 ID 直接显示，多个 ID 显示列表'
    if stripped.startswith('raw_tokens = [tokenizer.convert_ids_to_tokens'):
        s = '            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]  # 将 token ID 转换为原始的子词字符串'
    if stripped.startswith("print(f'Token ID:"):
        s = "            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')  # 输出：Token ID -> 原始子词 -> 解码字符串"
    if stripped == 'token_cache = []' and i > 150:
        s = '            token_cache = []  # 清空缓冲区，准备下一轮缓冲'

    # === main ===
    if stripped == "if __name__ == '__main__':":
        out.append('\n\n')
        s = "if __name__ == '__main__':  # 当脚本直接运行时执行以下代码"
    if stripped.startswith('train_tokenizer(DATA_PATH'):
        s = '    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)  # 第一步：训练 BPE 分词器'
    if stripped.startswith('eval_tokenizer(TOKENIZER_DIR)'):
        s = '    eval_tokenizer(TOKENIZER_DIR)  # 第二步：评估训练好的分词器'

    out.append(s + '\n')

with open('train_tokenizer.py', 'w', encoding='utf-8') as f:
    f.writelines(out)

print('All comments added successfully!')

