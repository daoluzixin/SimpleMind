# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import json  # JSON 解析与序列化，用于读取训练数据和保存配置
import os  # 操作系统接口，用于文件路径拼接和目录创建

from tokenizers import decoders, models, pre_tokenizers, trainers, \
    Tokenizer  # HuggingFace tokenizers 库，提供 BPE 分词器的构建、训练和解码功能

DATA_PATH = '../dataset/sft_t2t_mini.jsonl'  # 训练语料路径，JSONL 格式（每行一条 JSON 对话数据）
TOKENIZER_DIR = '../model_learn_tokenizer/'  # 训练后的 tokenizer 保存目录
VOCAB_SIZE = 6400  # 词表大小，即 BPE 合并后最终的词汇数量
SPECIAL_TOKENS_NUM = 36  # 特殊 token 的总数量（包括预定义特殊 token、附加 token 和缓冲 token）

def get_texts(data_path):
    """从 JSONL 文件中逐行读取对话文本，作为训练数据的生成器。

    Args:
        data_path: JSONL 格式的训练语料文件路径
    Yields:
        str: 每条对话中所有轮次的 content 拼接结果（用换行符分隔）
    """
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:  # 以 UTF-8 编码打开文件，忽略无法解码的字符
        for i, line in enumerate(f):  # 逐行遍历文件
            if i >= 10000: break  # 只取前 10000 行用于训练，限制数据量以加速测试
            try:
                data = json.loads(line)  # 将每行 JSON 字符串解析为字典
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]  # 提取对话中每一轮的 content 字段，过滤掉空内容
                if contents:  # 如果该条对话有有效内容
                    yield "\n".join(contents)  # 将所有轮次的 content 用换行符拼接后生成
            except json.JSONDecodeError:  # 跳过格式错误、无法解析的行
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    """训练 BPE 分词器并保存到指定目录。
    
    Args:
        data_path: 训练语料的文件路径
        tokenizer_dir: 训练后的 tokenizer 保存目录
        vocab_size: 目标词表大小
        special_tokens_num: 特殊 token 的总数量（默认 36）
    """
    tokenizer = Tokenizer(models.BPE())  # 创建一个基于 BPE（Byte Pair Encoding）算法的空分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 设置预分词器为字节级别，不在文本开头添加空格前缀

    # 预定义的特殊 token 列表，用于标记文本中的特殊语义边界
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",   # 未知 token、对话轮次起始/结束标记
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",   # 目标检测/框选相关的边界标记（用于多模态模型）
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",   # 视觉/图像/视频相关的边界与填充标记
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"  # 音频/TTS 语音合成相关的标记
    ]
    
    # 附加特殊 token，用于标记工具调用（Function Calling）的结构化边界
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)  # 计算需要预留的缓冲 token 数量
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]  # 生成缓冲 token，预留位置以便未来扩展
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens  # 合并所有特殊 token 为完整列表
    trainer = trainers.BpeTrainer(  # 创建 BPE 训练器，配置训练参数
        vocab_size=vocab_size,  # 目标词表大小
        show_progress=True,  # 训练时在终端显示进度条
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 初始字母表使用字节级别的 256 个字符
        special_tokens=all_special_tokens  # 注册所有特殊 token，使其在词表中占据固定位置
    )
    texts = get_texts(data_path)  # 获取训练文本的迭代器
    tokenizer.train_from_iterator(texts, trainer=trainer)  # 从文本迭代器训练 BPE 模型，执行合并操作直到达到目标词表大小
    tokenizer.decoder = decoders.ByteLevel()  # 设置解码器为字节级别，与预分词器保持一致，确保编码-解码的往返一致性
    tokenizer.add_special_tokens(special_tokens_list)  # 将预定义的特殊 token 添加到分词器中

    os.makedirs(tokenizer_dir, exist_ok=True)  # 创建 tokenizer 保存目录（若已存在则不报错）
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 将完整的 tokenizer 保存为 JSON 文件
    tokenizer.model.save(tokenizer_dir)  # 单独保存 BPE 模型文件（vocab.json 和 merges.txt）

    # ---- 修正 added_tokens 中非特殊 token 的 special 标志 ----
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")  # 重新读取刚保存的 tokenizer.json
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:  # 以 UTF-8 编码打开 tokenizer.json 用于读取
        tokenizer_data = json.load(f)  # 加载 tokenizer 的 JSON 数据
    for token_info in tokenizer_data.get('added_tokens', []):  # 遍历所有已添加的 token
        if token_info['content'] not in special_tokens_list:  # 如果该 token 不在预定义的特殊 token 列表中
            token_info['special'] = False  # 将其 special 标志设为 False（如附加 token 和缓冲 token 不应被当作特殊 token）
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:  # 以 UTF-8 编码打开 tokenizer.json 用于写入
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)  # 将修正后的数据写回 tokenizer.json，保留 Unicode 字符、缩进 2 空格
    

    # ---- 构建 added_tokens_decoder 字典，用于 tokenizer_config.json ----
    added_tokens_decoder = {}  # 键为 token ID（字符串），值为该 token 的详细属性
    for i, token in enumerate(all_special_tokens):  # 遍历所有特殊 token
        idx = tokenizer.token_to_id(token)  # 获取该 token 在词表中的 ID
        added_tokens_decoder[str(idx)] = {  # 以 ID 字符串为键，记录 token 属性
            "content": token,  # token 的文本内容
            "lstrip": False,  # 解码时不从左侧剥离空格
            "normalized": False,  # 该 token 不参与标准化处理
            "rstrip": False,  # 解码时不从右侧剥离空格
            "single_word": False,  # 该 token 不被视为单个词（即可以和其他 token 组合）
            "special": True if token in special_tokens_list else False  # 仅预定义特殊 token 标记为 True
        }


    # ---- 构建 tokenizer_config.json 配置，兼容 HuggingFace Transformers 的 PreTrainedTokenizerFast 接口 ----
    config = {
        "add_bos_token": False,  # 编码时是否自动在开头添加 BOS token
        "add_eos_token": False,  # 编码时是否自动在结尾添加 EOS token
        "add_prefix_space": False,  # 是否在文本前添加空格（ByteLevel 预分词器通常不需要）
        "added_tokens_decoder": added_tokens_decoder,  # 附加 token 的 ID -> 属性映射
        "additional_special_tokens": [t for t in special_tokens_list if t not in [""]],  # 除 UNK 外的额外特殊 token 列表
        "bos_token": "<|im_start|>",  # 句首/对话起始标记
        "clean_up_tokenization_spaces": False,  # 解码时是否清理多余空格
        "eos_token": "<|im_end|>",  # 句尾/对话结束标记
        "legacy": True,  # 是否使用旧版分词行为（兼容性）
        "model_max_length": 131072,  # 模型支持的最大序列长度（128K）
        "pad_token": "",  # 填充标记，用于 batch 中不等长序列的对齐
        "sp_model_kwargs": {},  # SentencePiece 模型的额外参数（本分词器不使用 SentencePiece）
        "spaces_between_special_tokens": False,  # 解码时是否在特殊 token 之间插入空格
        "unk_token": "",  # 未知 token 标记，遇到词表外的字符时使用
        "image_token": "<|image_pad|>",  # 图像填充 token（多模态使用）
        "audio_token": "<|audio_pad|>",  # 音频填充 token（多模态使用）
        "video_token": "<|video_pad|>",  # 视频填充 token（多模态使用）
        "vision_bos_token": "<|vision_start|>",  # 视觉输入起始标记
        "vision_eos_token": "<|vision_end|>",  # 视觉输入结束标记
        "audio_bos_token": "<|audio_start|>",  # 音频输入起始标记
        "audio_eos_token": "<|audio_end|>",  # 音频输入结束标记
        # Jinja2 格式的对话模板，定义了多轮对话、工具调用、推理思考等场景下的 prompt 组装规则
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"  # 指定使用 HuggingFace Transformers 的快速分词器类
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:  # 打开配置文件用于写入
        json.dump(config, f, ensure_ascii=False, indent=4)  # 将配置字典序列化为 JSON 并写入文件，保留 Unicode 字符、缩进 4 空格
    print("Tokenizer training completed.")  # 训练完成提示



def eval_tokenizer(tokenizer_dir):
    """加载训练好的 tokenizer 并进行多项评估测试。

    测试内容包括：
    1. 对话模板渲染（apply_chat_template）
    2. 编码-解码一致性验证
    3. 压缩率测试（字符数/Token数的比值）
    4. 流式解码（字节缓冲）测试

    Args:
        tokenizer_dir: tokenizer 文件所在目录
    """
    from transformers import AutoTokenizer  # 从 HuggingFace Transformers 导入自动分词器类
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)  # 从本地目录加载训练好的 tokenizer

    # ---- 1. 对话模板渲染测试 ----
    messages = [  # 构造一组多轮对话消息，用于测试 chat_template 的渲染效果
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},  # 系统提示
        {"role": "user", "content": '你来自哪里？'},  # 用户第一轮提问
        {"role": "assistant", "content": '我来自月球'},  # 助手第一轮回复
        {"role": "user", "content": '你到底来自哪里？'},  # 用户第二轮追问
        {"role": "assistant", "content": '我来自地球'},  # 助手第二轮回复
    ]
    new_prompt = tokenizer.apply_chat_template(  # 使用 tokenizer 的 chat_template 将消息列表渲染为完整的 prompt 字符串
        messages,
        tokenize=False  # 返回字符串而非 token ID 列表
    )
    print('-'*100)  # 打印分隔线
    print(new_prompt)  # 输出渲染后的完整 prompt
    print('-'*100)  # 打印分隔线

    # ---- 2. 编码-解码一致性验证 ----
    print('tokenizer词表长度：', len(tokenizer))  # 输出词表的总大小
    model_inputs = tokenizer(new_prompt)  # 将 prompt 文本编码为 token ID
    print('encoder长度：', len(model_inputs['input_ids']))  # 输出编码后的 token 数量
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)  # 将 token ID 解码回文本（保留特殊 token）
    print('decoder一致性：', response == new_prompt, "\n")  # 验证解码结果是否与原始文本完全一致
    print('-'*100)  # 打印分隔线

    # ---- 3. 压缩率测试（Chars/Tokens）：衡量分词器将文本压缩为 token 的效率 ----
    print('压缩率测试（Chars/Tokens）：')  # 压缩率 = 字符数 / Token数，值越高表示每个 token 承载的信息越多
    test_texts = [  # 测试样本：包含中文、英文和中英混合三种场景
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    

    total_compression = 0  # 累计所有样本的压缩率，用于计算平均值
    for i, text in enumerate(test_texts):  # 逐个测试样本
        encoded = tokenizer.encode(text)  # 将文本编码为 token ID 列表
        token_count = len(encoded)  # 编码后的 token 数量
        char_count = len(text)  # 原始文本的字符数
        compression_ratio = char_count / token_count  # 计算该样本的压缩率
        total_compression += compression_ratio  # 累加压缩率
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")  # 格式化输出每个样本的结果
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")  # 输出所有样本的平均压缩率
    print('-'*100)  # 打印分隔线

    # ---- 4. 流式解码（字节缓冲）测试：模拟大模型逐 token 生成时的解码过程 ----
    # 字节级 BPE 分词器可能出现一个 UTF-8 字符被拆到多个 token 中的情况，
    # 需要缓冲这些 token 直到能完整解码出一个字符
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']  # 获取之前编码的 token ID 列表
    token_cache = []  # 初始化 token 缓冲区，用于存储尚未完整解码的 token ID
    for tid in input_ids:  # 逐个 token 模拟流式输入
        token_cache.append(tid)  # 将当前 token ID 加入缓冲区
        current_decode = tokenizer.decode(token_cache)  # 尝试解码缓冲区中的所有 token
        if current_decode and '\ufffd' not in current_decode:  # 如果解码结果非空且不包含 Unicode 替换字符，说明缓冲区中的 token 已能完整解码
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache  # 格式化显示：单个 ID 直接显示，多个 ID 显示列表
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]  # 将 token ID 转换为原始的子词字符串
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')  # 输出：Token ID -> 原始子词 -> 解码字符串
            token_cache = []  # 清空缓冲区，准备下一轮缓冲



if __name__ == '__main__':  # 当脚本直接运行时执行以下代码
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)  # 第一步：训练 BPE 分词器
    eval_tokenizer(TOKENIZER_DIR)  # 第二步：评估训练好的分词器
