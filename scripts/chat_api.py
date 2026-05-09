"""MiniMind Chat API 客户端

通过 OpenAI 兼容接口与 MiniMind 模型进行交互式对话。
支持流式输出和思考模式（thinking）。

使用方式:
    1. 先启动 serve_openai_api.py 服务端
    2. 运行本脚本进行交互式对话

功能:
    - 支持流式/非流式输出
    - 支持思考模式（reasoning_content 显示为灰色）
    - 支持多轮对话历史
"""
from openai import OpenAI

# 初始化 OpenAI 客户端，连接到本地 MiniMind 服务
client = OpenAI(
    api_key="sk-123",                              # API 密钥（本地服务可随意设置）
    base_url="http://localhost:11434/v1"            # 服务端地址
)
stream = True                                        # 是否使用流式输出
conversation_history_origin = []                     # 原始对话历史（用于重置）
conversation_history = conversation_history_origin.copy()  # 当前对话历史
history_messages_num = 0  # 携带的历史对话轮数，必须设置为偶数（Q+A），为0则不携带历史对话

# 交互式对话循环
while True:
    query = input('[Q]: ')  # 用户输入
    conversation_history.append({"role": "user", "content": query})  # 添加用户消息到历史

    # 调用 OpenAI 兼容接口生成回复
    response = client.chat.completions.create(
        model="minimind-local:latest",              # 模型名称
        messages=conversation_history[-(history_messages_num or 1):],  # 截取历史消息
        stream=stream,                               # 是否流式输出
        temperature=0.8,                             # 采样温度
        max_tokens=2048,                             # 最大生成 token 数
        top_p=0.8,                                   # nucleus 采样阈值
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"}  # 思考开关
    )

    if not stream:
        # 非流式模式：直接获取完整回复
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        # 流式模式：逐 chunk 输出
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            delta = chunk.choices[0].delta
            r = getattr(delta, 'reasoning_content', None) or ""  # 思考内容
            c = delta.content or ""                                # 正式回复内容
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)  # 思考内容显示为灰色
            if c:
                print(c, end="", flush=True)                       # 正式回复正常显示
            assistant_res += c

    conversation_history.append({"role": "assistant", "content": assistant_res})  # 添加助手回复到历史
    print('\n\n')