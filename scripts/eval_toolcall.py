"""MiniMind Tool Call（工具调用）评测脚本

评测模型的工具调用（Function Calling / Tool Use）能力。
支持两种后端: local（本地模型推理）和 api（OpenAI兼容接口）。

工具调用流程:
1. 用户提出需要工具的问题
2. 模型识别需要调用的工具并生成工具调用请求
3. 执行工具获取结果
4. 将结果返回给模型，模型生成最终回答
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# ========== 工具定义 ==========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

# ========== Mock 工具执行结果 ==========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-").replace("\u00b2", "**2").replace("\u00b3", "**3").replace("\uff08", "(").replace("\uff09", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22\u00b0C", "humidity": "65%", "condition": "\u6674"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    """根据工具名列表获取工具定义"""
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "\u5e2e\u6211\u7b97\u4e00\u4e0b 256 \u4e58\u4ee5 37 \u7b49\u4e8e\u591a\u5c11", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "\u73b0\u5728\u51e0\u70b9\u4e86\uff1f", "tools": ["get_current_time", "random_number"]},
    {"prompt": "\u5e2e\u6211\u628a100\u516c\u91cc\u6362\u7b97\u6210\u82f1\u91cc", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "\u5e2e\u6211\u751f\u6210\u4e00\u4e2a1\u52301000\u7684\u968f\u673a\u6570\uff0c\u7136\u540e\u8ba1\u7b97\u5b83\u7684\u5e73\u65b9", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "\u5317\u4eac\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "\u67e5\u4e00\u4e0b\u7f8e\u5143\u5151\u4eba\u6c11\u5e01\u6c47\u7387", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "\u628a'\u4f60\u597d\u4e16\u754c'\u7ffb\u8bd1\u6210\u82f1\u6587", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    """初始化本地模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    """从模型输出文本中解析工具调用请求"""
    TC_OPEN = '\u2524'
    TC_CLOSE = '\u2518'
    matches = re.findall(rf'{TC_OPEN}(.*?){TC_CLOSE}', text, re.DOTALL)
    calls = []
    for m in matches:
        try: calls.append(json.loads(m.strip()))
        except Exception: pass
    return calls


def parse_tool_call_from_text(content):
    """从文本中解析工具调用（API模式）"""
    TC_OPEN = '\u2524'
    TC_CLOSE = '\u2518'
    pattern = rf'{TC_OPEN}\s*(\{{.*?\}})\s*{TC_CLOSE}'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches: return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({"id": f"call_{i}", "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}})
        except Exception: pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    """执行工具调用并返回Mock结果"""
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception: args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn: return {"error": f"\u672a\u77e5\u5de5\u5177: {name}"}
    try: return fn(args)
    except Exception as e: return {"error": f"\u5de5\u5177\u6267\u884c\u5931\u8d25: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    """本地模型生成回复（流式输出）"""
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('\U0001f9e0: ', end='')
    generated_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, top_p=args.top_p, temperature=args.temperature)
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    """API模式生成回复"""
    response = client.chat.completions.create(model=args.api_model, messages=messages, tools=tools, stream=stream, temperature=args.temperature, max_tokens=8192, top_p=args.top_p)
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls: tool_calls = parse_tool_call_from_text(content)
        print(f'\U0001f9e0: {content}')
        return content, tool_calls
    print('\U0001f9e0: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content: print(delta.content, end="", flush=True); content += delta.content
        if delta.tool_calls:
            if tool_calls is None: tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx: tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                if tc_chunk.id: tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name: tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments: tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls: tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    """运行单个测试用例（支持多轮工具调用）"""
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls: break
        tool_calls = [{"id": tc.id if hasattr(tc, 'id') else tc.get("id", ""), "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"], "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]} for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name, arguments = tc["name"], tc["arguments"]
            print(f'\U0001f4de [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'\U0001f4a1 [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall\u8bc4\u6d4b")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str)
    parser.add_argument('--load_from', default='../model', type=str)
    parser.add_argument('--save_dir', default='../out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--show_speed', default=0, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str)
    parser.add_argument('--api_key', default='sk-123', type=str)
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str)
    parser.add_argument('--stream', default=1, type=int)
    args = parser.parse_args()

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] \u81ea\u52a8\u6d4b\u8bd5\n[1] \u624b\u52a8\u8f93\u5165\n'))
    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('\U0001f4ac: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'\U0001f4e6 \u53ef\u7528\u5de5\u5177: {case["tool_names"]}\n'); print(f'\U0001f4ac: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
