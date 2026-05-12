"""快速测试 plan_warmup_v2 模型是否能在 RL prompt 格式下输出 <plan>"""
import torch, json, sys
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
                     num_key_value_heads=4, intermediate_size=2432)
model = MiniMindForCausalLM(cfg)
w = torch.load('out/plan_warmup_v2_768.pth', map_location='cpu')
model.load_state_dict(w, strict=False)
model.eval().cuda()
t = AutoTokenizer.from_pretrained('model/')

# Exactly replicate RL prompt construction from train_plan.py
PLAN_SYSTEM_PROMPT = "你是一个具备规划能力的 AI 助手。需要工具时，必须先输出 <plan>...</plan> 再行动。不需要工具时直接回答。"
PLAN_FEWSHOT_MESSAGES = [
    {"role": "user", "content": "北京今天天气怎么样？"},
    {"role": "assistant", "content": '<plan>\n[{"step": 1, "tool": "get_current_weather", "args_desc": "location=北京", "expect": "获取天气"}]\n</plan>'},
]

# Test case 1: translate + calculate (multi-tool, mimics agent_rl.jsonl)
tools = [
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

msgs = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}] + PLAN_FEWSHOT_MESSAGES + [
    {"role": "user", "content": "Translate Good morning to chinese and compute 2045*6994"}
]

for test_name, open_thinking in [("thinking=False", False), ("thinking=True", True)]:
    prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   tools=tools, open_thinking=open_thinking)
    print(f"\n=== {test_name} ===")
    print(f"Prompt ends with: ...{prompt[-200:]}")
    ids = t(prompt, return_tensors='pt').to('cuda')
    print(f"Prompt tokens: {ids.input_ids.shape[1]}")
    out = model.generate(ids.input_ids, max_new_tokens=300, temperature=0.7, do_sample=True)
    gen = t.decode(out[0][ids.input_ids.shape[1]:])
    print(f"GEN: {repr(gen[:400])}")

# Test case 2: simple single tool (weather)
print("\n=== Test 2: simple weather ===")
tools2 = [{"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
msgs2 = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}] + PLAN_FEWSHOT_MESSAGES + [
    {"role": "user", "content": "上海天气怎么样？"}
]
prompt2 = t.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True,
                                tools=tools2, open_thinking=False)
ids2 = t(prompt2, return_tensors='pt').to('cuda')
out2 = model.generate(ids2.input_ids, max_new_tokens=200, temperature=0.7, do_sample=True)
gen2 = t.decode(out2[0][ids2.input_ids.shape[1]:])
print(f"GEN: {repr(gen2[:300])}")

# Test case 3: no tools (should not output plan)
print("\n=== Test 3: no tools (should NOT output plan) ===")
msgs3 = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}] + PLAN_FEWSHOT_MESSAGES + [
    {"role": "user", "content": "你好，今天心情怎么样？"}
]
prompt3 = t.apply_chat_template(msgs3, tokenize=False, add_generation_prompt=True, open_thinking=False)
ids3 = t(prompt3, return_tensors='pt').to('cuda')
out3 = model.generate(ids3.input_ids, max_new_tokens=200, temperature=0.7, do_sample=True)
gen3 = t.decode(out3[0][ids3.input_ids.shape[1]:])
print(f"GEN: {repr(gen3[:300])}")
