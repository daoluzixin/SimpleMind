import json

with open('/root/dataset/xlam_agent_rl.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        s = json.loads(line)
        tools = []
        for c in s['conversations']:
            if c['role'] == 'system' and c.get('tools'):
                for t in json.loads(c['tools']):
                    tools.append(t['function']['name'])
        print(f'Sample {i}: tools={tools}, gt={s["gt"]}')
        print()

# 统计：GT 中的值是否能在 generic mock 结果中被找到
print("=" * 60)
print("GT matchability analysis (can GT be found in mock results?):")
print("=" * 60)

matchable = 0
total = 0
with open('/root/dataset/xlam_agent_rl.jsonl') as f:
    for i, line in enumerate(f):
        s = json.loads(line)
        gt = s.get('gt', [])
        total += 1
        
        # 模拟：如果模型正确调用了工具，generic mock 会把参数值返回
        # GT 应该出现在参数值中才能被 validate_gt_in_text 匹配到
        # 但 GT 实际上是什么？是参数值还是返回值？
        if i < 10:
            tools_in_sample = []
            for c in s['conversations']:
                if c['role'] == 'system' and c.get('tools'):
                    for t in json.loads(c['tools']):
                        tools_in_sample.append(t['function']['name'])
            print(f"Sample {i}: gt={gt}")
            print(f"  tools: {tools_in_sample}")
            print()
