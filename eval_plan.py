"""Plan-then-Execute 推理评测脚本 v2

从 plan_execute_500_part2.jsonl 采样 + 生成 none 样本，共 200 条。
覆盖所有场景（sequential/parallel/conditional/mixed/single/none），
评测 plan 正确性、误触率和执行质量。

用法:
  python eval_plan.py --model_path ./checkpoints_qwen_plan_v2/best --num_samples 200
"""
import os, sys, json, random, argparse, time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_plan_qwen import (
    init_model_qwen, TorchRolloutEngineHF,
    plan_rollout_single, parse_tool_calls, EXPERT_CONFIG,
)
from trainer.train_agent import validate_gt_in_text


# ==================== None 样本（不需要工具调用的问题） ====================
NONE_QUERIES = [
    "帮我解释一下强化学习中的reward是什么",
    "解释一下什么是梯度下降",
    "什么是 Transformer 架构？",
    "什么是强化学习？",
    "什么是卷积神经网络？",
    "Python 和 Java 哪个好？",
    "什么是反向传播算法？",
    "给我讲讲机器学习是什么",
    "什么是生成对抗网络？",
    "什么是知识蒸馏？",
    "什么是注意力机制？",
    "请解释一下什么是过拟合",
    "深度学习和传统机器学习有什么区别？",
    "什么是批量归一化（Batch Normalization）？",
    "解释一下 LSTM 和 GRU 的区别",
    "什么是迁移学习？举个例子",
    "为什么神经网络需要激活函数？",
    "什么是 Dropout？为什么能防止过拟合？",
    "解释一下 Adam 优化器的工作原理",
    "什么是 Embedding？有什么用？",
    "请介绍一下 BERT 模型的核心思想",
    "什么是自监督学习？",
    "解释一下什么是模型蒸馏",
    "什么是对比学习（Contrastive Learning）？",
    "请解释一下 LoRA 微调的原理",
    "什么是 RLHF？为什么 ChatGPT 需要它？",
    "解释一下什么是 tokenization",
    "什么是束搜索（Beam Search）？",
    "CPU 和 GPU 在深度学习中有什么区别？",
    "什么是混合精度训练？",
    "解释一下什么是 KV Cache",
    "什么是 Flash Attention？",
    "请解释一下 PPO 算法的核心思想",
    "什么是 DPO？和 RLHF 有什么区别？",
    "解释一下什么是 Chain-of-Thought 推理",
    "什么是量化（Quantization）？有什么优缺点？",
    "请介绍一下 Diffusion Model 的基本原理",
    "什么是 RAG（检索增强生成）？",
    "解释一下 MoE（混合专家模型）的架构",
    "什么是 Speculative Decoding？",
    "写一首关于春天的短诗",
    "给我讲一个关于时间管理的建议",
    "总结一下面向对象编程的三大特性",
    "如何写好一篇技术博客？",
    "给初学者推荐学习深度学习的路径",
]


def make_none_sample(query):
    """构造 none 类型的样本格式"""
    return {
        'query': query,
        'level': 0,
        'plan_gt': [],
        'gt': '',
        'step_gts': [],
        'num_steps': 0,
        'experts_needed': [],
        'dependency_type': 'none',
    }


def stratified_sample(data, n_tool=160, n_none=40, seed=42):
    """分层采样：n_tool 条工具调用样本 + n_none 条 none 样本"""
    random.seed(seed)

    # 1. 从工具调用数据中按 dependency_type 分层采样
    tool_data = [d for d in data if d['num_steps'] > 0]
    by_type = defaultdict(list)
    for item in tool_data:
        by_type[item['dependency_type']].append(item)

    type_keys = sorted(by_type.keys())
    total = sum(len(v) for v in by_type.values())
    tool_samples = []
    remaining = n_tool

    for i, key in enumerate(type_keys):
        pool = by_type[key]
        if i == len(type_keys) - 1:
            count = remaining
        else:
            count = max(5, round(n_tool * len(pool) / total))
            count = min(count, remaining, len(pool))
        chosen = random.sample(pool, min(count, len(pool)))
        tool_samples.extend(chosen)
        remaining -= len(chosen)

    # 2. 生成 none 样本
    none_queries = random.sample(NONE_QUERIES, min(n_none, len(NONE_QUERIES)))
    none_samples = [make_none_sample(q) for q in none_queries]

    # 3. 合并并打乱
    all_samples = tool_samples[:n_tool] + none_samples
    random.shuffle(all_samples)
    return all_samples


def evaluate_single(result, sample):
    """评估单条推理结果"""
    metrics = {
        'handoff_correct': False,
        'plan_correct': False,
        'expert_match': False,
        'gt_hit': False,
        'e2e_success': False,
        'false_trigger': False,
    }

    handoff = result.get('handoff_occurred', False)
    should_handoff = sample['num_steps'] > 0
    plan_executed = result.get('plan_executed', False)

    # 1. Handoff 正确性
    metrics['handoff_correct'] = (handoff == should_handoff)

    # 误触检测：不需要工具但发起了 handoff
    if not should_handoff and handoff:
        metrics['false_trigger'] = True

    # 2. Plan 正确性
    if sample['num_steps'] == 0:
        # none 样本：不该有 plan 也不该有 handoff
        metrics['plan_correct'] = (not plan_executed and not handoff)
    elif sample['num_steps'] == 1:
        # 单步：可以用 delegate 也可以用 plan，但必须 handoff
        metrics['plan_correct'] = handoff
    else:
        # 多步：应该使用 execute_plan
        metrics['plan_correct'] = plan_executed

    # 3. Expert 匹配
    if sample['num_steps'] == 0:
        # none 样本：没发 handoff 就算匹配正确
        metrics['expert_match'] = not handoff
    else:
        plan_info = result.get('plan_info', {})
        if plan_info and plan_info.get('steps'):
            called_experts = set()
            for step in plan_info['steps']:
                delegate = step.get('delegate', '')
                if not delegate.startswith('delegate_to_'):
                    delegate = f'delegate_to_{delegate}_agent'
                called_experts.add(delegate)
            expected_experts = set(f'delegate_to_{e}_agent' for e in sample['experts_needed'])
            metrics['expert_match'] = (called_experts == expected_experts)
        elif handoff and not plan_executed:
            delegated = result.get('delegated_expert', '')
            expected = set(f'delegate_to_{e}_agent' for e in sample['experts_needed'])
            metrics['expert_match'] = (delegated in expected)

    # 4. GT 命中
    if sample['num_steps'] == 0:
        # none 样本没有 GT，只要没误触就算 hit
        metrics['gt_hit'] = not handoff
    else:
        all_outputs = result.get('all_outputs', [])
        final_text = ' '.join(all_outputs) if all_outputs else ''
        gt = sample.get('gt', '')
        if gt:
            gt_result = validate_gt_in_text(final_text, gt)
            metrics['gt_hit'] = gt_result.get('hit', False) if isinstance(gt_result, dict) else bool(gt_result)

    # 5. E2E
    metrics['e2e_success'] = all([
        metrics['handoff_correct'],
        metrics['plan_correct'],
        metrics['expert_match'],
        metrics['gt_hit'],
    ])

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints_qwen_plan_v2/best')
    parser.add_argument('--data_path', default='./dataset/plan_execute_500_part2.jsonl')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--none_ratio', type=float, default=0.2,
                        help='none 样本占比 (default: 0.2 = 40/200)')
    parser.add_argument('--output', default='./logs/eval_plan_v2_results.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n_none = int(args.num_samples * args.none_ratio)
    n_tool = args.num_samples - n_none

    print(f'[Eval] Loading model from {args.model_path}...')
    model, tokenizer = init_model_qwen(args.model_path)
    model.eval()
    device = next(model.parameters()).device
    print(f'[Eval] Model loaded on {device}')

    rollout_engine = TorchRolloutEngineHF(model, tokenizer)

    # 加载数据并采样
    with open(args.data_path) as f:
        all_data = [json.loads(line) for line in f]
    samples = stratified_sample(all_data, n_tool=n_tool, n_none=n_none, seed=args.seed)

    print(f'[Eval] Total {len(samples)} samples (tool={n_tool}, none={n_none}):')
    dep_dist = defaultdict(int)
    for s in samples:
        dep_dist[s['dependency_type']] += 1
    for k, v in sorted(dep_dist.items()):
        print(f'  {k}: {v}')
    print()

    # 推理
    results = []
    agg_metrics = defaultdict(int)
    total = len(samples)
    false_trigger_count = 0
    none_count = 0

    for i, sample in enumerate(samples):
        t0 = time.time()
        try:
            with torch.no_grad():
                result = plan_rollout_single(
                    rollout_engine, tokenizer, sample['query'],
                    max_new_tokens=384, device=device,
                )
        except Exception as e:
            print(f'  [{i+1}/{total}] ERROR: {e}')
            result = {'handoff_occurred': False, 'plan_executed': False, 'all_outputs': []}

        elapsed = time.time() - t0
        metrics = evaluate_single(result, sample)

        for k, v in metrics.items():
            if k != 'false_trigger':
                agg_metrics[k] += int(v)

        if sample['num_steps'] == 0:
            none_count += 1
            if metrics['false_trigger']:
                false_trigger_count += 1

        entry = {
            'idx': i,
            'query': sample['query'],
            'dependency_type': sample['dependency_type'],
            'num_steps': sample['num_steps'],
            'experts_needed': sample['experts_needed'],
            'gt': sample.get('gt', ''),
            'handoff_occurred': result.get('handoff_occurred', False),
            'plan_executed': result.get('plan_executed', False),
            'plan_info': result.get('plan_info'),
            'final_output': result.get('all_outputs', [])[-1] if result.get('all_outputs') else '',
            'metrics': metrics,
            'time': round(elapsed, 2),
        }
        results.append(entry)

        status = 'OK' if metrics['e2e_success'] else 'FAIL'
        ft_mark = ' [FALSE_TRIGGER]' if metrics['false_trigger'] else ''
        print(f'  [{i+1}/{total}] {status} | {sample["dependency_type"]:12s} | '
              f'handoff={metrics["handoff_correct"]} plan={metrics["plan_correct"]} '
              f'expert={metrics["expert_match"]} gt={metrics["gt_hit"]}{ft_mark} | {elapsed:.1f}s')

    # 汇总
    print(f'\n{"="*60}')
    print(f'[RESULTS] {total} samples evaluated (tool={n_tool}, none={none_count}):')
    print(f'  Handoff Accuracy:  {agg_metrics["handoff_correct"]}/{total} = {agg_metrics["handoff_correct"]/total*100:.1f}%')
    print(f'  Plan Accuracy:     {agg_metrics["plan_correct"]}/{total} = {agg_metrics["plan_correct"]/total*100:.1f}%')
    print(f'  Expert Match:      {agg_metrics["expert_match"]}/{total} = {agg_metrics["expert_match"]/total*100:.1f}%')
    print(f'  GT Hit:            {agg_metrics["gt_hit"]}/{total} = {agg_metrics["gt_hit"]/total*100:.1f}%')
    print(f'  E2E Success:       {agg_metrics["e2e_success"]}/{total} = {agg_metrics["e2e_success"]/total*100:.1f}%')
    print(f'  ---')
    print(f'  False Trigger Rate: {false_trigger_count}/{none_count} = {false_trigger_count/max(1,none_count)*100:.1f}%')
    print(f'  Miss Rate (tool samples): {n_tool - sum(1 for r in results if r["num_steps"]>0 and r["handoff_occurred"])}/{n_tool}')

    # 按场景统计
    print(f'\n[By dependency_type]:')
    by_dep = defaultdict(lambda: defaultdict(int))
    dep_totals = defaultdict(int)
    for r in results:
        dep = r['dependency_type']
        dep_totals[dep] += 1
        for k, v in r['metrics'].items():
            if k != 'false_trigger':
                by_dep[dep][k] += int(v)
    for dep in sorted(dep_totals.keys()):
        t = dep_totals[dep]
        m = by_dep[dep]
        print(f'  {dep:12s} (n={t:2d}): handoff={m["handoff_correct"]}/{t} '
              f'plan={m["plan_correct"]}/{t} expert={m["expert_match"]}/{t} '
              f'gt={m["gt_hit"]}/{t} e2e={m["e2e_success"]}/{t}')

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = {
        'summary': {k: v for k, v in agg_metrics.items()},
        'total': total,
        'n_tool': n_tool,
        'n_none': none_count,
        'false_trigger_count': false_trigger_count,
        'false_trigger_rate': round(false_trigger_count / max(1, none_count) * 100, 1),
        'by_dependency_type': {dep: dict(by_dep[dep]) for dep in dep_totals},
        'details': results,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n[Eval] Results saved to {args.output}')


if __name__ == '__main__':
    main()
