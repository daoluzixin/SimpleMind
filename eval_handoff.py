"""Handoff RL 推理评测脚本

对训好的 Qwen2.5-1.5B checkpoint 进行全面的 Agent Handoff 能力评测。

评测指标：
  1. 路由准确率（Route Accuracy）: 需要工具时正确 handoff + 不需要时正确不 handoff
  2. 专家选择准确率（Expert Selection Accuracy）: handoff 时选对专家
  3. 误触率（False Trigger Rate）: 不需要工具时错误发起 handoff
  4. 漏触率（Miss Rate）: 需要工具时未发起 handoff
  5. GT 命中率（GT Hit Rate）: 最终回答包含正确答案
  6. 端到端成功率（E2E Success）: 路由正确 + 专家正确 + GT 命中

用法：
  python eval_handoff.py --model_path ./checkpoints_qwen_handoff_v2/best
  python eval_handoff.py --model_path ./checkpoints_qwen_handoff_v2/step_150 --num_samples 50
"""

import os
import sys
import re
import json
import random
import argparse
import time
import logging
from collections import defaultdict

import torch
from torch import Tensor
from contextlib import nullcontext


class SafeJSONEncoder(json.JSONEncoder):
    """Handle set, defaultdict, Tensor and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, defaultdict):
            return dict(obj)
        if isinstance(obj, Tensor):
            return obj.tolist()
        return super().default(obj)


def setup_logger(log_path):
    """设置日志，同时输出到终端和文件"""
    logger = logging.getLogger("eval_handoff")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 文件 handler（记录所有详情）
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # 终端 handler（简洁输出）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger

# 复用训练脚本中的核心组件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_handoff_qwen import (
    init_model_qwen,
    TorchRolloutEngineHF,
    handoff_rollout_single,
    parse_tool_calls,
    execute_tool,
    calculate_handoff_rewards,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_TOOLS,
    EXPERT_CONFIG,
    TOOLS,
)
from trainer.train_agent import validate_gt_in_text


# ==============================================================================
#  评测数据集（覆盖各类场景）
# ==============================================================================

EVAL_DATASET = [
    # ===== Math Agent（数学计算、单位换算）=====
    {"query": "123乘以456等于多少？", "gt": ["56088"], "needs_tool": True, "expert": "math"},
    {"query": "帮我算一下 (25 + 75) × 4", "gt": ["400"], "needs_tool": True, "expert": "math"},
    {"query": "3.14 × 10的平方是多少？", "gt": ["314"], "needs_tool": True, "expert": "math"},
    {"query": "一个正方形边长5cm，面积是多少？", "gt": ["25"], "needs_tool": True, "expert": "math"},
    {"query": "10公里等于多少英里？", "gt": ["6.2137", "6.21", "6.214"], "needs_tool": True, "expert": "math"},
    {"query": "50磅等于多少公斤？", "gt": ["22.68", "22.7"], "needs_tool": True, "expert": "math"},
    {"query": "100华氏度等于多少摄氏度？", "gt": ["37.78", "37.8"], "needs_tool": True, "expert": "math"},
    {"query": "999除以3等于多少？", "gt": ["333"], "needs_tool": True, "expert": "math"},
    {"query": "2的10次方是多少？", "gt": ["1024"], "needs_tool": True, "expert": "math"},
    {"query": "根号144等于多少？", "gt": ["12"], "needs_tool": True, "expert": "math"},

    # ===== Info Agent（天气、时间、汇率）=====
    {"query": "北京今天天气怎么样？", "gt": ["28°C", "晴", "28"], "needs_tool": True, "expert": "info"},
    {"query": "上海现在的气温是多少度？", "gt": ["26°C", "26"], "needs_tool": True, "expert": "info"},
    {"query": "深圳今天下雨了吗？", "gt": ["30°C", "多云", "30"], "needs_tool": True, "expert": "info"},
    {"query": "现在纽约几点了？", "gt": ["EST", "EDT", "纽约"], "needs_tool": True, "expert": "info"},
    {"query": "东京现在是什么时间？", "gt": ["JST", "东京"], "needs_tool": True, "expert": "info"},
    {"query": "100美元能换多少人民币？", "gt": ["7.21", "721"], "needs_tool": True, "expert": "info"},
    {"query": "1欧元等于多少日元？", "gt": ["157", "158", "156"], "needs_tool": True, "expert": "info"},
    {"query": "广州今天天气如何？", "gt": ["32°C", "晴", "32"], "needs_tool": True, "expert": "info"},
    {"query": "伦敦现在几点？", "gt": ["GMT", "BST", "伦敦"], "needs_tool": True, "expert": "info"},
    {"query": "50英镑等于多少人民币？", "gt": ["9", "45", "450"], "needs_tool": True, "expert": "info"},

    # ===== Translate Agent（翻译）=====
    {"query": "帮我把'你好世界'翻译成英文", "gt": ["Hello World", "Hello, World"], "needs_tool": True, "expert": "translate"},
    {"query": "'Machine Learning'翻译成中文是什么？", "gt": ["机器学习"], "needs_tool": True, "expert": "translate"},
    {"query": "帮我翻译一下'人工智能改变未来'", "gt": ["Artificial Intelligence", "AI", "future"], "needs_tool": True, "expert": "translate"},
    {"query": "'Deep Learning'用中文怎么说？", "gt": ["深度学习"], "needs_tool": True, "expert": "translate"},
    {"query": "把'今天天气真好'翻译成英文", "gt": ["weather", "nice", "good", "today"], "needs_tool": True, "expert": "translate"},
    {"query": "'Natural Language Processing'的中文是什么？", "gt": ["自然语言处理"], "needs_tool": True, "expert": "translate"},
    {"query": "帮我把'谢谢你的帮助'翻译成英文", "gt": ["Thank", "help"], "needs_tool": True, "expert": "translate"},
    {"query": "'Computer Science'中文怎么翻译？", "gt": ["计算机科学"], "needs_tool": True, "expert": "translate"},
    {"query": "翻译一下'知识就是力量'", "gt": ["Knowledge", "power"], "needs_tool": True, "expert": "translate"},
    {"query": "'Reinforcement Learning'是什么意思？", "gt": ["强化学习"], "needs_tool": True, "expert": "translate"},

    # ===== None（不需要工具，直接回答）=====
    {"query": "你好，今天过得怎么样？", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "你觉得人工智能的未来会怎样？", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "给我讲一个关于程序员的笑话", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "什么是深度学习？简单解释一下", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "推荐几本关于机器学习的好书", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "Python和Java哪个更适合初学者？", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "写一首关于春天的短诗", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "如何保持良好的编程习惯？", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "你能自我介绍一下吗？", "gt": [], "needs_tool": False, "expert": "none"},
    {"query": "解释一下什么是Transformer架构", "gt": [], "needs_tool": False, "expert": "none"},
]


def run_eval(args):
    """运行评测"""
    # 设置日志
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_handoff_{timestamp}.log")
    log = setup_logger(log_path)

    log.info(f"\n{'=' * 80}")
    log.info(f"  Handoff RL Evaluation")
    log.info(f"  Model: {args.model_path}")
    log.info(f"  Samples: {args.num_samples} / {len(EVAL_DATASET)}")
    log.info(f"  Temperature: {args.temperature}")
    log.info(f"  Log file: {log_path}")
    log.info(f"{'=' * 80}\n")

    # 加载模型
    model, tokenizer = init_model_qwen(
        args.model_path, device=args.device,
        gradient_checkpointing=False,
    )
    model.eval()

    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if "cuda" in args.device else nullcontext()
    rollout_engine = TorchRolloutEngineHF(
        policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
    )

    # 选择评测样本
    eval_data = EVAL_DATASET[:args.num_samples] if args.num_samples < len(EVAL_DATASET) else EVAL_DATASET
    random.seed(42)
    if args.shuffle:
        eval_data = eval_data.copy()
        random.shuffle(eval_data)

    # 统计变量
    stats = {
        "total": 0,
        "route_correct": 0,       # 路由决策正确（需要tool时handoff + 不需要时不handoff）
        "expert_correct": 0,      # 专家选择正确（在handoff发生的前提下）
        "expert_total": 0,        # 实际发生handoff的总数
        "false_trigger": 0,       # 不需要工具时错误handoff
        "false_trigger_total": 0, # 不需要工具的总数
        "miss": 0,                # 需要工具时未handoff
        "miss_total": 0,          # 需要工具的总数
        "gt_hit": 0,              # GT命中
        "gt_total": 0,            # 有GT的样本总数
        "e2e_success": 0,         # 端到端成功
    }

    # 按专家类型统计
    per_expert = defaultdict(lambda: {"total": 0, "route_ok": 0, "expert_ok": 0, "gt_ok": 0})

    results_log = []
    start_time = time.time()

    for i, sample in enumerate(eval_data):
        query = sample["query"]
        gt = sample["gt"]
        needs_tool = sample["needs_tool"]
        expected_expert = sample["expert"]

        stats["total"] += 1
        per_expert[expected_expert]["total"] += 1

        # 执行推理
        with torch.no_grad():
            result = handoff_rollout_single(
                rollout_engine, tokenizer, query,
                needs_tool=needs_tool,
                max_new_tokens=args.max_gen_len,
                thinking_ratio=0.0,
                device=args.device,
            )
            result["expected_expert"] = expected_expert

        handoff_occurred = result["handoff_occurred"]
        delegated_expert = result.get("delegated_expert", "")
        final_answer = result["final_answer"]

        # --- 评测各指标 ---

        # 1. 路由正确性
        route_ok = False
        if needs_tool and handoff_occurred:
            route_ok = True
        elif not needs_tool and not handoff_occurred:
            route_ok = True
        if route_ok:
            stats["route_correct"] += 1
            per_expert[expected_expert]["route_ok"] += 1

        # 2. 误触率
        if not needs_tool:
            stats["false_trigger_total"] += 1
            if handoff_occurred:
                stats["false_trigger"] += 1

        # 3. 漏触率
        if needs_tool:
            stats["miss_total"] += 1
            if not handoff_occurred:
                stats["miss"] += 1

        # 4. 专家选择正确性
        if handoff_occurred:
            stats["expert_total"] += 1
            expert_map = {
                "math": "delegate_to_math_agent",
                "info": "delegate_to_info_agent",
                "translate": "delegate_to_translate_agent",
            }
            if expected_expert in expert_map and delegated_expert == expert_map[expected_expert]:
                stats["expert_correct"] += 1
                per_expert[expected_expert]["expert_ok"] += 1

        # 5. GT 命中率
        if gt:
            stats["gt_total"] += 1
            # 检查最终回答中是否包含GT
            answer_text = final_answer
            if "<think>" in answer_text and "</think>" in answer_text:
                answer_text = answer_text.split("</think>")[-1].strip()
            gt_hit = validate_gt_in_text(answer_text, gt)
            if gt_hit:
                stats["gt_hit"] += 1
                per_expert[expected_expert]["gt_ok"] += 1

        # 6. 端到端成功
        e2e_ok = route_ok and (not needs_tool or (handoff_occurred and
                 delegated_expert == expert_map.get(expected_expert, "")))
        if e2e_ok and gt:
            e2e_ok = gt_hit
        if e2e_ok:
            stats["e2e_success"] += 1

        # 记录详细结果
        log_entry = {
            "idx": i,
            "query": query,
            "expected_expert": expected_expert,
            "needs_tool": needs_tool,
            "handoff_occurred": handoff_occurred,
            "delegated_expert": delegated_expert or "none",
            "route_ok": route_ok,
            "gt_hit": gt_hit if gt else "N/A",
            "e2e_ok": e2e_ok,
            "final_answer": final_answer[:500],
            "all_outputs": [o[:300] for o in result.get("all_outputs", [])],
        }
        results_log.append(log_entry)

        # 实时打印（终端简洁，日志详细）
        status = "✓" if e2e_ok else "✗"
        expert_display = delegated_expert.replace("delegate_to_", "").replace("_agent", "") if delegated_expert else "direct"
        log.info(f"  [{i+1:3d}/{len(eval_data)}] {status} | "
                 f"expect={expected_expert:10s} | got={expert_display:10s} | "
                 f"handoff={'Y' if handoff_occurred else 'N'} | "
                 f"query={query[:30]}")

        # 详细日志（只写文件）
        log.debug(f"  ┌─ Sample {i+1}: {query}")
        log.debug(f"  │  Expected: expert={expected_expert}, needs_tool={needs_tool}, gt={gt}")
        log.debug(f"  │  Result: handoff={handoff_occurred}, delegated={delegated_expert or 'none'}")
        for j, out in enumerate(result.get("all_outputs", [])):
            if j == 0:
                role = "Router(Route)"
            elif j == len(result.get("all_outputs", [])) - 1 and handoff_occurred:
                role = "Router(Synth)"
            else:
                role = f"Expert(turn{j})"
            log.debug(f"  │  [{role}] {out[:300]}")
        log.debug(f"  │  route_ok={route_ok}, gt_hit={gt_hit if gt else 'N/A'}, e2e={e2e_ok}")
        log.debug(f"  └─{'─' * 60}")

    elapsed = time.time() - start_time

    # ==== 汇总报告 ====
    log.info(f"\n{'=' * 80}")
    log.info(f"  EVALUATION REPORT")
    log.info(f"{'=' * 80}")
    log.info(f"\n  Total samples: {stats['total']}")
    log.info(f"  Time elapsed:  {elapsed:.1f}s ({elapsed/stats['total']:.2f}s/sample)")

    route_acc = stats["route_correct"] / stats["total"] * 100
    expert_acc = stats["expert_correct"] / stats["expert_total"] * 100 if stats["expert_total"] > 0 else 0
    false_trigger_rate = stats["false_trigger"] / stats["false_trigger_total"] * 100 if stats["false_trigger_total"] > 0 else 0
    miss_rate = stats["miss"] / stats["miss_total"] * 100 if stats["miss_total"] > 0 else 0
    gt_hit_rate = stats["gt_hit"] / stats["gt_total"] * 100 if stats["gt_total"] > 0 else 0
    e2e_rate = stats["e2e_success"] / stats["total"] * 100

    log.info(f"\n  {'Metric':<30s} {'Value':>10s}")
    log.info(f"  {'─' * 42}")
    log.info(f"  {'Route Accuracy':<30s} {route_acc:>9.1f}%")
    log.info(f"  {'Expert Selection Accuracy':<30s} {expert_acc:>9.1f}%")
    log.info(f"  {'False Trigger Rate':<30s} {false_trigger_rate:>9.1f}%")
    log.info(f"  {'Miss Rate':<30s} {miss_rate:>9.1f}%")
    log.info(f"  {'GT Hit Rate':<30s} {gt_hit_rate:>9.1f}%")
    log.info(f"  {'E2E Success Rate':<30s} {e2e_rate:>9.1f}%")

    log.info(f"\n  Per-Expert Breakdown:")
    log.info(f"  {'Expert':<12s} {'Total':>6s} {'Route%':>8s} {'Select%':>8s} {'GT%':>8s}")
    log.info(f"  {'─' * 44}")
    for expert in ["math", "info", "translate", "none"]:
        d = per_expert[expert]
        if d["total"] == 0:
            continue
        r_pct = d["route_ok"] / d["total"] * 100
        e_pct = d["expert_ok"] / d["total"] * 100 if expert != "none" else 0
        g_pct = d["gt_ok"] / d["total"] * 100 if expert != "none" else 0
        log.info(f"  {expert:<12s} {d['total']:>6d} {r_pct:>7.1f}% {e_pct:>7.1f}% {g_pct:>7.1f}%")

    # 保存详细结果
    output_path = args.output or os.path.join(log_dir, f"eval_results_{timestamp}.json")
    report = {
        "model_path": args.model_path,
        "num_samples": stats["total"],
        "elapsed_seconds": elapsed,
        "metrics": {
            "route_accuracy": route_acc,
            "expert_selection_accuracy": expert_acc,
            "false_trigger_rate": false_trigger_rate,
            "miss_rate": miss_rate,
            "gt_hit_rate": gt_hit_rate,
            "e2e_success_rate": e2e_rate,
        },
        "per_expert": {k: dict(v) for k, v in per_expert.items()},
        "details": results_log,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=SafeJSONEncoder)
    log.info(f"\n  Log saved to: {log_path}")
    log.info(f"  Results saved to: {output_path}")
    log.info(f"{'=' * 80}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handoff RL Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Checkpoint path (e.g. ./checkpoints_qwen_handoff_v2/best)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of eval samples (max 50)")
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Lower temperature for more deterministic eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--shuffle", action="store_true", default=False)
    args = parser.parse_args()

    run_eval(args)
