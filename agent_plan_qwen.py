"""Plan-then-Execute Multi-Agent RL 训练 — Qwen2.5-1.5B-Instruct

在 agent_handoff_qwen.py（单步 Handoff）基础上扩展 Plan-then-Execute 能力。
Router Agent 输出 execute_plan 工具调用 → 按依赖拓扑序执行多个 Expert → Synthesize。
同时向后兼容单步 delegate_to_* 路由。

核心扩展:
    1. execute_plan 工具: Router 输出结构化执行计划（steps + 依赖关系）
    2. 多步 Rollout: 按拓扑序依次调用 Expert，前序结果作为后序 context
    3. 分步 Reward: plan 结构奖励 + 分步 GT 验证 + 最终答案奖励
    4. 数据集: 兼容 plan_execute.jsonl（含 plan_gt, step_gts, level）

使用方式:
    # 单卡训练（1 epoch 测试）
    python agent_plan_qwen.py --mode train --epochs 1 \\
        --data_path ./dataset/plan_execute.jsonl \\
        --model_path Qwen/Qwen2.5-1.5B-Instruct

    # 从 Handoff checkpoint 继续训练
    python agent_plan_qwen.py --mode train --epochs 1 \\
        --data_path ./dataset/plan_execute.jsonl \\
        --model_path ./checkpoints_qwen_handoff/best
"""
import os
import sys
import re
import gc
import json
import math
import random
import argparse
import warnings
import time as _time
from contextlib import nullcontext
from typing import List, Optional, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---- 复用 agent_handoff_qwen.py 中的基础设施 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_handoff_qwen import (
    # 通用工具
    is_main_process, Logger, init_distributed_mode, setup_seed,
    SkipBatchSampler, QwenCheckpointManager,
    # 模型
    init_model_qwen, compute_per_token_logps_hf,
    TorchRolloutEngineHF,
    # 工具定义与执行
    parse_tool_calls, EXPERT_CONFIG,
)
from trainer.train_agent import (
    execute_tool, rep_penalty, validate_gt_in_text, CHECK_ARGS,
)


# ==============================================================================
#  Plan-then-Execute 扩展: System Prompt + Tools
# ==============================================================================

PLAN_ROUTER_SYSTEM_PROMPT = (
    "你是一个任务路由器（Router Agent）。你的职责是分析用户请求，"
    "决定是自己直接回答，还是制定执行计划委托给合适的专家 Agent。\n"
    "专家列表：\n"
    "- delegate_to_math_agent: 数学计算、公式求解、单位换算\n"
    "- delegate_to_info_agent: 天气查询、时间查询、汇率查询\n"
    "- delegate_to_translate_agent: 文本翻译\n\n"
    "对于需要多步处理的复杂问题，使用 execute_plan 工具制定分步计划。\n"
    "计划中的每个步骤需指定委托的专家、任务描述和依赖的前序步骤。\n"
    "对于简单的单步问题，直接使用 delegate_to_* 工具。\n"
    "如果问题不需要工具，直接回答即可。"
)

# 在原有 delegate_to_* 之上新增 execute_plan 工具
PLAN_ROUTER_TOOLS = [
    {"type": "function", "function": {
        "name": "delegate_to_math_agent",
        "description": "将数学计算、单位换算任务委托给数学专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要数学专家处理的任务描述"}
        }, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "delegate_to_info_agent",
        "description": "将信息查询任务（天气、时间、汇率）委托给信息查询专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要信息专家处理的任务描述"}
        }, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "delegate_to_translate_agent",
        "description": "将翻译任务委托给翻译专家 Agent",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "需要翻译专家处理的任务描述"}
        }, "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "execute_plan",
        "description": "制定并执行多步任务计划。每个步骤指定委托的专家和依赖关系，系统按拓扑序执行。",
        "parameters": {"type": "object", "properties": {
            "plan": {
                "type": "array",
                "description": "执行步骤列表",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "description": "步骤编号（从1开始）"},
                        "delegate": {"type": "string",
                                     "description": "委托的专家名称"},
                        "task": {"type": "string", "description": "该步骤的任务描述"},
                        "depends_on": {
                            "type": "array", "items": {"type": "integer"},
                            "description": "依赖的前序步骤编号列表（空=无依赖）"},
                    },
                    "required": ["step", "delegate", "task", "depends_on"],
                },
            }
        }, "required": ["plan"]}}},
]


# ==============================================================================
#  数据集（扩展支持 plan_execute.jsonl 格式）
# ==============================================================================

class PlanDataset(Dataset):
    """Plan-then-Execute 训练数据集

    兼容两种格式:
    - plan_execute.jsonl: 含 plan_gt, step_gts, level, dependency_type
    - agent_handoff.jsonl: 含 query, gt, expert（单步兼容）
    """

    def __init__(self, data_path, level_filter=None):
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if level_filter is not None:
                        if item.get("level", 0) not in level_filter:
                            continue
                    self.data.append(item)
        Logger(f"Loaded {len(self.data)} examples from {data_path}"
               + (f" (levels={level_filter})" if level_filter else ""))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item.get("query", ""),
            "gt": item.get("gt", []),
            "plan_gt": item.get("plan_gt", []),
            "step_gts": item.get("step_gts", []),
            "level": item.get("level", 0),
            "experts_needed": item.get("experts_needed", []),
            "dependency_type": item.get("dependency_type", "sequential"),
            "num_steps": item.get("num_steps", 1),
            "gt_branch": item.get("gt_branch", None),
            # 向后兼容单步 handoff 字段
            "needs_tool": item.get("needs_tool", len(item.get("plan_gt", [])) > 0),
            "expert": item.get("expert", "none"),
        }


# ==============================================================================
#  Plan Rollout: 解析 execute_plan 并按拓扑序执行多个 Expert
# ==============================================================================

def _topo_sort(steps):
    """对 plan steps 做拓扑排序，返回执行顺序"""
    graph = {}
    in_degree = {}
    for s in steps:
        sid = s["step"]
        graph[sid] = s.get("depends_on", [])
        in_degree[sid] = len(graph[sid])

    queue = [sid for sid in in_degree if in_degree[sid] == 0]
    order = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for sid in in_degree:
            if node in graph.get(sid, []):
                in_degree[sid] -= 1
                if in_degree[sid] == 0:
                    queue.append(sid)
    return order


def _run_expert_sub_rollout(rollout_engine, tokenizer, expert_name, task_description,
                            context_from_deps=None, max_new_tokens=384, device="cuda"):
    """执行单个 Expert Agent 的子 rollout"""
    expert_config = EXPERT_CONFIG.get(expert_name)
    if not expert_config:
        return {
            "expert_prompt_ids": [], "expert_response_ids": [],
            "expert_response_mask": [], "expert_old_logps": [],
            "expert_final_answer": f"未知专家: {expert_name}", "all_outputs": [],
        }

    full_task = task_description
    if context_from_deps:
        full_task = f"{task_description}\n\n前序步骤结果：\n{context_from_deps}"

    expert_messages = [
        {"role": "system", "content": expert_config["system_prompt"]},
        {"role": "user", "content": full_task},
    ]
    expert_tools = expert_config["tools"]

    expert_response_ids = []
    expert_response_mask = []
    expert_old_logps = []
    expert_prompt_ids = None
    expert_final_answer = ""
    all_outputs = []
    max_tool_turns = 2  # 模拟工具确定性输出，2轮足够

    for turn in range(max_tool_turns):
        expert_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False, add_generation_prompt=True,
            tools=expert_tools,
        )
        expert_inputs = tokenizer(
            expert_context, return_tensors="pt", add_special_tokens=False
        ).to(device)
        if expert_prompt_ids is None:
            expert_prompt_ids = expert_inputs["input_ids"][0].tolist()

        expert_result = rollout_engine.rollout(
            prompt_ids=expert_inputs["input_ids"],
            attention_mask=expert_inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = expert_result.completion_ids[0].tolist()
        new_logps = expert_result.per_token_logps[0].tolist()

        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = expert_result.completions[0]

        all_outputs.append(new_text)
        expert_response_ids.extend(new_ids)
        expert_response_mask.extend([1] * len(new_ids))
        expert_old_logps.extend(new_logps)

        calls = parse_tool_calls(new_text)
        if not calls:
            expert_final_answer = new_text
            break

        expert_messages.append({"role": "assistant", "content": new_text})

        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False)
                          if result else '{"error": "tool not found"}')[:2048]
            expert_messages.append({"role": "tool", "content": result_str})

        # 工具模板 token（mask=0，不参与 loss 计算）
        is_last_turn = (turn == max_tool_turns - 1)
        observe_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False,
            add_generation_prompt=not is_last_turn,
            tools=expert_tools,
        )
        observe_ids = tokenizer(
            observe_context, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
        current_len = len(expert_prompt_ids) + len(expert_response_ids)
        obs_delta = observe_ids[current_len:]
        expert_response_ids.extend(obs_delta)
        expert_response_mask.extend([0] * len(obs_delta))
        expert_old_logps.extend([0.0] * len(obs_delta))

    if not expert_final_answer:
        expert_final_answer = all_outputs[-1] if all_outputs else ""

    return {
        "expert_prompt_ids": expert_prompt_ids or [],
        "expert_response_ids": expert_response_ids,
        "expert_response_mask": expert_response_mask,
        "expert_old_logps": expert_old_logps,
        "expert_final_answer": expert_final_answer,
        "all_outputs": all_outputs,
    }


def plan_rollout_single(rollout_engine, tokenizer, user_query,
                        max_new_tokens=384, thinking_ratio=0.0, device="cuda"):
    """Plan-then-Execute 版本的 Rollout（单候选版本，用于 demo）

    Phase 1: RouterAgent 决策 → 输出 execute_plan 或 delegate_to_* 或直接回答
    Phase 2: 若 execute_plan → 按拓扑序执行多个 Expert；若 delegate_to_* → 单步执行
    Phase 3: RouterAgent 整合所有 Expert 结果
    """
    # ===== Phase 1: RouterAgent 决策 =====
    router_messages = [
        {"role": "system", "content": PLAN_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    router_context = tokenizer.apply_chat_template(
        router_messages, tokenize=False, add_generation_prompt=True,
        tools=PLAN_ROUTER_TOOLS,
    )
    router_inputs = tokenizer(
        router_context, return_tensors="pt", add_special_tokens=False
    ).to(device)
    router_prompt_ids = router_inputs["input_ids"][0].tolist()

    router_result = rollout_engine.rollout(
        prompt_ids=router_inputs["input_ids"],
        attention_mask=router_inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
    )
    router_gen_ids = router_result.completion_ids[0].tolist()
    router_gen_logps = router_result.per_token_logps[0].tolist()

    pairs = [(t, lp) for t, lp in zip(router_gen_ids, router_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    router_gen_ids = [t for t, _ in pairs]
    router_gen_logps = [lp for _, lp in pairs]
    router_text = router_result.completions[0]

    # 复用 Phase 2/3 逻辑
    return _continue_from_router(
        rollout_engine, tokenizer, user_query,
        router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
        max_new_tokens=max_new_tokens, device=device,
    )


def _continue_from_router(rollout_engine, tokenizer, user_query,
                           router_prompt_ids, router_gen_ids, router_gen_logps,
                           router_text, max_new_tokens=384, device="cuda"):
    """Phase 2/3: 接收 Router 的输出，执行 Expert 和 Synthesize

    从已生成的 Router 结果继续：解析工具调用 → Expert 执行 → 综合回答。
    供 plan_rollout_single（demo）和 plan_train_step（批量训练）共用。
    """
    all_outputs = [router_text]

    # ===== 解析 Router 输出 =====
    tool_calls = parse_tool_calls(router_text)

    # 情况 1: 无工具调用（直接回答）
    if not tool_calls:
        return _make_result(
            router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
            handoff_occurred=False, plan_executed=False, all_outputs=all_outputs,
        )

    plan_call = None
    single_call = None
    for call in tool_calls:
        name = call.get("name", "")
        if name == "execute_plan":
            plan_call = call
            break
        elif name.startswith("delegate_to_") and name in EXPERT_CONFIG:
            single_call = call

    # 情况 2: 单步 delegate（向后兼容）
    if plan_call is None and single_call is not None:
        delegated_expert = single_call["name"]
        task_args = single_call.get("arguments", {})
        if isinstance(task_args, str):
            try:
                task_args = json.loads(task_args)
            except (json.JSONDecodeError, ValueError):
                task_args = {}
        delegated_task = task_args.get("task", user_query)

        expert_result = _run_expert_sub_rollout(
            rollout_engine, tokenizer, delegated_expert, delegated_task,
            max_new_tokens=max_new_tokens, device=device,
        )
        all_outputs.extend(expert_result["all_outputs"])

        synth_data = _run_synthesize(
            rollout_engine, tokenizer, user_query, router_text,
            {1: expert_result["expert_final_answer"]},
            max_new_tokens=max_new_tokens // 2, device=device,
        )
        all_outputs.append(synth_data["synth_text"])

        return _make_result(
            router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
            handoff_occurred=True, plan_executed=False,
            expert_data=[expert_result], synth_data=synth_data,
            delegated_expert=delegated_expert, delegate_task=delegated_task,
            all_outputs=all_outputs,
            plan_info={"type": "single", "steps": [
                {"step": 1, "delegate": delegated_expert, "task": delegated_task}]},
        )

    # 情况 3: execute_plan（多步计划）
    if plan_call is not None:
        plan_args = plan_call.get("arguments", {})
        if isinstance(plan_args, str):
            try:
                plan_args = json.loads(plan_args)
            except (json.JSONDecodeError, ValueError):
                plan_args = {}
        plan_steps = plan_args.get("plan", [])

        if not plan_steps:
            return _make_result(
                router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
                handoff_occurred=False, plan_executed=False, all_outputs=all_outputs,
            )

        exec_order = _topo_sort(plan_steps)
        step_map = {s["step"]: s for s in plan_steps}
        step_results = {}
        expert_data_list = []

        for step_id in exec_order:
            step = step_map.get(step_id)
            if not step:
                continue

            delegate_name = step.get("delegate", "")
            task_desc = step.get("task", "")
            deps = step.get("depends_on", [])

            dep_context = None
            if deps:
                dep_parts = []
                for d in deps:
                    if d in step_results:
                        dep_parts.append(f"步骤{d}结果: {step_results[d]}")
                if dep_parts:
                    dep_context = "\n".join(dep_parts)

            expert_result = _run_expert_sub_rollout(
                rollout_engine, tokenizer, delegate_name, task_desc,
                context_from_deps=dep_context,
                max_new_tokens=max_new_tokens, device=device,
            )
            step_results[step_id] = expert_result["expert_final_answer"]
            expert_result["step_id"] = step_id
            expert_data_list.append(expert_result)
            all_outputs.extend(expert_result["all_outputs"])

        synth_data = _run_synthesize(
            rollout_engine, tokenizer, user_query, router_text,
            step_results, max_new_tokens=max_new_tokens // 2, device=device,
        )
        all_outputs.append(synth_data["synth_text"])

        return _make_result(
            router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
            handoff_occurred=True, plan_executed=True,
            expert_data=expert_data_list, synth_data=synth_data,
            all_outputs=all_outputs,
            plan_info={
                "type": "multi",
                "steps": plan_steps,
                "exec_order": exec_order,
                "step_results": step_results,
            },
        )

    # 情况 4: 工具调用但不识别
    return _make_result(
        router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
        handoff_occurred=False, plan_executed=False, all_outputs=all_outputs,
    )


def _run_synthesize(rollout_engine, tokenizer, user_query, router_text,
                    step_results, max_new_tokens=192, device="cuda"):
    """RouterAgent 整合阶段"""
    results_summary = {}
    for step_id, answer in step_results.items():
        results_summary[f"step_{step_id}"] = answer

    synth_messages = [
        {"role": "system", "content": PLAN_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": router_text},
        {"role": "tool", "content": json.dumps(
            {"expert_results": results_summary}, ensure_ascii=False)},
    ]

    synth_context = tokenizer.apply_chat_template(
        synth_messages, tokenize=False, add_generation_prompt=True,
        tools=PLAN_ROUTER_TOOLS,
    )
    synth_inputs = tokenizer(
        synth_context, return_tensors="pt", add_special_tokens=False
    ).to(device)
    synth_prompt_ids = synth_inputs["input_ids"][0].tolist()

    synth_result = rollout_engine.rollout(
        prompt_ids=synth_inputs["input_ids"],
        attention_mask=synth_inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    )
    synth_gen_ids = synth_result.completion_ids[0].tolist()
    synth_gen_logps = synth_result.per_token_logps[0].tolist()

    pairs = [(t, lp) for t, lp in zip(synth_gen_ids, synth_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    synth_gen_ids = [t for t, _ in pairs]
    synth_gen_logps = [lp for _, lp in pairs]
    synth_text = synth_result.completions[0]

    return {
        "synth_prompt_ids": synth_prompt_ids,
        "synth_response_ids": synth_gen_ids,
        "synth_response_mask": [1] * len(synth_gen_ids),
        "synth_old_logps": synth_gen_logps,
        "synth_text": synth_text,
    }


def _make_result(router_prompt_ids, router_gen_ids, router_gen_logps, router_text,
                 handoff_occurred, plan_executed, all_outputs,
                 expert_data=None, synth_data=None,
                 delegated_expert=None, delegate_task="", plan_info=None):
    """构建统一的 rollout 结果字典"""
    merged_expert_prompt_ids = []
    merged_expert_response_ids = []
    merged_expert_response_mask = []
    merged_expert_old_logps = []

    if expert_data:
        for i, ed in enumerate(expert_data):
            if i == 0:
                merged_expert_prompt_ids = ed["expert_prompt_ids"]
            else:
                # 后续 Expert 的 prompt 作为 mask=0 的上下文
                merged_expert_response_ids.extend(ed["expert_prompt_ids"])
                merged_expert_response_mask.extend([0] * len(ed["expert_prompt_ids"]))
                merged_expert_old_logps.extend([0.0] * len(ed["expert_prompt_ids"]))

            merged_expert_response_ids.extend(ed["expert_response_ids"])
            merged_expert_response_mask.extend(ed["expert_response_mask"])
            merged_expert_old_logps.extend(ed["expert_old_logps"])

    return {
        "final_answer": synth_data["synth_text"] if synth_data else router_text,
        "router_prompt_ids": router_prompt_ids,
        "router_response_ids": router_gen_ids,
        "router_response_mask": [1] * len(router_gen_ids),
        "router_old_logps": router_gen_logps,
        "expert_prompt_ids": merged_expert_prompt_ids,
        "expert_response_ids": merged_expert_response_ids,
        "expert_response_mask": merged_expert_response_mask,
        "expert_old_logps": merged_expert_old_logps,
        "synth_prompt_ids": synth_data["synth_prompt_ids"] if synth_data else [],
        "synth_response_ids": synth_data["synth_response_ids"] if synth_data else [],
        "synth_response_mask": synth_data["synth_response_mask"] if synth_data else [],
        "synth_old_logps": synth_data["synth_old_logps"] if synth_data else [],
        "handoff_occurred": handoff_occurred,
        "plan_executed": plan_executed,
        "delegated_expert": delegated_expert,
        "delegate_task": delegate_task,
        "all_outputs": all_outputs,
        "plan_info": plan_info,
    }


# ==============================================================================
#  Plan-aware Reward 计算
# ==============================================================================

def calculate_plan_rewards(results_batch, gt_batch, plan_gt_batch, step_gts_batch,
                           level_batch, num_gen, device="cuda"):
    """Plan-then-Execute 场景的奖励计算

    奖励组成:
    1. Plan 结构奖励: plan 步骤数/专家/依赖关系与 GT 的匹配度 (+1.0)
    2. 路由正确性: 多步问题是否正确使用 execute_plan (+0.5)
    3. 分步 GT 验证: 各步骤 Expert 结果是否包含正确答案 (+1.5)
    4. 最终答案 GT 验证: 综合回答是否包含所有关键信息 (+2.0)
    5. 重复惩罚
    """
    total = len(results_batch)
    rewards = torch.zeros(total, device=device)
    router_rewards = torch.zeros(total, device=device)
    expert_rewards = torch.zeros(total, device=device)

    for idx, result in enumerate(results_batch):
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx] if sample_idx < len(gt_batch) else []
        plan_gt = plan_gt_batch[sample_idx] if sample_idx < len(plan_gt_batch) else []
        step_gts = step_gts_batch[sample_idx] if sample_idx < len(step_gts_batch) else []
        level = level_batch[sample_idx] if sample_idx < len(level_batch) else 0

        r_reward = 0.0
        e_reward = 0.0

        handoff_occurred = result["handoff_occurred"]
        plan_executed = result.get("plan_executed", False)
        final_answer = result["final_answer"]
        plan_info = result.get("plan_info")

        # 去除思考部分
        answer_text = final_answer
        if "<think>" in answer_text and "</think>" in answer_text:
            answer_text = answer_text.split("</think>")[-1].strip()

        is_multi_step = len(plan_gt) > 1  # GT 有多步 → 应该 execute_plan
        is_single_step = len(plan_gt) == 1  # GT 仅 1 步 → delegate_to_*
        is_direct = len(plan_gt) == 0  # GT 无步骤 → 直接回答

        # ---- 1. 路由正确性 ----
        if is_multi_step and plan_executed:
            r_reward += 0.5  # 正确使用 execute_plan
        elif is_multi_step and handoff_occurred and not plan_executed:
            r_reward += 0.1  # 用了 delegate_to_ 但至少 handoff 了
        elif is_multi_step and not handoff_occurred:
            r_reward -= 0.5  # 多步问题却不路由
        elif is_single_step and handoff_occurred:
            r_reward += 0.5  # 单步正确 handoff
        elif is_single_step and not handoff_occurred:
            r_reward -= 0.3
        elif is_direct and not handoff_occurred:
            r_reward += 0.3  # 直接回答正确
        elif is_direct and handoff_occurred:
            r_reward -= 0.3

        # ---- 2. Plan 结构奖励（仅对 execute_plan）----
        if plan_executed and plan_info and plan_info.get("type") == "multi":
            generated_steps = plan_info.get("steps", [])
            if plan_gt:
                # 2a. 步骤数匹配
                gen_n = len(generated_steps)
                gt_n = len(plan_gt)
                if gen_n == gt_n:
                    r_reward += 0.3
                elif abs(gen_n - gt_n) == 1:
                    r_reward += 0.1
                else:
                    r_reward -= 0.2

                # 2b. 专家选择匹配
                gt_experts = [s.get("expert", s.get("delegate", "")) for s in plan_gt]
                gen_experts = [s.get("delegate", "") for s in generated_steps]
                # 归一化专家名称
                expert_name_map = {
                    "math": "delegate_to_math_agent",
                    "info": "delegate_to_info_agent",
                    "translate": "delegate_to_translate_agent",
                }
                gt_experts_norm = [expert_name_map.get(e, e) for e in gt_experts]
                expert_match = sum(1 for a, b in zip(gt_experts_norm, gen_experts) if a == b)
                expert_ratio = expert_match / max(len(gt_experts_norm), 1)
                r_reward += 0.4 * expert_ratio

                # 2c. 依赖关系匹配
                gt_deps = {s.get("step", i + 1): s.get("depends_on", []) for i, s in enumerate(plan_gt)}
                gen_deps = {s.get("step", i + 1): s.get("depends_on", []) for i, s in enumerate(generated_steps)}
                dep_match = 0
                dep_total = 0
                for sid in gt_deps:
                    if sid in gen_deps:
                        dep_total += 1
                        if set(gt_deps[sid]) == set(gen_deps.get(sid, [])):
                            dep_match += 1
                    else:
                        dep_total += 1
                if dep_total > 0:
                    r_reward += 0.3 * (dep_match / dep_total)

        # ---- 3. 分步 GT 验证 ----
        if plan_info and step_gts:
            step_results_map = plan_info.get("step_results", {})
            step_score = 0.0
            step_count = 0
            for i, step_gt in enumerate(step_gts):
                step_id = i + 1
                if step_id in step_results_map and step_gt:
                    step_answer = step_results_map[step_id]
                    verified = validate_gt_in_text(step_answer, step_gt)
                    if verified:
                        step_score += len(verified) / len(step_gt)
                    step_count += 1
            if step_count > 0:
                avg_step_score = step_score / step_count
                e_reward += 1.5 * avg_step_score

        # ---- 4. 最终答案 GT 验证（核心奖励）----
        if gt:
            verified = validate_gt_in_text(answer_text, gt)
            gt_score = 2.0 * len(verified) / len(gt)
            if handoff_occurred:
                e_reward += gt_score * 0.6
                r_reward += gt_score * 0.4
            else:
                r_reward += gt_score
        elif not handoff_occurred:
            r_reward += 0.3 if 5 <= len(answer_text) <= 800 else -0.3

        # ---- 5. Thinking 格式奖励 ----
        for output in result["all_outputs"]:
            if "<think>" in output and "</think>" in output:
                think_part = output.split("<think>")[1].split("</think>")[0]
                if 20 <= len(think_part.strip()) <= 300:
                    r_reward += 0.2
                    break

        # ---- 6. 重复惩罚 ----
        penalty = rep_penalty(answer_text)
        r_reward -= penalty * 0.5
        e_reward -= penalty * 0.5

        # ---- 7. Level 加权（更难的样本更高的 reward 上限）----
        level_scale = 1.0 + 0.1 * level  # L0=1.0, L1=1.1, L2=1.2, L3=1.3, L4=1.4

        # 汇总
        total_reward = (r_reward + e_reward) * level_scale
        rewards[idx] = max(min(total_reward, 5.0), -3.0)
        router_rewards[idx] = max(min(r_reward * level_scale, 3.0), -2.0)
        expert_rewards[idx] = max(min(e_reward * level_scale, 3.0), -2.0)

    return rewards, router_rewards, expert_rewards


# ==============================================================================
#  训练步（Plan 版本）
# ==============================================================================

def plan_train_step(model, ref_model, rollout_engine, tokenizer, batch, args,
                    optimizer, scheduler, autocast_ctx, step, reward_model=None):
    """Plan-then-Execute 训练步

    结构与 handoff_train_step 一致（三段拼接 + GRPO/CISPO loss），
    但使用 plan_rollout_single 和 calculate_plan_rewards。
    """
    queries = batch["queries"]
    gt_batch = batch["gt"]
    plan_gt_batch = batch.get("plan_gt", [[] for _ in queries])
    step_gts_batch = batch.get("step_gts", [[] for _ in queries])
    level_batch = batch.get("level", [0 for _ in queries])

    # ===== Rollout（优化: Router 阶段批量生成，省去重复 prefill）=====
    # 关键优化: rollout 是纯推理，关闭 gradient_checkpointing 以启用 KV cache
    # gradient_checkpointing 会强制 use_cache=False，导致 generate 每步重算全部 attention
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if args.gradient_checkpointing:
        raw_model.gradient_checkpointing_disable()
        raw_model.config.use_cache = True

    results_batch = []
    with torch.no_grad():
        for i, query in enumerate(queries):
            # Phase 1 批量: 同一条 query 一次性生成 N 个 Router 候选
            router_messages = [
                {"role": "system", "content": PLAN_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            router_context = tokenizer.apply_chat_template(
                router_messages, tokenize=False, add_generation_prompt=True,
                tools=PLAN_ROUTER_TOOLS,
            )
            router_inputs = tokenizer(
                router_context, return_tensors="pt", add_special_tokens=False
            ).to(args.device)
            router_prompt_ids = router_inputs["input_ids"][0].tolist()

            router_result = rollout_engine.rollout(
                prompt_ids=router_inputs["input_ids"],
                attention_mask=router_inputs["attention_mask"],
                num_generations=args.num_generations,
                max_new_tokens=args.max_gen_len,
                temperature=0.8,
            )

            # 拆分 N 个候选，各自走 Phase 2/3
            for gen_idx in range(args.num_generations):
                gen_ids = router_result.completion_ids[gen_idx].tolist()
                gen_logps = router_result.per_token_logps[gen_idx].tolist()
                gen_text = router_result.completions[gen_idx]

                pairs = [(t, lp) for t, lp in zip(gen_ids, gen_logps)
                         if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
                gen_ids = [t for t, _ in pairs]
                gen_logps = [lp for _, lp in pairs]

                result = _continue_from_router(
                    rollout_engine, tokenizer, query,
                    router_prompt_ids, gen_ids, gen_logps, gen_text,
                    max_new_tokens=args.max_gen_len, device=args.device,
                )
                result["original_query"] = query
                results_batch.append(result)

    # 恢复 gradient_checkpointing（后续 policy forward + backward 需要）
    if args.gradient_checkpointing:
        raw_model.gradient_checkpointing_enable()
        raw_model.config.use_cache = False

    torch.cuda.empty_cache()
    gc.collect()

    # ===== Reward =====
    rewards, router_rewards, expert_rewards = calculate_plan_rewards(
        results_batch, gt_batch, plan_gt_batch, step_gts_batch,
        level_batch, args.num_generations, device=args.device,
    )

    # ===== 打包序列（三段拼接: router + expert(s) + synth）=====
    packed_samples = []
    segment_info = []

    for result in results_batch:
        prompt_ids = result["router_prompt_ids"]
        response_ids = list(result["router_response_ids"])
        response_mask = list(result["router_response_mask"])
        old_logps = list(result["router_old_logps"])

        router_len = len(response_ids)

        expert_start = router_len
        if result["handoff_occurred"] and result["expert_response_ids"]:
            expert_prompt = result["expert_prompt_ids"]
            response_ids.extend(expert_prompt)
            response_mask.extend([0] * len(expert_prompt))
            old_logps.extend([0.0] * len(expert_prompt))

            response_ids.extend(result["expert_response_ids"])
            response_mask.extend(result["expert_response_mask"])
            old_logps.extend(result["expert_old_logps"])
        expert_end = len(response_ids)

        synth_start = expert_end
        if result["handoff_occurred"] and result["synth_response_ids"]:
            synth_prompt = result["synth_prompt_ids"]
            response_ids.extend(synth_prompt)
            response_mask.extend([0] * len(synth_prompt))
            old_logps.extend([0.0] * len(synth_prompt))

            response_ids.extend(result["synth_response_ids"])
            response_mask.extend(result["synth_response_mask"])
            old_logps.extend(result["synth_old_logps"])
        synth_end = len(response_ids)

        ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + response_mask
        full_old_logps = [0.0] * max(len(prompt_ids) - 1, 0) + old_logps

        if len(ids) > args.max_total_len:
            overflow = len(ids) - args.max_total_len
            ids = ids[overflow:]
            mask = mask[overflow:]
            full_old_logps = full_old_logps[-(len(ids) - 1):]

        prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
        packed_samples.append((ids, mask, prompt_len, full_old_logps))

        segment_info.append({
            "router_range": (0, router_len),
            "expert_range": (expert_start, expert_end),
            "synth_range": (synth_start, synth_end),
        })

    # ===== Padding =====
    seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
    max_len = seq_lens.max().item()
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples],
        device=args.device,
    )
    full_response_masks = torch.tensor(
        [mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
        device=args.device, dtype=torch.float32,
    )
    old_per_token_logps = torch.tensor(
        [lps + [0.0] * ((max_len - 1) - len(lps)) for _, _, _, lps in packed_samples],
        device=args.device, dtype=torch.float32,
    )
    full_mask = (input_ids != tokenizer.pad_token_id).long()

    # ===== 策略 Loss =====
    with autocast_ctx:
        res = model(input_ids, attention_mask=full_mask)
        logits = res.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        T = logits.size(1)
        _chunk = 128
        _logps_chunks = []
        for _s in range(0, T, _chunk):
            _e = min(_s + _chunk, T)
            _chunk_logits = logits[:, _s:_e, :]
            _lp = F.log_softmax(_chunk_logits, dim=-1)
            _gathered = torch.gather(
                _lp, 2, target_ids[:, _s:_e].unsqueeze(-1)
            ).squeeze(-1)
            _logps_chunks.append(_gathered)
        per_token_logps = torch.cat(_logps_chunks, dim=1)
        del logits, _logps_chunks

    # ref_model 可能在不同 GPU 上，需要搬运 input
    ref_device = next(ref_model.parameters()).device
    with torch.no_grad():
        ref_per_token_logps = compute_per_token_logps_hf(
            ref_model,
            input_ids.to(ref_device),
            input_ids.size(1) - 1,
            attention_mask=full_mask.to(ref_device),
        ).to(input_ids.device)

    # Completion mask
    completion_mask = full_response_masks[:, 1:]
    is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
    eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1,
                         device=args.device, dtype=torch.long)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
    completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
    token_counts = completion_mask.sum(dim=1)
    valid_rows = token_counts > 0

    # ===== Credit Assignment =====
    grouped_rewards = rewards.view(-1, args.num_generations)
    grouped_router = router_rewards.view(-1, args.num_generations)
    grouped_expert = expert_rewards.view(-1, args.num_generations)

    def normalize_group(grouped):
        mean = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        return (grouped.view(-1) - mean) / (std + 1e-4)

    total_advantages = normalize_group(grouped_rewards)
    router_advantages = normalize_group(grouped_router)
    expert_advantages = normalize_group(grouped_expert)

    B = completion_mask.size(0)
    T_mask = completion_mask.size(1)
    per_token_advantages = torch.zeros(B, T_mask, device=args.device)

    for i in range(B):
        per_token_advantages[i] = total_advantages[i]
        if results_batch[i]["handoff_occurred"]:
            per_token_advantages[i] = (router_advantages[i] + expert_advantages[i]) / 2

    # KL + 策略 loss
    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps - old_per_token_logps)

    if args.loss_type == "cispo":
        clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
        per_token_loss = -(clamped_ratio * per_token_advantages * per_token_logps
                           - args.beta * per_token_kl)
    else:
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss1 = ratio * per_token_advantages
        per_token_loss2 = clipped_ratio * per_token_advantages
        per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                           - args.beta * per_token_kl)

    policy_loss = (
        ((per_token_loss * completion_mask).sum(dim=1)[valid_rows]
         / token_counts[valid_rows].clamp(min=1)).mean()
        if valid_rows.any() else per_token_loss.sum() * 0.0
    )

    loss = policy_loss / args.accumulation_steps
    loss.backward()

    if step % args.accumulation_steps == 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    del per_token_logps, ref_per_token_logps, per_token_advantages

    plan_rate = sum(1 for r in results_batch if r.get("plan_executed", False)) / max(len(results_batch), 1)

    return {
        "loss": loss.item() * args.accumulation_steps,
        "reward_mean": rewards.mean().item(),
        "router_reward_mean": router_rewards.mean().item(),
        "expert_reward_mean": expert_rewards.mean().item(),
        "handoff_rate": sum(1 for r in results_batch if r["handoff_occurred"]) / len(results_batch),
        "plan_rate": plan_rate,
        "kl": (kl_div * completion_mask).sum().item() / max(token_counts.sum().item(), 1),
    }


# ==============================================================================
#  完整训练流程
# ==============================================================================

def run_plan_training(args):
    """Qwen2.5-1.5B-Instruct Plan-then-Execute 训练（支持 DDP）"""

    # 初始化分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if device_type == "cpu"
                    else torch.cuda.amp.autocast(dtype=dtype))

    # CheckpointManager
    ckpt_mgr = QwenCheckpointManager(
        save_dir=args.save_dir,
        max_keep=args.max_keep,
        track_metric="reward",
        metric_mode="max",
    )

    ckp_data = ckpt_mgr.load(resume_mode=args.resume_mode) if args.from_resume else None

    # 模型初始化
    model_path = ckp_data["model_path"] if ckp_data else args.model_path
    model, tokenizer = init_model_qwen(
        model_path, device=args.device, dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ref_model offload 到第二张 GPU（若可用），避免与 policy 争抢显存
    ref_device = "cuda:1" if torch.cuda.device_count() > 1 else args.device
    ref_model, _ = init_model_qwen(
        args.model_path, device=ref_device, dtype=dtype,
        gradient_checkpointing=False,
    )
    ref_model = ref_model.eval().requires_grad_(False)
    Logger(f"  Ref model on: {ref_device}")

    # wandb
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_id = ckp_data.get("wandb_id") if ckp_data else None
            resume = "must" if wandb_id else None
            wandb.init(
                project=args.wandb_project,
                name=f"Qwen1.5B-Plan-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
                id=wandb_id, resume=resume,
            )
        except ImportError:
            Logger("Warning: swanlab not installed, wandb disabled")
            wandb = None

    # Rollout 引擎
    rollout_engine = TorchRolloutEngineHF(
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    # 数据集
    level_filter = None
    if args.curriculum_phase is not None:
        # 课程学习：Phase 0=[L0,L1], Phase 1=[L0-L2], Phase 2=[L0-L3], Phase 3=[L0-L4]
        phase_levels = {
            0: [0, 1],
            1: [0, 1, 2],
            2: [0, 1, 2, 3],
            3: [0, 1, 2, 3, 4],
        }
        level_filter = phase_levels.get(args.curriculum_phase, None)

    train_ds = PlanDataset(args.data_path, level_filter=level_filter)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    def collate_fn(batch):
        return {
            "queries": [b["query"] for b in batch],
            "gt": [b["gt"] for b in batch],
            "plan_gt": [b["plan_gt"] for b in batch],
            "step_gts": [b["step_gts"] for b in batch],
            "level": [b["level"] for b in batch],
            "needs_tool": [b["needs_tool"] for b in batch],
            "expert": [b["expert"] for b in batch],
        }

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps,
                                  eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # DDP
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=True,
        )
    rollout_engine.update_policy(model)

    # ===== 训练循环 =====
    Logger("=" * 80)
    Logger("Plan-then-Execute RL Training (Qwen2.5-1.5B-Instruct)")
    Logger(f"  Model: {args.model_path}")
    Logger(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    Logger(f"  Data: {len(train_ds)} examples, num_generations={args.num_generations}")
    if level_filter:
        Logger(f"  Curriculum: Phase {args.curriculum_phase}, levels={level_filter}")
    Logger(f"  Experts: math_agent, info_agent, translate_agent")
    Logger(f"  Loss: {args.loss_type}, beta={args.beta}")
    Logger(f"  DDP: {'Yes' if dist.is_initialized() else 'No'}"
           f"{f', world_size={dist.get_world_size()}' if dist.is_initialized() else ''}")
    Logger(f"  Gradient Checkpointing: {args.gradient_checkpointing}")
    Logger("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)

        last_step = skip
        for step_offset, batch in enumerate(loader, start=skip + 1):
            global_step = step_offset
            last_step = global_step

            metrics = plan_train_step(
                model=model,
                ref_model=ref_model,
                rollout_engine=rollout_engine,
                tokenizer=tokenizer,
                batch=batch,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
                autocast_ctx=autocast_ctx,
                step=global_step,
                reward_model=None,
            )

            # 日志
            if global_step % args.log_interval == 0 and is_main_process():
                Logger(
                    f"[Epoch {epoch + 1}/{args.epochs}] Step {global_step}/{iters} | "
                    f"loss={metrics['loss']:.4f} | reward={metrics['reward_mean']:.3f} | "
                    f"router_r={metrics['router_reward_mean']:.3f} | "
                    f"expert_r={metrics['expert_reward_mean']:.3f} | "
                    f"handoff={metrics['handoff_rate']:.0%} | "
                    f"plan={metrics['plan_rate']:.0%} | kl={metrics['kl']:.4f}"
                )
                if wandb:
                    wandb.log({
                        "reward": metrics["reward_mean"],
                        "router_reward": metrics["router_reward_mean"],
                        "expert_reward": metrics["expert_reward_mean"],
                        "handoff_rate": metrics["handoff_rate"],
                        "plan_rate": metrics["plan_rate"],
                        "policy_loss": metrics["loss"],
                        "kl_ref": metrics["kl"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })

            # 保存 checkpoint
            if (global_step % args.save_interval == 0 or global_step == iters) and is_main_process():
                model.eval()
                ckpt_mgr.save(
                    model=model, tokenizer=tokenizer,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, step=global_step,
                    metric_value=metrics["reward_mean"],
                    extra_metrics={
                        "router_reward": metrics["router_reward_mean"],
                        "expert_reward": metrics["expert_reward_mean"],
                        "handoff_rate": metrics["handoff_rate"],
                        "plan_rate": metrics["plan_rate"],
                    },
                    wandb=wandb,
                )
                model.train()
            if (global_step % args.save_interval == 0 or global_step == iters) and dist.is_initialized():
                dist.barrier()

            if global_step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        # Epoch 末尾梯度累积
        if last_step > skip and last_step % args.accumulation_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    Logger("\nPlan-then-Execute Training complete.")
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
#  Demo
# ==============================================================================

def run_plan_demo(args):
    """Plan-then-Execute 推理 Demo"""
    setup_seed(42)

    model, tokenizer = init_model_qwen(
        args.model_path, device=args.device,
        gradient_checkpointing=False,
    )
    autocast_ctx = (torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if "cuda" in args.device else nullcontext())
    rollout_engine = TorchRolloutEngineHF(
        policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
    )

    test_queries = [
        # L0: 直接回答
        ("你好，请介绍一下你自己", [], [], []),
        # L1: 单步 → 信息 → 数学
        ("查询纽约的天气，然后根据温度华氏转摄氏", ["摄氏"], [],
         [{"step": 1, "expert": "info"}, {"step": 2, "expert": "math"}]),
        # L2: 并行 → 聚合
        ("分别查询东京和伦敦的天气，比较两地温度差",
         ["温度差"], [],
         [{"step": 1, "expert": "info"}, {"step": 2, "expert": "info"},
          {"step": 3, "expert": "math"}]),
        # L3: 条件分支（router 需判断）
        ("查询北京天气，如果温度超过30度计算需要多少冰块降温",
         [], [], []),
        # L4: 多步流水线
        ("查询东京时间，换算成北京时间，翻译成英文，计算两地时差",
         [], [],
         [{"step": 1, "expert": "info"}, {"step": 2, "expert": "math"},
          {"step": 3, "expert": "translate"}, {"step": 4, "expert": "math"}]),
    ]

    Logger("=" * 80)
    Logger("Plan-then-Execute Demo — Qwen2.5-1.5B-Instruct")
    Logger("  RouterAgent → execute_plan → [Experts by topo order] → Synthesize")
    Logger("=" * 80)

    for query, gt, step_gts, plan_gt in test_queries:
        Logger(f"\n{'─' * 60}")
        Logger(f"[User] {query}")
        Logger(f"[Expected] gt={gt}, plan_steps={len(plan_gt)}")
        Logger(f"{'─' * 60}")

        with torch.no_grad():
            result = plan_rollout_single(
                rollout_engine, tokenizer, query,
                max_new_tokens=args.max_gen_len,
                thinking_ratio=0.0,
                device=args.device,
            )

        Logger(f"  Handoff: {'Yes' if result['handoff_occurred'] else 'No'}")
        Logger(f"  Plan: {'Yes' if result.get('plan_executed') else 'No'}")
        if result.get("plan_info"):
            pi = result["plan_info"]
            Logger(f"  Plan Type: {pi.get('type', 'N/A')}")
            if pi.get("steps"):
                for s in pi["steps"]:
                    Logger(f"    Step {s.get('step')}: {s.get('delegate')} "
                           f"(depends={s.get('depends_on', [])}) -> {s.get('task', '')[:60]}")
            if pi.get("step_results"):
                for sid, ans in pi["step_results"].items():
                    Logger(f"    Result[{sid}]: {str(ans)[:100]}")

        Logger(f"  # Outputs: {len(result['all_outputs'])}")
        for i, out in enumerate(result["all_outputs"]):
            if i == 0:
                role = "RouterAgent (Plan)"
            elif i == len(result["all_outputs"]) - 1 and result["handoff_occurred"]:
                role = "RouterAgent (Synthesize)"
            else:
                role = f"ExpertAgent (output {i})"
            display = out[:200] + "..." if len(out) > 200 else out
            Logger(f"  [{role}] {display}")

        # 计算奖励
        rewards, r_rewards, e_rewards = calculate_plan_rewards(
            [result], [gt], [plan_gt], [step_gts], [0],
            num_gen=1, device=args.device,
        )
        Logger(f"  Reward: total={rewards[0].item():.3f}, "
               f"router={r_rewards[0].item():.3f}, expert={e_rewards[0].item():.3f}")

    Logger(f"\n{'=' * 80}")
    Logger("Demo Complete.")


# ==============================================================================
#  Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniMind-Agent Plan-then-Execute (Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "demo"])

    # 模型参数
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model id 或本地路径")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1],
                        help="是否开启梯度检查点（24GB 4090 必须开）")

    # 训练参数
    parser.add_argument("--save_dir", type=str, default="./checkpoints_qwen_plan")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="per-GPU batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # RL 参数
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO group 内候选数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数")
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--thinking_ratio", type=float, default=0.0)

    # 序列长度
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=384,
                        help="每阶段最大生成 token 数")
    parser.add_argument("--max_total_len", type=int, default=4096,
                        help="三段拼接后的最大总长度")

    # 数据
    parser.add_argument("--data_path", type=str, default="./dataset/plan_execute.jsonl")

    # 课程学习
    parser.add_argument("--curriculum_phase", type=int, default=None,
                        help="课程学习阶段: 0=[L0,L1], 1=[L0-L2], 2=[L0-L3], 3=[L0-L4]。"
                             "不设则加载全部级别。")

    # 断点续训
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--resume_mode", type=str, default="latest")

    # 工程
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--max_keep", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-Plan-Qwen")

    args = parser.parse_args()
    args.gradient_checkpointing = bool(args.gradient_checkpointing)

    if args.mode == "train":
        run_plan_training(args)
    else:
        run_plan_demo(args)
