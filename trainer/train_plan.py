"""MiniMind Plan-then-Execute — 显式规划 + 工具执行训练 (v5 修复版)

v5 关键修复:
1. plan_rollout_single: Plan 轮结束后自动注入 plan 作为 assistant 消息，
   继续进入执行轮（之前 plan 输出没有 ゅ...゜ 标记导致直接 break，模型永远无法执行工具）
2. KL 控制: beta 0.04→0.1，加 per-token KL clip (上限 15)
3. 退化检测: 检测重复字符/token 生成，提前终止并给重惩罚
4. Reward 简化: 去掉 Plan Critic / 复杂度自适应权重等噪声源，让信号更清晰

核心思想：在多轮工具调用之前，强制模型先输出一段结构化执行计划（<plan> 标签），
然后再进入执行循环。通过 plan_adherence_reward 验证实际执行是否遵循了计划，
训练模型同时具备「规划能力」和「执行能力」，对应 Planner-Executor 架构。

与 train_agent.py 的区别:
    train_agent.py: 模型边想边调工具，没有显式规划
    train_plan.py:  Phase 1 强制输出 plan → Phase 2 按计划执行工具 → Phase 3 总结回答
                    reward 额外包含 plan 对齐分和 replanning 奖励

Plan 格式:
    <plan>
    [
        {"step": 1, "tool": "get_current_weather", "args": {"location": "北京"}, "expect": "获取温度"},
        {"step": 2, "tool": "none", "args": {}, "expect": "组织最终回答"}
    ]
    </plan>

训练方式:
    同 GRPO 框架。rollout 时第一轮生成必须包含 <plan>；如果模型没有生成 plan
    直接开始调工具，给予格式惩罚。reward 由 4 部分组成：
    1. plan_format_reward: plan 格式是否正确（JSON 可解析、步骤合理）
    2. plan_adherence_reward: 实际执行路径是否遵循 plan
    3. plan_quality_reward: plan 是否合理（步骤数适当、工具选择正确）
    4. execution_reward: 复用 train_agent.py 的 GT 验证 + 工具调用正确性

使用方式:
    python trainer/train_plan.py --mode train --data_path ../dataset/agent_rl.jsonl
    python trainer/train_plan.py --mode demo --hidden_size 768

依赖:
    复用 train_agent.py 的工具系统和 rollout 引擎。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import gc
import json
import math
import random
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, CheckpointManager, init_distributed_mode,
    setup_seed, SkipBatchSampler, init_model, LMForRewardModel
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps
from trainer.train_agent import (
    TOOLS, MOCK_RESULTS, CHECK_ARGS,
    parse_tool_calls, execute_tool, rep_penalty, validate_gt_in_text
)
from dataset.lm_dataset import AgentRLDataset

warnings.filterwarnings('ignore')


# ================================ Plan 系统提示与解析 ================================

PLAN_SYSTEM_PROMPT = """你是一个具备规划能力的 AI 助手。需要工具时，必须先输出 <plan>...</plan> 再行动。不需要工具时直接回答。"""

# Few-shot 示例：作为对话注入 prompt，让小模型直接看到 plan 格式的实际用法
PLAN_FEWSHOT_MESSAGES = [
    {"role": "user", "content": "北京今天天气怎么样？"},
    {"role": "assistant", "content": '<plan>\n[{"step": 1, "tool": "get_current_weather", "args_desc": "location=北京", "expect": "获取天气"}]\n</plan>'},
]


def parse_plan(text):
    """从模型输出中提取 <plan>...</plan> 内容并解析为结构化步骤

    Returns:
        plan_steps: list of dicts, 每个 dict 包含 step/tool/args_desc/expect
        plan_raw: 原始 plan 文本（用于展示）
        has_plan: 是否成功解析到 plan
    """
    match = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
    if not match:
        return [], "", False

    plan_raw = match.group(1).strip()
    try:
        steps = json.loads(plan_raw)
        if isinstance(steps, list) and all(isinstance(s, dict) for s in steps):
            return steps, plan_raw, True
    except (json.JSONDecodeError, ValueError):
        pass

    # 容错：尝试修复常见 JSON 格式问题
    try:
        # 去掉可能的 markdown code block
        cleaned = re.sub(r'```json\s*', '', plan_raw)
        cleaned = re.sub(r'```\s*', '', cleaned)
        steps = json.loads(cleaned)
        if isinstance(steps, list):
            return steps, plan_raw, True
    except (json.JSONDecodeError, ValueError):
        pass

    return [], plan_raw, False


def extract_execution_trace(all_outputs):
    """从多轮 rollout 输出中提取实际执行路径

    Returns:
        trace: list of dicts, 每个 dict 包含 tool/args/success
    """
    trace = []
    for output in all_outputs:
        calls = parse_tool_calls(output)
        for call in calls:
            name = call.get("name", "")
            raw_args = call.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    raw_args = {}
            trace.append({"tool": name, "args": raw_args})
    return trace


# ================================ 退化检测 ================================

def _is_degenerate(text, threshold=0.6):
    """检测生成文本是否退化（大量重复字符/token）

    检测模式：
    1. 连续重复字符占比过高（如 ' " ' " ' "...）
    2. 同一个短 token 重复出现次数过多
    """
    if len(text) < 10:
        return False
    chars = text.strip()
    if not chars:
        return True
    # 最常见字符占比
    char_counts = Counter(chars)
    most_common_ratio = char_counts.most_common(1)[0][1] / len(chars)
    if most_common_ratio > threshold:
        return True
    # 检测短 token 重复模式（如 '" ' 重复）
    for pat_len in range(1, 5):
        if len(chars) < pat_len * 4:
            continue
        pat = chars[:pat_len]
        repeat_count = 0
        for i in range(0, len(chars) - pat_len + 1, pat_len):
            if chars[i:i + pat_len] == pat:
                repeat_count += 1
        if repeat_count >= len(chars) / pat_len * threshold:
            return True
    return False


# ================================ Plan-aware Rollout ================================

# Plan/Execute 阶段标记常量（用于信用分离）
PHASE_PLAN = 1      # plan 生成阶段的 token
PHASE_EXECUTE = 2   # 工具调用执行阶段的 token
PHASE_SYNTHESIZE = 3  # 最终回答合成阶段的 token


def plan_rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=4,
                        max_new_tokens=512, thinking_ratio=0.5, device="cuda",
                        active_replan=True):
    """Plan-then-Execute 的单条样本 Rollout（v5 修复版）

    v5 关键修复:
    1. Plan 轮结束后自动注入 plan 作为 assistant 消息，继续进入执行轮
       （之前 plan 输出没有 ゅ...゜ 标记导致直接 break，模型永远无法执行工具）
    2. 检测退化生成（重复字符），提前终止避免浪费 token
    3. 简化 phase 标记逻辑

    Args:
        rollout_engine: 推理引擎
        tokenizer: 分词器
        messages: 对话消息列表（含 PLAN_SYSTEM_PROMPT）
        tools: 可用工具列表
        max_turns: 最大交互轮数（包含 plan 轮）
        max_new_tokens: 每轮最大生成 token 数
        thinking_ratio: 开启思考模式的概率
        device: 运行设备
        active_replan: 是否启用主动 replan 触发

    Returns:
        final_output: 最终回复文本
        prompt_ids: prompt 的 token IDs
        response_ids: 所有回复的 token IDs
        response_mask: 回复 mask（0=环境token, 1=模型生成token）
        response_old_logps: old logps
        all_outputs: 每轮的回复文本列表
        plan_info: 增强版 plan 信息 dict
        unfinished: 是否因达到最大轮数而未完成
        phase_labels: 每个 response token 的阶段标签 (PHASE_PLAN/EXECUTE/SYNTHESIZE)
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    phase_labels = []
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    plan_info = {
        "plan_steps": [],
        "plan_raw": "",
        "has_plan": False,
        "execution_trace": [],
        "step_adherence": [],
        "replanned": False,
        "replan_trigger": None,
        "plan_token_range": (0, 0),
        "degenerate": False,  # v5 新增：是否检测到退化
    }

    # 追踪当前 plan 预期的下一步
    expected_step_idx = 0

    for turn in range(max_turns):
        # 构建当前上下文
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=open_thinking
        )
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids

        # 模型生成
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()

        # 过滤 pad/eos
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        all_outputs.append(new_text)

        # ===== v5 新增：退化检测 =====
        # 如果生成了大量重复字符，提前终止（仍记录 token 用于训练，让惩罚信号传播）
        if _is_degenerate(new_text):
            plan_info["degenerate"] = True
            current_phase = PHASE_PLAN if turn == 0 else PHASE_EXECUTE
            response_ids.extend(new_ids)
            response_mask.extend([1] * len(new_ids))
            response_old_logps.extend(new_logps)
            phase_labels.extend([current_phase] * len(new_ids))
            if turn == 0:
                plan_info["plan_token_range"] = (len(response_ids) - len(new_ids), len(response_ids))
                steps, raw, has = parse_plan(new_text)
                plan_info["plan_steps"] = steps
                plan_info["plan_raw"] = raw
                plan_info["has_plan"] = has
            break

        # 确定当前 turn 的阶段标签
        if turn == 0:
            current_phase = PHASE_PLAN
        else:
            calls = parse_tool_calls(new_text)
            if calls:
                current_phase = PHASE_EXECUTE
            else:
                current_phase = PHASE_SYNTHESIZE

        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        phase_labels.extend([current_phase] * len(new_ids))

        # 记录 plan token 边界
        if turn == 0:
            plan_start = len(response_ids) - len(new_ids)
            plan_end = len(response_ids)
            plan_info["plan_token_range"] = (plan_start, plan_end)

        # ===== 第一轮（Plan 轮）：特殊处理 =====
        if turn == 0:
            # 解析 plan
            steps, raw, has = parse_plan(new_text)
            plan_info["plan_steps"] = steps
            plan_info["plan_raw"] = raw
            plan_info["has_plan"] = has

            # 检查 plan 轮是否同时包含了工具调用（模型跳过纯 plan 直接调工具）
            calls = parse_tool_calls(new_text)
            if not calls:
                # Plan 输出格式是 <plan>...</plan>，不包含 ゅ...゜ 工具调用标记，
                # 所以 parse_tool_calls 会返回空。但这不意味着 rollout 应该结束——
                # 如果有工具可用且 plan 中包含工具步骤，应该继续进入执行轮。
                if tools and has and any(s.get("tool", "none") != "none" for s in steps):
                    # Plan 成功解析且包含工具步骤 → 注入 plan 作为 assistant 消息，继续执行
                    messages.append({"role": "assistant", "content": new_text})
                    # 注入一个 user 提示，引导模型开始执行 plan
                    messages.append({"role": "user", "content": "请按照你的计划执行。"})
                    # 添加环境 token（user 消息模板）
                    observe_context = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        tools=tools, open_thinking=open_thinking
                    )
                    observe_ids = tokenizer(observe_context, return_tensors="pt",
                                            add_special_tokens=False)["input_ids"][0].tolist()
                    current_len = len(prompt_ids) + len(response_ids)
                    obs_delta = observe_ids[current_len:]
                    response_ids.extend(obs_delta)
                    response_mask.extend([0] * len(obs_delta))
                    response_old_logps.extend([0.0] * len(obs_delta))
                    phase_labels.extend([0] * len(obs_delta))
                    continue  # 进入下一轮（执行轮）
                else:
                    # 无工具 / plan 中没有工具步骤 → 正常结束
                    break
            # 如果 plan 轮同时包含了工具调用（模型跳过了纯 plan 直接调工具），继续处理

        else:
            # 非 plan 轮：正常检查工具调用
            calls = parse_tool_calls(new_text)
            if not calls:
                break  # 没有工具调用，结束

        unfinished = turn == max_turns - 1
        # turn==0 且有 calls 时，messages.append 在这里处理
        # turn==0 且无 calls 时，已在上面 continue 或 break
        if turn > 0 or calls:
            # 避免重复 append（turn==0 且走了 continue 分支的已经 append 过了）
            if not (turn == 0 and not calls):
                messages.append({"role": "assistant", "content": new_text})

        # 执行工具调用并记录 trace + 逐步对齐
        for call in calls:
            name, raw_args = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    raw_args = {}
            result = execute_tool(name, raw_args)
            tool_failed = result is None
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found or execution failed"}')[:2048]

            # 逐步对齐验证（Step-level Adherence）
            step_aligned = False
            if plan_info["has_plan"] and expected_step_idx < len(plan_info["plan_steps"]):
                expected = plan_info["plan_steps"][expected_step_idx]
                expected_tool = expected.get("tool", "")
                if expected_tool == name:
                    step_aligned = True
                elif expected_tool == "none":
                    pass
                expected_step_idx += 1

            plan_info["step_adherence"].append({
                "expected_tool": plan_info["plan_steps"][expected_step_idx - 1].get("tool", "?")
                    if plan_info["has_plan"] and 0 < expected_step_idx <= len(plan_info["plan_steps"])
                    else "?",
                "actual_tool": name,
                "aligned": step_aligned,
                "tool_failed": tool_failed,
            })

            plan_info["execution_trace"].append({
                "tool": name,
                "args": raw_args,
                "success": not tool_failed,
            })

            # 主动 Replanning 触发：工具返回失败时
            if active_replan and tool_failed and not unfinished:
                replan_hint = (
                    '{"error": "tool execution failed", "hint": "请修正计划后继续"}'
                )
                messages.append({"role": "tool", "content": replan_hint})
                plan_info["replan_trigger"] = "tool_failure_at_step_{}".format(
                    len(plan_info["execution_trace"]))
            else:
                messages.append({"role": "tool", "content": result_str})

        # 工具模板 token（mask=0）
        observe_context = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not unfinished,
            tools=tools, open_thinking=open_thinking
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        phase_labels.extend([0] * len(obs_delta))  # 0 = 环境 token，不参与信用分配

        # 检测 replanning：如果后续轮次又出现了 <plan>，说明模型自主修正了计划
        if turn > 0 and '<plan>' in new_text:
            plan_info["replanned"] = True
            # 重新解析 plan 用于后续步骤的对齐（直接搜索 <plan> 标签）
            new_steps, _, new_has = parse_plan(new_text)
            if new_has:
                plan_info["plan_steps"] = new_steps
                expected_step_idx = 0  # 重置对齐指针

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return (final_output, prompt_ids, response_ids, response_mask,
            response_old_logps, all_outputs, plan_info, unfinished, phase_labels)


# ================================ Reward 计算 ================================

def _lcs_len(a, b):
    """最长公共子序列长度（复用工具函数）"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def calculate_plan_adherence(plan_steps, execution_trace, step_adherence=None):
    """计算 plan 与实际执行的对齐度

    对齐维度:
    1. 步骤数对齐: plan 步骤数 vs 实际调用数（满分 0.25）
    2. 工具名对齐: plan 中预定的工具 vs 实际调用的工具（满分 0.35）
    3. 顺序对齐: LCS 衡量执行顺序与 plan 的一致性（满分 0.4）

    Returns:
        adherence_score: 0~1 之间的对齐分数
        details: 对齐详情 dict
    """
    if not plan_steps or not execution_trace:
        return 0.0, {"reason": "empty plan or trace"}

    # 提取 plan 中预定的工具调用步骤（过滤 "none"）
    planned_tools = [s.get("tool", "") for s in plan_steps if s.get("tool", "none") != "none"]
    actual_tools = [t["tool"] for t in execution_trace]

    if not planned_tools:
        return 0.2 if not actual_tools else 0.0, {"reason": "plan says no tools"}

    # 1. 步骤数对齐（满分 0.25）
    count_diff = abs(len(planned_tools) - len(actual_tools))
    count_score = max(0, 0.25 - 0.1 * count_diff)

    # 2. 工具名对齐（满分 0.35）
    planned_set = set(planned_tools)
    actual_set = set(actual_tools)
    if planned_set:
        precision = len(planned_set & actual_set) / len(actual_set) if actual_set else 0.0
        recall = len(planned_set & actual_set) / len(planned_set)
        name_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        name_f1 = 1.0 if not actual_set else 0.0
    name_score = 0.35 * name_f1

    # 3. 顺序对齐（满分 0.4）
    if planned_tools and actual_tools:
        lcs = _lcs_len(planned_tools, actual_tools)
        order_score = 0.4 * lcs / max(len(planned_tools), len(actual_tools))
    else:
        order_score = 0.0

    total = count_score + name_score + order_score
    details = {
        "planned_tools": planned_tools,
        "actual_tools": actual_tools,
        "count_score": count_score,
        "name_score": name_score,
        "order_score": order_score,
    }
    return total, details


def calculate_plan_rewards(completions, gt_batch, tools_batch, num_gen,
                           plan_infos, all_turn_outputs_batch, unfinished_batch,
                           reward_model=None, device="cuda"):
    """计算 Plan-then-Execute 场景的综合奖励（v5 简化版）

    v5 简化:
    - 去掉 Plan Critic（derive_optimal_plan）和复杂度自适应权重
    - 对 64M 小模型，信号越简单越好
    - 加强退化检测惩罚

    奖励组成:
    1. plan_format_reward: plan 格式正确性（+0.5/-0.3）
    2. plan_quality_reward: 步骤合理性 + 工具选择
    3. plan_adherence_reward: 逐步对齐
    4. replanning_reward: replan 有效性
    5. execution_reward: GT 验证 + 工具调用正确性
    6. 格式/重复/退化惩罚

    Returns:
        rewards: 总奖励 [B*num_gen]
        plan_scores: plan 相关奖励（用于分析）
        exec_scores: 执行相关奖励（用于分析）
    """
    total = len(completions)
    rewards = torch.zeros(total, device=device)
    plan_scores = torch.zeros(total, device=device)
    exec_scores = torch.zeros(total, device=device)

    for idx, response in enumerate(completions):
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        tools = tools_batch[sample_idx]
        plan_info = plan_infos[idx]
        turn_outputs = all_turn_outputs_batch[idx]
        unfinished = unfinished_batch[idx]

        p_reward = 0.0  # plan 相关奖励
        e_reward = 0.0  # 执行相关奖励

        # ===== v5 新增：退化惩罚 =====
        if plan_info.get("degenerate", False):
            # 退化生成：给重惩罚，跳过其他 reward 计算
            rewards[idx] = -3.0
            plan_scores[idx] = -1.5
            exec_scores[idx] = -1.5
            continue

        # 提取最终回答
        answer = response
        if '゜' in response:
            answer = response.split('゜')[-1].strip()

        # 解析所有轮次的工具调用
        all_tool_calls = []
        turn_answers = []
        for turn in turn_outputs:
            turn_answer = turn.split('゜')[-1].strip() if '゜' in turn else turn.strip()
            turn_answers.append(turn_answer)
            all_tool_calls.extend(parse_tool_calls(turn_answer))

        valid_names = {t['function']['name'] for t in tools} if tools else set()
        needs_tools = len(gt) > 0  # 有 GT 通常需要工具

        # ======== Plan 相关奖励 ========

        if needs_tools:
            # ---- 1. Plan 格式奖励 ----
            if plan_info["has_plan"]:
                p_reward += 0.5
            else:
                first_output = turn_outputs[0] if turn_outputs else ""
                if '<plan>' in first_output:
                    p_reward -= 0.1  # 尝试了但 JSON 格式错误
                else:
                    p_reward -= 0.3  # 完全没有输出 plan

            # ---- 2. Plan 质量奖励（v5 简化版）----
            if plan_info["has_plan"] and plan_info["plan_steps"]:
                steps = plan_info["plan_steps"]
                planned_tools = [s.get("tool", "") for s in steps if s.get("tool", "none") != "none"]

                # 2a. 步骤数合理性
                if 1 <= len(steps) <= 5:
                    p_reward += 0.15
                elif len(steps) > 5:
                    p_reward -= 0.1

                # 2b. 工具选择正确性
                if planned_tools:
                    valid_ratio = sum(1 for t in planned_tools if t in valid_names) / len(planned_tools)
                    p_reward += 0.2 * valid_ratio
                    if valid_ratio < 0.5:
                        p_reward -= 0.2

            # ---- 3. Plan 对齐奖励 ----
            if plan_info["has_plan"] and plan_info["execution_trace"]:
                adherence, _ = calculate_plan_adherence(
                    plan_info["plan_steps"],
                    plan_info["execution_trace"],
                    step_adherence=plan_info.get("step_adherence"),
                )
                p_reward += 0.8 * adherence

            # ---- 4. Replanning 奖励 ----
            if plan_info["replanned"]:
                if gt:
                    verified = validate_gt_in_text(answer, gt)
                    if verified:
                        p_reward += 0.2
                    else:
                        p_reward += 0.05
                else:
                    p_reward += 0.05

        else:
            # 不需要工具的问题
            if plan_info["has_plan"]:
                p_reward -= 0.2
            else:
                p_reward += 0.2

        # ======== 执行相关奖励（复用 train_agent 逻辑）========

        # 标签匹配扣分
        e_reward -= 0.5 * sum(abs(turn.count('ゅ') - turn.count('゜')) for turn in turn_answers)

        if not all_tool_calls:
            if needs_tools:
                e_reward -= 0.5
            else:
                e_reward += 0.3 if 5 <= len(answer) <= 800 else -0.3
                if reward_model is not None:
                    try:
                        rm_msgs = [{"role": "user", "content": ""},
                                   {"role": "assistant", "content": answer}]
                        score = reward_model.get_score(rm_msgs, answer)
                        e_reward += score
                    except Exception:
                        pass
        else:
            # 有工具调用
            valid_call_count = 0
            for tc in all_tool_calls:
                name = tc.get("name", "")
                raw = tc.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        raw = {}
                check = CHECK_ARGS.get(name, lambda a: bool(a))  # 通用检查: 有参数即有效
                valid_call_count += int(bool(name in valid_names and check(raw)))

            # 工具对齐分
            tool_gap = abs(valid_call_count - max(len(gt), 1)) + max(0, len(all_tool_calls) - valid_call_count)
            e_reward += 0.5 if tool_gap == 0 else -0.3 * min(tool_gap, 2)

            # GT 验证（核心奖励）
            final_text = "" if unfinished else answer
            if gt:
                verified = validate_gt_in_text(final_text, gt)
                e_reward += 2.5 * len(verified) / len(gt)

            if unfinished:
                e_reward -= 0.5

        # 重复惩罚（v5 加强版）
        penalty = rep_penalty(answer)
        # 额外检测空白/引号重复模式
        if answer:
            whitespace_ratio = sum(1 for c in answer if c in ' \t\n"\'') / len(answer)
            if whitespace_ratio > 0.5:
                penalty += 1.0  # 额外重惩罚
        e_reward -= penalty

        # Thinking 格式奖励
        if '゜' in response:
            think_part = response.split('゜')[0]
            if 20 <= len(think_part.strip()) <= 300:
                e_reward += 0.3

        # 汇总
        total_reward = p_reward + e_reward
        rewards[idx] = max(min(total_reward, 5.0), -3.0)
        plan_scores[idx] = max(min(p_reward, 2.5), -1.5)
        exec_scores[idx] = max(min(e_reward, 3.0), -3.0)

    return rewards, plan_scores, exec_scores


# ================================ 训练主循环 ================================

def plan_train_epoch(epoch, loader, iters, rollout_engine, ref_model, tokenizer, model,
                     lm_config, args, optimizer, scheduler, autocast_ctx,
                     reward_model=None, start_step=0, wandb=None, ckpt_mgr=None):
    """Plan-then-Execute 训练一个 epoch"""
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        # ===== Rollout =====
        all_completions = []
        all_prompt_ids = []
        all_response_ids = []
        all_response_masks = []
        all_response_old_logps = []
        all_turn_outputs = []
        all_plan_infos = []
        all_unfinished = []

        all_phase_labels = []

        with torch.no_grad():
            for messages, tools in zip(messages_batch, tools_batch):
                for _ in range(args.num_generations):
                    # === 裁剪多轮历史：只保留 system + 最后一个 user ===
                    last_user_msg = None
                    system_msg = None
                    for m in messages:
                        if m["role"] == "system":
                            system_msg = dict(m)
                        if m["role"] == "user":
                            last_user_msg = dict(m)
                    msgs_copy = []
                    if system_msg:
                        system_msg["content"] = PLAN_SYSTEM_PROMPT
                        msgs_copy.append(system_msg)
                    else:
                        msgs_copy.append({"role": "system", "content": PLAN_SYSTEM_PROMPT})
                    # 注入 few-shot 示例
                    for fshot in PLAN_FEWSHOT_MESSAGES:
                        msgs_copy.append(dict(fshot))
                    # 添加最后一个 user query（实际的工具请求）
                    if last_user_msg:
                        msgs_copy.append(last_user_msg)

                    # DEBUG: 在前几步打印完整 prompt 用于排查
                    if step <= 2 and args.debug_mode:
                        _dbg_ctx = tokenizer.apply_chat_template(
                            msgs_copy, tokenize=False, add_generation_prompt=True,
                            tools=tools, open_thinking=False
                        )
                        print(f"[PROMPT_DEBUG] step={step} tools={'Yes' if tools else 'No'} "
                              f"msgs={len(msgs_copy)} prompt_chars={len(_dbg_ctx)}")
                        print(f"[PROMPT_DEBUG] prompt=\n{_dbg_ctx}")
                        print("[PROMPT_DEBUG] ---END---")

                    (completion, prompt_ids, response_ids, response_mask,
                     response_old_logps, turn_outputs, plan_info, unfinished,
                     phase_labels) = plan_rollout_single(
                        rollout_engine, tokenizer, msgs_copy, tools,
                        max_turns=args.max_turns,
                        max_new_tokens=args.max_gen_len,
                        thinking_ratio=args.thinking_ratio,
                        device=args.device,
                        active_replan=args.active_replan,
                    )
                    all_completions.append(completion)
                    all_prompt_ids.append(prompt_ids)
                    all_response_ids.append(response_ids)
                    all_response_masks.append(response_mask)
                    all_response_old_logps.append(response_old_logps)
                    all_turn_outputs.append(turn_outputs)
                    all_plan_infos.append(plan_info)
                    all_unfinished.append(unfinished)
                    all_phase_labels.append(phase_labels)

        # ===== 打包序列 =====
        packed_samples = []
        for p, r, m, old_lp in zip(all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            packed_samples.append((ids, mask, old_logps))

        # Padding
        seq_lens = torch.tensor([len(ids) for ids, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor(
            [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _ in packed_samples],
            device=args.device
        )
        full_response_masks = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for _, mask, _ in packed_samples],
            device=args.device, dtype=torch.float32
        )
        old_per_token_logps = torch.tensor(
            [lps + [0.0] * ((max_len - 1) - len(lps)) for _, _, lps in packed_samples],
            device=args.device, dtype=torch.float32
        )
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        # ===== 前向传播 =====
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-2).gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(
                ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask
            )

        # ========== 构建 completion mask（到 EOS 为止的有效 token） ==========
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

        # ========== 计算奖励 ==========
        rewards, plan_scores, exec_scores = calculate_plan_rewards(
            completions=all_completions,
            gt_batch=gt_batch,
            tools_batch=tools_batch,
            num_gen=args.num_generations,
            plan_infos=all_plan_infos,
            all_turn_outputs_batch=all_turn_outputs,
            unfinished_batch=all_unfinished,
            reward_model=reward_model,
            device=args.device,
        )

        # ========== 调试模式 ==========
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger("[DEBUG] step={}, gt[{}]: {}".format(step, i, repr(gt_batch[i])))
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    pi = all_plan_infos[idx]
                    Logger("  gen[{}][{}] has_plan={}, plan_steps={}, degenerate={}".format(
                        i, j, pi["has_plan"],
                        json.dumps(pi["plan_steps"], ensure_ascii=False)[:200],
                        pi.get("degenerate", False)))
                    Logger("  gen[{}][{}] turns={}, exec_trace={}".format(
                        i, j, len(all_turn_outputs[idx]),
                        json.dumps(pi["execution_trace"], ensure_ascii=False)[:200]))
                    Logger("  gen[{}][{}] text={}".format(
                        i, j, repr(all_completions[idx][:300])))
                    Logger("  gen[{}][{}] reward={:.4f} (plan={:.3f}, exec={:.3f})".format(
                        i, j, rewards[idx].item(), plan_scores[idx].item(), exec_scores[idx].item()))
                Logger('-' * 80)

        # ========== GRPO 优势估计 + 信用分离 ==========
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # Plan/Execute 信用分离
        use_credit_sep = getattr(args, 'credit_separation', True)
        if use_credit_sep:
            grouped_plan = plan_scores.view(-1, args.num_generations)
            plan_mean = grouped_plan.mean(dim=1).repeat_interleave(args.num_generations)
            plan_std = grouped_plan.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
            plan_advantages = (plan_scores - plan_mean) / (plan_std + 1e-4)

            grouped_exec = exec_scores.view(-1, args.num_generations)
            exec_mean = grouped_exec.mean(dim=1).repeat_interleave(args.num_generations)
            exec_std = grouped_exec.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
            exec_advantages = (exec_scores - exec_mean) / (exec_std + 1e-4)

            # 构建 phase_mask tensor 并计算 per-token advantage
            phase_mask_list = []
            for pl_labels, p_ids in zip(all_phase_labels, all_prompt_ids):
                full_phases = [0] * len(p_ids) + pl_labels
                if len(full_phases) > max_len:
                    full_phases = full_phases[-max_len:]
                full_phases = full_phases + [0] * (max_len - len(full_phases))
                phase_mask_list.append(full_phases)
            phase_tensor = torch.tensor(phase_mask_list, device=args.device)[:, 1:]  # shift 对齐 logps

            is_plan_token = (phase_tensor == PHASE_PLAN).float()
            is_exec_token = ((phase_tensor == PHASE_EXECUTE) | (phase_tensor == PHASE_SYNTHESIZE)).float()
            per_token_advantages = (
                plan_advantages.unsqueeze(1) * is_plan_token +
                exec_advantages.unsqueeze(1) * is_exec_token +
                advantages.unsqueeze(1) * (1.0 - is_plan_token - is_exec_token)  # fallback
            )
        else:
            per_token_advantages = advantages.unsqueeze(1).expand_as(per_token_logps)

        # ========== 计算 KL 散度和策略损失（v5: 加 KL clip） ==========
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # v5 新增：per-token KL clip，防止 KL 爆炸
        per_token_kl = torch.clamp(per_token_kl, max=args.kl_clip)

        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * per_token_advantages * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * per_token_advantages
            per_token_loss2 = clipped_ratio * per_token_advantages
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        policy_loss = (
            ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
            if valid_rows.any() else per_token_loss.sum() * 0.0
        )
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ========== 梯度累积 + 优化器更新 ==========
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ========== 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl_val = (per_token_kl * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            adv_std = advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            plan_rate = sum(1 for p in all_plan_infos if p["has_plan"]) / max(len(all_plan_infos), 1)
            exec_rate = sum(1 for p in all_plan_infos if p["execution_trace"]) / max(len(all_plan_infos), 1)
            degen_rate = sum(1 for p in all_plan_infos if p.get("degenerate", False)) / max(len(all_plan_infos), 1)
            Logger(
                'Epoch:[{}/{}]({}/{}), Reward:{:.4f}, KL:{:.4f}, GrpStd:{:.4f}, AdvStd:{:.4f}, '
                'Loss:{:.4f}, AvgLen:{:.2f}, PlanRate:{:.2%}, ExecRate:{:.2%}, DegenRate:{:.2%}, LR:{:.8f}'.format(
                    epoch + 1, args.epochs, step, iters,
                    ar, kl_val, gs, adv_std, pl, al, plan_rate, exec_rate, degen_rate, lr))
            if wandb and is_main_process():
                wandb.log({
                    "reward": ar, "plan_reward": plan_scores.mean().item(),
                    "exec_reward": exec_scores.mean().item(),
                    "kl_ref": kl_val, "group_reward_std": gs,
                    "advantages_std": adv_std, "policy_loss": pl,
                    "avg_response_len": al, "plan_rate": plan_rate,
                    "exec_rate": exec_rate, "degen_rate": degen_rate,
                    "learning_rate": lr,
                })

        # ========== 保存检查点 ==========
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckpt_mgr.save(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=step, wandb=wandb,
                metric_value=rewards.mean().item(),
                extra_metrics={'plan_reward': plan_scores.mean().item(), 'exec_reward': exec_scores.mean().item()}
            )
            model.train()

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del per_token_logps, ref_per_token_logps
        del all_completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        gc.collect()
        torch.cuda.empty_cache()

    # 处理 epoch 末尾未完成的梯度累积步
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


# ================================ 入口函数 ================================

def run_plan_training(args):
    """Plan-then-Execute RL 训练入口"""
    os.makedirs(args.save_dir, exist_ok=True)
    ddp = init_distributed_mode()
    setup_seed(args.seed)
    device = args.device

    # Wandb
    wandb_inst = None
    if args.use_wandb and is_main_process():
        import wandb as _wandb
        _wandb.init(project="minimind-plan-rl", config=vars(args))
        wandb_inst = _wandb

    # Model config
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        use_moe=args.use_moe,
    )
    # 初始化模型和 tokenizer
    model, tokenizer = init_model(lm_config, args.model_weight, device=device)
    model.train()

    # Ref model（冻结）
    ref_model, _ = init_model(lm_config, args.model_weight, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # DDP
    if ddp:
        model = DistributedDataParallel(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))])

    # Rollout engine
    rollout_engine = create_rollout_engine(engine_type="torch", policy_model=model, tokenizer=tokenizer, device=device, autocast_ctx=None)

    # Dataset & DataLoader
    dataset = AgentRLDataset(args.data_path, tokenizer)
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        drop_last=True, collate_fn=lambda batch: {"messages": [b["messages"] for b in batch], "tools": [b["tools"] for b in batch], "gt": [b["gt"] for b in batch]},
    )
    iters = len(loader)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=iters * args.epochs, eta_min=args.learning_rate * 0.1)

    # AMP context
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if args.use_bf16 else nullcontext()

    # Reward model（可选）
    reward_model = None
    if args.reward_model_path:
        reward_model = LMForRewardModel(lm_config)
        rm_state = torch.load(args.reward_model_path, map_location=device)
        reward_model.load_state_dict(rm_state, strict=False)
        reward_model.to(device).eval()
        for p in reward_model.parameters():
            p.requires_grad = False
        del rm_state

    # 初始化 CheckpointManager
    ckpt_mgr = CheckpointManager(
        lm_config=lm_config,
        weight=args.save_weight,
        save_dir='../checkpoints',
        max_keep=getattr(args, 'max_keep', 5),
        track_metric='reward',
        metric_mode='max'
    )

    # Resume
    start_epoch, start_step = 0, 0
    if args.resume_ckpt:
        ckp_data = ckpt_mgr.load(resume_mode=getattr(args, 'resume_mode', 'latest'))
        if ckp_data:
            model.load_state_dict(ckp_data['model'])
            optimizer.load_state_dict(ckp_data['optimizer'])
            scheduler.load_state_dict(ckp_data['scheduler'])
            start_epoch = ckp_data['epoch']
            start_step = ckp_data.get('step', 0)
            Logger("Resumed from epoch={}, step={}".format(start_epoch, start_step))

    Logger("=== Plan-then-Execute RL Training (v5) ===")
    Logger("  Epochs: {}, Steps/epoch: {}, Batch: {}, Generations: {}".format(
        args.epochs, iters, args.batch_size, args.num_generations))
    Logger("  Max turns: {}, Max gen len: {}, Loss type: {}".format(
        args.max_turns, args.max_gen_len, args.loss_type))
    Logger("  Beta: {}, KL clip: {}, LR: {}".format(args.beta, args.kl_clip, args.learning_rate))
    Logger("  Plan reward weight included in total reward")

    for epoch in range(start_epoch, args.epochs):
        if ddp and sampler is not None:
            sampler.set_epoch(epoch)
        plan_train_epoch(
            epoch=epoch, loader=loader, iters=iters,
            rollout_engine=rollout_engine, ref_model=ref_model,
            tokenizer=tokenizer, model=model, lm_config=lm_config,
            args=args, optimizer=optimizer, scheduler=scheduler,
            autocast_ctx=autocast_ctx, reward_model=reward_model,
            start_step=start_step if epoch == start_epoch else 0,
            wandb=wandb_inst, ckpt_mgr=ckpt_mgr,
        )
        start_step = 0

    if wandb_inst:
        wandb_inst.finish()
    Logger("Plan-then-Execute training completed!")


def get_args():
    parser = argparse.ArgumentParser(description="MiniMind Plan-then-Execute RL Training (v5)")
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--num_key_value_heads', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=16)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    parser.add_argument('--use_moe', action='store_true')
    # Data
    parser.add_argument('--data_path', type=str, default='../dataset/agent_rl.jsonl')
    parser.add_argument('--model_weight', type=str, default='minimind')
    parser.add_argument('--save_weight', type=str, default='minimind_plan_rl')
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--resume_ckpt', type=str, default='')
    parser.add_argument('--reward_model_path', type=str, default='')
    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-6)  # v5: 5e-6 → 3e-6
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # GRPO / ClSPO
    parser.add_argument('--num_generations', type=int, default=4)
    parser.add_argument('--loss_type', type=str, default='grpo', choices=['grpo', 'cispo'])
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--epsilon_high', type=float, default=5.0)
    parser.add_argument('--beta', type=float, default=0.1)  # v5: 0.04 → 0.1
    parser.add_argument('--kl_clip', type=float, default=15.0)  # v5 新增：per-token KL 上限
    # Rollout
    parser.add_argument('--max_turns', type=int, default=4)
    parser.add_argument('--max_gen_len', type=int, default=512)
    parser.add_argument('--max_total_len', type=int, default=4096)
    parser.add_argument('--thinking_ratio', type=float, default=0.5)
    # Plan 策略增强
    parser.add_argument('--active_replan', action='store_true',
                        help='启用主动 replan: 工具调用失败时自动触发重新规划')
    parser.add_argument('--credit_separation', action='store_true', default=True,
                        help='启用 Plan/Execute 信用分离: 不同阶段 token 使用独立 advantage')
    parser.add_argument('--no_credit_separation', dest='credit_separation', action='store_false',
                        help='禁用信用分离，使用统一 advantage')
    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--debug_interval', type=int, default=20)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_plan_training(args)
