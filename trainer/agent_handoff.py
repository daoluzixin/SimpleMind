"""MiniMind Agent Handoff — 多 Agent 协同训练

核心思想：将 Agent 暴露为另一个 Agent 的 Tool（A2A 协作的最小实现）。

架构:
    RouterAgent —— 负责意图识别、任务分发、最终结果整合
        ↓ (通过 tool_call "delegate_to_math_agent" / "delegate_to_info_agent" / "delegate_to_translate_agent")
    专家 Agent（多个）—— 负责实际的工具调用和结果整合
        - MathAgent: 数学计算、单位换算
        - InfoAgent: 天气、时间、汇率查询
        - TranslateAgent: 文本翻译

训练方式:
    所有 Agent 共享同一模型权重，但用不同的 system prompt 区分角色。
    Rollout 四阶段流程：
        Phase 1: RouterAgent 决策（路由到哪个专家 or 直接回答）
        Phase 2: 专家 Agent 执行工具调用（可多轮）
        Phase 3: RouterAgent 整合专家结果，生成最终回答
    Reward 按最终结果统一计算，同时分离 Router/Expert 的 credit assignment。
    支持 DDP 分布式训练、断点续训、wandb 日志、MoE 辅助损失。

使用方式:
    # 训练
    python trainer/agent_handoff.py --mode train --data_path ../dataset/agent_handoff.jsonl
    # Demo
    python trainer/agent_handoff.py --mode demo --hidden_size 768

依赖:
    复用 train_agent.py 中的工具定义、Mock 数据和 reward 计算逻辑。
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
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, CheckpointManager, init_distributed_mode,
    setup_seed, SkipBatchSampler, init_model, LMForRewardModel
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps
from trainer.train_agent import (
    TOOLS, MOCK_RESULTS, CHECK_ARGS, WEATHER_DATA, TIME_DATA,
    EXCHANGE_DATA, TRANSLATE_DATA, UNIT_DATA,
    parse_tool_calls, execute_tool, rep_penalty, validate_gt_in_text
)

warnings.filterwarnings('ignore')


# ================================ 多专家 Agent 角色定义 ================================

ROUTER_SYSTEM_PROMPT = """你是一个任务路由 Agent。你的职责是判断用户的问题类型，然后决定：
1. 如果问题涉及数学计算或单位换算，调用 delegate_to_math_agent。
2. 如果问题涉及天气、时间、汇率查询，调用 delegate_to_info_agent。
3. 如果问题涉及文本翻译，调用 delegate_to_translate_agent。
4. 如果问题是简单的闲聊或知识问答，你直接回答即可。

你只有三个委托工具可用，不要尝试自己调用 calculate_math 等具体工具。
当你收到专家返回的结果后，请基于结果向用户给出简洁明确的最终回答。"""

ROUTER_SYNTHESIZE_PROMPT = """专家已完成任务并返回了以下结果：
{expert_result}

请基于以上结果，向用户给出简洁明确的最终回答。不要重复工具调用，直接用自然语言回答。"""

MATH_AGENT_SYSTEM_PROMPT = """你是一个数学计算专家 Agent。你接收到的任务已被确认为数学计算或单位换算类。
请分析任务，调用 calculate_math 或 unit_converter 工具，并根据结果给出精确答案。
注意：只回答事实性结果，不要做多余的寒暄。"""

INFO_AGENT_SYSTEM_PROMPT = """你是一个信息查询专家 Agent。你接收到的任务已被确认为天气、时间或汇率查询类。
请分析任务，调用 get_current_weather / get_current_time / get_exchange_rate 工具，并根据结果给出精确答案。
注意：只回答事实性结果，不要做多余的寒暄。"""

TRANSLATE_AGENT_SYSTEM_PROMPT = """你是一个翻译专家 Agent。你接收到的任务已被确认为文本翻译类。
请分析任务，调用 translate_text 工具，并给出翻译结果。
注意：只回答翻译结果，不要做多余的寒暄。"""

# RouterAgent 可用的工具：多专家委托
ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_math_agent",
            "description": "将数学计算或单位换算任务委托给数学专家。当用户问题涉及算术运算、公式计算、单位换算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "需要计算的具体任务描述，保留关键数字和运算符"}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_info_agent",
            "description": "将信息查询任务委托给信息专家。当用户问题涉及天气查询、时间查询、汇率查询时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "需要查询的具体任务描述，保留关键城市名、货币等信息"}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_translate_agent",
            "description": "将翻译任务委托给翻译专家。当用户问题涉及文本翻译、语言转换时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "需要翻译的具体任务描述，保留原文和目标语言"}
                },
                "required": ["task"]
            }
        }
    },
]

# 专家 → 工具子集映射
EXPERT_CONFIG = {
    "delegate_to_math_agent": {
        "system_prompt": MATH_AGENT_SYSTEM_PROMPT,
        "tools": [t for t in TOOLS if t["function"]["name"] in ("calculate_math", "unit_converter")],
    },
    "delegate_to_info_agent": {
        "system_prompt": INFO_AGENT_SYSTEM_PROMPT,
        "tools": [t for t in TOOLS if t["function"]["name"] in ("get_current_weather", "get_current_time", "get_exchange_rate")],
    },
    "delegate_to_translate_agent": {
        "system_prompt": TRANSLATE_AGENT_SYSTEM_PROMPT,
        "tools": [t for t in TOOLS if t["function"]["name"] in ("translate_text",)],
    },
}


# ================================ 外部数据集 ================================

class HandoffDataset(Dataset):
    """Agent Handoff 训练数据集

    JSONL 格式，每行:
    {"query": "...", "gt": [...], "needs_tool": true/false, "expert": "math/info/translate/none"}

    如果 data_path 不存在，使用内置示例数据。
    """

    # 内置示例数据（兜底）
    BUILTIN_EXAMPLES = [
        {"query": "北京今天天气怎么样？", "gt": ["28°C", "晴"], "needs_tool": True, "expert": "info"},
        {"query": "帮我算一下 123 * 456", "gt": ["56088"], "needs_tool": True, "expert": "math"},
        {"query": "100美元能换多少人民币？", "gt": ["7.21", "721"], "needs_tool": True, "expert": "info"},
        {"query": "现在东京几点了？", "gt": ["15:30"], "needs_tool": True, "expert": "info"},
        {"query": "把'你好世界'翻译成英文", "gt": ["Hello World"], "needs_tool": True, "expert": "translate"},
        {"query": "5公里等于多少英里？", "gt": ["3.1069", "3.107"], "needs_tool": True, "expert": "math"},
        {"query": "上海天气如何？", "gt": ["15°C", "多云"], "needs_tool": True, "expert": "info"},
        {"query": "帮我算 2^10", "gt": ["1024"], "needs_tool": True, "expert": "math"},
        {"query": "1英镑等于多少人民币？", "gt": ["9.12"], "needs_tool": True, "expert": "info"},
        {"query": "把 Good morning 翻译成中文", "gt": ["早上好"], "needs_tool": True, "expert": "translate"},
        {"query": "(15 + 27) * 3 等于多少？", "gt": ["126"], "needs_tool": True, "expert": "math"},
        {"query": "把'机器学习很有趣'翻译成英文", "gt": ["Machine learning is interesting"], "needs_tool": True, "expert": "translate"},
        {"query": "什么是机器学习？", "gt": [], "needs_tool": False, "expert": "none"},
        {"query": "你好，请介绍一下你自己", "gt": [], "needs_tool": False, "expert": "none"},
        {"query": "Python 是什么语言？", "gt": [], "needs_tool": False, "expert": "none"},
        {"query": "请解释什么是深度学习", "gt": [], "needs_tool": False, "expert": "none"},
        {"query": "HTTP 和 HTTPS 有什么区别？", "gt": [], "needs_tool": False, "expert": "none"},
        {"query": "什么是递归？", "gt": [], "needs_tool": False, "expert": "none"},
    ]

    def __init__(self, data_path=None):
        self.data = []
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            Logger(f"Loaded {len(self.data)} examples from {data_path}")
        if not self.data:
            self.data = self.BUILTIN_EXAMPLES
            Logger(f"Using {len(self.data)} built-in examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "gt": item.get("gt", []),
            "needs_tool": item.get("needs_tool", len(item.get("gt", [])) > 0),
            "expert": item.get("expert", "none"),
        }


# ================================ Handoff Rollout（四阶段） ================================

def handoff_rollout_single(rollout_engine, tokenizer, user_query, needs_tool=None,
                           max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """单条样本的四阶段 Handoff Rollout

    Phase 1: RouterAgent 接收用户问题，决定路由到哪个专家（或直接回答）
    Phase 2: 专家 Agent 执行工具调用（可多轮，最多 3 轮）
    Phase 3: RouterAgent 整合专家结果，生成最终回答

    Returns: dict with keys:
        final_answer, router_prompt_ids, router_response_ids, router_response_mask,
        router_old_logps, expert_prompt_ids, expert_response_ids, expert_response_mask,
        expert_old_logps, synth_prompt_ids, synth_response_ids, synth_response_mask,
        synth_old_logps, handoff_occurred, delegated_expert, delegate_task,
        all_outputs, needs_tool
    """
    open_thinking = random.random() < thinking_ratio

    # ===== Phase 1: RouterAgent 决策 =====
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    router_context = tokenizer.apply_chat_template(
        router_messages, tokenize=False, add_generation_prompt=True,
        tools=ROUTER_TOOLS, open_thinking=open_thinking
    )
    router_inputs = tokenizer(router_context, return_tensors="pt", add_special_tokens=False).to(device)
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

    # 过滤 pad/eos
    pairs = [(t, lp) for t, lp in zip(router_gen_ids, router_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    router_gen_ids = [t for t, _ in pairs]
    router_gen_logps = [lp for _, lp in pairs]
    router_text = router_result.completions[0]

    all_outputs = [router_text]

    # ===== Phase 2: 解析委托决策 =====
    router_calls = parse_tool_calls(router_text)
    handoff_call = None
    delegated_expert = None
    for call in router_calls:
        name = call.get("name", "")
        if name in EXPERT_CONFIG:
            handoff_call = call
            delegated_expert = name
            break

    # 没有 Handoff → RouterAgent 直接回答
    if not handoff_call:
        return {
            "final_answer": router_text,
            "router_prompt_ids": router_prompt_ids,
            "router_response_ids": router_gen_ids,
            "router_response_mask": [1] * len(router_gen_ids),
            "router_old_logps": router_gen_logps,
            "expert_prompt_ids": [],
            "expert_response_ids": [],
            "expert_response_mask": [],
            "expert_old_logps": [],
            "synth_prompt_ids": [],
            "synth_response_ids": [],
            "synth_response_mask": [],
            "synth_old_logps": [],
            "handoff_occurred": False,
            "delegated_expert": None,
            "delegate_task": "",
            "all_outputs": all_outputs,
            "needs_tool": needs_tool,
        }

    # 提取委托任务描述
    task_args = handoff_call.get("arguments", {})
    if isinstance(task_args, str):
        try:
            task_args = json.loads(task_args)
        except (json.JSONDecodeError, ValueError):
            task_args = {}
    delegated_task = task_args.get("task", user_query)

    # ===== Phase 3: 专家 Agent 子 Rollout =====
    expert_config = EXPERT_CONFIG[delegated_expert]
    expert_messages = [
        {"role": "system", "content": expert_config["system_prompt"]},
        {"role": "user", "content": delegated_task}
    ]
    expert_tools = expert_config["tools"]

    expert_response_ids = []
    expert_response_mask = []
    expert_old_logps = []
    expert_prompt_ids = None
    expert_final_answer = ""
    max_tool_turns = 3

    for turn in range(max_tool_turns):
        expert_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False, add_generation_prompt=True,
            tools=expert_tools, open_thinking=open_thinking
        )
        expert_inputs = tokenizer(expert_context, return_tensors="pt", add_special_tokens=False).to(device)
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

        # 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            expert_final_answer = new_text
            break

        expert_messages.append({"role": "assistant", "content": new_text})

        # 执行工具
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
            expert_messages.append({"role": "tool", "content": result_str})

        # 工具模板 token（mask=0）
        is_last_turn = (turn == max_tool_turns - 1)
        observe_context = tokenizer.apply_chat_template(
            expert_messages, tokenize=False,
            add_generation_prompt=not is_last_turn,
            tools=expert_tools, open_thinking=open_thinking
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(expert_prompt_ids) + len(expert_response_ids)
        obs_delta = observe_ids[current_len:]
        expert_response_ids.extend(obs_delta)
        expert_response_mask.extend([0] * len(obs_delta))
        expert_old_logps.extend([0.0] * len(obs_delta))

    if not expert_final_answer:
        expert_final_answer = all_outputs[-1] if len(all_outputs) > 1 else ""

    # ===== Phase 4: RouterAgent 整合回答 =====
    # 把专家结果注入 Router 对话，让 Router 生成最终面向用户的回答
    synth_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": router_text},
        {"role": "tool", "content": json.dumps({"expert_result": expert_final_answer}, ensure_ascii=False)},
    ]

    synth_context = tokenizer.apply_chat_template(
        synth_messages, tokenize=False, add_generation_prompt=True,
        tools=ROUTER_TOOLS, open_thinking=False  # 整合阶段不需要 thinking
    )
    synth_inputs = tokenizer(synth_context, return_tensors="pt", add_special_tokens=False).to(device)
    synth_prompt_ids = synth_inputs["input_ids"][0].tolist()

    synth_result = rollout_engine.rollout(
        prompt_ids=synth_inputs["input_ids"],
        attention_mask=synth_inputs["attention_mask"],
        num_generations=1,
        max_new_tokens=max_new_tokens // 2,  # 整合回答通常较短
        temperature=0.7,
    )
    synth_gen_ids = synth_result.completion_ids[0].tolist()
    synth_gen_logps = synth_result.per_token_logps[0].tolist()

    pairs = [(t, lp) for t, lp in zip(synth_gen_ids, synth_gen_logps)
             if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
    synth_gen_ids = [t for t, _ in pairs]
    synth_gen_logps = [lp for _, lp in pairs]
    synth_text = synth_result.completions[0]

    all_outputs.append(synth_text)
    final_answer = synth_text  # 最终回答来自 Router 的整合

    return {
        "final_answer": final_answer,
        "router_prompt_ids": router_prompt_ids,
        "router_response_ids": router_gen_ids,
        "router_response_mask": [1] * len(router_gen_ids),
        "router_old_logps": router_gen_logps,
        "expert_prompt_ids": expert_prompt_ids or [],
        "expert_response_ids": expert_response_ids,
        "expert_response_mask": expert_response_mask,
        "expert_old_logps": expert_old_logps,
        "synth_prompt_ids": synth_prompt_ids,
        "synth_response_ids": synth_gen_ids,
        "synth_response_mask": [1] * len(synth_gen_ids),
        "synth_old_logps": synth_gen_logps,
        "handoff_occurred": True,
        "delegated_expert": delegated_expert,
        "delegate_task": delegated_task,
        "all_outputs": all_outputs,
        "needs_tool": needs_tool,
    }


# ================================ Reward 计算 ================================

def calculate_handoff_rewards(results_batch, gt_batch, num_gen, reward_model=None, device="cuda"):
    """计算 Handoff 场景的奖励（分离 Credit Assignment）

    奖励组成:
    1. 路由正确性: 需要工具的问题是否触发了 handoff（+0.5/-0.5）
    2. 专家选择正确性: 是否路由到了正确的专家（+0.3/-0.3）
    3. 委托质量: delegate task 描述是否保留了关键信息（+0.3）
    4. 工具调用正确性: Expert 的工具调用是否正确
    5. GT 验证: 最终回答是否包含正确答案（核心，+2.5）
    6. 协作效率: 不需要工具的问题没有触发 handoff（+0.3）
    7. Thinking 格式奖励
    8. RM 评分（无工具直接回答时）
    9. 重复惩罚

    Returns:
        rewards: [B*num_gen] 总奖励
        router_rewards: RouterAgent 的独立奖励（路由 + 整合）
        expert_rewards: Expert Agent 的独立奖励（工具调用 + GT）
    """
    total = len(results_batch)
    rewards = torch.zeros(total, device=device)
    router_rewards = torch.zeros(total, device=device)
    expert_rewards = torch.zeros(total, device=device)

    for idx, result in enumerate(results_batch):
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        r_reward = 0.0  # Router 独立奖励
        e_reward = 0.0  # Expert 独立奖励

        needs_tool = result.get("needs_tool")
        if needs_tool is None:
            needs_tool = len(gt) > 0
        handoff_occurred = result["handoff_occurred"]
        final_answer = result["final_answer"]

        # 去思考部分
        answer_text = final_answer.split('゜')[-1].strip() if '゜' in final_answer else final_answer.strip()

        # ---- 1. 路由正确性 ----
        if needs_tool and handoff_occurred:
            r_reward += 0.5
        elif needs_tool and not handoff_occurred:
            r_reward -= 0.5
        elif not needs_tool and not handoff_occurred:
            r_reward += 0.3
        elif not needs_tool and handoff_occurred:
            r_reward -= 0.3

        # ---- 2. 专家选择正确性 ----
        if handoff_occurred and result.get("delegated_expert"):
            expected_expert_map = {
                "math": "delegate_to_math_agent",
                "info": "delegate_to_info_agent",
                "translate": "delegate_to_translate_agent",
            }
            # 从数据集获取期望专家（如果有标注）
            expected = result.get("expected_expert")
            if expected and expected in expected_expert_map:
                if result["delegated_expert"] == expected_expert_map[expected]:
                    r_reward += 0.3
                else:
                    r_reward -= 0.3

        # ---- 3. 委托质量 ----
        if handoff_occurred:
            delegate_task = result.get("delegate_task", "")
            query = result.get("original_query", "")
            if delegate_task and query:
                # 检查关键实体是否被保留（数字、专有名词）
                query_nums = set(re.findall(r'\d+(?:\.\d+)?', query))
                task_nums = set(re.findall(r'\d+(?:\.\d+)?', delegate_task))
                if query_nums and query_nums.issubset(task_nums):
                    r_reward += 0.15
                # 检查文本实体
                query_entities = set(re.findall(r'[\u4e00-\u9fff]+', query))
                task_entities = set(re.findall(r'[\u4e00-\u9fff]+', delegate_task))
                if query_entities and len(query_entities & task_entities) >= len(query_entities) * 0.5:
                    r_reward += 0.15

        # ---- 4. Thinking 格式奖励 ----
        for output in result["all_outputs"]:
            if '゜' in output:
                think_part = output.split('゜')[0]
                if 20 <= len(think_part.strip()) <= 300:
                    r_reward += 0.2
                    break

        # ---- 5. GT 验证（核心奖励）----
        if gt:
            verified = validate_gt_in_text(answer_text, gt)
            gt_score = 2.5 * len(verified) / len(gt)
            if handoff_occurred:
                # 分配：60% 归 Expert（工具调用正确），40% 归 Router（整合正确）
                e_reward += gt_score * 0.6
                r_reward += gt_score * 0.4
            else:
                r_reward += gt_score
        elif not handoff_occurred:
            # 无工具直接回答 → RM 评分
            if reward_model is not None:
                try:
                    messages = [{"role": "user", "content": result.get("original_query", "")},
                                {"role": "assistant", "content": answer_text}]
                    score = reward_model.get_score(messages, answer_text)
                    r_reward += score
                except Exception:
                    pass
            # 长度奖励
            r_reward += 0.3 if 5 <= len(answer_text) <= 800 else -0.3

        # ---- 6. 工具调用质量（仅在 Handoff 发生时）----
        if handoff_occurred:
            tool_calls = []
            for output in result["all_outputs"][1:-1]:  # 跳过 Router 和 Synth
                tool_calls.extend(parse_tool_calls(output))

            if tool_calls:
                expert_name = result.get("delegated_expert", "")
                expert_tools = EXPERT_CONFIG.get(expert_name, {}).get("tools", TOOLS)
                valid_names = {t['function']['name'] for t in expert_tools}
                valid_count = 0
                for tc in tool_calls:
                    name = tc.get("name", "")
                    raw = tc.get("arguments", {})
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            raw = {}
                    check = CHECK_ARGS.get(name)
                    valid_count += int(bool(name in valid_names and check and check(raw)))

                tool_gap = abs(valid_count - max(len(gt), 1)) + max(0, len(tool_calls) - valid_count)
                e_reward += 0.5 if tool_gap == 0 else -0.3 * min(tool_gap, 2)

        # ---- 7. 重复惩罚 ----
        penalty = rep_penalty(answer_text)
        r_reward -= penalty * 0.5
        e_reward -= penalty * 0.5

        # ---- 汇总 ----
        total_reward = r_reward + e_reward
        rewards[idx] = max(min(total_reward, 4.0), -3.0)
        router_rewards[idx] = max(min(r_reward, 2.5), -2.0)
        expert_rewards[idx] = max(min(e_reward, 2.5), -2.0)

    return rewards, router_rewards, expert_rewards


# ================================ 训练步 ================================

def handoff_train_step(model, ref_model, rollout_engine, tokenizer, batch, args, optimizer,
                       scheduler, autocast_ctx, step, lm_config, reward_model=None):
    """单个训练步的 Handoff 训练逻辑

    将 RouterAgent（Phase 1 + Phase 4）和 Expert Agent（Phase 2）的生成序列
    拼接后统一计算策略 loss。三段生成共享同一组模型权重。

    Credit Assignment: Router 段和 Expert 段使用各自的 advantage 进行加权。
    """
    queries = batch["queries"]
    gt_batch = batch["gt"]
    needs_tool_batch = batch.get("needs_tool", [None] * len(queries))
    expert_batch = batch.get("expert", ["none"] * len(queries))

    # ===== Rollout =====
    results_batch = []
    with torch.no_grad():
        for i, query in enumerate(queries):
            for _ in range(args.num_generations):
                result = handoff_rollout_single(
                    rollout_engine, tokenizer, query,
                    needs_tool=needs_tool_batch[i],
                    max_new_tokens=args.max_gen_len,
                    thinking_ratio=args.thinking_ratio,
                    device=args.device,
                )
                # 注入额外信息供 reward 使用
                result["original_query"] = query
                result["expected_expert"] = expert_batch[i]
                results_batch.append(result)

    # ===== Reward =====
    rewards, router_rewards, expert_rewards = calculate_handoff_rewards(
        results_batch, gt_batch, args.num_generations,
        reward_model=reward_model, device=args.device
    )

    # ===== 打包序列（三段拼接）=====
    # 结构: [router_prompt | router_response | expert_prompt(mask=0) | expert_response | synth_prompt(mask=0) | synth_response]
    packed_samples = []
    # 记录每个样本中 router/expert/synth 各段的 token 位置范围，用于分离 advantage
    segment_info = []

    for result in results_batch:
        prompt_ids = result["router_prompt_ids"]
        response_ids = list(result["router_response_ids"])
        response_mask = list(result["router_response_mask"])
        old_logps = list(result["router_old_logps"])

        router_len = len(response_ids)  # Router 段长度

        # 如果有 Handoff，拼接 Expert + Synth
        expert_start = router_len
        if result["handoff_occurred"] and result["expert_response_ids"]:
            # Expert prompt（mask=0）
            expert_prompt = result["expert_prompt_ids"]
            response_ids.extend(expert_prompt)
            response_mask.extend([0] * len(expert_prompt))
            old_logps.extend([0.0] * len(expert_prompt))

            # Expert 生成
            response_ids.extend(result["expert_response_ids"])
            response_mask.extend(result["expert_response_mask"])
            old_logps.extend(result["expert_old_logps"])

        expert_end = len(response_ids)

        # Synth 段（Router 的整合回答）
        synth_start = expert_end
        if result["handoff_occurred"] and result["synth_response_ids"]:
            # Synth prompt（mask=0）
            synth_prompt = result["synth_prompt_ids"]
            response_ids.extend(synth_prompt)
            response_mask.extend([0] * len(synth_prompt))
            old_logps.extend([0.0] * len(synth_prompt))

            # Synth 生成
            response_ids.extend(result["synth_response_ids"])
            response_mask.extend(result["synth_response_mask"])
            old_logps.extend(result["synth_old_logps"])

        synth_end = len(response_ids)

        # 拼接并截断
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

        # 段位置信息（相对于 response 起点）
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
        device=args.device
    )
    full_response_masks = torch.tensor(
        [mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
        device=args.device, dtype=torch.float32
    )
    old_per_token_logps = torch.tensor(
        [lps + [0.0] * ((max_len - 1) - len(lps)) for _, _, _, lps in packed_samples],
        device=args.device, dtype=torch.float32
    )
    full_mask = (input_ids != tokenizer.pad_token_id).long()

    # ===== 策略 Loss =====
    model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    with autocast_ctx:
        res = model_unwrapped(input_ids, attention_mask=full_mask)
        aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        logits = res.logits[:, :-1, :]
        per_token_logps = F.log_softmax(logits, dim=-1).gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    with torch.no_grad():
        ref_per_token_logps = compute_per_token_logps(
            ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask
        )

    # Completion mask（到 EOS 为止）
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

    # ===== Credit Assignment: 分离 Advantage =====
    # Router advantage 由 router_rewards 驱动，Expert advantage 由 expert_rewards 驱动
    grouped_rewards = rewards.view(-1, args.num_generations)
    grouped_router = router_rewards.view(-1, args.num_generations)
    grouped_expert = expert_rewards.view(-1, args.num_generations)

    # 各自归一化
    def normalize_group(grouped):
        mean = grouped.mean(dim=1).repeat_interleave(args.num_generations)
        std = grouped.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        return (grouped.view(-1) - mean) / (std + 1e-4)

    total_advantages = normalize_group(grouped_rewards)
    router_advantages = normalize_group(grouped_router)
    expert_advantages = normalize_group(grouped_expert)

    # 构建逐 token 的 advantage（Router 段用 router_adv，Expert 段用 expert_adv，Synth 段用 router_adv）
    B = completion_mask.size(0)
    T = completion_mask.size(1)
    per_token_advantages = torch.zeros(B, T, device=args.device)

    for i in range(B):
        # 默认使用总 advantage（对于没有 handoff 的样本）
        per_token_advantages[i] = total_advantages[i]

        if results_batch[i]["handoff_occurred"]:
            seg = segment_info[i]
            prompt_len_i = next((j for j, v in enumerate(
                [0] * len(results_batch[i]["router_prompt_ids"]) + list(results_batch[i]["router_response_mask"])
            ) if v == 1), 0)

            # 在 completion_mask 坐标系下（shift 1），映射段范围
            # response 起点在 input_ids 中的位置
            r_start, r_end = seg["router_range"]
            e_start, e_end = seg["expert_range"]
            s_start, s_end = seg["synth_range"]

            # Router 段 + Synth 段用 router_advantages
            # Expert 段用 expert_advantages
            # 简化实现：整行用加权 advantage
            per_token_advantages[i] = (router_advantages[i] + expert_advantages[i]) / 2

    # KL + 策略 loss
    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
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

    # 梯度更新
    if step % args.accumulation_steps == 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 释放中间变量
    del per_token_logps, ref_per_token_logps, per_token_advantages

    return {
        "loss": loss.item() * args.accumulation_steps,
        "reward_mean": rewards.mean().item(),
        "router_reward_mean": router_rewards.mean().item(),
        "expert_reward_mean": expert_rewards.mean().item(),
        "handoff_rate": sum(1 for r in results_batch if r["handoff_occurred"]) / len(results_batch),
        "kl": ((kl_div) * completion_mask).sum().item() / max(token_counts.sum().item(), 1),
        "aux_loss": aux_loss.item(),
    }


# ================================ 完整训练流程 ================================

def run_handoff_training(args):
    """完整的 Agent Handoff 训练流程（支持 DDP、断点续训、wandb）"""

    # 初始化分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe)
    )

    # 初始化 CheckpointManager
    ckpt_mgr = CheckpointManager(
        lm_config=lm_config,
        weight='agent_handoff',
        save_dir='../checkpoints',
        max_keep=getattr(args, 'max_keep', 5),
        track_metric='reward',
        metric_mode='max'
    )

    # 断点续训
    ckp_data = ckpt_mgr.load(resume_mode=getattr(args, 'resume_mode', 'latest')) if args.from_resume else None

    # 混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"Handoff-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}",
            id=wandb_id, resume=resume
        )

    # 初始化模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 奖励模型（可选）
    reward_model = None
    if args.reward_model_path and os.path.exists(args.reward_model_path):
        try:
            reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
            Logger(f'Loaded reward model from {args.reward_model_path}')
        except Exception as e:
            Logger(f'Warning: Failed to load reward model: {e}')

    # Rollout 引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=getattr(args, 'sglang_base_url', None),
        sglang_model_path=getattr(args, 'sglang_model_path', None),
        sglang_shared_path=getattr(args, 'sglang_shared_path', None),
    )

    # 数据集
    train_ds = HandoffDataset(args.data_path)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    def collate_fn(batch):
        return {
            'queries': [b['query'] for b in batch],
            'gt': [b['gt'] for b in batch],
            'needs_tool': [b['needs_tool'] for b in batch],
            'expert': [b['expert'] for b in batch],
        }

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # 恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # torch.compile
    if args.use_compile:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)

    # DDP
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ===== 训练循环 =====
    Logger("=" * 80)
    Logger("MiniMind Agent Handoff Training (Multi-Expert)")
    Logger(f"  Model: hidden_size={args.hidden_size}, layers={args.num_hidden_layers}, MoE={args.use_moe}")
    Logger(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    Logger(f"  Handoff: {len(train_ds)} examples, num_generations={args.num_generations}")
    Logger(f"  Experts: math_agent, info_agent, translate_agent")
    Logger(f"  Loss: {args.loss_type}, beta={args.beta}")
    Logger("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_fn)

        last_step = skip
        for step_offset, batch in enumerate(loader, start=skip + 1):
            global_step = step_offset
            last_step = global_step

            metrics = handoff_train_step(
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
                lm_config=lm_config,
                reward_model=reward_model,
            )

            # 日志
            if global_step % args.log_interval == 0 and is_main_process():
                Logger(
                    f"[Epoch {epoch+1}/{args.epochs}] Step {global_step}/{iters} | "
                    f"loss={metrics['loss']:.4f} | reward={metrics['reward_mean']:.3f} | "
                    f"router_r={metrics['router_reward_mean']:.3f} | expert_r={metrics['expert_reward_mean']:.3f} | "
                    f"handoff={metrics['handoff_rate']:.0%} | kl={metrics['kl']:.4f} | aux={metrics['aux_loss']:.4f}"
                )
                if wandb:
                    wandb.log({
                        "reward": metrics["reward_mean"],
                        "router_reward": metrics["router_reward_mean"],
                        "expert_reward": metrics["expert_reward_mean"],
                        "handoff_rate": metrics["handoff_rate"],
                        "policy_loss": metrics["loss"],
                        "kl_ref": metrics["kl"],
                        "aux_loss": metrics["aux_loss"],
                        "learning_rate": optimizer.param_groups[0]['lr'],
                    })

            # 保存检查点
            if (global_step % args.save_interval == 0 or global_step == iters) and is_main_process():
                model.eval()
                ckpt_mgr.save(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, step=global_step, wandb=wandb,
                    metric_value=metrics['reward_mean'],
                    extra_metrics={
                        'router_reward': metrics['router_reward_mean'],
                        'expert_reward': metrics['expert_reward_mean'],
                        'handoff_rate': metrics['handoff_rate']
                    }
                )
                model.train()
                Logger(f"  Saved checkpoint at step {global_step}")

            # 同步 rollout 引擎
            if global_step % args.save_interval == 0:
                rollout_engine.update_policy(model)

        # Epoch 末尾梯度累积处理
        if last_step > skip and last_step % args.accumulation_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    Logger("\nTraining complete.")
    if dist.is_initialized():
        dist.destroy_process_group()


# ================================ Demo ================================

def run_handoff_demo(args):
    """Handoff 推理 Demo：展示 RouterAgent → Expert → Router 整合的协作流程"""
    setup_seed(42)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe)
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if "cuda" in args.device else nullcontext()
    rollout_engine = create_rollout_engine(
        engine_type="torch",
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    test_queries = [
        ("北京今天天气怎么样？", ["28°C", "晴"], True, "info"),
        ("帮我算一下 (15 + 27) * 3", ["126"], True, "math"),
        ("100美元能换多少人民币？", ["7.21", "721"], True, "info"),
        ("你觉得人工智能的未来会怎样？", [], False, "none"),
        ("帮我把'机器学习很有趣'翻译成英文", ["Machine learning is interesting"], True, "translate"),
        ("5公里等于多少英里？", ["3.1069", "3.107"], True, "math"),
    ]

    Logger("=" * 80)
    Logger("Agent Handoff Demo - Multi-Expert Routing")
    Logger("  RouterAgent → [MathAgent | InfoAgent | TranslateAgent] → RouterAgent (Synthesize)")
    Logger("=" * 80)

    for query, gt, needs_tool, expected_expert in test_queries:
        Logger(f"\n{'─' * 60}")
        Logger(f"[User] {query}")
        Logger(f"[Expected] expert={expected_expert}, gt={gt}")
        Logger(f"{'─' * 60}")

        with torch.no_grad():
            result = handoff_rollout_single(
                rollout_engine, tokenizer, query,
                needs_tool=needs_tool,
                max_new_tokens=args.max_gen_len,
                thinking_ratio=0.0,
                device=args.device,
            )
            result["original_query"] = query
            result["expected_expert"] = expected_expert

        Logger(f"  Handoff: {'Yes' if result['handoff_occurred'] else 'No'}")
        if result['handoff_occurred']:
            Logger(f"  Expert: {result['delegated_expert']}")
            Logger(f"  Delegate Task: {result['delegate_task']}")
        Logger(f"  # Outputs: {len(result['all_outputs'])}")
        for i, out in enumerate(result['all_outputs']):
            if i == 0:
                role = "RouterAgent (Route)"
            elif i == len(result['all_outputs']) - 1 and result['handoff_occurred']:
                role = "RouterAgent (Synthesize)"
            else:
                role = f"ExpertAgent (turn {i})"
            display = out[:200] + "..." if len(out) > 200 else out
            Logger(f"  [{role}] {display}")

        # 计算 reward
        rewards, r_rewards, e_rewards = calculate_handoff_rewards(
            [result], [gt], num_gen=1, device=args.device
        )
        Logger(f"  Reward: total={rewards[0].item():.3f}, router={r_rewards[0].item():.3f}, expert={e_rewards[0].item():.3f}")

    Logger(f"\n{'=' * 80}")
    Logger("Demo Complete.")


# ================================ Main ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent Handoff (Multi-Expert)")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "demo"],
                        help="运行模式: train=训练, demo=推理演示")
    # 训练参数
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 模型参数
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3000)
    # RL 参数
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--thinking_ratio", type=float, default=0.1)
    # 数据和权重
    parser.add_argument("--data_path", type=str, default="../dataset/agent_handoff.jsonl")
    parser.add_argument("--from_weight", default="full_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    # 工程参数
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-Handoff")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    # Rollout 引擎
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_handoff")
    # Debug
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=5)
    parser.add_argument("--max_keep", type=int, default=5, help="最多保留的checkpoint数量")
    parser.add_argument("--resume_mode", type=str, default="latest", help="恢复模式: latest/best/步数")
    args = parser.parse_args()

    if args.mode == "train":
        run_handoff_training(args)
    else:
        run_handoff_demo(args)
