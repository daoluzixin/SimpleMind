"""
训练工具函数集合

本文件提供 MiniMind 训练过程中通用的工具函数和类，包括：
- 模型参数统计
- 分布式训练辅助（进程判断、初始化）
- 日志输出
- 学习率调度（余弦退火）
- 断点续训（Checkpoint 保存/加载）
- 模型初始化
- 数据采样（SkipBatchSampler）
- 奖励模型封装（LMForRewardModel）
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """统计并打印模型参数量，区分总参数和活跃参数（MoE 场景）

    对于 MoE 模型，总参数量远大于活跃参数量（每个 token 只激活部分专家）。
    格式: "Model Params: {total}M-A{active}M" 或 "Model Params: {total}M"

    Args:
        model: 模型实例
        config: 模型配置，用于获取 MoE 相关参数
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6  # 总参数量（百万）
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))  # 路由专家数
    n_active = getattr(config, 'num_experts_per_tok', 0)  # 每个 token 激活的专家数
    n_shared = getattr(config, 'n_shared_experts', 0)  # 共享专家数
    # 统计单个路由专家的参数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    # 统计共享专家的参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 基础参数量 = 总参数 - 所有路由专家参数 - 共享专家参数
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 活跃参数量 = 基础参数 + 激活的路由专家参数 + 共享专家参数
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断当前是否为主进程（rank 0）

    在分布式训练中，只有主进程应该打印日志和保存模型。
    非分布式模式下始终返回 True。

    Returns:
        bool: 当前是否为主进程
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """主进程日志输出

    仅在主进程（rank 0）上打印内容，避免分布式训练中日志重复。

    Args:
        content: 要打印的内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """计算当前步的学习率（余弦退火调度）

    公式: lr_t = lr * (0.1 + 0.45 * (1 + cos(π * t / T)))
    - 初始阶段: lr * 0.55 ≈ 55% 的基础学习率（warm-up 效果）
    - 中期: 逐渐升高到 lr
    - 末期: 逐渐降低到 lr * 0.1（退火收敛）

    这个调度器同时实现了 warm-up 和 cosine decay 的效果。

    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率

    Returns:
        当前步的学习率
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """初始化分布式训练环境

    检测环境变量 RANK 判断是否为分布式训练：
    - 如果 RANK == -1: 非分布式模式，返回 local_rank=0
    - 否则: 初始化 NCCL 进程组，设置当前 GPU 设备

    环境变量说明:
    - RANK: 全局进程编号
    - LOCAL_RANK: 当前节点上的进程编号

    Returns:
        int: 本地 GPU 编号（local_rank）
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")  # 使用 NCCL 后端（GPU 通信）
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # 绑定当前进程到对应 GPU
    return local_rank


def setup_seed(seed: int):
    """设置全局随机种子，确保实验可复现

    同时设置 Python、NumPy、PyTorch 的随机种子，
    并关闭 CuDNN 的非确定性优化。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # 关闭确定性算法，提升性能
    torch.backends.cudnn.benchmark = True        # 开启自动调优，选择最快kernel


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """断点续训 Checkpoint 保存/加载

    该函数有两种工作模式:
    1. 保存模式（model 不为 None）: 保存模型权重、优化器状态、训练进度等
    2. 加载模式（model 为 None）: 从文件加载 Checkpoint 恢复训练

    保存策略:
    - 权重文件: {weight}_{hidden_size}[_moe].pth（仅模型权重，用于推理加载）
    - 续训文件: {weight}_{hidden_size}[_moe]_resume.pth（完整训练状态，用于断点续训）
    - 使用临时文件 + 原子重命名（os.replace），防止保存过程中断导致文件损坏

    支持保存的额外状态（通过 kwargs 传入）:
    - scaler: GradScaler 状态
    - scheduler: 学习率调度器状态
    - critic_model: PPO 中的 Critic 模型权重
    - critic_optimizer/critic_scheduler: Critic 优化器和调度器状态

    Args:
        lm_config: 模型配置
        weight: 权重名称前缀（如 'pretrain', 'full_sft', 'dpo' 等）
        model: 模型实例（None 时为加载模式）
        optimizer: 优化器实例
        epoch: 当前 epoch
        step: 当前步数
        wandb: wandb 实例（用于保存 run id）
        save_dir: 保存目录
        **kwargs: 其他需要保存的状态对象

    Returns:
        加载模式时返回 checkpoint 字典，保存模式时返回 None
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'          # 权重文件路径
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'  # 续训文件路径

    if model is not None:
        # ===== 保存模式 =====
        # 解包 DDP / torch.compile 包装，获取原始模型
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}  # 转 FP16 节省空间

        # 原子保存权重文件：先写临时文件，再重命名（防止中断导致文件损坏）
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 wandb run id（用于续训时恢复 wandb 日志）
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建续训数据字典
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # 处理额外的状态对象（如 scaler, scheduler, critic_model 等）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 对于有 state_dict 方法的对象（如 scheduler, critic_model），解包后保存
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 原子保存续训文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()  # 释放 GPU 缓存
    else:  # ===== 加载模式 =====
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # 处理 GPU 数量变化：将步数按比例转换
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """初始化模型和分词器

    执行流程:
    1. 加载分词器
    2. 创建模型实例
    3. 可选: 加载预训练权重
    4. 打印参数统计信息

    Args:
        lm_config: 模型配置
        from_weight: 预训练权重名称（'none' 表示从头训练）
        tokenizer_path: 分词器路径
        save_dir: 权重文件目录
        device: 运行设备

    Returns:
        (model, tokenizer): 模型和分词器的元组
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)  # strict=False 允许部分加载

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class CheckpointManager:
    """训练 Checkpoint 管理器

    在 lm_checkpoint 的基础上提供更完善的 checkpoint 生命周期管理：
    - top-k 保留：只保留最近 k 个 checkpoint，自动清理旧文件
    - best model 追踪：根据指标（如 loss）自动保存最优模型
    - 元信息记录：每次保存时记录 metrics、时间戳等到 JSON 日志
    - epoch 级别保存：支持按 epoch 保存独立 checkpoint

    典型用法:
        ckpt_mgr = CheckpointManager(lm_config, weight='pretrain', save_dir='../checkpoints', max_keep=3)
        # 训练循环中:
        ckpt_mgr.save(model, optimizer, epoch, step, metrics={'loss': 0.5}, wandb=wandb, scaler=scaler)
        # 加载:
        ckp_data = ckpt_mgr.load()

    Args:
        lm_config: 模型配置
        weight: 权重名称前缀
        save_dir: checkpoint 保存目录
        max_keep: 最多保留的 checkpoint 数量（0 表示不限制）
        track_metric: 追踪的指标名称（用于 best model 判断）
        metric_mode: 'min' 表示指标越小越好（如 loss），'max' 表示越大越好（如 accuracy）
    """
    def __init__(self, lm_config, weight='full_sft', save_dir='../checkpoints',
                 max_keep=3, track_metric='loss', metric_mode='min'):
        self.lm_config = lm_config
        self.weight = weight
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.track_metric = track_metric
        self.metric_mode = metric_mode

        self.moe_suffix = '_moe' if lm_config.use_moe else ''
        self.prefix = f'{weight}_{lm_config.hidden_size}{self.moe_suffix}'

        # best metric 追踪
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')
        self.best_step = -1

        # checkpoint 历史记录
        self._history = []  # [(step, epoch, path, metrics), ...]

        os.makedirs(save_dir, exist_ok=True)
        self._load_history()

    def _history_path(self):
        """元信息 JSON 文件路径"""
        return os.path.join(self.save_dir, f'{self.prefix}_history.json')

    def _load_history(self):
        """从磁盘加载 checkpoint 历史记录"""
        import json
        path = self._history_path()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._history = data.get('history', [])
                self.best_metric = data.get('best_metric', self.best_metric)
                self.best_step = data.get('best_step', -1)
            except (json.JSONDecodeError, KeyError):
                self._history = []

    def _save_history(self):
        """将 checkpoint 历史记录持久化到磁盘"""
        import json
        data = {
            'history': self._history,
            'best_metric': self.best_metric,
            'best_step': self.best_step,
            'track_metric': self.track_metric,
            'metric_mode': self.metric_mode,
        }
        path = self._history_path()
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def _step_resume_path(self, step):
        """按 step 编号的 checkpoint 文件路径"""
        return os.path.join(self.save_dir, f'{self.prefix}_step{step}_resume.pth')

    def _best_path(self):
        """best model 的保存路径"""
        return os.path.join(self.save_dir, f'{self.prefix}_best.pth')

    def _is_better(self, metric_value):
        """判断当前指标是否优于历史最优"""
        if self.metric_mode == 'min':
            return metric_value < self.best_metric
        return metric_value > self.best_metric

    def save(self, model, optimizer, epoch, step, metrics=None, wandb=None, **kwargs):
        """保存 checkpoint 并管理历史版本

        执行流程:
        1. 调用 lm_checkpoint 保存最新的 resume 文件（覆盖式，用于快速恢复）
        2. 额外保存一份带 step 编号的 checkpoint（用于 top-k 管理）
        3. 如果当前指标是最优的，额外保存 best model
        4. 清理超出 max_keep 的旧 checkpoint
        5. 更新元信息日志

        Args:
            model: 模型实例
            optimizer: 优化器实例
            epoch: 当前 epoch
            step: 当前全局步数
            metrics: 指标字典，如 {'loss': 0.5, 'lr': 1e-4}
            wandb: wandb 实例
            **kwargs: 传递给 lm_checkpoint 的额外状态（scaler, scheduler 等）
        """
        import time as _time

        if not is_main_process():
            return

        metrics = metrics or {}

        # 1. 保存最新的 resume 文件（覆盖式，兼容原有逻辑）
        lm_checkpoint(self.lm_config, weight=self.weight, model=model, optimizer=optimizer,
                      epoch=epoch, step=step, wandb=wandb, save_dir=self.save_dir, **kwargs)

        # 2. 保存带 step 编号的 checkpoint（用于 top-k 管理）
        step_path = self._step_resume_path(step)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
            'metrics': metrics,
            'timestamp': _time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        tmp = step_path + '.tmp'
        torch.save(resume_data, tmp)
        os.replace(tmp, step_path)

        # 3. 判断是否为 best model
        tracked_value = metrics.get(self.track_metric)
        is_best = False
        if tracked_value is not None and self._is_better(tracked_value):
            self.best_metric = tracked_value
            self.best_step = step
            is_best = True
            best_path = self._best_path()
            tmp_best = best_path + '.tmp'
            torch.save(resume_data, tmp_best)
            os.replace(tmp_best, best_path)
            Logger(f'[Checkpoint] New best {self.track_metric}={tracked_value:.6f} at step {step}')

        # 4. 记录历史
        record = {
            'step': step,
            'epoch': epoch,
            'path': step_path,
            'metrics': metrics,
            'timestamp': resume_data['timestamp'],
            'is_best': is_best,
        }
        self._history.append(record)

        # 5. 清理超出 max_keep 的旧 checkpoint
        if self.max_keep > 0 and len(self._history) > self.max_keep:
            to_remove = self._history[:-self.max_keep]
            self._history = self._history[-self.max_keep:]
            for old in to_remove:
                old_path = old['path']
                if os.path.exists(old_path) and old_path != self._best_path():
                    os.remove(old_path)
                    Logger(f'[Checkpoint] Removed old checkpoint: {os.path.basename(old_path)}')

        # 6. 持久化元信息
        self._save_history()
        del state_dict, resume_data
        torch.cuda.empty_cache()

        Logger(f'[Checkpoint] Saved step {step} (epoch {epoch+1}), '
               f'metrics={{{", ".join(f"{k}={v:.4f}" for k, v in metrics.items())}}}')

    def load(self, resume_mode='latest'):
        """加载 checkpoint

        Args:
            resume_mode: 加载模式
                - 'latest': 加载最新的 resume 文件（默认，最快恢复）
                - 'best': 加载 best model
                - int: 加载指定 step 的 checkpoint

        Returns:
            checkpoint 字典，或 None（无可用 checkpoint）
        """
        if resume_mode == 'best':
            path = self._best_path()
            if os.path.exists(path):
                Logger(f'[Checkpoint] Loading best model (step {self.best_step}, {self.track_metric}={self.best_metric:.6f})')
                return torch.load(path, map_location='cpu')
            Logger('[Checkpoint] No best model found, falling back to latest')
            resume_mode = 'latest'

        if isinstance(resume_mode, int):
            path = self._step_resume_path(resume_mode)
            if os.path.exists(path):
                Logger(f'[Checkpoint] Loading checkpoint at step {resume_mode}')
                return torch.load(path, map_location='cpu')
            Logger(f'[Checkpoint] Step {resume_mode} not found, falling back to latest')
            resume_mode = 'latest'

        # latest: 使用原有的 lm_checkpoint 加载逻辑
        return lm_checkpoint(self.lm_config, weight=self.weight, save_dir=self.save_dir)

    def get_history(self):
        """获取所有 checkpoint 的历史记录"""
        return self._history.copy()

    def get_best_info(self):
        """获取 best model 的信息"""
        return {
            'step': self.best_step,
            'metric_name': self.track_metric,
            'metric_value': self.best_metric,
            'path': self._best_path() if self.best_step >= 0 else None,
        }


class SkipBatchSampler(Sampler):
    """支持跳过前 N 个 batch 的批采样器

    用于断点续训时，跳过已训练过的 batch，从断点处继续。
    当 skip_batches > 0 时，前 skip_batches 个 batch 会被静默跳过。

    Args:
        sampler: 底层采样器（如 DistributedSampler 或索引列表）
        batch_size: 批大小
        skip_batches: 需要跳过的 batch 数量
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 跳过已训练过的 batch
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理最后一个不完整的 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """奖励模型封装类

    用于 PPO/GRPO 等 RL 训练中，对模型生成的回复进行评分。
    封装了 HuggingFace 的 AutoModel，提供 get_score 接口。

    评分流程:
    1. 将对话历史和回复组装为评估消息格式
    2. 调用底层模型的 get_score 方法获取分数
    3. 将分数裁剪到 [-3.0, 3.0] 范围内

    Args:
        model_path: 奖励模型路径
        device: 运行设备
        dtype: 模型精度
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """计算回复的奖励分数

        Args:
            messages: 对话历史列表，格式 [{"role": "user/assistant", "content": "..."}]
            response: 待评分的回复文本

        Returns:
            float: 奖励分数，裁剪到 [-3.0, 3.0]
        """
        # 构造对话历史文本
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        # 组装评估消息（包含问题和待评分的回复）
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)  # 裁剪到 [-3.0, 3.0]
