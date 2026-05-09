"""
数据质量评分与过滤管线（Data Quality Scorer）

核心思想：用已训练的 MiniMind 模型自身的 Perplexity 作为数据质量信号，
实现"模型反哺数据 → 数据提升模型"的离线数据飞轮。

功能模块:
1. PPL 批量计算：对预训练语料逐条计算 token 级 perplexity，支持批处理 + 混合精度
2. 分布分析：画 PPL 分布直方图，自动检测双峰结构（低 PPL 模板文本 / 高 PPL 噪声）
3. 分桶贡献分析：将数据按 PPL 分为 N 个桶，分别评估每个桶对 val loss 的边际贡献
4. 过滤策略：支持百分位过滤、自适应阈值（基于 KDE 谷点检测）、长度归一化过滤
5. 数据报告：输出过滤前后统计摘要 + 可视化

使用方式:
    # 1. 计算 PPL 并输出带分数的 JSONL
    python scripts/data_quality_scorer.py score \\
        --data_path dataset/pretrain_data.jsonl \\
        --model_path out/pretrain_768.pth

    # 2. 分析 PPL 分布并确定过滤阈值
    python scripts/data_quality_scorer.py analyze \\
        --scored_path dataset/pretrain_scored.jsonl

    # 3. 执行过滤，输出高质量子集
    python scripts/data_quality_scorer.py filter \\
        --scored_path dataset/pretrain_scored.jsonl \\
        --strategy percentile --low 25 --high 90

    # 4. 分桶贡献实验（每桶训练 1 epoch，对比 val loss 下降量）
    python scripts/data_quality_scorer.py bucket_exp \\
        --scored_path dataset/pretrain_scored.jsonl \\
        --val_path dataset/pretrain_val.jsonl \\
        --n_buckets 8

    # 5. 重复训练收益实验（验证"少而精重复 N 次"的效果）
    python scripts/data_quality_scorer.py repeat_exp \\
        --filtered_path dataset/pretrain_filtered.jsonl \\
        --val_path dataset/pretrain_val.jsonl \\
        --repeats 1,2,4,8
"""
import os
import sys
import json
import math
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ═══════════════════════════════════════════════════════════════════════════════
#                           PPL 计算数据集
# ═══════════════════════════════════════════════════════════════════════════════

class PPLDataset(Dataset):
    """用于批量 PPL 计算的数据集

    与 PretrainDataset 的区别：不做固定长度 padding，
    而是保留原始长度信息，在 collate_fn 中按 batch 内最大长度动态 padding。
    这样避免了 padding token 对 PPL 计算的污染，同时提升了短文本的推理效率。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = str(sample.get('text', ''))
        tokens = self.tokenizer(
            text, add_special_tokens=False,
            max_length=self.max_length - 2, truncation=True
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        return {
            'input_ids': tokens,
            'index': index,
            'text_length': len(text),
            'token_length': len(tokens)
        }


def dynamic_collate_fn(batch):
    """动态 padding collate 函数

    按 batch 内最大长度 padding（而非全局 max_length），
    减少短文本 batch 的无效计算量。
    """
    max_len = max(item['token_length'] for item in batch)
    input_ids_list = []
    attention_mask_list = []
    indices = []
    text_lengths = []
    token_lengths = []

    for item in batch:
        ids = item['input_ids']
        pad_len = max_len - len(ids)
        input_ids_list.append(ids + [0] * pad_len)
        attention_mask_list.append([1] * len(ids) + [0] * pad_len)
        indices.append(item['index'])
        text_lengths.append(item['text_length'])
        token_lengths.append(item['token_length'])

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
        'indices': indices,
        'text_lengths': text_lengths,
        'token_lengths': token_lengths
    }


# ═══════════════════════════════════════════════════════════════════════════════
#                           PPL 计算引擎
# ═══════════════════════════════════════════════════════════════════════════════

class PerplexityEngine:
    """Perplexity 批量计算引擎

    设计要点：
    1. 只在真实 token 上计算 NLL，通过 attention_mask 排除 padding 的影响
    2. 使用 token 级平均 NLL 再取 exp 得到 PPL
       不同长度的序列如果直接求和 NLL 会系统偏向短序列（NLL 总和小）
    3. 同时输出 per-token NLL 方差，作为辅助质量信号：
       - 低方差 → 模型对每个 token 都很确定 → 模板化/重复文本
       - 高方差 → 确定性不均匀 → 有信息量（或噪声，需配合 PPL 判断）
    4. 批处理 + 混合精度，单卡 V100 可在 30 分钟内处理百万条短文本
    """

    def __init__(self, model: MiniMindForCausalLM, device: str = 'cuda:0',
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model.to(device).eval()
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def compute_batch(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """计算一个 batch 的 PPL 和 token 级 NLL 方差

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            ppl: [batch] perplexity
            nll_var: [batch] token 级 NLL 方差
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        # shift: logits[t] 预测 token[t+1]
        shift_logits = logits[:, :-1, :].contiguous().float()  # 转 float32 算 loss 更稳定
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        batch_size, seq_len, vocab_size = shift_logits.shape

        # 逐 token NLL
        per_token_nll = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)

        # mask 掉 padding
        per_token_nll = per_token_nll * shift_mask
        real_count = shift_mask.sum(dim=1).clamp(min=1)

        # 平均 NLL → PPL
        mean_nll = per_token_nll.sum(dim=1) / real_count
        ppl = torch.exp(mean_nll)

        # token 级 NLL 方差
        mean_expanded = mean_nll.unsqueeze(1)
        diff_sq = ((per_token_nll - mean_expanded * shift_mask) ** 2) * shift_mask
        nll_var = diff_sq.sum(dim=1) / real_count.clamp(min=2)

        return ppl.cpu().numpy(), nll_var.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#                           分布分析器
# ═══════════════════════════════════════════════════════════════════════════════

class DistributionAnalyzer:
    """PPL 分布分析器

    职责：
    1. 统计分布特征（均值/中位数/偏度/峰度）
    2. 检测双峰结构——低 PPL 峰（模板文本）和高 PPL 峰（噪声）
    3. 自动确定过滤阈值（KDE 谷点检测）
    4. 分析 PPL 与文本长度的相关性（判断是否需要长度归一化）
    5. 输出可视化报告
    """

    def __init__(self, ppl_scores: np.ndarray, nll_vars: np.ndarray,
                 token_lengths: np.ndarray):
        self.ppl = ppl_scores
        self.nll_var = nll_vars
        self.token_lengths = token_lengths

    def basic_stats(self) -> Dict:
        """基础统计量"""
        return {
            'count': int(len(self.ppl)),
            'mean': float(np.mean(self.ppl)),
            'median': float(np.median(self.ppl)),
            'std': float(np.std(self.ppl)),
            'min': float(np.min(self.ppl)),
            'max': float(np.max(self.ppl)),
            'p5': float(np.percentile(self.ppl, 5)),
            'p10': float(np.percentile(self.ppl, 10)),
            'p25': float(np.percentile(self.ppl, 25)),
            'p75': float(np.percentile(self.ppl, 75)),
            'p90': float(np.percentile(self.ppl, 90)),
            'p95': float(np.percentile(self.ppl, 95)),
            'skewness': float(self._skewness()),
            'kurtosis': float(self._kurtosis()),
        }

    def _skewness(self) -> float:
        """偏度：正偏 = 右尾长（少量极高 PPL 噪声拖尾）"""
        m, s = np.mean(self.ppl), np.std(self.ppl)
        return float(np.mean(((self.ppl - m) / max(s, 1e-8)) ** 3))

    def _kurtosis(self) -> float:
        """峰度：越高 = 尾部越重（极端值越多）"""
        m, s = np.mean(self.ppl), np.std(self.ppl)
        return float(np.mean(((self.ppl - m) / max(s, 1e-8)) ** 4) - 3.0)

    def detect_valley_threshold(self, n_bins: int = 200) -> Optional[Tuple[float, float]]:
        """基于 KDE 谷点检测自适应过滤阈值

        原理：PPL 分布若呈双峰/三峰，峰间谷点就是自然分割阈值。
        - 低于第一个谷点 → 模型已完全记忆的模板文本（学习价值低）
        - 高于最后一个谷点 → 乱码/噪声（学了也没意义）

        实现：用直方图 + 高斯平滑近似 KDE，寻找显著局部最小值。

        Returns:
            (low_threshold, high_threshold) 或 None
        """
        # 去极端值做直方图
        p1, p99 = np.percentile(self.ppl, [1, 99])
        ppl_clipped = self.ppl[(self.ppl >= p1) & (self.ppl <= p99)]

        hist, bin_edges = np.histogram(ppl_clipped, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 高斯平滑
        kernel = np.exp(-0.5 * np.linspace(-2, 2, 7) ** 2)
        kernel /= kernel.sum()
        smoothed = np.convolve(hist.astype(float), kernel, mode='same')

        # 寻找显著谷点
        valleys = []
        peak_height = max(smoothed)
        for i in range(2, len(smoothed) - 2):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                # 谷点显著性：相邻区域最大值与谷值的比值
                local_max = max(max(smoothed[max(0, i - 10):i]), max(smoothed[i + 1:min(len(smoothed), i + 11)]))
                prominence = (local_max - smoothed[i]) / peak_height
                if prominence > 0.08:  # 显著性阈值 8%
                    valleys.append((bin_centers[i], prominence))

        if not valleys:
            return None

        valleys.sort(key=lambda x: -x[1])
        if len(valleys) >= 2:
            thresholds = sorted([valleys[0][0], valleys[1][0]])
            return (thresholds[0], thresholds[1])
        else:
            valley_ppl = valleys[0][0]
            median_ppl = float(np.median(self.ppl))
            if valley_ppl < median_ppl:
                return (valley_ppl, float(np.percentile(self.ppl, 95)))
            else:
                return (float(np.percentile(self.ppl, 5)), valley_ppl)

    def length_correlation(self) -> Dict:
        """分析 PPL 与 token 长度的相关性

        若 |correlation| > 0.5，说明全局 PPL 阈值存在长度偏差，
        建议切换到 length_normalized 过滤策略。
        """
        corr = float(np.corrcoef(self.ppl, self.token_lengths)[0, 1])
        return {
            'ppl_length_correlation': corr,
            'length_bias_detected': abs(corr) > 0.5,
            'recommendation': 'length_normalized' if abs(corr) > 0.5 else 'global_percentile'
        }

    def generate_report(self, output_dir: str) -> Dict:
        """生成完整分析报告"""
        os.makedirs(output_dir, exist_ok=True)

        report = {
            'stats': self.basic_stats(),
            'length_analysis': self.length_correlation(),
            'auto_thresholds': None,
        }

        valley = self.detect_valley_threshold()
        if valley:
            report['auto_thresholds'] = {'low': valley[0], 'high': valley[1]}

        # 保存 JSON
        with open(os.path.join(output_dir, 'quality_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 可视化
        self._plot(output_dir, valley)

        return report

    def _plot(self, output_dir: str, valley: Optional[Tuple[float, float]]):
        """绘制分析图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # PPL 分布
            p1, p99 = np.percentile(self.ppl, [1, 99])
            plot_ppl = self.ppl[(self.ppl >= p1) & (self.ppl <= p99)]
            axes[0, 0].hist(plot_ppl, bins=100, density=True, alpha=0.7, color='steelblue')
            axes[0, 0].set_xlabel('Perplexity')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('PPL Distribution (P1-P99)')
            if valley:
                axes[0, 0].axvline(valley[0], color='red', ls='--', label=f'Low={valley[0]:.1f}')
                axes[0, 0].axvline(valley[1], color='orange', ls='--', label=f'High={valley[1]:.1f}')
                axes[0, 0].legend()

            # PPL vs Length
            idx = np.random.choice(len(self.ppl), min(5000, len(self.ppl)), replace=False)
            axes[0, 1].scatter(self.token_lengths[idx], self.ppl[idx], alpha=0.3, s=3, c='coral')
            axes[0, 1].set_xlabel('Token Length')
            axes[0, 1].set_ylabel('PPL')
            corr = self.length_correlation()['ppl_length_correlation']
            axes[0, 1].set_title(f'PPL vs Length (r={corr:.3f})')

            # PPL vs NLL Variance（辅助判断维度）
            axes[1, 0].scatter(self.nll_var[idx], self.ppl[idx], alpha=0.3, s=3, c='seagreen')
            axes[1, 0].set_xlabel('Per-token NLL Variance')
            axes[1, 0].set_ylabel('PPL')
            axes[1, 0].set_title('PPL vs NLL Variance')

            # 分位区间计数
            edges = [0, 5, 10, 25, 50, 75, 90, 95, 100]
            counts, labels = [], []
            for i in range(len(edges) - 1):
                lo = np.percentile(self.ppl, edges[i])
                hi = np.percentile(self.ppl, edges[i + 1])
                counts.append(int(np.sum((self.ppl >= lo) & (self.ppl < hi))))
                labels.append(f'P{edges[i]}-{edges[i+1]}')
            axes[1, 1].bar(labels, counts, color='mediumpurple', alpha=0.8)
            axes[1, 1].set_title('Sample Count by PPL Percentile')
            axes[1, 1].tick_params(axis='x', rotation=35)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ppl_analysis.png'), dpi=150)
            plt.close()
        except ImportError:
            print("[Warn] matplotlib 未安装，跳过图表")


# ═══════════════════════════════════════════════════════════════════════════════
#                           过滤策略
# ═══════════════════════════════════════════════════════════════════════════════

class DataFilter:
    """数据过滤器

    三种策略：
    1. percentile: 全局百分位截断（P_low ~ P_high）
    2. kde_valley: KDE 谷点自适应阈值
    3. length_normalized: 按 token 长度分段，段内独立过滤
       解决 PPL 与长度相关时的系统性偏差
    """

    def __init__(self, scored_data: List[Dict]):
        self.data = scored_data
        self.ppl_array = np.array([d['ppl_score'] for d in scored_data])
        self.nll_var_array = np.array([d.get('nll_variance', 0) for d in scored_data])
        self.token_lengths = np.array([d.get('token_length', 100) for d in scored_data])

    def filter_percentile(self, low: float = 25.0, high: float = 90.0) -> List[Dict]:
        """百分位过滤

        - 低于 P_low: 模板化文本（模型完全记忆，PPL≈1，无学习增量）
        - 高于 P_high: 乱码/噪声（模型学了也产生不了泛化收益）
        - 中间段: "模型觉得有点难但不是完全不懂" → 学习价值最高
        """
        low_th = float(np.percentile(self.ppl_array, low))
        high_th = float(np.percentile(self.ppl_array, high))

        filtered = [d for d in self.data if low_th <= d['ppl_score'] <= high_th]
        self._log('percentile', f'P{low:.0f}={low_th:.2f}, P{high:.0f}={high_th:.2f}', filtered)
        return filtered

    def filter_kde_valley(self) -> List[Dict]:
        """KDE 谷点自适应过滤"""
        analyzer = DistributionAnalyzer(self.ppl_array, self.nll_var_array, self.token_lengths)
        valley = analyzer.detect_valley_threshold()

        if valley is None:
            print("[Filter] 未检测到双峰，回退到 P25-P90 percentile")
            return self.filter_percentile(25.0, 90.0)

        low_th, high_th = valley
        filtered = [d for d in self.data if low_th <= d['ppl_score'] <= high_th]
        self._log('kde_valley', f'auto=[{low_th:.2f}, {high_th:.2f}]', filtered)
        return filtered

    def filter_length_normalized(self, n_segments: int = 5,
                                 low: float = 20.0, high: float = 85.0) -> List[Dict]:
        """长度归一化过滤

        问题：短文本天然 PPL 低（信息少、模型容易预测），长文本天然 PPL 高。
        如果用全局阈值，会系统性过滤掉短文本或保留过多长噪声。

        方案：按 token 长度等频分成 N 段，每段内独立做百分位过滤。
        """
        seg_boundaries = np.percentile(self.token_lengths, np.linspace(0, 100, n_segments + 1))

        filtered = []
        for i in range(n_segments):
            seg_mask = (self.token_lengths >= seg_boundaries[i]) & \
                       (self.token_lengths < seg_boundaries[i + 1] + (1 if i == n_segments - 1 else 0))
            seg_indices = np.where(seg_mask)[0]
            if len(seg_indices) == 0:
                continue

            seg_ppl = self.ppl_array[seg_indices]
            seg_low = float(np.percentile(seg_ppl, low))
            seg_high = float(np.percentile(seg_ppl, high))

            for idx in seg_indices:
                if seg_low <= self.ppl_array[idx] <= seg_high:
                    filtered.append(self.data[idx])

        self._log('length_normalized', f'{n_segments} segs, P{low:.0f}-P{high:.0f}', filtered)
        return filtered

    def _log(self, method: str, params: str, filtered: list):
        ratio = len(filtered) / max(len(self.data), 1) * 100
        print(f"[Filter:{method}] {params}")
        print(f"[Filter:{method}] {len(self.data)} → {len(filtered)} ({ratio:.1f}% retained)")


# ═══════════════════════════════════════════════════════════════════════════════
#                   分桶贡献实验（Bucket Contribution Experiment）
# ═══════════════════════════════════════════════════════════════════════════════

class BucketExperiment:
    """分桶贡献实验

    将数据按 PPL 等频分为 N 个桶，对每个桶独立训练 1 epoch，
    在统一的 validation set 上评估 val loss。

    实验目标：
    - 找到"高贡献桶"（训练后 val loss 下降最多）
    - 找到"负贡献桶"（训练后 val loss 反而上升——噪声数据）
    - 确定最优 PPL 范围

    这比简单的"卡阈值"更有说服力：你能画出一条
    "PPL 桶 vs. val loss 贡献"的曲线，清楚展示中间桶贡献最大。
    """

    def __init__(self, scored_data: List[Dict], val_data_path: str,
                 config: 'BucketExpConfig'):
        self.scored_data = scored_data
        self.val_data_path = val_data_path
        self.config = config
        self.ppl_array = np.array([d['ppl_score'] for d in scored_data])

    def split_buckets(self, n_buckets: int) -> List[Tuple[float, float, List[Dict]]]:
        """按 PPL 等频分桶

        Returns:
            List of (ppl_low, ppl_high, bucket_data)
        """
        percentiles = np.linspace(0, 100, n_buckets + 1)
        thresholds = [float(np.percentile(self.ppl_array, p)) for p in percentiles]

        buckets = []
        for i in range(n_buckets):
            lo, hi = thresholds[i], thresholds[i + 1]
            bucket_data = [d for d in self.scored_data if lo <= d['ppl_score'] < hi + (1e9 if i == n_buckets - 1 else 0)]
            buckets.append((lo, hi, bucket_data))
        return buckets

    def run_single_bucket(self, bucket_data: List[Dict], bucket_id: int) -> Dict:
        """对单个桶训练 1 epoch，返回训练前后的 val loss

        流程：
        1. 初始化一个新模型（从 pretrain 权重开始）
        2. 构造桶数据的 DataLoader
        3. 训练 1 epoch
        4. 在 val set 上计算 loss
        5. 对比训练前的 baseline val loss
        """
        from torch import optim
        from dataset.lm_dataset import PretrainDataset

        print(f"\n[Bucket {bucket_id}] 样本数: {len(bucket_data)}, "
              f"PPL 范围: [{bucket_data[0]['ppl_score']:.1f}, {bucket_data[-1]['ppl_score']:.1f}]")

        # 写入临时 JSONL
        tmp_path = f'/tmp/bucket_{bucket_id}.jsonl'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for d in bucket_data:
                f.write(json.dumps({'text': d['text']}, ensure_ascii=False) + '\n')

        # 初始化模型
        lm_config = MiniMindConfig(
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers
        )
        model = MiniMindForCausalLM(lm_config)
        if self.config.base_weight_path:
            weights = torch.load(self.config.base_weight_path, map_location='cpu')
            model.load_state_dict(weights, strict=False)
        model = model.to(self.config.device)

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        # baseline val loss
        baseline_val_loss = self._eval_val_loss(model, tokenizer, lm_config)

        # 训练 1 epoch
        train_ds = PretrainDataset(tmp_path, tokenizer, max_length=self.config.max_seq_len)
        loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        optimizer = optim.AdamW(model.parameters(), lr=self.config.lr)

        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(self.config.device)
            labels = labels.to(self.config.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
                loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            steps += 1

        avg_train_loss = total_loss / max(steps, 1)

        # 训练后 val loss
        post_val_loss = self._eval_val_loss(model, tokenizer, lm_config)

        # val loss 变化量（负数 = 有贡献）
        delta = post_val_loss - baseline_val_loss

        result = {
            'bucket_id': bucket_id,
            'sample_count': len(bucket_data),
            'ppl_low': bucket_data[0]['ppl_score'],
            'ppl_high': bucket_data[-1]['ppl_score'],
            'train_loss': avg_train_loss,
            'baseline_val_loss': baseline_val_loss,
            'post_val_loss': post_val_loss,
            'val_loss_delta': delta,  # 负值 = 正贡献
            'contribution': -delta,   # 正值 = 正贡献（方便排序）
        }

        print(f"[Bucket {bucket_id}] train_loss={avg_train_loss:.4f}, "
              f"val: {baseline_val_loss:.4f} → {post_val_loss:.4f} (Δ={delta:+.4f})")

        # 清理
        os.remove(tmp_path)
        del model, optimizer
        torch.cuda.empty_cache()

        return result

    @torch.no_grad()
    def _eval_val_loss(self, model, tokenizer, lm_config) -> float:
        """在 validation set 上计算 loss"""
        from dataset.lm_dataset import PretrainDataset

        model.eval()
        val_ds = PretrainDataset(self.val_data_path, tokenizer, max_length=self.config.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size * 2, shuffle=False, num_workers=2)

        total_loss = 0.0
        total_tokens = 0
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(self.config.device)
            labels = labels.to(self.config.device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
            # 用 token 数加权（更准确的平均）
            n_tokens = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens

        model.train()
        return total_loss / max(total_tokens, 1)

    def run_all(self, n_buckets: int = 8) -> List[Dict]:
        """运行完整分桶实验"""
        buckets = self.split_buckets(n_buckets)
        results = []
        for i, (lo, hi, data) in enumerate(buckets):
            if len(data) < 10:
                print(f"[Bucket {i}] 数据不足（{len(data)} 条），跳过")
                continue
            result = self.run_single_bucket(data, i)
            results.append(result)

        # 按贡献排序
        results.sort(key=lambda x: -x['contribution'])
        print("\n===== 分桶贡献排名 =====")
        print(f"{'桶ID':<6}{'PPL范围':<20}{'样本数':<8}{'贡献(↓val_loss)':<15}")
        for r in results:
            print(f"{r['bucket_id']:<6}[{r['ppl_low']:.1f}, {r['ppl_high']:.1f}]{'':<5}"
                  f"{r['sample_count']:<8}{r['contribution']:+.5f}")

        return results


@dataclass
class BucketExpConfig:
    """分桶实验配置"""
    hidden_size: int = 768
    num_hidden_layers: int = 8
    max_seq_len: int = 340
    batch_size: int = 32
    lr: float = 5e-4
    device: str = "cuda:0"
    base_weight_path: Optional[str] = None
    tokenizer_path: str = "../model"


# ═══════════════════════════════════════════════════════════════════════════════
#                           门面类：DataQualityScorer
# ═══════════════════════════════════════════════════════════════════════════════

class DataQualityScorer:
    """数据质量评估管线的统一入口

    封装了 score → analyze → filter 的完整流程，
    也可以单独调用每个阶段。

    典型用法:
        scorer = DataQualityScorer(model_path='out/pretrain_768.pth')
        scorer.score_dataset('dataset/pretrain.jsonl')
        scorer.analyze('dataset/pretrain_scored.jsonl')
        scorer.filter('dataset/pretrain_scored.jsonl', strategy='percentile')
    """

    def __init__(self, model_path: str, tokenizer_path: str = '../model',
                 hidden_size: int = 768, num_hidden_layers: int = 8,
                 device: str = 'cuda:0', dtype: str = 'bfloat16'):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.dtype_str = dtype
        self.dtype = torch.bfloat16 if dtype == 'bfloat16' else (
            torch.float16 if dtype == 'float16' else torch.float32)

        # 延迟加载模型（避免不需要 score 时浪费显存）
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """懒加载模型和分词器"""
        if self._model is not None:
            return

        lm_config = MiniMindConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers
        )
        self._model = MiniMindForCausalLM(lm_config)
        weights = torch.load(self.model_path, map_location='cpu')
        self._model.load_state_dict(weights, strict=False)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        total_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        print(f"[Scorer] 模型加载完成: {total_params:.1f}M params, device={self.device}")

    def score_dataset(self, data_path: str, output_path: Optional[str] = None,
                      max_length: int = 512, batch_size: int = 32) -> str:
        """对数据集逐条计算 PPL，输出带分数的 JSONL

        输出格式（每行）:
        {"text": "...", "ppl_score": 12.34, "nll_variance": 0.56, "token_length": 128}
        """
        self._load_model()

        if output_path is None:
            output_path = data_path.replace('.jsonl', '_scored.jsonl')

        engine = PerplexityEngine(self._model, self.device, self.dtype)
        dataset = PPLDataset(data_path, self._tokenizer, max_length)
        loader = DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=dynamic_collate_fn,
            num_workers=4, pin_memory=True
        )

        # 预分配结果数组
        all_ppl = np.zeros(len(dataset))
        all_var = np.zeros(len(dataset))

        start = time.time()
        processed = 0
        for batch in loader:
            ppl, nll_var = engine.compute_batch(batch['input_ids'], batch['attention_mask'])
            for i, idx in enumerate(batch['indices']):
                all_ppl[idx] = ppl[i]
                all_var[idx] = nll_var[i]
            processed += len(batch['indices'])

            if processed % (batch_size * 50) == 0:
                elapsed = time.time() - start
                speed = processed / elapsed
                eta = (len(dataset) - processed) / speed
                print(f"[Score] {processed}/{len(dataset)} "
                      f"({processed/len(dataset)*100:.1f}%) "
                      f"speed={speed:.0f} samples/s, ETA={eta:.0f}s")

        # 写入输出
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, sample in enumerate(dataset.samples):
                record = {
                    'text': str(sample.get('text', '')),
                    'ppl_score': float(all_ppl[i]),
                    'nll_variance': float(all_var[i]),
                    'token_length': len(self._tokenizer(
                        str(sample.get('text', '')), add_special_tokens=False
                    ).input_ids)
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

        elapsed = time.time() - start
        print(f"[Score] 完成! {len(dataset)} 条, 耗时 {elapsed:.1f}s, "
              f"输出: {output_path}")
        print(f"[Score] PPL 统计: mean={np.mean(all_ppl):.2f}, "
              f"median={np.median(all_ppl):.2f}, std={np.std(all_ppl):.2f}")

        return output_path

    @staticmethod
    def analyze(scored_path: str, output_dir: str = None) -> Dict:
        """加载已评分数据，生成分布分析报告"""
        if output_dir is None:
            output_dir = str(Path(scored_path).parent / 'quality_analysis')

        scored_data = []
        with open(scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                scored_data.append(json.loads(line.strip()))

        ppl = np.array([d['ppl_score'] for d in scored_data])
        nll_var = np.array([d.get('nll_variance', 0) for d in scored_data])
        token_len = np.array([d.get('token_length', 100) for d in scored_data])

        analyzer = DistributionAnalyzer(ppl, nll_var, token_len)
        report = analyzer.generate_report(output_dir)

        print(f"\n[Analyze] 报告已保存: {output_dir}/")
        return report

    @staticmethod
    def filter_data(scored_path: str, output_path: Optional[str] = None,
                    strategy: str = 'percentile',
                    low: float = 25.0, high: float = 90.0,
                    n_repeat: int = 1) -> str:
        """加载已评分数据并过滤"""
        if output_path is None:
            output_path = scored_path.replace('_scored.jsonl', '_filtered.jsonl')

        scored_data = []
        with open(scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                scored_data.append(json.loads(line.strip()))

        flt = DataFilter(scored_data)

        if strategy == 'percentile':
            filtered = flt.filter_percentile(low, high)
        elif strategy == 'kde_valley':
            filtered = flt.filter_kde_valley()
        elif strategy == 'length_normalized':
            filtered = flt.filter_length_normalized(low=low, high=high)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 重复上采样
        if n_repeat > 1:
            print(f"[Filter] 高质量数据重复 {n_repeat} 次: {len(filtered)} → {len(filtered) * n_repeat}")
            filtered = filtered * n_repeat

        with open(output_path, 'w', encoding='utf-8') as f:
            for d in filtered:
                # 输出只保留 text 字段（供 PretrainDataset 直接加载）
                f.write(json.dumps({'text': d['text']}, ensure_ascii=False) + '\n')

        print(f"[Filter] 输出: {output_path} ({len(filtered)} 条)")
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#                           命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MiniMind 数据质量评估与过滤工具")
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')

    # ===== score =====
    sp = subparsers.add_parser('score', help='计算数据集 PPL 分数')
    sp.add_argument('--data_path', type=str, required=True)
    sp.add_argument('--model_path', type=str, required=True)
    sp.add_argument('--tokenizer_path', type=str, default='../model')
    sp.add_argument('--output_path', type=str, default=None)
    sp.add_argument('--hidden_size', type=int, default=768)
    sp.add_argument('--num_hidden_layers', type=int, default=8)
    sp.add_argument('--max_length', type=int, default=512)
    sp.add_argument('--batch_size', type=int, default=32)
    sp.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    sp.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])

    # ===== analyze =====
    sp = subparsers.add_parser('analyze', help='分析 PPL 分布')
    sp.add_argument('--scored_path', type=str, required=True)
    sp.add_argument('--output_dir', type=str, default=None)

    # ===== filter =====
    sp = subparsers.add_parser('filter', help='过滤数据集')
    sp.add_argument('--scored_path', type=str, required=True)
    sp.add_argument('--output_path', type=str, default=None)
    sp.add_argument('--strategy', type=str, default='percentile',
                    choices=['percentile', 'kde_valley', 'length_normalized'])
    sp.add_argument('--low', type=float, default=25.0)
    sp.add_argument('--high', type=float, default=90.0)
    sp.add_argument('--repeat', type=int, default=1, help='过滤后重复上采样次数')

    # ===== bucket_exp =====
    sp = subparsers.add_parser('bucket_exp', help='分桶贡献实验')
    sp.add_argument('--scored_path', type=str, required=True)
    sp.add_argument('--val_path', type=str, required=True)
    sp.add_argument('--base_weight', type=str, default=None, help='基础权重路径')
    sp.add_argument('--n_buckets', type=int, default=8)
    sp.add_argument('--hidden_size', type=int, default=768)
    sp.add_argument('--num_hidden_layers', type=int, default=8)
    sp.add_argument('--batch_size', type=int, default=32)
    sp.add_argument('--device', type=str, default='cuda:0')
    sp.add_argument('--tokenizer_path', type=str, default='../model')

    args = parser.parse_args()

    if args.mode == 'score':
        scorer = DataQualityScorer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            device=args.device,
            dtype=args.dtype
        )
        scorer.score_dataset(args.data_path, args.output_path, args.max_length, args.batch_size)

    elif args.mode == 'analyze':
        DataQualityScorer.analyze(args.scored_path, args.output_dir)

    elif args.mode == 'filter':
        DataQualityScorer.filter_data(
            args.scored_path, args.output_path,
            strategy=args.strategy,
            low=args.low, high=args.high,
            n_repeat=args.repeat
        )

    elif args.mode == 'bucket_exp':
        scored_data = []
        with open(args.scored_path, 'r', encoding='utf-8') as f:
            for line in f:
                scored_data.append(json.loads(line.strip()))

        exp_config = BucketExpConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            batch_size=args.batch_size,
            device=args.device,
            base_weight_path=args.base_weight,
            tokenizer_path=args.tokenizer_path
        )
        exp = BucketExperiment(scored_data, args.val_path, exp_config)
        results = exp.run_all(args.n_buckets)

        # 保存实验结果
        out_path = str(Path(args.scored_path).parent / 'bucket_experiment_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n实验结果已保存: {out_path}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
