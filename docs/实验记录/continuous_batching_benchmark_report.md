# Continuous Batching 推理引擎压测实验报告

> 2026-05-21，云服务器 RTX 4090 24GB（AutoDL），模型 Qwen2.5-7B-Instruct (7616M 参数)。
> 本文记录从零实现 Continuous Batching 推理引擎、适配 HuggingFace transformers 5.9.0 DynamicCache API、到完成 4 组压测实验的完整过程。核心发现：Python 层面的 KV Cache 管理开销完全抵消了 Batched Decode 的理论吞吐收益，但 CB 的真正价值体现在延迟指标——P95 延迟从 18.8s 降至 2.5s，改善 7.5 倍。

---

## 背景与目标

前期的 Agent RL 实验（MiniMind 64M → Qwen2.5-1.5B）聚焦于训练侧，模型规模最大做到 1.5B。推理侧一直使用最朴素的串行逐条生成，每次只处理一个请求，处理完才接下一个。这种方式在单用户场景没有问题，但在并发场景下用户体验极差——后到的请求必须等前面所有请求生成完毕才能开始，排队延迟随请求数线性增长。

Continuous Batching 是工业推理引擎（vLLM、TGI、TensorRT-LLM）的核心调度策略。其核心思想是：不再等一个请求完整生成完毕才处理下一个，而是在每个 decode step 的粒度上动态管理多个请求——当某个请求生成完毕时立刻释放资源、从等待队列补入新请求。这样 GPU 始终在做有效计算，不会因为某个请求结束而"空转等待"。

本次实验的目标是从零实现一个教学级的 Continuous Batching 引擎，在 Qwen2.5-7B-Instruct 上验证其性能表现，并回答一个核心问题：**纯 Python 实现的 CB 引擎能获得多少吞吐提升？** 这个问题的答案将帮助理解为什么 vLLM 需要用 C++/CUDA 实现 PagedAttention——不仅是为了显存效率，更是为了消除 KV Cache 管理的计算开销。


## 实验环境

硬件是 AutoDL 的 RTX 4090 24GB 单卡服务器。模型从 ModelScope 下载到本地（`/root/autodl-tmp/Qwen2.5-7B-Instruct`），因为 hf-mirror.com 的下载速度时好时坏，modelscope 的 `snapshot_download` 反而更稳定。环境是 miniconda3 + transformers 5.9.0 + PyTorch 2.x + CUDA（FP16 推理）。

选择 Qwen2.5-7B-Instruct 而非更小的模型是有意为之的：7B 模型有 28 层 Transformer、GQA（Grouped-Query Attention）、KV Cache 体量可观。这个规模下 KV Cache 管理的开销才会真正暴露出来——如果用 0.5B 模型做实验，Python 开销相对于模型计算几乎可以忽略，会得出误导性的结论。


## 三种推理模式的设计

压测脚本实现了三种模式的严格对比：

**Serial（串行逐条）** 是 baseline。一次处理一个请求：加载 prompt → prefill → 逐 token decode → 生成完毕 → 处理下一个。最简单、最朴素的实现方式。

**CB Sequential Decode（连续批处理 + 逐序列解码）** 引入了 CB 的调度逻辑，但 decode 阶段仍然是逐序列 forward。它的价值在于：请求不再排队等待，而是到达后立刻进入 prefill 和 decode 循环。从用户视角看，每个请求的 TTFT 不再取决于前面排了多少个请求。

**CB Batched Decode（连续批处理 + 批量解码）** 是理论上最优的方案。将 running batch 中所有序列的最新 token 拼成一个 batch tensor，通过左 padding 对齐 KV Cache，一次 model forward 处理所有序列。GPU 的并行计算单元（CUDA cores / Tensor cores）在 batch_size > 1 时利用率更高，decode 阶段又是 memory-bandwidth-bound 的，理论上 batch 越大吞吐越高。

三种模式使用完全相同的 prompt 数据（固定 seed=42），相同的采样参数（temperature=0.8, top_k=50, top_p=0.9），相同的最大生成长度。唯一的区别是调度和 forward 策略。


## DynamicCache API 适配：从踩坑到通关

这部分是整个实验中最耗时间的环节。原始代码参考了旧版 HuggingFace 的 KV Cache 接口——`past_key_values` 是一个 tuple of tuple，通过 `past_key_values[layer_idx][0]`（keys）和 `past_key_values[layer_idx][1]`（values）访问。但 transformers 5.9.0 已经将 KV Cache 统一为 `DynamicCache` 类。

### 第一个错误：`'DynamicCache' object is not subscriptable`

直接跑旧代码时报了这个错。`DynamicCache` 不支持 `cache[layer_idx]` 的下标访问语法。查了 transformers 源码后，第一次修复尝试使用了 `cache.key_cache[layer_idx]` 和 `cache.value_cache[layer_idx]`——这是某些版本文档中提到的接口。

### 第二个错误：`'DynamicCache' object has no attribute 'key_cache'`

实际在 transformers 5.9.0 中，`DynamicCache` 的内部结构是 `cache.layers`，每个 layer 是一个包含 `.keys` 和 `.values` 属性的对象。最终确认的正确 API 是：

```python
from transformers.cache_utils import DynamicCache

# 读取
num_layers = len(cache)  # layer 数量
k = cache.layers[layer_idx].keys    # shape: [batch, num_kv_heads, seq_len, head_dim]
v = cache.layers[layer_idx].values

# 写入/更新
cache.update(key_tensor, value_tensor, layer_idx)
```

`DynamicCache.update()` 方法会在指定 layer 追加新的 KV 状态。如果该 layer 不存在则创建。这个设计很优雅——构造新 cache 时只需创建空实例，然后逐 layer 调用 `update` 即可。

### Batched Decode 的 KV Cache 对齐

Batched Decode 的难点在于：running batch 中每个序列的 KV Cache 长度不同（因为 prompt 长度不同、已生成 token 数不同）。要将它们拼成一个 batch 送入模型，必须做对齐。

选择了左 padding 方案：找到 batch 中最长的 KV 长度 `max_kv_len`，对较短的序列在 seq_len 维度（dim=2）的**左侧**补零。这样有效内容都在右侧，与 causal attention 的因果掩码方向一致。同时构造对应的 `attention_mask`，将 padding 位置标记为 0。

forward 完成后，需要将 batched 的输出 KV Cache 拆分回各序列。关键操作是从右侧取每个序列的有效长度部分（`k_full[:, :, -new_valid_len:, :]`），并调用 `.contiguous()` 确保内存连续。这个 slice + contiguous 操作在 28 层 × batch_size 个序列上执行，是主要的 Python 开销来源。


## 实验结果

### 实验 1: 标准配置

| 参数 | 值 |
|------|-----|
| 请求数 | 16 |
| Prompt 长度 | ~37 tokens (avg) |
| 最大生成 | 64 tokens/request |
| 最大 Batch | 4 |
| 到达间隔 | 0.02s |

| 模式 | 吞吐 (tok/s) | 加速比 | P95 延迟 | TTFT (avg) |
|------|-------------|--------|----------|-----------|
| Serial | 54.4 | 1.00x | 18839ms | 29.3ms |
| CB Sequential Decode | 54.1 | 1.00x | 2458ms | 21.4ms |
| CB Batched Decode | 48.7 | 0.90x | 2901ms | 21.4ms |

最引人注目的数字是 P95 延迟：串行模式下最后完成的请求需要等待 18.8 秒，而 CB 模式下只需要 2.5 秒。这是因为串行模式的延迟 = 请求在队列中的等待时间 + 自身生成时间，而 CB 模式下请求几乎到达即开始处理。

吞吐方面，Batched Decode（48.7 tok/s）反而比 Serial（54.4 tok/s）慢了 10%。这完全出乎最初的理论预期。

### 实验 2: 大 Batch 配置

| 参数 | 值 |
|------|-----|
| 请求数 | 32 |
| 最大 Batch | 8 |
| 最大生成 | 64 tokens/request |

| 模式 | 吞吐 (tok/s) | 加速比 | P95 延迟 |
|------|-------------|--------|----------|
| Serial | 54.6 | 1.00x | 36367ms |
| CB Sequential Decode | 54.3 | 1.00x | 2428ms |
| CB Batched Decode | 49.8 | 0.91x | 2546ms |

32 个请求时串行模式的 P95 延迟飙到 36.4 秒——最后几个请求要等前面 30 个都生成完。CB 模式的 P95 仍然只有 2.4 秒，延迟改善达到 **15 倍**。

吞吐没有变化，Batched Decode 仍然比 Serial 慢约 9%。batch_size 从 4 提到 8 并没有改善吞吐，进一步印证了 Python 开销是瓶颈的判断。

### 实验 3: 长生成配置

| 参数 | 值 |
|------|-----|
| 请求数 | 16 |
| 最大生成 | 128 tokens/request |
| 最大 Batch | 4 |

| 模式 | 吞吐 (tok/s) | 加速比 | P95 延迟 |
|------|-------------|--------|----------|
| Serial | 54.6 | 1.00x | 37513ms |
| CB Sequential Decode | 54.3 | 0.99x | 4918ms |
| CB Batched Decode | 49.0 | 0.90x | 5721ms |

生成长度翻倍后，吞吐保持稳定（54.6 vs 54.4），说明 4090 的 decode 性能不受 KV Cache 长度影响——在 7B 模型 + 128 tokens 的规模下还远未触及显存带宽瓶颈。

P95 延迟方面，CB Sequential Decode 从 2.5s 涨到 4.9s，基本是 2 倍关系，符合生成长度翻倍的预期。

### 实验 4: Batch Size Scaling

只跑 Batched Decode 模式，对比 max_batch_size = 1, 2, 4, 8 的吞吐差异：

| Batch Size | 吞吐 (tok/s) | P95 延迟 |
|-----------|-------------|----------|
| 1 | 48.2 | 3126ms |
| 2 | 47.7 | 3414ms |
| 4 | 48.1 | 3179ms |
| 8 | 48.4 | 3119ms |

四种 batch size 的吞吐几乎相同（47.7 ~ 48.4 tok/s），差异在噪声范围内。这个结果直接证明了：**增大 batch size 完全没有带来吞吐提升**。GPU 的并行计算能力被 Python 层面的 KV Cache pad/cat/slice/contiguous 操作所掩盖——batch size 越大，每个 decode step 的 Python 开销越大，恰好抵消了 GPU 侧的计算效率增益。


## 根因分析：为什么 Batched Decode 不如串行

理论上 decode 阶段是 memory-bandwidth-bound 的：每个 token 只做一次 attention 和一次 FFN，计算量很小，瓶颈在于从显存加载模型权重和 KV Cache。当多个序列共用同一次 forward 时，权重只需加载一次即可服务所有序列，理论吞吐应随 batch size 线性增长。

但这个理论假设前提是 KV Cache 的管理操作是零开销的——在 vLLM 中，PagedAttention 通过 C++/CUDA kernel 直接在 GPU 显存中管理 KV blocks，对齐和拼接操作在 kernel 层面完成，Python 不参与。

而本实验的 Python 实现中，每个 decode step 需要执行：

1. **遍历所有序列取 KV 长度**：`kv_lengths = [seq.kv_cache.layers[0].keys.shape[2] for seq in active_seqs]`
2. **逐层逐序列做左 padding**：28 层 × batch_size 次 `F.pad` 调用
3. **逐层拼接**：28 次 `torch.cat` 沿 batch 维度
4. **构造 DynamicCache 并 update**：28 次 `batched_cache.update()`
5. **forward 后拆分**：逐序列逐层 slice + contiguous，28 × batch_size 次

对于 batch_size=4、28 层的 7B 模型，每个 decode step 有 28×4 = 112 次 `F.pad`、28 次 `torch.cat`、28×4 = 112 次 slice + contiguous。这些操作虽然每个都很快（微秒级），但累积起来远超一次 model forward 的 GPU 计算时间。

做一个粗略估算：4090 上 7B FP16 模型单 token 的 forward 时间约 18ms（= 1/54.4s × 1000 ÷ 1 token）。而 Python 管理 KV Cache 的开销，从实际数据反推，每个 step 额外增加了约 2ms（48.7 tok/s 对应 20.5ms/token，比 serial 的 18.4ms 多了 2.1ms）。这 2ms 的额外开销看似不大，但在 64 个 decode step 中累积为 134ms，对总时间（1164ms/request @ serial）的贡献约 11.5%——恰好对应 0.90x 的吞吐下降。

### 为什么 Sequential Decode 没有这个问题

CB Sequential Decode 模式虽然也维护多个序列的 KV Cache，但 decode 时逐序列 forward，每次直接传 `seq.kv_cache` 给模型，无需任何 pad/cat/slice 操作。模型 forward 返回的 `past_key_values` 直接赋值回 `seq.kv_cache`，开销为零。所以它的吞吐（54.1 tok/s）几乎与 Serial（54.4 tok/s）相同。


## 核心结论

### Continuous Batching 的真正价值是延迟，不是吞吐

在 Python 实现层面，CB 无法提升吞吐——甚至 Batched Decode 还会降低吞吐。但 CB 的调度机制本身带来了延迟的本质性改善：

| 指标 | Serial | CB Sequential | 改善倍数 |
|------|--------|---------------|---------|
| P95 延迟 (16 req) | 18.8s | 2.5s | **7.5x** |
| P95 延迟 (32 req) | 36.4s | 2.4s | **15.2x** |
| P95 延迟 (16 req, 128 tok) | 37.5s | 4.9s | **7.7x** |

这个改善来自调度层面：请求到达即开始处理，无需排队。即使 decode 阶段仍然是逐序列 forward（不做 batch），仅靠调度策略就能将尾部延迟降一个数量级。

### 吞吐提升需要 C++/CUDA 级别的 KV Cache 管理

从实验数据可以明确：Python 层面操作 KV Cache tensor（pad, cat, slice, contiguous, 对象创建/销毁）的开销太大，无法兑现 Batched Decode 的理论收益。这就是 vLLM 设计 PagedAttention 的根本原因——KV Cache 的分配、对齐、拼接必须在 CUDA kernel 中完成，跳过 Python 解释器的开销。

Batch Size Scaling 实验（bs=1/2/4/8 吞吐无差异）更是直接证明：**瓶颈不在 GPU 计算侧，而在 Python 层面的 KV Cache 管理操作**。如果瓶颈在 GPU，增大 batch size 应该能看到吞吐线性增长；实际没有任何增长，说明 GPU 端的 forward 时间相对于 Python 开销可以忽略不计。

### 工程启示

对于教学和原型验证目的，CB Sequential Decode 是性价比最高的方案——它获得了 CB 的全部延迟优势，吞吐没有任何损失（因为不需要操作 KV Cache），实现复杂度也远低于 Batched Decode。

如果要追求真正的吞吐提升（3-5x），必须走向 C++ 实现的 PagedAttention 或类似方案。这也解释了为什么所有生产级推理框架（vLLM、TGI、TensorRT-LLM）都不是纯 Python 的——Python 在热路径上的开销对于推理性能来说是不可接受的。


## 附录：环境踩坑备忘

模型下载方面，hf-mirror.com 下载到一半速度归零是常见问题，最终改用 modelscope 的 `snapshot_download` 拉完全部 4 个 safetensors 分片（~15GB）。另外 transformers 5.9.0 中 `huggingface-cli` 已废弃，新版命令是 `hf`。

服务器连接方面，用 `expect` 自动化了 `ssh-copy-id` 实现免密登录，之后 scp 传脚本和拉日志都不再需要交互输入。run_benchmark.sh 中需要显式指定 Python 路径（AutoDL 的 miniconda3 默认不在 PATH 中）。


## 附录：完整数据表

| 实验 | 模式 | 请求数 | Max Batch | Max Tokens | 吞吐 (tok/s) | 总 tokens | 总耗时 | P95 延迟 | TTFT (avg) |
|------|------|--------|-----------|-----------|-------------|-----------|--------|----------|-----------|
| 标准 | Serial | 16 | - | 64 | 54.4 | 1024 | 18.84s | 18839ms | 29.3ms |
| 标准 | CB Seq | 16 | 4 | 64 | 54.1 | 1024 | 18.93s | 2458ms | 21.4ms |
| 标准 | CB Batch | 16 | 4 | 64 | 48.7 | 1024 | 21.02s | 2901ms | 21.4ms |
| 大 Batch | Serial | 32 | - | 64 | 54.6 | 2048 | 37.53s | 36367ms | 25.1ms |
| 大 Batch | CB Seq | 32 | 8 | 64 | 54.3 | 2048 | 37.69s | 2428ms | 21.2ms |
| 大 Batch | CB Batch | 32 | 8 | 64 | 49.8 | 2048 | 41.10s | 2546ms | 21.0ms |
| 长生成 | Serial | 16 | - | 128 | 54.6 | 2048 | 37.51s | 37513ms | 28.8ms |
| 长生成 | CB Seq | 16 | 4 | 128 | 54.3 | 2048 | 37.73s | 4918ms | 21.4ms |
| 长生成 | CB Batch | 16 | 4 | 128 | 49.0 | 2048 | 41.81s | 5721ms | 21.4ms |
| Scaling | CB Batch (bs=1) | 16 | 1 | 64 | 48.2 | 1024 | 21.24s | 3126ms | 33.8ms |
| Scaling | CB Batch (bs=2) | 16 | 2 | 64 | 47.7 | 1024 | 21.47s | 3414ms | 34.0ms |
| Scaling | CB Batch (bs=4) | 16 | 4 | 64 | 48.1 | 1024 | 21.29s | 3179ms | 34.3ms |
| Scaling | CB Batch (bs=8) | 16 | 8 | 64 | 48.4 | 1024 | 21.17s | 3119ms | 33.4ms |
