# Speculative Decoding：性能优化与推理系统实现

Speculative Decoding（投机解码，下文简称 Spec Decoding）用轻量 draft model 连续预测多个候选 token，再由 target model 一次验证，以减少串行 decode 次数。

它的核心不是单纯提高接受率，而是在以下因素之间取得平衡：

- draft 的预测准确率与运行成本；
- target verification 的额外计算量；
- draft 与 target 的并行方式；
- batch、Tensor Parallelism（TP）、KV cache、CUDA Graph 等系统能力；
- 延迟、吞吐和显存占用。

## 1. 核心性能判断

投机解码本质上是用额外计算换取更少的串行 decode 步数。它最适合 GPU 算力尚未饱和的 memory-bound 场景，而不是所有推理任务。

### 适合开启

- 小 batch、低 QPS 或单请求；
- decode 主要受显存带宽限制；
- GPU 计算单元仍有余量；
- draft 接受率高；
- 业务关注 TPOT（Time Per Output Token）和单请求延迟。

此时 target 一次验证多个 token 所增加的 FLOPs 接近“利用闲置算力”，却能显著减少串行步数。

### 通常不适合开启

- 大 batch、高 QPS；
- target 已经 compute-bound；
- 目标是最大化整体吞吐；
- draft 与 target 分布差异大，接受率低；
- draft 权重和 KV cache 会明显挤占 target 的并发容量。

大 batch 下，verification 的额外计算不再免费，还会与其他请求争抢算力，因此可能比普通 batched decode 更慢。

投机解码还存在以下固定成本：

- **额外显存**：需要保存 draft 权重和 draft KV cache，会压缩 target KV pool。
- **兼容成本**：tree attention、CUDA Graph、MoE、混合注意力等组合需要单独适配。
- **维护成本**：涉及回滚、KV 槽位整理和变长 batch，容易产生边界问题。
- **TTFT 无收益**：投机解码只优化 decode；prefill 和首 token 通常不受益。

因此，即使存在合适的 draft model，也不应默认无条件开启。生产环境应根据 batch size、接受率、负载和显存动态决策。

## 2. Draft model 的优化

Draft model 的目标是在较低运行成本下尽可能接近 target model 的预测结果。主要矛盾是 efficiency 与 accuracy 的权衡。

### 2.1 Draft model 的来源

#### 从头训练

从头训练一个与 target model 配套的 draft model。

优点：

- 可以专门设计成复用 target hidden states 或 KV cache；
- 架构和训练目标可针对投机解码优化。

缺点：

- 训练成本高；
- 每个 target model 可能都需要单独训练。

#### 直接使用已有小模型

例如使用 Qwen 0.6B 为 32B 模型生成候选。

优点：

- 开箱即用；
- 不需要额外训练。

缺点：

- 并非所有 target model 都有同系列小模型；
- 独立模型与 target 的分布可能不一致，接受率受限；
- 无法直接利用 target 的中间特征。

#### 从已有小模型蒸馏

以现有小模型为初始化，将其蒸馏为能够接收 target feature 或 KV cache 的多层 draft model。

当大小模型的训练数据和词表高度接近时，这种方案通常比从头训练更高效。例如 Qwen3-0.6B 与 32B 的训练分布较接近，适合作为蒸馏起点。

这条路线兼顾：

- 已有小模型的语言建模能力；
- target 中间状态带来的条件信息；
- 低于从头训练的训练成本。

### 2.2 模型规模与形状

Draft model 的预测能力与参数规模强相关。当前很多方案让 draft 层宽与 target 单层一致，但这不是唯一选择。

一个值得研究的方向是**宽而浅的 draft model**：

- 增加宽度以扩展参数量和表达能力；
- 保持较少层数，控制串行执行时间；
- 在相同延迟预算下寻找更合适的宽度、深度和接受率组合。

已有观察显示，4 层 EAGLE 的预测准确率可明显高于 1 层版本，但层数增加也会迅速提高 draft 时间。因此不能只看接受率，需要比较端到端加速比。

相关论文：[Scaling Laws for Speculative Decoding](https://arxiv.org/pdf/2505.07858)

### 2.3 长距离预测衰减

使用 target hidden states 作为 draft 输入时，预测距离越远，draft 的准确率通常越低。

Target attention 可简化表示为：

```text
o = softmax(qk) · v
h = o · w
```

Hidden state `h` 已经聚合了 KV cache，并包含 target 对 prefix token 的历史注意力权重。直接复用 hidden state，相当于用过去的注意力分布近似后续预测：

- 预测距离较近时，注意力分布变化较小，近似通常有效；
- 预测距离较远时，新的 query 对 prefix 的注意力可能明显变化，历史聚合结果会产生偏差。

可能的优化方向：

1. **从复用 feature 转向复用 KV cache**  
   让 draft 使用 target 的 K/V，但自行重新计算对 prefix 的注意力权重，减少 hidden state 中历史注意力偏见的影响。（dflash idea）

2. **增加 draft model 容量**  
   使用更多层或更多参数提升长距离预测能力。

3. **提高 draft 自身建模能力**  
   改进蒸馏数据、训练目标和架构，而不是只依赖 target feature。

4. **调整 verification 准则**  
   放宽接受条件可提高接受长度，但会失去严格无损保证。

### 2.4 Draft 运行时间

Draft 不能只追求高接受率。以 `depth=10` 为例，EAGLE-3 的 draft 时间可能达到 target 时间的一半，继续增加预测深度未必能提高端到端性能。

可考虑两类优化：

- **并行执行 draft 与 target**：以 SSD、PEARL 类方案为代表；
- **并行生成多个候选位置**：以 Medusa 类方案为代表。

Medusa 的各预测头缺少完整的顺序依赖，预测能力通常弱于 EAGLE 或 MTP，但并行度更高。两者仍然是准确率与并行度的权衡。

### 2.5 Adaptive draft length

Draft length 是一轮投机中 draft model 连续预测的 token 数。固定长度难以适配不同 batch 和运行负载：

- batch 小、GPU 空闲时，可增加 draft steps；
- batch 大、target 接近 compute-bound 时，应减少或关闭 draft；
- 接受率下降时，应缩短 draft length，避免为大量必然被拒绝的 token 付费。

动态 draft length 是生产系统比固定参数更重要的能力。

## 3. Verification 的优化

### 3.1 Tree attention

Tree attention 在每个位置保留多个候选 token，并使用自定义 attention mask 让 target 一次验证整棵候选树。

优点：

- 提高命中 target 路径的概率；
- 有机会增加单轮接受 token 数。

缺点：

- draft token 数量显著增加；
- verification 计算和 KV 占用增大；
- `batch_size > 1` 时，扩展后的候选树更容易使 target compute-bound；
- 接受路径可能不连续，需要整理 KV 槽位。

Tree attention 更适合小 batch 和较高分支质量的场景。

### 3.2 Lossy verification

严格 verification 会拒绝一部分语义质量较高、但不符合无损采样条件的 draft token。Lossy verification 通过放宽接受标准提高接受率和加速比。

代价是：

- 输出分布与原 target model 不再严格一致；
- 需要单独评估任务质量、安全性和可复现性；
- 不适合要求无损等价的场景。

## 4. Draft-target 并行与分离

### 4.1 SSD 的关键取舍

SSD 能异步执行的根本原因，是 draft 与 target 没有数据依赖。独立小模型可以放到另一张 GPU 上提前运行。

EAGLE-3、dFlash 等高接受率方案依赖 target hidden states。Draft 必须等待 target 产生这些状态，因此难以完全异步。

两种方向形成基本矛盾：

- **追求接受率**：draft 使用 target hidden states，耦合更强；
- **追求异步并行**：draft 独立运行，但通常损失预测准确率。

SSD 论文附录 E 尝试让 EAGLE-3 使用自身 hidden state 代替 target hidden state，从而支持异步执行，但也削弱了 EAGLE-3 的核心信息优势。

### 4.2 为什么需要 draft-target 分离

Draft 与 target 的推理行为不同：

- draft 参数量小，适合的 TP degree 通常更小；
- draft 连续执行多次 autoregressive decode；
- target 通常执行一次多 token verification；
- 两者同卡运行会争抢计算、显存和 KV cache；
- target 开启 TP 后速度提高，draft 的相对耗时会进一步放大。

传统 EAGLE 部署中，draft 可能位于 GPU 0，而 target 横跨 GPU 0–7。GPU 0 同时保存两套状态，上下文增长后更容易 OOM。

Draft-target 分离的主要目标是：

- 为两者选择独立并行策略；
- 避免计算资源争抢；
- 隔离 KV cache 和权重显存；
- 为 draft-target overlap 提供执行基础。

## 5. SGLang 的 Spec Decoding

### 5.1 TP 策略

SGLang 中 draft 与 target：

- 共用 GPU；
- 共用 TP degree；
- 共用 TP/NCCL group；
- TP degree 由唯一的 `--tp-size` 决定。

资源隔离主要依赖 `mem_fraction_static` 和 KV pool 预算，而不是物理 GPU 切分。

启用 `--enable-dp-attention` 时，draft 会切换到 attention TP group 做 all-reduce，但物理 GPU 仍然共享。

### 5.2 Draft 为什么仍跟随 target 做 TP

Draft 通常只有一层或少量层，看起来没有必要切到多卡。它跟随 target TP 的原因不是自身规模，而是系统耦合：

- 对齐 target hidden state 的分片；
- 共享或复用 KV；
- 避免额外 gather、scatter 和跨组通信；
- 简化 draft-target 数据流。

代价是 tiny draft layer 的计算很少，TP all-reduce 延迟可能占据较大比例。极端情况下，将一层 draft 切到 8 卡，通信时间可能高于计算时间。

因此，TP 布局跟随 target 是工程约束，真正的优化方向是让 draft 在既定 TP 下尽量轻。

### 5.3 TP 下的 draft 轻量化

#### Top-k=1 时跳过全词表 softmax

当 `topk == 1` 时，候选树退化为单链，下游只需要选中的 token id，不需要 `topk_p`：

```text
argmax(logits) = argmax(softmax(logits))
```

因此可以直接对 logits 做 argmax，省去在约 12.8 万维词表上的 softmax。对于只有一层的 draft，LM head 和 softmax 可能占据较大开销。

已知边界：AMD FP8 模型在 logits 相等时可能出现 tie-break 行为差异，进而影响 token 选择和接受率。

#### 截断 LM head 词表

只加载 hot token 子集，例如将词表从约 120000 缩小到 32000：

1. 在子词表上执行 LM head 和 softmax；
2. 得到 draft token id；
3. 通过 D2T 映射恢复为 target token id。

其他减负策略还包括：

- `topk == 1` 快路径；
- 大 batch 下设置 `num_steps=0`，直接跳过 draft；
- FR-Spec 等词表截断方案。

### 5.4 Batch inference 的核心设计

一个 batch 中真正变化的是：

1. 每个请求已提交的前缀长度；
2. verification 后每个请求接受的 token 数。

SGLang 的策略是：

- 输入始终按 `[batch_size × num_draft_tokens]` 定宽执行；
- 前缀长度差异通过 `positions`、前缀和和 `indptr` 编码；
- 接受长度差异通过 per-request `accept_lens` 和 `accept_index` 编码；
- verification 后再 compact 并释放 overshoot KV；
- kernel 和 CUDA Graph 始终接收规整张量。

下文以两个请求、`num_draft_tokens = 4` 为例。这里的 `accept_lens` 表示示例中接受的 draft 节点数；具体实现若包含 bonus token，需要按对应字段语义额外加 1。

#### Draft 构造定宽候选

设请求 A、B 的前缀长度分别为 5 和 3。`topk=1` 时，每个请求生成一条 4 节点单链：

```text
A: nA0 → nA1 → nA2 → nA3
B: nB0 → nB1 → nB2 → nB3
```

打平后：

```text
draft_token = [nA0 nA1 nA2 nA3 | nB0 nB1 nB2 nB3]
index           0   1   2   3     4   5   6   7
```

两个请求都贡献 4 个节点，因此 query 张量形状固定为 `[8]`。

#### Positions 吸收前缀差异

每个节点的绝对位置等于请求前缀长度加树内深度：

```text
A: [5, 6, 7, 8]
B: [3, 4, 5, 6]

positions = [5, 6, 7, 8, 3, 4, 5, 6]
```

前缀长度不同只影响 position 起点，不改变张量形状。

#### Tree mask 编码可见性

每个 draft 节点可以看到：

- 本请求的全部 prefix；
- 树内自己的祖先节点。

Mask 由两部分组成：

```text
prefix mask size = seq_lens_sum × num_verify_tokens
tree mask size   = num_verify_tokens² × batch_size
```

`seq_lens_sum = 5 + 3 = 8`，所以两部分大小均为 32。

`topk=1` 是单链，树内 mask 为下三角：

```text
     n0 n1 n2 n3
n0 [ ✓  .  .  . ]
n1 [ ✓  ✓  .  . ]
n2 [ ✓  ✓  ✓  . ]
n3 [ ✓  ✓  ✓  ✓ ]
```

`build_tree_kernel_efficient` 可按 `grid=(batch_size,)` 让每个 program 处理一个请求，并用 `seq_lens` 的前缀和定位各请求的 prefix 区段。

#### Verification 的 KV 与 query 布局

每个请求贡献 4 个 query：

```text
qo_indptr = [0, 4, 8]
```

每个请求的 KV 长度为 prefix 加 4 个 draft token：

```text
paged_kernel_lens = [5 + 4, 3 + 4] = [9, 7]
kv_indptr = [0, 9, 16]
```

布局为：

```text
KV(A): [prefix 0..4][nA0 nA1 nA2 nA3]
KV(B): [prefix 0..2][nB0 nB1 nB2 nB3]
```

Target 一次 forward 为 8 个 query 打分。Query 数量定宽，KV 长度差异由 `kv_indptr` 分段表示。

#### 接受数不同

假设验证结果为：

```text
A: n0 ✓  n1 ✓  n2 ✓  n3 ✗  → 接受 3 个
B: n0 ✓  n1 ✗                 → 接受 1 个
```

变长结果编码为：

```text
accept_lens = [3, 1]
new_seq_lens = [5 + 3, 3 + 1] = [8, 4]
```

Batch 形状没有变化，接受数量只保存在 `[batch_size]` 的向量中。

#### 释放 overshoot KV

Verification 为每个请求预分配 4 个 KV 槽，但实际只接受 3 个和 1 个：

```text
A: [t0 t1 t2 | x3]     释放 1 个 overshoot
B: [t0 | x1 x2 x3]     释放 3 个 overshoot
```

`topk=1` 时，接受路径必然是单链前缀，已经位于块的最前面，无需 compact，只需释放尾部未接受槽位。

CPU 可按固定 stride 切片：

```text
A: predict[0:3]
B: predict[4:5]
```

#### Top-k>1 时的差异

`topk=2` 时，4 个节点可能形成以下树：

```text
    n0
   /  \
 n1    n2
 |
 n3
```

Tree mask 不再是简单下三角：

```text
     n0 n1 n2 n3
n0 [ ✓  .  .  . ]
n1 [ ✓  ✓  .  . ]
n2 [ ✓  .  ✓  . ]
n3 [ ✓  ✓  .  ✓ ]
```

如果接受路径是 `n0 → n2`，对应块内索引 `{0, 2}`，KV 不连续。此时需要：

1. `_compact_accept_to_front` 根据 `accept_index` gather 接受节点；
2. 将 `{n0, n2}` 移到块前部；
3. `move_accept_tokens_to_target_kvcache` 将对应 KV 搬到连续 target 槽位；
4. 释放其他 overshoot 槽位。

因此，`topk=1` 的 compact 是恒等操作，可以跳过；`topk>1` 才需要真正搬运。

### 5.5 CUDA Graph

SGLang 通过定宽 query 布局稳定 CUDA Graph 输入形状。变长信息不改变主张量结构，而是编码在：

- `positions`；
- `qo_indptr` / `kv_indptr`；
- `accept_lens`；
- `accept_index`；
- KV block table。

这使 kernel 和 CUDA Graph 可以持续复用规整执行路径。

### 5.6 Draft worker 的功能边界

SGLang/SRT 中大量逻辑通过 `is_draft_worker` 区分 target 与 draft。准确理解不是“draft 要砍掉大量能力”，而是将能力分成三类。

#### 可直接复用

Draft 复用大部分计算栈：

- 模型结构；
- attention backend；
- CUDA Graph；
- KV pool；
- 量化；
- ModelRunner 基础代码。

#### 必须关闭

面向外部服务或具有进程级全局副作用的能力只应由 target 管理：

- EPLB 和专家分布；
- routed-experts capturer；
- two-batch overlap；
- tokenizer / detokenizer 初始化；
- metrics；
- 部分服务级 prefix cache 和 HiCache L3 写入。

Draft 是 target 的内部加速器，不应重复产生对外状态或统计，否则可能污染 target 的服务状态。

#### 需要专门适配

单 token、单模型语义下设计的模块，需要适配“多 token + 树形候选”：

- Mamba track 和 page table 需要扩展尺寸；
- KV indices 需要按候选树重算；
- `qo_indptr` 需要定宽构造；
- 接受路径需要 compact 和 rollback。

Target 是完整的对外服务模型；draft 复用其计算栈，但剥离服务级副作用，并适配投机解码的数据布局。

## 6. SGLang Adaptive Speculative Decoding

Adaptive Speculative Decoding 只支持部分配置，适用范围是：

- EAGLE / EAGLE3；
- `topk=1`。

以下场景会自动禁用：

- DP attention；
- multi-layer EAGLE；
- TBO；
- MTP；
- 其他尚未适配的组合。

具体支持范围可能随 SGLang 版本变化，应以当前代码和启动日志为准。

### 6.1 第一级：按 batch size 限制 draft steps

每次 decode forward 开始时调用 `activate_step_by_batch(bs)`。Batch size 变化后，系统切换到预建的 `SpecRuntimeState`，只替换状态引用，不重新 capture CUDA Graph。

默认 batch size 档位为 1、8、32、64：

- `bs = 1–7`：允许 `[1, 3, 7]`，最大可 draft 7 步；
- `bs = 8–31`：允许 `[0, 1, 3]`，可降到 0 步；
- `bs = 32–63`：允许 `[0, 1]`，只能浅 draft 或关闭；
- `bs >= 64`：只允许 `[0]`，强制关闭 draft。

这组阈值直接体现了“大 batch 下 verification 额外计算更昂贵”的系统判断。

### 6.2 第二级：按接受率动态调整

系统在 batch size 允许的候选集合内，根据近期接受率选择 draft steps：

- 接受率高时增加 steps；
- 接受率低时减少 steps；
- 最低可降到 `step=0`。

Batch size 决定可选范围，接受率决定范围内的具体档位。

### 6.3 Step=0 的执行方式

降到 `step=0` 后：

1. 不执行实际 draft；
2. 构造一个 `draft_token_num=1` 的 trivial verify；
3. 复用 `TARGET_VERIFY` CUDA Graph；
4. 效果等价于普通 decode。

默认仍可执行 `draft_extend`：即使当前不 draft，也继续将 target hidden states 写入 draft KV，使 draft cache 保持温热。Batch 之后缩小时，可以直接恢复 `steps>0`，避免冷启动。

## 7. 优化问题的统一视角

Spec Decoding 的性能优化可以归纳为四个问题：

1. **候选是否足够准**  
   由 draft 架构、训练方式、target feature/KV 复用方式和预测距离决定。

2. **候选是否足够便宜**  
   由 draft 层数、LM head、softmax、TP 通信和 CUDA Graph 决定。

3. **验证是否充分利用硬件**  
   由 batch size、tree width、定宽布局、paged KV 和 target 当前瓶颈决定。

4. **draft 与 target 是否合理协同**  
   由同卡或分离部署、数据依赖、TP group、显存预算和异步执行能力决定。

最终应以端到端指标判断，而不是单独优化接受率：

- TPOT；
- 吞吐；
- draft 时间占比；
- 平均接受长度；
- target verification FLOPs；
- draft/target KV cache 占用；
- 不同 batch size 下的性能曲线。
