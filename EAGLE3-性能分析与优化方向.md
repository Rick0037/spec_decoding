# EAGLE3-SGL 性能分析与优化方向

## 1. 分析对象与结论口径

数据来源：`/tmp/eagle3_vl_profile.txt` 与同配置的 `time_stages` 墙钟结果。

运行配置：

- target：`Qwen3-VL-8B-Instruct`
- draft：`taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3`
- verify：`full_model_tree(triton_tree)`
- 树配置：`topk=8, num_steps=8, max_tree_nodes=64`
- dtype：`float16`
- 输出：130 tokens，50 轮，平均接受长度 `tau=2.60`

注意：profile 运行时 wall=20.91s、6.22 tok/s；无 profiler 的分阶段墙钟为 4.60s、28.28 tok/s。profiler 会引入明显的 CPU 和同步开销，因此：

- profile 用于定位耗时原因；
- `time_stages` 用于分析阶段墙钟占比；
- 正式 TPOT、吞吐和 HF 加速比必须使用关闭插桩后的 `bench_accept_tpot.py`。

## 2. 表一：分阶段 device 时间

| 阶段 | 总 device 时间 | ms/轮 | 占比 |
|---|---:|---:|---:|
| target verify | 1203.27ms | 24.07 | 43.4% |
| target replay | 1104.46ms | 22.09 | 39.9% |
| draft build tree | 304.64ms | 6.09 | 11.0% |
| target prefill | 101.24ms | 仅一次 | 3.7% |
| draft extend | 53.68ms | 1.07 | 1.9% |
| draft prefill init | 2.87ms | 仅一次 | 0.1% |

### 结论

1. verify + replay 占 83.3% device 时间，是主要优化对象。
2. replay 只处理平均约 2.6 个接受 token，却与 64 节点的 verify 耗时接近，说明其固定前向成本很高。
3. draft 建树排第三，但明显低于两次 target 前向。
4. prefill 只发生一次，不应优先于 decode 循环优化。

### 对应优化方向

#### P0：删除 target replay

目标：复用 verify 已经计算出的接受路径 KV、logits 和 EAGLE hidden，不再对 append 做一次完整 target forward。

按当前干净墙钟估算，replay 约为 35.35ms/轮；这是理论可消除部分，实际收益以实现为准。

实现时不能直接取消 verify 后的 cache crop：

- 树 KV 中包含未选中的分支；
- 接受路径节点在树中的物理顺序不一定等于最终序列顺序；
- 必须按接受路径重排并写回正式 KV cache；
- recovery logits 必须来自接受路径最后一个位置；
- EAGLE hidden 应通过 hook 只采集所需三层，避免返回全部层 hidden。

#### P1：StaticCache + verify CUDA Graph

64 节点 verify 的 device 时间约 24ms，但 CPU 时间更高，说明除真实计算外还有固定调度成本。

方向：

- 使用静态 KV buffer，按 cache position 原地写；
- 固定树节点数 L，并将 mask、tokens、positions 写入预分配 buffer；
- warmup 后捕获 verify CUDA Graph；
- VL 路径必须把 `rope_delta` 纳入 position buffer；
- 实际 L 或容量不满足捕获条件时回退 eager。

CUDA Graph 只能减少 launch 和 CPU 间隙，不能减少 GEMM 的真实 GPU 计算。

#### P1：减少 verify 树节点

当前 64 节点 verify 有真实 GEMM 成本，减小树可以同时降低：

- draft 建树开销；
- verify GEMM 工作量；
- mask 和 accept select 开销。

优先复测：

- `depth=2, topk=4, nodes=20`
- `depth=3, topk=2, nodes=14`
- 当前 `depth=2, topk=8, nodes=64` 等效形状

每组必须同时比较 tau、tree_nodes/round、ms/round、TPOT 和 lossless，不能只比较接受长度。

#### P2：draft 按层批量建树

将逐节点 BFS、单 token draft step 改为按深度组成 batch，一次计算同一层所有父节点的 children。

目标是减少小 GEMM、kernel launch、临时 tensor 和逐节点 D2H，暂时不必优先给 draft 做 CUDA Graph。

## 3. 表二：关键事件计数

| 事件 | 总次数 | 每轮约次数 | CPU 时间 | 主要含义 |
|---|---:|---:|---:|---|
| cudaLaunchKernel | 260875 | 5218 | 1770.8ms | eager launch 很重 |
| aten::copy_ | 50235 | 1005 | 777.5ms | 大量拷贝和 dispatch |
| aten::cat | 17396 | 348 | 330.4ms | KV/激活拼接 |
| cudaStreamSynchronize | 5303 | 106 | 66.5ms | 同步点较多 |
| aten::item | 3406 | 68 | 84.7ms | 标量 D2H 读取 |
| Memcpy DtoH | 3850 | 77 | GPU 3.3ms | 传输本身不贵 |
| Memcpy HtoD | 1453 | 29 | GPU 1.2ms | 传输本身不贵 |

### 结论

1. launch 数量是 CPU 侧最突出的指标之一。
2. copy/cat 的 CPU 成本高于其 GPU 成本，说明临时对象、dispatch 和 cache 管理值得优化。
3. HtoD/DtoH 的实际传输时间很小，主要问题是同步和调用次数，而不是 PCIe 带宽。

### 对应优化方向

#### 静态 KV 和输入 buffer

- 用 StaticCache 减少逐层 KV cat/copy；
- 复用 tokens、positions、mask、cache_position、indptr buffer；
- 对可复用输入采用原地 copy_；
- 检查 `acts_all = torch.cat(...)`，避免无界逐轮拼接或改为预分配特征区域。

StaticCache 只能消除 KV 相关 copy/cat，不会自动消除所有激活拷贝。

#### 合并 argmax 和 D2H

verify accept select：

- 对全部树节点一次 argmax；
- 拼接 root logits 的预测；
- 只做一次 tolist()；
- 后续路径比较在 host list 上完成，避免沿不同叶路径重复计算同一节点的 argmax。

draft 建树：

- 不要每个节点单独 tolist()；
- 按层批量计算 topk；
- 尽量让 token tensor 留在 GPU，在层间直接传递；
- 整层或整棵树结束后统一取回需要的 host 数据。

## 4. 表三：Top self CUDA

### 4.1 aten::mm 是主要 GPU 计算项

- 调用 28812 次；
- Self CUDA 为 2.196s；
- 占 Self CUDA 76.78%；
- 平均约 43.9ms/轮。

这说明当前不只是 CPU launch 问题，target 前向确实存在大量真实 GEMM 计算。

优先方向：

1. 删除 replay，减少一次完整 target 前向；
2. 减小或剪枝 draft 树，减少 verify 的 GEMM 工作量；
3. draft 按层 batch，减少大量小 GEMM；
4. 暂时不优先替换 CUTLASS/cuBLAS GEMM 微内核。

### 4.2 Triton tree attention 不是瓶颈

自定义 `_fwd_kernel`：

- 调用 1800 次，即每轮 36 次；
- Self CUDA 总计约 90.09ms；
- 占 Self CUDA 3.15%；
- 平均每轮约 1.8ms。

因此不建议当前优先重写 Triton extend attention。verify 的主要成本在其他 GEMM、临时操作和 eager 调度。

### 4.3 小算子融合属于第二梯队

`mean / pow / rsqrt / mul / add` 等算子调用频繁，单次 GPU 时间只有几微秒。

可考虑：

- 融合 RMSNorm；
- 融合 RoPE 相关 elementwise 操作；
- 通过 CUDA Graph 减少小 kernel 的启动成本。

这些优化有助于降低 launch 数量，但整体优先级低于删除 replay 和减少 target 计算。

## 5. 表四：Top self CPU

主要项目：

| 算子/事件 | Self CPU | 调用次数 | 优化含义 |
|---|---:|---:|---|
| cudaLaunchKernel | 1.655s | 247978 | CUDA Graph、融合 |
| aten::mm | 762.3ms | 28812 | 减少前向和小 GEMM |
| aten::mul | 591.3ms | 53512 | 融合 elementwise |
| aten::add | 387.8ms | 32957 | 融合 elementwise |
| aten::copy_ | 387.2ms | 50235 | 静态 buffer、原地写 |
| empty_strided / empty | 约 420ms | 67081 | 减少临时分配 |
| as_strided | 94.6ms | 118083 | 减少视图和临时操作 |
| transpose | 112.2ms | 58607 | 减少重复布局变换 |
| reshape | 81.9ms | 42772 | 减少临时 reshape |
| aten::to | 78.0ms | 56526 | 避免重复 device/dtype 转换 |

### 对应优化方向

- 用 CUDA Graph 减少 launch 和 eager dispatch；
- 预分配并复用静态输入输出 buffer；
- 减少重复 reshape、transpose、to 和小 tensor 构造；
- 按层批量 draft，减少逐节点 Python 调用；
- 批量 argmax，减少 accept select 的重复计算。

`Activity Buffer Request` 属于 profiler 自身，不是模型优化对象。

## 6. 三个新增方案的可行性与收益评估

当前基线为：98.15ms/轮、tau=2.504、HF TPOT=34.10ms。若每轮成本不变，仅靠提高接受长度超过 HF，需要：

```text
tau > 98.15 / 34.10 = 2.878
```

即至少需要增加约 0.374 token/轮。若方案同时增加 draft 或 verify 开销，实际需要更高的 tau。

### 6.1 使用 EAGLE2 风格的累计分数 draft 树

#### 方案

当前 EAGLE3-SGL 的建树方式是：

- 每个父节点展开 topk 个子节点；
- 按 BFS 顺序创建节点；
- 达到 max_tree_nodes 后直接截断。

当前 `topk=8, max_tree_nodes=64` 时，根层 8 个节点，第二层最多创建 56 个节点，因此实际主要只有两层，num_steps=8 没有充分发挥作用。

EAGLE2 风格可以改为：

1. 每个节点记录从根到当前节点的累计 log 概率；
2. 每层展开后，按累计分数只保留全局 top-K 条活跃路径进入下一层；
3. 最后按分数保留高分节点及其祖先，压缩到 verify 节点预算内。

目标是在有限的 64 个 verify 节点中保留更深、更高概率的路径，而不是让大量低概率兄弟分支提前耗尽预算。

#### 可行性

算法上可行，而且 verify 不需要改变。只要树掩码和接受逻辑保持正确，改变树形状不会改变 greedy 无损性，只会改变 tau 和每轮成本。

但不能直接调用当前仓库的 EAGLE2 `expand_draft_tree_topk`：

- EAGLE3-SGL 的 draft 是自回归递归，每个 frontier 节点必须携带自己的 hidden 和 KV；
- EAGLE2 示例中的 hidden 传递和 target_hidden_refresh 逻辑与当前 AR draft 不完全相同；
- 应复用的是“累计 logp + 全局 beam + 按分数剪枝”的策略，而不是直接替换函数。

#### 收益与代价

潜在收益：中到高。它直接针对当前 BFS 硬截断导致树过浅的问题，有机会在相同 verify 节点数下提高接受长度。

主要代价：

- 每层可能先创建多于 64 个候选，再进行剪枝，draft 扩展成本会上升；
- 必须保存每条活跃路径的累计分数、hidden 和 KV；
- 如果不做按层批量化，更多 draft step 和 D2H 可能抵消 tau 收益。

因此建议先做固定 verify 节点预算的 A/B：比较当前 BFS 与 EAGLE2 风格树的实际深度、tau、tree_nodes/round、draft_build_tree ms/round 和最终 TPOT。只有 tau 增长大于额外建树成本时才保留。

结论：值得优先实验，预期比单纯继续加宽更合理，但不能预先保证端到端加速。

实现状态：已在 `tree_draft.py` 加入 `draft_tree_mode=eagle2/eagle2_batch/bfs`，默认使用 `eagle2_batch`，逐节点 EAGLE2 和 BFS 保留为对照组。实现保留了每个 frontier 节点的 AR hidden/KV，并用完整 draft 词表的 logsumexp 计算真实 log 概率。最终剪枝严格保证节点数不超过 max_tree_nodes 且 parent 链完整。合成测试已验证同预算下可以形成更深树，真实权重的 lossless 和速度仍需 A/B 确认。

随后新增 `eagle2_batch` 实验路径：每层 frontier 只做一次 batch draft forward，在 GPU 上完成累计分数和全局 beam topk，再用 index_select 重排 AR hidden/KV，整棵树结束后统一 D2H。合成测试中 serial 与 batch 得到相同 TreeLayout；topk=2、num_steps=4 时，serial 为 6 次单节点 forward，batch 降为 3 次 batch=2 forward。真实 VL 上仍需比较 bfs、eagle2、eagle2_batch 三组，批量版不预设必然更快。

### 6.2 draft 预消费 target 已确定的 token 与末位 feature

#### 方案

这里的“target prefill 的第一个 logits”应明确为：target prefill **末位 recovery_logits**，它预测第一个新生成 token xP，而不是 prompt 第 0 个位置的 logits。

当前 target prefill 后已经有：

```text
recovery_logits.argmax() -> 确定的 xP
acts_all[:, -1:, :]      -> 最后一个 prompt 位置的特征 f(P-1)
```

当前 draft 仍会在根层重新预测 xP。优化后可以：

1. 正常完成 draft prefill；
2. 将 `(xP, f(P-1))` 作为一对新输入追加给 draft；
3. draft 再输出用于预测 x(P+1) 的 root_logits；
4. target verify 将 xP 视为已确定根节点，后续 draft 候选从 x(P+1) 开始。

注意：不能把 target 的 recovery_logits 直接赋给 draft.root_logits。target 与 draft 的输出词表不同，正确做法是把 target 已确定的 token 和对应 feature 输入 draft，让 draft 自己计算下一位置分布。

#### 可行性

算法上可行，与利用 target 已确定 token 推进 draft 状态的思路一致。潜在收益包括：

- 不再让 draft 为已经确定的 xP 提供根层候选；
- 减少根层不匹配造成的整棵树浪费；
- 在相同节点预算下，将更多节点用于后续位置；
- 每轮可能多推进一个有效位置，从而提高 tau。

如果 tau 能从 2.504 提升到约 2.9 以上，同时每轮成本接近不变，就有机会越过当前 HF 基线。这里的数值只是敏感性分析，不是预期必然结果。

#### 代价与风险

- 需要额外一次 draft 单 token 前向；
- target 仍需处理 xP，才能得到验证 x(P+1) 所需的 logits，因此该方案本身不减少 target forward；
- draft 会比 out_ids 提前消费一个 token，prefix_len、draft KV 长度和 RoPE 位置都必须同步调整；
- verify 后更新 draft 时不能再次写入 xP，否则会产生重复 KV；
- 若要在每一轮都使用该方案，而不只是首次 prefill，必须同样处理每轮的 correction/bonus 和 draft 状态对齐。

结论：可行性中等，潜在 tau 收益中等，但工程复杂度较高。适合先做首次 prefill 的小实验验证正确性和接受率，不应直接假设“与论文对齐就一定提速”。

实现状态：已按 feature/modity_frist_token 分支提交 9fd02c8 移植到当前 eagle2_batch + 批量 D2H 代码，并补齐 VL 的 rope_delta。开关为 `preconsume_first_token`（bench 加 `--preconsume`），默认关闭以便与基线 A/B。draft prefill 追加消费 (xP, f(P-1))，建树时 xP 作为确定根节点、draft 候选全部挂为后代；首轮更新 draft 时跳过 xP 防止重复写 KV，只接受根 token 时不再 extend。proposed 统计不计确定根节点。合成状态对齐测试与随机小 GPT2 端到端 greedy 对照均通过（bfs/eagle2_batch × preconsume 开/关 四组全部无损）。真实权重的 tau 与速度收益待 A/B。

### 6.3 复用 verify KV 并删除 target replay

#### 需要先修正的前提

当前每轮 append 都包含：

```text
接受的 draft tokens + 最后一个 target 预测 token
```

全部接受时，最后这个 token 叫 bonus；中途不匹配时，叫 correction。

因此“没有产生 bonus token”并不代表没有最后的 target token。部分接受时仍有 correction，而 correction 和 bonus 都有一个共同点：

- 它已经由某个 root/树节点 logits 预测出来；
- 但它本身尚未经过 target forward；
- 它的 KV 尚未写入；
- 用于预测再下一个 token 的 recovery_logits 也尚未产生。

所以不能仅凭“复用 verify KV + 没有 bonus”就直接删掉全部 replay。

#### 两种实现层级

##### 方案 A：复用接受路径 KV，只重放最后一个 token

verify 后保留接受路径的 KV 和 hidden，只对 correction/bonus 做一次单 token target forward，以获得下一轮 recovery_logits。

收益：避免重放已接受的 draft token，减少部分 KV cat/copy 和 hidden 计算。

限制：此前微基准显示 eager 前向 N=1 与 N=3 的墙钟接近，主要成本是 launch 和调度。因此只把 replay 从 A 个 token 缩到 1 个 token，不一定能省掉完整 35ms/轮。

结论：可行，但单独使用时收益可能有限。

##### 方案 B：将 correction/bonus 并入下一轮 verify，完全删除 replay

把本轮最后的 correction/bonus 作为下一轮 verify 的已确定根节点，与下一棵 draft 树一起前向：

- 该节点写入正式 KV；
- 它的输出 logits 用于验证下一轮根层 draft token；
- 不再单独执行 target replay；
- 接受路径的 KV 和 EAGLE hidden 尽量从上一轮 verify 复用。

这是能够真正删除一次 target forward 的方案，理论收益最高，对应 replay 约 22.09ms device/轮、约 35.35ms 干净墙钟/轮。

主要风险：

- 树 KV 中包含拒绝分支，不能直接取消 crop，必须按接受路径重排并提交；
- correction/bonus 的 KV、root logits、draft hidden 和 out_ids 必须保持同一位置；
- EOS、根层完全不匹配、整棵树全部接受等边界都要单独验证；
- VL 的 position_ids 和 rope_delta 必须连续。

结论：方向可行且收益最高，但应把目标明确为“将最后 target token 并入下一轮 verify”，而不是简单复用 KV 后直接删除 replay。

## 7. 推荐实施顺序

| 顺序 | 优化项 | 主要收益来源 | 风险点 |
|---|---|---|---|
| P0 | 复用接受路径 KV，并将 correction/bonus 并入下轮 verify | 真正删除一次完整 target replay | KV 顺序、correction/bonus logits、hidden 对齐 |
| P1 | StaticCache + verify CUDA Graph | launch、KV cat/copy、CPU 空闲 | 固定 shape、VL position、cache crop |
| P1 | 使用 EAGLE2 风格累计分数 beam 树并剪枝 | 固定节点预算内保留更深的高概率路径 | draft 扩展成本、AR hidden/KV 对齐、tau 收益不确定 |
| P2 | draft 预消费 target 确定 token + 末位 feature | 减少无效根层候选，可能提高 tau | draft/target 位置错位、重复 KV、额外 draft step |
| P2 | draft 按层批量建树 + 批量 D2H | 建树、小 GEMM、同步 | 必须保持树结构和输出不变 |
| P2 | 融合 RMSNorm/elementwise，复用临时 buffer | 小 kernel launch、临时分配 | 数值一致性和后端兼容性 |
| 暂不做 | 重写 Triton attention、调 GEMM 微内核 | 当前占比较低 | 投入大、预期收益有限 |

## 8. 每项优化的验证标准

### 正确性

- greedy 下与 HF baseline 逐 token 一致；
- `lossless(all)=True`；
- StaticCache-eager 与 CUDA Graph 输出一致；
- VL 必须覆盖带图输入，不能只测纯文本；
- 出现分叉时记录首个分叉位置、两侧 token 和 top1-top2 logit gap。

### 性能

- 所有正式数字均来自无 profiler 的干净运行；
- EAGLE 与 HF 使用相同 prompt、max_new_tokens、dtype、batch=1 和 greedy；
- 双方分别 warmup；
- 至少报告 TPOT、端到端吞吐、wall、tau、tree_nodes/round、ms/round 和 speedup；
- 调树参数时必须同时看 tau 和 ms/round，避免用更多计算换取无效接受长度增长。

## 9. 复现命令

### 分阶段墙钟

```bash
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. \
.venv/bin/python spec_decoding/eagle3_sgl/example/time_stages_eagle3_sgl.py \
    --mode vl --max-new-tokens 128 --explain \
    --report-out /tmp/eagle3_vl_stages.txt
```

### torch.profiler

```bash
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. \
.venv/bin/python spec_decoding/eagle3_sgl/example/profile_eagle3_sgl.py \
    --mode vl --max-new-tokens 128 --explain \
    --trace-out /tmp/eagle3_vl_trace.json \
    --report-out /tmp/eagle3_vl_profile.txt
```

trace 默认只保留 `/tmp/eagle3_vl_trace.json.gz`。

### 无插桩 HF 对比

```bash
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. \
.venv/bin/python spec_decoding/eagle3_sgl/example/bench_accept_tpot.py \
    --mode vl --num-prompts 10 --max-new-tokens 128 --explain
```

将 `--mode vl` 改为 `--mode text` 即可分析纯文本模型。

## 10. 总结

当前瓶颈由三类问题共同构成：

1. 冗余的 target replay；
2. verify/replay 中真实的 GEMM 计算；
3. 大量 eager launch、临时 tensor、KV/激活拷贝和标量同步。

因此不能只依赖 CUDA Graph。最合理的路径是：先将 correction/bonus 并入下轮 verify 以删除 replay，再用 StaticCache + CUDA Graph 降低固定调度成本。EAGLE2 风格树和 draft 预消费 target 确定 token 主要用于提高 tau，必须与额外 draft/verify 成本一起评估。Triton attention 和树掩码目前都不是主要瓶颈。
