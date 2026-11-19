# 本目录介绍

本目录为基于eagle3_Sgl的profile结果的深一步优化，一开始eagle3_Sgl accept length为1.2左右，意味着draft tokens几乎没有被接收，但已经做了draft extend功能以及inc acts功能，但性能差Huggingface很多，最多只有0.38x

## 1. eagle3_sgl 哪个操作最耗时（配置 topk=8/steps=8/nodes=64）

分阶段墙钟（18.6 tok/s，accept/round 2.61）：

| 阶段 | %wall | ms/轮 |
|---|---|---|
| target_verify（树过 8B target） | 54% | 75.8 |
| target_replay（重放刷新 KV） | 13% | 18.3 |
| draft_build_tree | 5% | 7.0 |
| unmeasured / Python（BFS 建树 + 小张量构造 + .item()/.tolist() 同步） | 27% | — |

算子级最大项是 aten::mm（GEMM，45% GPU 时间）。

## 2. 对比 SGLang EAGLE3，最终结论

| 指标 | eagle3_sgl | SGLang EAGLE3 | 比值 |
|---|---|---|---|
| 吞吐 | ~18.6 tok/s | ~332 tok/s | ~18× |
| 每轮耗时 | ~140 ms | ~10.7 ms | ~13× |
| 单次 target verify | ~76 ms | ~3.7 ms | ~20× |
| cudaStreamSynchronize | 246,016 | 592 | ~400× |
| Memcpy HtoD 事件 | 243,701 | 395 | ~600× |
| aten::copy_ 调用 | 261,225 | 3,185 | ~80× |
| CUDA graph | 无 | 107 次 | — |

关键结论：sglang和eagle3_sgl的GEMM时间完全相同，说明差距不在Gemm kernel上，在除Gemm外的kernel上有差距。另外，由上表可知，当前差距几乎是调度/同步/拷贝开销，ealge3_sgl实现是 CPU/同步 bound（GPU 净利用率仅 ~23%）。

分析哪些操作待优化（按收益排序）：

1) 【估计收益最高】去掉每步 host 标量同步 + 对 decode 步做 CUDA graph 捕获 —— 猜测24.6 万次 cudaStreamSynchronize 来自 Python 逐节点 .item()/.tolist() 和每轮 torch.tensor(..., device=)。具体是
* expand_draft_tree_ar函数里的如下代码, child_idx本来在GPU上，但是经过tolist必须要拷贝回cpu，这里涉及一次拷贝和一次同步，64个树节点就有64次
```
child_idx, _ = draft_model.topk_target_ids(logits, K)
for ct in [int(c) for c in child_idx[0, :K].tolist()]:
```

* _select_append_root_child函数里的item()取出单个元素，涉及leaves × path_len同步，同一个节点的 argmax 在不同路径里被重复算多次（路径的前缀都相同）
```
pred = int(src.argmax(dim=-1).item())
```

* host2device 的 pageable 小拷贝
  * tree_draft.py:376 torch.tensor([[tok]], device=)
  * draft_model.py:233 torch.tensor([position], device=)
  * tree_verify_full.py:340/369/190 torch.tensor([layout.token_ids], device=)
  * runner.py:341/364/392 torch.tensor([append_list], device=)
  * tree_verify_attn.py:120-125 qo/kv/mask indptr
  * tree_verify_mask.py:69 depths

* draft.step 一次只处理 1 个 token，配合 expand_draft_tree_ar 的串行 BFS，等于每轮做 64 次微型 eager 前向 + 64 次 D2H 同步 可能是造成GPU 净利用率只有 ~23%（大部分时间在等同步/启动 kernel）的一个重要原因；且这套结构无法被 CUDA graph 捕获（图里不能有 .item()/动态 Python 分支/动态 shape）。做法是固定 topk 下一次 batch 出某一层所有 children，去掉循环，且整个 draft+verify 步用 CUDA graph 整体捕获。

2) output_hidden_states=True 返回全部 32 层 hidden（只需 3 个 eagle 层 + 末位 logits），造成大量多余 copy/cat，触发HtoD，aten::copy_和cudaStreamSynchronize事件。

3) DynamicCache 每轮 cat/clone 前缀 KV（造成aten::copy_数量激增）—— 应预分配 + 原地写， 即staticCache。

4) 逐节点 argmax/Python 循环应融合成单 kernel。

根据profile结果的初步总结：target_verify 最耗时，其次是target replay，但其中的 GEMM kernel还好，其余kernel和sglang比差点；优化收益最高的是"CUDA graph + 消除标量同步"，最没必要的是"全层 hidden 返回 + 逐轮 mask/KV clone"。 HF eager + Python 建树这套动态架构本身无法被 graph 捕获，需要修改结构为静态结构以被捕获

## 针对以上的优化历程

### Step 1：优化tree verify里面的冗余argmax

- 改动：tree_verify_full.py::_select_append_root_child。原本逐 (leaf, k) 做 argmax().item()（标量同步 + 路径重叠节点重复算 argmax）→ 改为一次性 node_logits.argmax(dim=-1)（[L]）+ root 拼接后单次 D2H tolist()，匹配逻辑全在 host list 上跑。
- 结果（topk8/steps8/nodes64，128 tokes）：

  | 指标 | 原版 | Step1 | 变化 |
  |---|---|---|---|
  | 输出逐位一致 | — | — | True（无损） |
  | cudaStreamSynchronize | 246,016 | 244,048 | −0.8% |

- 结论：收益主要来自消除重复 argmax 计算，不是去同步（同步次数仅 −0.8%）。

### Step 2：draft tree展开逐层批量化（去掉 64 次微前向 + 每节点的 .tolist()/torch.tensor）

- 改动：tree_draft.py::expand_draft_tree_ar。把"逐节点 pop + 单 token draft.step"改为"每个 BFS 深度合成一个 batch、一次 draft.step 前向"；child token 全程以 GPU tensor在层间传递，整棵树末尾只一次 torch.cat().tolist() 。
- 结果（干净 A/B，无 profiler，topk8/steps8/nodes64，128 tok，取最优 rep）：

  | 实现 | tok/s | acc/round | 逐位一致 |
  |---|---|---|---|
  | 原版 | 20.97 | 2.61 | — |
  | Step1+2 | 22.47 | 2.61 | True（无损） |

  累计 +7.1%（1.07×）；acc/round 不变。

- 关键结论（修正认知）：Step2 后 cudaStreamSynchronize 仍几乎不降（原版逐节点fwd vs 批量版fwd次数基本相同）→ 看来24 万次同步不在 draft 建树里，而在 target侧（verify/extend/replay，占墙钟 ~67%），draft 侧去同步收益有限。下一步应转向 target 侧的优化。

### Step 3：target 前向只 hook 采集 eagle 层 hidden（去 output_hidden_states=True）

- 改动：hidden.py 新增 EagleActCapture（forward_pre_hook 只采集 eagle_layers 这三层必需的输入残差流）；runner.py 的 prefill / replay 两处 target 前向去掉 output_hidden_states=True，改用 hook 采集。
- 结果（干净 A/B，topk8/steps8/nodes64，128 tok）：

  | 实现 | tok/s | acc/round | 逐位一致 |
  |---|---|---|---|
  | 原版 | 21.18 | 2.61 | — |
  | Step1+2+3 | 22.74 | 2.61 | True（无损） |

  累计 +7.4%，相对 Step2（+7.1%）几乎无新增。

- 关键结论1：output_hidden_states=True 在 HuggingFace 里只是收集已算好的中间激活的引用，而不是拷贝，所以去掉它无损但基本不提速。至此 draft 侧 + hidden 侧的微优化已基本榨干（累计仅 +7%）。

- 关键结论2：基于1，墙钟 67% 在 HF eager 的 target 前向，单次 verify  跑 64 个树 token 要 ~76ms，初步判断差距是 无 CUDA graph 的逐层 eager 启动开销。于是收敛可能高收益的方向只剩两条：1.消除 replay 冗余前向（verify 已对被接受路径跑过 target，可复用其 KV/logits，省掉占比 ~13%的targetreplay）；②对 target decode 步做 CUDA graph 捕获 / 换 paged-KV+融合注意力（SGLang 那套）。

### Step 4（原计划消除冗余replay）：但微观分析后发现真正瓶颈是 树掩码 Python 双重循环，向量化后 整体 2.95×

- 先做微基准，做了后否决了 replay 方向（example/probe_fwd_cost.py）：target 前向墙钟随新 token 数几乎不变 —— N=1: 14.9ms、N=3: 17.1ms、N=64: 16.6ms，证明这完全 launch-bound。即 replay 3 token的前向时间 ≈ 1-token 前向时间，复用 verify KV 把 replay 缩到 1 token目前并不是第一瓶颈。
- 改测 verify 内部构成（example/probe_verify_internals.py），重大发现：

  | verify 子步 | ms/round | 占总墙钟 |
  |---|---|---|
  | build_triton_tree_verify_metadata（建树掩码） | 34.8 | ~29% |
  | _extend_then_crop（真正的 8B 前向） | 30.5 | ~25% |
  | accept_select（Step1 后）/ pos_ids / ctx 安装 | <0.6 | — |

  树掩码构建竟比前向本身还贵，这不科学。根本原因：tree_verify_mask.py::build_tree_cross_attn_bias_with_prefix 用 for i in range(L): for j in range(past+L) 双重 Python 循环（64×564 ≈ 3.6 万次/轮），每次一个标量 GPU 写 bias[i,j]=0.0 + 一次 _on_tree_path 祖先链遍历，所以有了 24 万次同步 / HtoD。
- 改动：新增 _ancestor_self_matrix(parent, n) —— 用所有节点沿 parent 同步上爬、每层一次向量化 scatter一次性构造 [L,L] 祖先布尔矩阵（循环次数 = 树深 ~8，而非 3.6 万），再用掩码一次写出 bias。build_tree_cross_attn_bias_with_prefix / build_tree_sdpa_attn_bias / build_tree_allow_mask_4d 全部改用它。
- 结果：
  - 掩码构建 34.8ms → 0.41ms/轮（85×），verify 总耗 66 → 42ms/轮。
  - 干净 A/B（无 profiler，topk8/steps8/nodes64，81 tok，取最优 rep）：

  | 实现 | tok/s | wall(s) | acc/round | 逐位一致 |
  |---|---|---|---|---|
  | 原版 eagle3_sgl | 22.13 | 3.66 | 2.61 | — |
  | Step1+2+3+4 | 65.34 | 1.24 | 2.61 | True（无损） |

  累计 2.95×（+195%），输出逐位一致、acc/round 接受率不变。
- 关键结论（修正之前的初步结论）：前三步只 +7%、误以为只剩 CUDA graph 一条路，真正的瓶颈一直藏在建树掩码的 Python 双重循环里，profiler 把它摊在大量小 kernel/同步上，这并不显眼，所以需要针对耗时占比最长的阶段，深入下去，肉眼查看或者写脚本分函数测试看看到底是哪一段代码耗时如此之久。下一步若要再提速才轮到CUDA graph。

### step1-4后得profile结果

同配置复测（topk8/steps8/nodes64，greedy，81 new tokens，31 轮，acc/round 2.61）。

表 1 —— 分阶段墙钟（CUDA event 同步计时，example/time_stages_opti.py）

| 阶段 | %wall | ms/轮（原版 → 现在） |
|---|---|---|
| target_verify（树过 8B target） | 54.2% | 75.8 → 24.3 |
| target_replay（重放刷新 KV） | 37.9% | 18.3 → 17.0 |
| draft_build_tree | 2.8% | 7.0 → 1.25 |
| draft_extend | 2.4% | — → 1.07 |
| target_prefill / draft_prefill_init | ~1.2% | — |
| unmeasured / Python（BFS 建树 + 小张量构造 + .item()/.tolist() 同步） | 1.5% | （原版 27%）→ 0.69 |

> ms/round：~140 → 44.8；spec tok/s：~18.6 → 58–65（time_stages 58.3、干净 A/B 65.3）。两次 eager 8B 前向（verify+replay）已占墙钟 ~92%。

表 2 —— 对照原版 / SGLang（torch.profiler 计数，example/profile_opti.py）

| 指标 | 原版 eagle3_sgl | opti（现在） | SGLang EAGLE3 | opti 改善 |
|---|---|---|---|---|
| 吞吐 | ~18.6 tok/s | ~65 tok/s | ~332 tok/s | ~3.5× |
| 每轮耗时 | ~140 ms | 44.8 ms | ~10.7 ms | ~3.1× |
| 单次 target verify | ~76 ms | ~24 ms | ~3.7 ms | ~3.2× |
| cudaStreamSynchronize | 246,016 | 813 | 592 | ~300× |
| Memcpy HtoD 事件 | 243,701 | 404 | 395 | ~600× |
| aten::copy_ 调用 | 261,225 | 14,828 | 3,185 | ~18× |
| aten::item | （大量） | 316 | — | — |

读数：

- 同步/拷贝已降到与 SGLang 同量级（sync 813 vs 592、HtoD 404 vs 395）——证实此前 24 万次同步/H2D 几乎全来自建树掩码的 Python 双重循环，现已消除。
- 现在墙钟 92% = 两次 eager 8B 前向（verify 24ms + replay 17ms）。但纯 GPU 的 verify 仅 ~10ms 而 wall 24ms，因此推测差值是逐层 kernel 启动间隙（cudaLaunchKernel 90,638 次）。CUDA graph就是干这个事的
- 剩余可榨点：① aten::copy_ 14,828 + aten::cat 8,659 ≈ DynamicCache 每轮 cat/copy 前缀 KV（通过 staticCache或paged-KV 去掉，aten::cat 8,659 ≈ 32 层 × 2(K/V) × 31 轮 × ~2 次前向(verify+replay) —— 几乎全是 HF DynamicCache 每次前向对前缀 KV 做的 torch.cat([past, new])。预分配 paged/static KV + 按 slot 原地写可直接消掉这一类 cat 和它伴随的 copy_，但 copy_ 14,828次里不止 KV cat（还有 _extend_then_crop 的 crop 切片、RoPE、激活搬运等），staticCache或paged KV 只能去掉其中 KV 那部分。如果 patch DynamicCache，直接换 StaticCache， 那么cat会完全消失，copy_ 从每层拷整段 past+new缩成只写 new 的几行，量级和耗时都大降。注意： 但是会与现有verify代码有几处摩擦：_extend_then_crop 的 past.crop(before)，triton-tree attention 手写 KV 的那段得让它也往同一个 static buffer 原地写，而不是走 DynamicCache 的 update方法，保证cache_position 要正确贯穿）；② target verify/replay 两次前向未合并。

### Step 5：static KV 缓存 + 固定 shape masked SDPA + verify 前向 CUDA graph（不做块表 paged KV）

动机：Step4 后墙钟 92% = 两次 target eager前向，纯 GPU 的 verify 仅 ~10ms 但 wall 24ms，差值多数是逐层 kernel 启动间隙（cudaLaunchKernel 9 万次）。要消除它必须 CUDA graph， 而 graph 要求固定地址 + 固定 shape + 无 reallocation，DynamicCache 的逐步 cat 不满足。故需要先把 target KV 换成连续 static cache。DynamicCache和StaticCache的区别见DynamicCache-StaticCache.md

实现（target_graph.py + runner._generate_static_graph，开关 cfg.decode_cuda_graph）：

1. target 换 HF StaticCache：预分配 [B,H,max_len,D]，按 cache_position 原地写，没有 cat/realloc。
2. verify阶段的注意力 = 固定 shape 4D 树 mask + 普通 SDPA（mask 宽度 padding 到 max_len，前缀=0，树段按祖先矩阵分配0或-inf，未写&未来 slot=-inf）；verify 写的树 KV 落在 [cur_len:cur_len+L)，等价于之前DynamicCache的 extend+crop。
3. 对固定L的 verify 前向做一次 CUDA graph 捕获（warmup 3 次后 capture），后续每轮只把 tokens/position_ids/cache_position/mask copy_ 进静态 buffer 后 graph.replay()，读固定输出 buffer。static cache + graph 跨 generate() 复用（否则每次触发 cudaMalloc + 重新捕获会非常慢）。
4. replay / draft 仍 eager，但 replay 也在同一 static cache 上原地写。实际L 与捕获时的L不同时退回 eager 静态前向。

验证（example/validate_step5.py，topk8/steps8/nodes64）：

| 实现 | tok/s | ms/round | acc/round | 说明 |
|---|---|---|---|---|
| EAGER-opti（Step4，triton verify） | 64.3 | 44.8 | 2.61 | 基线 |
| StaticCache-eager（static KV + SDPA，无 graph） | 69.7 | — | 2.59 | static KV 去 cat/copy 的贡献 +8% |
| CUDA-graph（Step5） | 83.3 | 31.1 | 2.59 | vs Step4 +29.5%（1.30×）；graph 本身 +20%（1.20×） |

- 无损性已证明：StaticCache-eager vs CUDA-graph 输出逐位完全一致。
- 优化后各种操作的计数：cudaLaunchKernel 90,638→54,167（CUDA graph减少kernel launch数量）、aten::cat 8,659→2,790（static KV 干掉逐层 verify KV cat）、aten::copy_ 14,828→9,684（static KV 干掉的）、cudaStreamSynchronize 813→511。
- 累计：相对原版 eagle3_sgl（~22 tok/s）现 ~83 tok/s ≈ 3.8×。
- 下一步：replay（~38% 墙钟）仍 eager 未CUDA graph；draft 侧也可图化。若要再逼近 SGLang，需把 replay 也纳入 graph + 最终的块表 paged KV / flashinfer 融合注意力。

### Step5（CUDA graph ON）重新 profile：两张表的新变化

> 计时方法：① 端到端 tok/s、ms/round 用无插桩的干净 A/B（validate_step5.py）；② 单次 verify 用隔离的 graph.replay() 纯 GPU 计时（probe_graph_replay.py）；

表 1 —— 再次分阶段（Step4 eager → Step5 graph，ms/轮）

| 阶段 | Step4 (eager) | Step5 (graph) | 变化 |
|---|---|---|---|
| target_verify | 24.3 | ~9.9（graph.replay 纯 GPU） | −59% |
| target_replay（刷新 KV） | 17.0 | ~17.0（仍 eager，未图化） | 不变 → 现在是最大耗时的部分（~55%） |
| draft_build_tree | 1.25 | ~1.3 | — |
| draft_extend | 1.07 | ~1.0 | — |
| 整轮墙钟 ms/round | 44.8 | 31.1 | −31% |

表 2 —— Step4 → Step5（torch.profiler 计数 + 干净吞吐）

| 指标 | 原版 | Step4 (eager) | Step5 (graph) | SGLang |
|---|---|---|---|---|
| 吞吐 | ~18.6 tok/s | ~65 tok/s | ~83 tok/s | ~332 tok/s |
| 每轮耗时 | ~140 ms | 44.8 ms | 31.1 ms | ~10.7 ms |
| 单次 target verify | ~76 ms | ~24 ms | ~9.9 ms（graph GPU） | ~3.7 ms |
| cudaLaunchKernel | —(极多) | 90,638 | 54,167 | — |
| aten::cat | —(极多) | 8,659 | 2,790（static KV 去逐层 KV cat） | — |
| aten::copy_ | 261,225 | 14,828 | 9,684 | 3,185 |
| cudaStreamSynchronize | 246,016 | 813 | 511 | 592 |
| Memcpy HtoD 事件 | 243,701 | 404 | 241 | 395 |
| aten::item | （大量） | 316 | 210 | — |

读数：

- verify 已差不多：graph.replay 9.9ms ≈ 同前向 eager 15.7ms 去掉 ~6ms 启动间隙；相对 Step4 的 triton-verify 24ms wall 降到 ~9.9ms。cudaLaunchKernel 90,638→54,167（少掉的 ~3.6 万是 verify 那 32 层逐 kernel 启动，聚集成一次 graph replay）。
- static KV对消除部分算子的作用：aten::cat 8,659→2,790、aten::copy_ 14,828→9,684。
- 瓶颈转移：现在最大耗时部分是 target_replay（~17ms，~55%），因为它仍 eager；verify 反而成了小头。累计相对原版 ~3.8×（~83 tok/s）。
- 无损：static-eager vs CUDA-graph 输出逐位一致，证明cuda graph 无损；与 triton 路径的差异仅 fp16 归约顺序、决策等价。
- draft build_tree和extend需要cuda graph吗?：draft build_tree和extend阶段目前耗时只有1.3ms左右，看起来不需要cuda graph
- 下一步最高收益：把 replay 也图化（变长 → padding 到固定长 + mask 后 capture），看能否再吃掉这 ~55% 里的启动间隙。当然，如果想个办法去掉target replay阶段，那肯定能完全干掉这55%的开销。其实推荐做法是省掉replay，复用verify阶段的kv

### 端到端实测：example/run_eagle3_sgl_llama31.py vs HF greedy

EAGLE3-SGL-profile-opti（Llama-3.1-8B target + yuhuili EAGLE3-8B draft），topk8/steps8/nodes64，max_new_tokens=128，fp16，单卡 batch=1。预热一次后计时（eagle3_sgl_profile_opti预热含 CUDA graph 捕获 + static cache 分配；HF 侧预热含 DynamicCache 冷启动），保证公平性

| 实现 | spec tok/s | acc/round | 逐位一致 | HF greedy tok/s | speedup |
|---|---|---|---|---|---|
| EAGER-opti（Step4，triton verify） | 66.6 | 2.61 | False（fp16 平局，第 47 token 后微分叉，决策等价） | 48 | 1.40× |
| CUDA-graph（Step5） | 86.3 | 2.59 | True | 48 | 1.8× |

读数：

- CUDA graph 把 verify 的逐kernel启动间隙去除（24ms→~10ms），spec 升到 86 tok/s，相对 eager-opti +30%、相对 HF greedy 1.8×。
- HF greedy 实测 ~48 tok/s

> 复现：PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl_profile_opti.example.run_eagle3_sgl_llama31 --max-new-tokens 128 --topk 8 --num-steps 8 --max-tree-nodes 64 --cuda-graph（去掉 --cuda-graph 即 eager-opti）


