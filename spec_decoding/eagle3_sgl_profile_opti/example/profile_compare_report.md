# EAGLE3-SGL vs SGLang EAGLE3 —— torch.profiler 对比结论

环境：25.04版本的nvidia torch镜像 (torch 2.7.0a0) 与 lmsysorg/sglang:v0.5.12.post1，单卡 H100 80GB。
配置对齐：Llama-3.1-8B target，topk=8，num_steps=8，draft_tokens/max_tree_nodes=64，fp16，greedy。

## 0. 吞吐总览

| 实现 | 吞吐 (tok/s) | accept/round | 每轮耗时 | 备注 |
|---|---|---|---|---|
| eagle3_sgl | ~18.6 | 2.61 | ~140 ms | HF eager + Python 建树 + DynamicCache |
| SGLang EAGLE3 | ~332 | ~4.2 | ~10.7 ms | CUDA graph + flashinfer paged tree attn |

→ 约 13–18× 差距，且主要不是 acceptance 差距（4.2 vs 2.6 只解释 ~1.6×），而是每一步的执行效率。

## 1. eagle3_sgl 分阶段墙钟（CUDA event 同步计时，无 profiler 干扰）

| stage | %wall | ms/call | 说明 |
|---|---|---|---|
| target_verify | 54.0% | 75.8 | 树节点过 target（8B）—— 最耗时 |
| target_replay | 13.0% | 18.3 | 接受 token 重放刷新 target KV |
| draft_build_tree | 5.0% | 7.0 | draft 建树（含 draft prefill/step） |
| draft_extend | 0.8% | 1.1 | 增量 draft KV |
| target_prefill | 0.4% | 16.2 | 仅一次 |
| unmeasured / Python | 26.8% | — | BFS 建树 + 小张量构造 + .item()/.tolist() 同步 |

## 2. 算子级对比（torch.profiler）

| 指标 | eagle3_sgl（81 tok/31 轮） | SGLang（128 tok/36 轮） | 比值 |
|---|---|---|---|
| GPU 时间最大项 | aten::mm 45% | nvjet_*(cuBLAS GEMM) ~75% | 都是 GEMM |
| aten::copy_ 调用数 | 261,225 | 3,185 | ~80× |
| Memcpy HtoD 事件 | 243,701 | 395 | ~600× |
| cudaStreamSynchronize | 246,016 | 592 | ~400× |
| 单次 target verify | ~76 ms | ~3.7 ms | ~20× |
| CUDA graph | 无 | 107 次 graph launch | — |
| GPU 利用率(净) | ~23% | 高（GEMM-bound） | — |

## 3. 结论

### 最耗时的操作
- 阶段维度：target_verify（target 8B 跑投机树）占墙钟 54%；其次 target_replay 13%。
- 算子维度：两边的 GPU 时间都被 GEMM（aten::mm / nvjet cuBLAS）主导——这是 target/draft/lm_head 的必要矩阵乘，flops 完全相同。

### 为什么慢 20×（注意以下都还只是对profile结果的初步分析，并不一定会有收益）
本实现是 CPU/启动/同步 bound（GPU 净利用率仅 ~23%），而非算力 bound。多余开销来自：
1. 没有 CUDA graph：每个 kernel 单独 eager 启动 + Python 分发；SGLang 把整个 draft+verify 步整体 graph 捕获（107 次）。
2. 海量 host↔device 同步：24.6 万次 cudaStreamSynchronize + 24.4 万次小 HtoD 拷贝（vs SGLang 几百次）。来源：Python BFS 逐节点 .item()/.tolist()、每轮 torch.tensor([...], device=) 构造 append/path/mask。
3. output_hidden_states=True：每次 target 前向都返回全部 32 层 hidden（只需 3 个 eagle 层 + 末位 logits），产生大量 copy/cat。
4. DynamicCache 的 cat/clone：每轮在前缀 KV 上 extend/clone，aten::copy_ 调用 ~80× 于 SGLang。
5. 逐节点 argmax/Python 循环（1999 次 argmax）而非一个融合 kernel。
