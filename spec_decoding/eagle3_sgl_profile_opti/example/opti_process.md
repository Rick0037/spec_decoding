# 性能分析思路

## 阶段 0 —— 基线定位（在原始 eagle3_sgl 上）

先宏观分析，搞清瓶颈在哪个阶段、哪个算子，两个一起看：

| 文件 | 作用 |
|---|---|
| profile_eagle3_sgl.py | torch.profiler：分阶段 device 时间 + Top 算子 + 同步/拷贝计数（aten::item / cudaSynchronize / copy_）。回答哪些 op 耗时 |
| time_stages_eagle3_sgl.py | CUDA event + sync 的分阶段真实墙钟（prefill / draft_build / verify / replay / draft_extend）。回答哪个投机解码阶段最耗时 |
| parse_sglang_trace.py | 解析 SGLang 的 chrome trace，按 kernel / CPU op 聚合耗时。用来做对照 |

## 阶段 1 —— 优化后的正确性 + 吞吐校验

每做完一步改动就跑：

| 文件 | 作用 |
|---|---|
| validate_step1.py | Step1（接受循环张量化）：原版 vs opti 输出逐位一致 + 吞吐收益 |
| validate_step5.py | Step5（cuda graph verify）：原版 vs opti 输出逐位一致 + 吞吐收益 |
| ab_speed.py | 无 profiler 的干净 A/B 吞吐（原版 vs opti）。每一步都能复用，看真实吞吐 |

## 阶段 2 —— 再找微观瓶颈 + 测bound

找出bound具体到底是哪段哪行代码，拆分测量某阶段的子阶段，优化后再check是否等价

| 文件 | 作用 |
|---|---|
| probe_fwd_cost.py | 微基准：target 前向墙钟随token 数量几乎不变 → 证明launch-bound（指向 CUDA graph） |
| probe_verify_internals.py | 把 verify 那 ~76ms/轮拆成 position_ids / triton metadata / attn-context / 真正前向 / accept → 发现建树 mask 的 Python 双重循环才是最耗时 |
| check_mask_equiv.py | 向量化后的树 mask 与原始双重循环逐元素等价校验（Step4 改动的正确性测量） |

## 阶段 3 复测 —— 刷新两张表（基于优化后的各版本）

| 文件 | 作用 |
|---|---|
| time_stages_opti.py | 优化后的版本的分阶段墙钟，刷新墙钟 |
| profile_opti.py | 优化后的版本的算子耗时 / 同步数量 |

## 阶段 4 —— CUDA graph（先确定风险，再实现，再验证 / 计时）

| 文件 | 作用 | 次序 |
|---|---|---|
| spike_static_verify.py | 前置要求：StaticCache + cache_position + padded 4D mask + 普通 SDPA 的 node_logits，与 DynamicCache extend verify 逐元素一致（成立才敢上 graph） | 先 |
| validate_step5.py | eager-opti vs CUDA-graph 的输出一致性 / acc/round / tokens/s | 中 |
| stage_events_opti.py | event 计时 | 中 |
| probe_graph_replay.py | 隔离测 verify graph.replay() 的纯 GPU 时间 vs eager triton verify，量化 graph 收益 | 后 |
