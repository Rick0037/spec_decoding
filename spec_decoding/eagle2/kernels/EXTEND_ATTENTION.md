# Extend Attention Kernel 技术说明

本文档说明 `extend_attention.py` 中的 Triton kernel:做什么、输入输出长什么样、内部两阶段怎么算、能否用于 prefill/decode、在 `spec_decoding.eagle`(EAGLE1)里如何调用。

---

## 1. 一句话定义

Extend Attention:在一次 attention 里,让 「本步新写入的 Q(extend 段)」 去 attend 「KV cache 里已有的前缀(prefix)」+「本步新算出的 K/V(extend 段)」,并支持 自定义 bool mask(EAGLE 树 verify)或 因果 mask(普通 extend prefill)。

它不是「整段序列从零开始」的独立 prefill kernel,也不是 FlashAttention `flash_attn_with_kvcache` 那种单 token decode 专用 API;而是「带 KV 前缀的 extend / chunked prefill」 算子。

---

## 2. Prefill 和 Decode 都支持吗？

| 场景 | 是否适合 | 说明 |
|------|----------|------|
| 带 KV 前缀的多 token 前向(extend) | ✅ 主要场景 | `prefix_len > 0`,`extend_len ≥ 1`;Q 只来自 extend 段 |
| 无前缀的纯 prefill | ✅ 退化情况 | `prefix_len = 0`,`extend_len = L`;Stage 1 循环为空,Stage 2 处理全部 K/V |
| 单 token decode | ✅ 特例 | `extend_len = 1`;每个 seq 只有 1 个 query,仍走同一 kernel |

结论:

- 从 算子语义 上:同时覆盖 「有 prefix 的多 token 计算」(常被叫 extend prefill)和 `extend_len=1` 的 decode 步。
- 从 spec_decoding 用途 上:当前只在 EAGLE1 的 `verify_mode=full_model_tree` 且 `verify_attn_backend` 解析为 `triton_tree`(`topk>1` 且 Triton 可用)时调用(见 `tree_verify_attn.py`),一次 forward 过 `L` 个 draft 节点(`extend_len = L`);EAGLE1 默认/测试用的 `reference_paths` 与 `eager`/`flash_attn` 后端不走本 kernel。

---

## 3. 与标准 Attention 的对应关系

设某条序列在 KV pool 中已有前缀长度 `P`(prefix),本步新算 `E` 个 token(extend):

```
逻辑序列:  [  prefix: P 个 token  |  extend: E 个 token  ]
Query:     仅 extend 段的 E 个位置
Key/Value:  prefix 从 k_buffer/v_buffer 经 kv_indices 间接索引
           extend 从 k_extend/v_extend 直接读
```

一次 kernel launch 输出 `o_extend`,形状与 `q_extend` 对齐,即 只填 extend 段的 attention 输出。

在 EAGLE1 verify 中(batch=1):

- `P = past_len`(已接受 token 的 KV)
- `E = L`(draft 树节点数)
- `custom_mask`:树形 allow mask(兄弟不可互看)

---

## 4. 对外 API:`extend_attention_fwd`

### 4.1 张量形状(连续 memory,token-major)

| 参数 | 形状 | 含义 |
|------|------|------|
| `q_extend` | `[ΣE, H_q, D_q]` | 本步所有 extend query |
| `k_extend` | `[ΣE, H_kv, D_k]` | 本步 extend 的 K(未 repeat GQA) |
| `v_extend` | `[ΣE, H_kv, D_v]` | 本步 extend 的 V |
| `o_extend` | `[ΣE, H_q, D_v]` | 输出(与 Q 行对齐) |
| `k_buffer` | `[T_pool, H_kv, D_k]` | KV pool 中 K;逻辑上含 prefix+extend |
| `v_buffer` | `[T_pool, H_kv, D_v]` | KV pool 中 V |

- `H_q / H_kv`:query head 数 / KV head 数;GQA 时 `kv_group_num = H_q // H_kv`,在 kernel 内由 `cur_kv_head = cur_head // kv_group_num` 处理。
- `ΣE`:batch 内 extend token 总数,由 `qo_indptr` 描述每条序列的切片。

### 4.2 CSR 式 batch 描述

| 参数 | 形状 | 含义 |
|------|------|------|
| `qo_indptr` | `[B+1]` int32 | 第 `i` 条序列的 extend Q 在 `q_extend` 中的 `[qo_indptr[i], qo_indptr[i+1])` |
| `kv_indptr` | `[B+1]` int32 | 第 `i` 条序列逻辑 KV 长度(prefix+extend);注意:实现里用 `kv_indptr` 相邻差作为 prefix 长度 |
| `kv_indices` | `[总逻辑 KV 数]` int32 | 逻辑位置 → `k_buffer`/`v_buffer` 行号 |
| `mask_indptr` | `[B+1]` int32 | 每条序列 `custom_mask` 扁平段起始 |
| `custom_mask` | `[总 mask 元素]` bool 或 0/1 | `True` = 允许 attend;见第 7 节 |
| `max_len_extend` | int | batch 内最大 extend 长度,用于 grid 第三维 |

### 4.3 其它参数

| 参数 | 默认 | 作用 |
|------|------|------|
| `is_causal` | 调用方传入 | 无 custom_mask 时是否用因果三角 |
| `skip_prefix_custom_mask` | `True` | 不读 custom_mask(前缀对 extend Q 通常全可见) |
| `sm_scale` | `1/sqrt(D_q)` | softmax 缩放 |
| `sliding_window_size` | `-1` | `>0` 启用滑动窗口 mask |
| `logit_cap` | `0` | `>0` 时对 logits 做 `cap * tanh(x/cap)` |
| `sinks` | `None` | attention sink token |
| `xai_temperature_len` | `-1` | 位置相关温度缩放 |

`spec_decoding.eagle` 调用示例(`tree_verify_attn.py` 的 `_triton_tree_attn_forward`):

```python
extend_attention_fwd(
    q, k_extend, v_extend, o,
    k_all, v_all,           # 逻辑连续的 prefix+extend buffer
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, False,     # is_causal=False,树 mask 全权负责
    mask_indptr, L,
    sm_scale=head_dim ** -0.5,
    skip_prefix_custom_mask=True,
)
```

---

## 5. Kernel 网格与分块

### 5.1 Launch grid

```text
grid = (batch_size, head_num, ceil(max_len_extend / BLOCK_M))
```

- `program_id(0)`:batch 内序列 id  
- `program_id(1)`:head id  
- `program_id(2)`:extend 序列方向上的 Q 块 id(每块 `BLOCK_M` 个 query)

### 5.2 块大小 `_get_block_sizes_for_extend_attention`

按 head 维 `Lq/Lv` 和 GPU compute capability 选择 `BLOCK_M`、`BLOCK_N`、`num_warps`:

- 大 head(如 576)拆成 `BLOCK_DMODEL + BLOCK_DPE`(主维 + partial RoPE 维),`qk` 两次 `tl.dot` 相加。
- Hopper (sm90+)、Ampere (sm80+)、Blackwell (sm120) 等有不同 `(BLOCK_M, BLOCK_N)` 表。

---

## 6. 核心算法:`_fwd_kernel` 两阶段

采用 online softmax,维护每个 query 行的 `e_max`、`deno`、`acc`,避免物化完整 `[Q, K]` 分数矩阵。

### 6.1 每个 program 先读序列元信息

```text
cur_seq_len_extend = qo_indptr[seq+1] - qo_indptr[seq]
cur_seq_len_prefix = kv_indptr[seq+1] - kv_indptr[seq]   # 实现约定:此处表示 prefix 长度
cur_seq_len          = prefix + extend
```

加载当前 Q 块:`q` shape `[BLOCK_M, BLOCK_DMODEL]`(及可选 `qpe`)。

### 6.2 Stage 1:对 prefix 的 K/V(在 buffer 里)

```text
for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
    通过 kv_indices 从 K_Buffer / V_Buffer 取 K/V
    qk = scale * (Q @ K^T)   [+ logit_cap / xai_temperature]
    应用 final_mask:
      - 默认: 有效 query × 有效 key
      - USE_CUSTOM_MASK 且 not SKIP_PREFIX_CUSTOM_MASK: 与 custom_mask 相与
      - SLIDING_WINDOW_SIZE > 0: 滑动窗口
    在线 softmax 累加到 acc, deno, e_max
```

EAGLE1 默认 `skip_prefix_custom_mask=True`:前缀段不看 `custom_mask`,即 所有 extend query 均可 attend 全部 prefix KV(与树 verify 语义一致:历史 prompt 全局可见)。

### 6.3 Stage 2:对 extend 段的 K/V(在 k_extend/v_extend 里)

```text
cur_block_m_end = extend_len  若 is_causal=False
                = min(extend_len, (block_m+1)*BLOCK_M)  若 is_causal=True  # 因果三角上界

for start_n in range(0, cur_block_m_end, BLOCK_N):
    从 K_Extend / V_Extend 读当前 extend 块的 K/V
    final_mask:
      - USE_CUSTOM_MASK: 读 custom_mask(含 prefix+extend 列偏移 cur_seq_len_prefix)
      - elif IS_CAUSAL: query index >= key index(extend 内因果)
      - else: 全连接
    同样在线 softmax 累加
```

EAGLE1:`is_causal=False` + `USE_CUSTOM_MASK=True`,Stage 2 完全由 树形 custom_mask 决定谁能看谁,而不是下三角因果。

### 6.4 写回

```text
O_Extend[query_row, head, :] = acc / deno
```

可选 `HAS_SINK`:在分母加 sink 项

---

## 7. Custom mask 内存布局(读懂 EAGLE 树 verify 的关键)

对 batch 中第 `seq` 条序列,设:

- `P = cur_seq_len_prefix`
- `E = cur_seq_len_extend`
- `mask` 段长度 = `E * (P + E)`(与 SGLang `EagleVerifyInput` 一致)

扁平下标(行优先,按 extend 内 query 行 `i = 0..E-1`,逻辑 KV 列 `j = 0..P+E-1`):

```text
idx(i, j) = mask_indptr[seq] + i * (P + E) + j
```

- `j < P`:第 `i` 个 extend query 对 前缀 第 `j` 个 KV 是否允许。  
- `j >= P`:对 extend 段中第 `j-P` 个 KV 是否允许(树掩码子块)。

Stage 1 若启用 custom_mask(`skip_prefix_custom_mask=False`),列 `j` 仅遍历 `[0, P)`。  
Stage 2 列在 extend 段时加上偏移 `cur_seq_len_prefix`。

spec_decoding 由 `build_tree_cross_attn_bias_with_prefix` 得到 `[E, P+E]` 的 0/-inf bias,再 `(bias == 0).reshape(-1)` 转为 bool `custom_mask`。

示例(`P=5`, `E=3`,两分支兄弟):

- query 1 可看 prefix 全部 + 节点 {0,1}  
- query 2 可看 prefix 全部 + 节点 {0,2}  
- 兄弟 1 与 2 不可 互看 → `custom_mask[1, P+2] = False` 等

---

## 9. 在 spec_decoding 中的调用链

```text
Eagle1Generator.generate_step_payload
  → full_model_tree_verify_logits (verify_attn_backend=triton_tree)
    → tree_verify_triton_attention_context
      → 每层 self_attn.forward 被替换
        → _triton_tree_attn_forward
          → extend_attention_fwd(...)
```

Patch 层(`tree_verify_attn.py`)负责:

1. HF 标准 `q_proj/k_proj/v_proj`、RoPE、`past_key_value.update`  
2. 将 `key_states/value_states` 转为 `[total, H_kv, D]` 的 `k_all/v_all`,`k_extend = k_all[-E:]`  
3. 填入 `TritonTreeVerifyMetadata`(indptr、mask)  
4. 调用本 kernel,再 `o_proj`

未走本 kernel 时:`verify_attn_backend=eager` 用 HF forward + 4D 树 mask patch;`flash_attn` 在 `topk=1` 用官方 FA2。

---

## 10. 与 FlashAttention / SDPA 的对比

| 维度 | Extend Attention (本 kernel) | FlashAttention decode | HF eager + 4D mask |
|------|---------------------------|------------------------|---------------------|
| Q 范围 | 仅 extend 段 | 通常 1 token | 整段 forward |
| KV 来源 | pool 索引 + extend 张量 | 单一 k_cache | 单层 matmul |
| 树形 mask | bool `custom_mask` | 不支持 | additive 4D mask |
| 批量形式 | CSR indptr,适合 serving | 固定 decode API | 简单但慢 |

---

## 11. 限制与注意事项

1. 依赖 Triton + CUDA;无 GPU 时 `triton_extend_available()` 为 false,回退 eager。  
2. GQA:`k_extend/v_extend` 必须是 KV head 数,不要手动 repeat 成 Q head 数。  
3. `kv_indptr` 语义:相邻差在 Stage 1 当作 prefix 长度;`kv_indices` 长度应覆盖 `P+E` 个逻辑位置。  
4. 数值:在线 softmax 与 eager 可能有微小浮点差;应用时用 `eager` 对照 logits。  
5. 支持模型:patch 仅覆盖 `LlamaAttention` / `Qwen2Attention` 等;需 decoder 传入 `position_embeddings`。  
6. Sliding window / sink / logit_cap:kernel 支持,EAGLE1 当前调用未启用。

---


## 13. 相关文件

| 文件 | 关系 |
|------|------|
| `kernels/extend_attention.py` | Kernel 实现 |
| `tree_verify_attn.py` | EAGLE1 调用与 HF patch |
| `tree_verify_mask.py` | 树掩码 → `custom_mask` |
| `tree_verify_full.py` | `verify_attn_backend` 分发 |
| `config.py` | `verify_attn_backend` 配置项 |
