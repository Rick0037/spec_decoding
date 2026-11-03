# EAGLE1 函数调用栈(`spec_decoding.eagle`)

本包在普通链式投机之上补三件事,但用 静态 top-k 树:

1. feature-level draft:轻量 draft 复用 target 的 embedding/lm_head,输入 `fc(concat(token_emb, target_hidden))`,通常 1 层 decoder 且首层跳过 `input_layernorm`。
2. top-k>1 静态树:每节点固定展开 top-k 子节点。
3. 树注意力 verify:一次 target forward 验证整棵树(树掩码防兄弟泄漏)。

---

## 0) 模块一览

| 文件 | 主要符号 |
|------|----------|
| `config.py` | `Eagle1Config` |
| `draft_model.py` | `Eagle1DraftModel`、`build_eagle1_draft`、`make_eagle1_draft_topk_fn`、`_DraftDecoderLayer`、`_RMSNorm` |
| `hidden.py` | `forward_target_last_hidden` |
| `tree_draft.py` | `expand_draft_tree`、`tree_draft_to_verify_layout`、`TreeDraftResult`、`DraftTopkFn` |
| `runner.py` | `Eagle1Generator`、`make_hidden_residual_draft_topk_fn`、`_verify_path_clean` |
| `tree_verify_mask.py` | `TreeVerifyLayout`、`build_tree_sdpa_attn_bias`、`build_tree_cross_attn_bias_with_prefix` |
| `tree_verify_full.py` | `full_model_tree_verify_logits`、`select_append_from_full_tree_logits`、`tree_verify_attention_context` |
| `tree_verify_parallel.py` | `path_node_indices_root_to_leaf`、`verify_leaf_path_against_parallel_logits`(供 full_model_tree 选最优路径) |
| `tree_verify_attn.py` | `resolve_verify_attn_backend`、`flash_attn_available`、`triton_extend_available` |
| `kernels/extend_attention.py` | Triton `extend_attention_fwd`(树掩码 attention) |

---

## 1) 总览调用图(端到端)

```text
# 准备阶段(一次)
build_eagle1_draft(target)                 # draft_model.py
  -> Eagle1DraftModel.__init__
       reuse target.get_input_embeddings() / get_output_embeddings()
       fc = Linear(2H, H); layers=[_DraftDecoderLayer(skip_input_layernorm=i==0)]; norm
make_eagle1_draft_topk_fn(draft, topk)     # -> Draft作spec并求得Topk ids

# 生成阶段
Eagle1Generator(target, cfg, draft_topk_fn).generate(input_ids, max_new_tokens)
  -> target prefill (out0, past)
  -> while new_tok < max_new_tokens:
       -> Eagle1Generator.generate_step_payload(out_ids, past, last_token)
          -> forward_target_last_hidden(target, out_ids, use_cache=False)   # 返回last token hidden, 一般情况下,这个函数只在while循环第一次调用,其余都不调用,而是last token hidden直接从target replay处就获取了
          -> expand_draft_tree(cfg, h_root, last_token, draft_topk_fn)      # 静态 top-k 树
               -> draft_topk_fn(hidden, parent_token)  (每节点)
                    -> Eagle1DraftModel.forward -> topk
          -> tree_draft_to_verify_layout(draft_res)
          -> verify(按 cfg.verify_mode):
               reference_paths    -> _verify_path_clean(每个叶子路径)
               full_model_tree    -> full_model_tree_verify_logits + select_append_from_full_tree_logits
          -> 返回 append(本轮采纳 token)
       -> target replay append -> 更新 past / out_ids
```

---

## 2) 准备:轻量 EAGLE1 draft

### `draft_model.build_eagle1_draft(target, num_layers=1)`
```text
build_eagle1_draft
  -> Eagle1DraftModel.__init__(target, num_layers)
       tm = _unwrap(target)                      # 取完整 *ForCausalLM(保留 lm_head)
       embed   = tm.get_input_embeddings()        # 复用(list 持有,不注册为子模块)
       lm_head = tm.get_output_embeddings()       # 复用;None 表示与 embedding tie
       fc      = Linear(2H, H)
       layers  = [_DraftDecoderLayer(H, n_heads, intermediate,
                                     skip_input_layernorm=(i == 0)) ...]
       norm    = _RMSNorm(H)
```

### `draft_model.make_eagle1_draft_topk_fn(draft, topk)` → `DraftTopkFn`
```text
fn(hidden[B,H], parent_token_id[B,1]) -> (topk_ids[B,K], topk_scores[B,K])
  -> Eagle1DraftModel.forward(parent_token_id, hidden)
       emb = embed(token)                         # [B,1,H]
       x   = fc(concat([emb, target_hidden], -1)) # feature-level 融合
       for layer in layers: x = layer(x)          # _DraftDecoderLayer
            (skip input_layernorm@layer0) -> self-attn -> +res -> post_ln -> SwiGLU -> +res
       x = norm(x)
       logits = lm_head(x[:, -1])                 # 复用 target 输出头
  -> torch.topk(logits, k=topk)
```

注:EAGLE1 只用 `topk_ids` 建树;`topk_scores` 仅为接口兼容保留。

---

## 3) 生成主循环

### `runner.Eagle1Generator.generate(input_ids, max_new_tokens, eos_token_id)`
```text
generate
  -> out0 = target_m(out_ids, use_cache=True); past = out0.past_key_values
  -> while new_tok < max_new_tokens and not eos:
       last_t = out_ids[:, -1:]
       append, layout, bias, draft_res = self.generate_step_payload(out_ids, past, last_t)
       out_sync = target_m(append, past_key_values=past, use_cache=True)
       past = out_sync.past_key_values
       out_ids = cat([out_ids, append]); new_tok += append.numel()
```
说明:`generate()` 支持 `full_model_tree` / `reference_paths`。

---

## 4) 单轮:建树 + verify

### `runner.Eagle1Generator.generate_step_payload(full_prefix_ids, past_before, last_token_id)`
```text
generate_step_payload
  A. h_root = forward_target_last_hidden(target, full_prefix_ids, use_cache=False)
       # 干净前向,不带 past_before(避免原地污染共享 DynamicCache)
  B. draft_res = expand_draft_tree(cfg, h_root, last_token_id, draft_topk_fn,
                                   target_hidden_refresh=refresh,
                                   initial_prefix_ids=full_prefix_ids)
  C. layout = tree_draft_to_verify_layout(draft_res)
  D. leaves = _leaf_indices(draft_res.parent)
       若无叶子: _verify_path_clean(target, prefix, [])  -> 直接取前缀末 greedy
  E. 按 cfg.verify_mode 分派(见 §6)
  F. 返回 (append, layout, attn_bias, draft_res)
```

---

## 5) Draft 树(静态 top-k,无累计剪枝)

### `tree_draft.expand_draft_tree(cfg, root_hidden, root_parent_token_id, draft_topk_fn, ...)`
```text
expand_draft_tree                              # EAGLE1:静态 BFS top-k
  -> draft_topk_fn(root_hidden, root_token) -> 取前 topk 个 id 作为根层节点
  -> 队列 BFS:
       while 队列非空 and 节点数 < max_tree_nodes:
         node = pop()
         if depth(node) >= num_steps: continue
         h = _expand_hidden(node.hidden, node.path,
                            target_hidden_refresh, initial_prefix_ids)  # 可选刷新 target hidden
         child_ids,_ = draft_topk_fn(h, node.last_token)
         for ct in 前 topk 个 child: 追加为子节点(记录 parent / depth)
  -> TreeDraftResult(token_ids, parent, node_depth, bfs_index)
```
要点(与 EAGLE2 的差异):不计算 `cum_log_probs`,不做 beam / 全局按分剪枝;树形状只由 `topk` / `num_steps` / `max_tree_nodes` 静态决定。

### `tree_draft.tree_draft_to_verify_layout(draft)`
```text
-> TreeVerifyLayout(token_ids, parent, dfs_order=range(n))
```

---

## 6) Verify 三条路径

### 6a. `reference_paths`(默认 e2e/smoke,模型无关、无损)
```text
for leaf in leaves:
  path = _path_tokens_to_leaf(parent, token_ids, leaf)
  mlen, nxt = _verify_path_clean(target_m, full_prefix_ids, path)
       # 一次干净整段前向(use_cache=False);按位置对齐:
       #   path[j] 真值 = logits[P-1+j]; 全中 bonus = logits[P-1+len(path)]
  选 mlen 最大(同长取更前叶子)
append = path[:best_m] + [best_next]   (全中则 path + [bonus])
```

### 6b. `full_model_tree`(树掩码一次性 verify,推荐生产路径)
```text
full_model_tree_verify_logits(target_m, layout, past_before, past_len,
                              verify_attn_backend, topk)
  -> resolve_verify_attn_backend(...)          # tree_verify_attn.py
  -> tree_verify_attention_context(...)        # 给 target attention 注入树掩码
  -> target_m.forward(tree_tokens, past_before) -> 各节点 logits
select_append_from_full_tree_logits(logits, parent, token_ids, leaves)
  -> 沿各叶子路径按 greedy 比对,选最长接受路径 -> append
build_tree_cross_attn_bias_with_prefix(layout, past_len)   # 树+前缀 cross 掩码
```

---

## 7) 树掩码与后端

```text
tree_verify_mask.TreeVerifyLayout                 # token_ids / parent / dfs_order
build_tree_sdpa_attn_bias(layout)                 # 树内 [L,L]:只允许 attend 祖先
build_tree_cross_attn_bias_with_prefix(layout, P) # [L, P+L]:前缀全可见 + 树内祖先

tree_verify_attn.resolve_verify_attn_backend(cfg) # auto -> flash_attn / triton_tree / eager
  flash_attn_available() / triton_extend_available()
kernels/extend_attention.extend_attention_fwd     # Triton:带 custom_mask 的树 attention
```

---

## 8) 关键数据流

```text
target hidden (forward_target_last_hidden)
   -> draft 条件 (Eagle1DraftModel: fc(concat(emb, hidden)))
   -> 静态 top-k 树 (expand_draft_tree) -> TreeDraftResult(token_ids/parent/depth)
   -> TreeVerifyLayout (+ 树掩码 bias)
   -> target verify (reference_paths / full_model_tree) -> append
   -> replay append 更新 KV/out_ids -> 下一轮
```

EAGLE1 = feature 条件的轻量 draft + 静态 top-k 树 + 树掩码 verify;正确性始终由 target verify 保证(无损),draft 只决定接受率/速度。

---

## 9) 入口一:`expand_draft_tree`(建静态 top-k 树)

> `tree_draft.expand_draft_tree`(被 `runner.generate_step_payload` 调用)

```text
expand_draft_tree(cfg, root_hidden, root_parent_token_id, draft_topk_fn,
                  target_hidden_refresh, initial_prefix_ids)
  │
  ├─ draft_topk_fn(root_hidden, root_parent_token_id)        # 根层:父 token + target hidden -> 下一 token 分布
  │    └─(make_eagle1_draft_topk_fn 时)Eagle1DraftModel.forward(token_ids, target_hidden)
  │         ├─ self.embed(token_ids)                          # 复用 target embedding
  │         ├─ self.fc(cat([emb, target_hidden]))             # feature 融合(2H -> H)
  │         ├─ for layer in self.layers: _DraftDecoderLayer.forward
  │         │     ├─ input_layernorm(x)   (layer0 为恒等; 否则 _RMSNorm.forward)
  │         │     ├─ _DraftDecoderLayer._attn -> F.scaled_dot_product_attention(is_causal)
  │         │     ├─ post_attention_layernorm -> _RMSNorm.forward
  │         │     └─ SwiGLU: down(silu(gate(x)) * up(x))
  │         ├─ self.norm -> _RMSNorm.forward
  │         └─ self._lm_logits(x[:, -1]) -> 复用 target lm_head
  ├─ _topk_ids(topk_ids, K)                                   # 取前 K 个候选 id(忽略分数)
  │
  └─ BFS 循环  while q 非空 and 节点数 < max_tree_nodes:
       ├─ _expand_hidden(parent_hidden, path,                 # 扩展子节点前刷新 draft 条件
       │     target_hidden_refresh, initial_prefix_ids)
       │     └─(若提供 refresh)target_hidden_refresh(prefix+path)
       │           └─ Eagle1Generator._refresh_hidden(prefix)
       │                └─ forward_target_last_hidden(target, prefix, use_cache=False)
       │                     ├─ _unwrap(target).forward(output_hidden_states=True)
       │                     └─ 取末 token 第 hidden_layer_index 层 hidden  [1,H]
       ├─ draft_topk_fn(h, 父节点末 token)                    # 同上 draft 前向 -> top-k
       └─ _topk_ids -> 追加子节点(token_ids/parent/node_depth)
  ▼
  TreeDraftResult(token_ids, parent, node_depth, bfs_index)   # 静态树(无累计概率/剪枝)
```

要点:根层 + 每个非叶节点都各调用一次 `draft_topk_fn`(即一次 draft 前向);`depth >= num_steps` 的节点不再展开;树形状只由 `topk/num_steps/max_tree_nodes` 决定。

---

## 10) 入口二:`full_model_tree_verify_logits`(一次树掩码 verify)

> `tree_verify_full.full_model_tree_verify_logits`(`verify_mode=full_model_tree` 时被 `runner.generate_step_payload` 调用)

```text
full_model_tree_verify_logits(target_m, layout, past_key_values, past_len,
                              verify_attn_backend, topk)
  │
  ├─ layout_tree_position_ids(layout, past_len)               # 各节点 position_ids = 根到该节点深度 + past_len
  │    └─ node_depth_from_root(layout.parent, i)   (每节点沿 parent 上溯)
  ├─ build_tree_allow_mask_4d(layout) -> [1,1,L,L]            # 1=允许 / 0=屏蔽(兄弟分支不可互看)
  │    └─ _on_tree_path(layout.parent, j, i)   (每对 (i,j) 判 j 是否 i 的祖先/自身)
  ├─ _decoder_stack(target_m)                                 # 取 target_m.model(decoder 栈)
  ├─ resolve_verify_attn_backend(backend, topk, layout_len)   # -> 'flash_attn' / 'triton_tree' / 'eager'
  │    └─ flash_attn_available() / triton_extend_available()
  │
  └─ 按 backend 分派一次 target forward(use_cache=False,不把树写进 KV):
       ├─ eager(默认回退):
       │    with tree_verify_attention_context(decoder, tree_block):   # patch _update_causal_mask
       │         └─ _apply_tree_block_to_causal(causal, tree_block)     # 把树掩码叠到因果 mask 右下角 [-L:,-L:]
       │    └─ target_m(draft_ids, past_key_values, position_ids, use_cache=False)
       │
       ├─ flash_attn(topk==1):
       │    with tree_verify_attention_context_flash(decoder):          # 切 _attn_implementation=flash_attention_2
       │    └─ target_m(draft_ids, ...)                                 # 链式因果,不叠树 mask
       │
       └─ triton_tree(topk>1):
            ├─ build_triton_tree_verify_metadata(layout, past_len)
            │    └─ build_tree_cross_attn_bias_with_prefix(...) -> custom_mask(bool, 展平)
            with tree_verify_triton_attention_context(decoder, meta):   # patch 每层 self_attn.forward
            │    └─ _triton_tree_attn_forward(module, hidden_states, ...)
            │         ├─ q_proj/k_proj/v_proj
            │         ├─ _apply_rotary(module, q, k, position_embeddings)
            │         ├─ past_key_value.update(...)                      # 写/取 KV(_layer_idx 定位层)
            │         ├─ kernels/extend_attention.extend_attention_fwd   # Triton:带 custom_mask 的树 attention
            │         └─ o_proj
            └─ target_m(draft_ids, ...)
  ▼
  return out.logits   # [1, L, vocab]:每行=树第 i 个节点处 target 对“下一 token”的预测
```

要点:无论哪个 backend,都是一次整模前向算出全部 L 个节点的 logits;`eager`/`triton_tree` 用树掩码保证兄弟分支不互相 attend,`flash_attn` 仅在 `topk==1`(退化为链)时用。返回的 `logits` 随后交给 `select_append_from_full_tree_logits` 选最长接受路径。

---

## 真实模型端到端:模型选择、修复、测试结果与结论

> example:`spec_decoding/eagle/example/run_eagle1_llama3.py`;
> `spec_decoding/eagle/example/perf.log`。全程在 docker 镜像为nvcr.io/nvidia/pytorch:25.04-py3的容器运行。

### 模型选择(与 sglang / vllm 的 EAGLE-1 配置一致)
- **target**:`meta-llama/Meta-Llama-3-8B-Instruct`(无 HF token 时自动回退等价镜像
  `NousResearch/Meta-Llama-3-8B-Instruct`)。
- **draft**:`yuhuili/EAGLE-LLaMA3-Instruct-8B`(官方 EAGLE-1 权重)。结构正好匹配本包:
  `fc(2H→H)` + 1 层 **GQA**(32 q / 8 kv)decoder(首层无 input_layernorm)+ 复用 target
  embedding/lm_head,且**无 draft 最终 norm**。EAGLE-1 的条件 = target 最后一层 hidden(lm_head
  之前的 post-norm),本包 `hidden_layer_index=-1` 已对。

### 为跑真实 draft 做的改动
- `draft_model.py`:加 **GQA**(k/v_proj=`n_kv*head_dim` + `_sdpa_gqa`,否则装不下 32q/8kv 权重)、
  RMSNorm **eps 从 config 读**(1e-5)、`use_final_norm` 开关(官方无 draft norm → 加载真实权重时
  自动 False)、新增 **`load_eagle1_draft_weights`**(去 `self_attn.`/`mlp.`、跳过 `embed_tokens`);
  `build_eagle1_draft` 增 `draft_weights_path`。
- `runner.py`:加 `refresh_target_each_node` 开关(大模型设 False)、`n_rounds`/`n_accepted_tokens` 计数。

### full_model_tree 升级为复用前缀 KV 的 extend verify
原 full_model_tree 走 `full_model_tree_verify_logits` + `select_append_from_full_tree_logits`,
该路径在 transformers 5.x 上有两处 bug(triton kernel + 接受 off-by-one)。方案:
- 新增 `tree_verify_full.full_tree_verify_extend`(HF sdpa + 4D 树 cross mask)/
  `full_tree_verify_triton`,**只前向 L 个树节点、复用前缀 KV、
  不重算前缀**;共用 `_select_append_root_child`(**正确 parent→child 接受**:节点 i 的输出预测其子节点,
  root 用 `recovery_logits`)。
- 修 triton metadata 的 `kv_indptr/kv_indices`(只描述前缀,修兄弟泄漏)+ transformer5x版本的attention kwarg/返回值兼容。
- runner:`generate` 透传 `recovery_logits` 与 `root_hidden`(从 prefill/append-replay 的
  `output_hidden_states` 取,省掉每轮整段前缀 forward);**KV-replay 用全长 mask**(否则
  `recovery_logits` 错算);verify 在 past 上临时 extend 再 `crop` 还原(不深拷贝、不污染 past)。
- backend `auto`:topk>1 且 triton 可用 → `triton_tree`,否则 HF sdpa。

### 测试结果(真实 Llama-3-8B + EAGLE-1 draft,64 tokens,topk=4/num_steps=4/max_nodes=24)

| verify | accept/round | 无损 | wall | spec tok/s | vs HF 贪心 |
|---|---|---|---|---|---|
| reference_paths(逐叶干净前向,旧) | 2.21 | ✅ | 9.74s | 6.6 | 0.11× |
| full_model_tree extend(clone 缓存) | 2.21 | ✅ | 4.00s | 16.0 | 0.29× |
| **full_model_tree extend(crop 还原,auto→triton)** | 2.21 | ✅ | **3.27s** | **19.6** | **0.38×** |

- extend verify 比旧 reference_paths **快约 3×**(9.74s→3.27s),全程**无损**、accept/round=2.21 不变。

### 结论
- HF greedy推理的基线 = 不用投机、直接 `target.generate(do_sample=False, use_cache=True)` 让 8B target 自己逐 token 生成同样数量 token 的墙钟(wall time)时间。
- extend verify 达成了只前向 L 个树节点、不重算前缀的目标(3× 于 reference_paths、无损、accept 2.21),
  但端到端**仍 0.38× 于 HF 贪心(更慢)**。原因是这套**参考实现的逐轮开销**:Python 建树/选叶、
  `output_hidden_states`、带 4D mask 的 sdpa extend(未用融合 kernel)、单层 draft 的多次小前向、cache crop。
- 8B 在 H100 上 HF 本就很快(~52 tok/s 的紧凑 KV-cache 解码)。要靠投机**净加速**,需要更大/更memory-bound 的 target,或 sglang/vllm 那样的生产级融合 kernel。本包算法**无损正确、接受率真实**,但墙钟在此规模/实现下不占优。

### 思考

- 在4090这种消费显卡试一下,HF greedy应该就没有这么快
- target模型换成更大一点的,比如32B的(已试32B,对speedup无帮助,还是0.38x)
- 看一下eagle2,eagle3,是否是类似结论(已试eagle2,对speedup无帮助,还是0.38x)

