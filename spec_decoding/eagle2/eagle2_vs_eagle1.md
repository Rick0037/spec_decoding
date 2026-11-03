# EAGLE2 相比 EAGLE1 的改动

本文对比本仓库的两个包:

- `spec_decoding.eagle`(EAGLE1)
- `spec_decoding.eagle2`(EAGLE2,本目录)

两者共享同一套「融合 target input embedding / lm_head 和深层特征的 draft + top-k draft tree + tree attn verify」框架;EAGLE2 的核心增量是把 EAGLE1 的**静态树**升级为**累计概率驱动的动态树(beam + 全局剪枝)**。

**Draft 权重完全共用**:EAGLE2 与 EAGLE1 可以使用**同一份 EAGLE draft checkpoint**(如 `yuhuili/EAGLE-LLaMA3-Instruct-8B`);即draft 网络结构与权重都不变。

---

## 总结

> EAGLE2 = EAGLE1 + 「累计 log 概率的动态树展开与剪枝」。draft 网络、draft 权重、verify 思路、Runner 调度能力基本一致;差异集中在树展开与 topk_fn 分数语义。

---

## 共同点(两包一致)

- **轻量 draft 网络**:复用 target 的 embedding/lm_head + `fc(2H→H)` + 通常 1 层 decoder(首层跳过 `input_layernorm`)。
  - EAGLE1:`draft_model.Eagle1DraftModel` / `build_eagle1_draft` / `make_eagle1_draft_topk_fn`
  - EAGLE2:`draft_model.Eagle2DraftModel` / `build_eagle2_draft` / `make_eagle2_draft_topk_fn`
  - 结构完全相同;**checkpoint 可互换**;差异仅在 topk_fn 返回的 score 形式(见下)。
- **集成feature-level tensor作为输入**:draft 输入 = `fc(concat(token_emb, target_hidden))`。
- **verify 模式**:`full_model_tree`(extend verify,推荐)与 `reference_paths`(逐叶路径校验,慢、作对照)。
- **树掩码 / 后端**:`tree_verify_mask` / `tree_verify_full` / `tree_verify_attn` / `tree_verify_parallel`;后端 eager / flash_attn / triton_tree;
- **Runner 调度能力**(两包 `runner.py` 对齐,详见「两包已对齐的 Runner 能力」):
  - `refresh_target_each_node`、`root_hidden` / `recovery_logits` 透传
  - `n_rounds` / `n_accepted_tokens` 诊断计数
  - extend verify 路径(`full_tree_verify_extend` / `full_tree_verify_triton`)

---

## 核心差异速查

| 维度 | EAGLE1 (`eagle`) | EAGLE2 (`eagle2`) |
|------|------------------|-------------------|
| 树展开策略 | **静态 top-k 树**(固定每节点展开 topk,BFS 填到 `max_tree_nodes`) | **累计 log 概率 beam + 按累积分数剪枝**(动态、上下文感知) |
| 展开函数 | `tree_draft.expand_draft_tree` | `tree_draft.expand_draft_tree_topk`(cumulative)/ `expand_draft_tree_bfs` |
| 是否算累计概率 | 否(无打分、无剪枝) | 是(`cum_log_probs` 逐层累加) |
| 剪枝 | 仅按 BFS 顺序截断到 `max_tree_nodes` | `_compact_tree_by_score`:按累计分保留 top-N(含祖先) |
| 配置类 | `Eagle1Config`(无 `tree_expand_mode`) | `Eagle2TreeConfig`(多 `tree_expand_mode: cumulative/bfs`) |
| `TreeDraftResult` | `token_ids/parent/node_depth/bfs_index` | 额外含 `cum_log_probs` |
| draft topk_fn 分数 | 原始 logits 的 top-k(`make_eagle1_draft_topk_fn`) | **log_softmax 后**的 top-k(`make_eagle2_draft_topk_fn`),供累计概率用 |
| 生成器类 | `Eagle1Generator` | `Eagle2Generator` |
| `reference_paths` 校验 | `_verify_path_clean`(串行 verify 整段,位置对齐,**无损**、不污染 KV) | **已同步:同 EAGLE1 的 `_verify_path_clean`** |
| `h_root` 取 hidden | `use_cache=False`、不传 past(不污染共享 KV) | **已同步:同 EAGLE1(`use_cache=False`、不传 past)** |

---

## 逐项详述

### 1. 树展开:静态 vs 动态(最核心)

EAGLE1(`eagle/tree_draft.py`):

```text
expand_draft_tree(cfg, ...)
  - 根层 draft_topk_fn 取 top-k -> k 个根节点
  - BFS:每个深度 < num_steps 的节点固定展开 top-k 子节点
  - 节点数达 max_tree_nodes 即停(按 BFS 顺序截断)
  - 根层 K = cfg.topk(不做 max_tree_nodes 截断)
```

EAGLE2(`eagle2/tree_draft.py`):

```text
expand_draft_tree_topk(cfg, ...)   # tree_expand_mode="cumulative"(默认)
  - 每一步:cum_logp(child) = cum_logp(parent) + log p(child|parent)
  - 每步只保留累计分最高的 topk 条活跃路径(beam)
  - 根层 K = min(topk, max_tree_nodes)
  - 结束后若超 max_tree_nodes,用 _compact_tree_by_score 按累计分全局剪枝(保留祖先)

expand_draft_tree_bfs(cfg, ...)    # tree_expand_mode="bfs"(对照 / 调试)
  - BFS 展开,不做 beam 剪枝;树形状接近 EAGLE1 静态树
  - 仍会记录 cum_log_probs(EAGLE1 无此字段)
  - 根层 K = min(topk, max_tree_nodes - 1)(与 cumulative 的 K 计算略有不同)
```

观察现象:EAGLE1 的树形状只由超参(`topk/num_steps/max_tree_nodes`)静态决定;EAGLE2 用 draft 的置信度(累计 log 概率)动态选择更可能被接受的路径,提高节点利用率。

#### `_compact_tree_by_score` 算法(EAGLE2 独有)

```text
1. 若节点数 n <= max_nodes,直接返回
2. 按 cum_log_probs 降序排列所有节点索引
3. 取前 max_nodes 个高分节点,并沿 parent 链向上补全所有祖先(保证 parent 链有效)
4. 按原 BFS 顺序重排保留节点,重映射 parent 索引
```

### 2. 配置差异

- EAGLE1 `Eagle1Config`:无 `tree_expand_mode`(恒为静态树)。
- EAGLE2 `Eagle2TreeConfig`:多 `tree_expand_mode ∈ {cumulative, bfs}`。
- 其余字段(`topk`/`num_steps`/`max_tree_nodes`/`hidden_layer_index`/`verify_mode`/`verify_attn_backend`)一致。

### 3. draft topk_fn 分数语义

- EAGLE1 `make_eagle1_draft_topk_fn`:`torch.topk(logits)` 返回原始 logits 值(静态树不依赖分数)。
- EAGLE2 `make_eagle2_draft_topk_fn`:`torch.topk(log_softmax(logits))` 返回 **log 概率**,便于树展开把每步分数当 log 概率累加做 beam/剪枝。
- **注意**:`make_hidden_residual_draft_topk_fn`(两包均有)始终返回 log_softmax 分数,与 EAGLE2 draft topk_fn 一致;只有 `make_eagle1_draft_topk_fn` 返回原始 logits。

### 4. 数据结构

`TreeDraftResult`:EAGLE2 多一个 `cum_log_probs`(每节点累计 log 概率);EAGLE1 已移除该字段。EAGLE2 另有内部类 `_FrontierNode`(beam 展开用)。

### 5. verify 实现

#### reference_paths(两包已对齐)

- `_verify_path_clean`:对 `prefix + path` 做一次 `use_cache=False` 的干净整段前向,按位置 `logits[P-1+j]` 对齐(`path[j]` 的真值用「吃完真实前缀后」的预测),**无损**且不污染 `DynamicCache`。
- `h_root`:用 `use_cache=False` 取末 token hidden、不传 `past_before`,避免原地修改共享 KV。

#### full_model_tree(Runner 实际调用链,两包相同)

文档常简称为「一次整模型 forward + 树掩码」,这里经过一点小优化, **Runner 实际走的是 extend verify 路径**,而非旧的 `full_model_tree_verify_logits` + `select_append_from_full_tree_logits`(后者仍在 `tree_verify_full.py` 导出,供单独调用/单测):

```text
Eagle1/2Generator.generate_step_payload(verify_mode=full_model_tree)
  → full_tree_verify_extend(HF sdpa + 4D cross mask)
     或 full_tree_verify_triton(SGLang extend_attention_fwd + custom_mask)
  → _select_append_root_child(parent→child 接受 + recovery_logits root 接受)
```

**Extend verify 的 KV 语义**:在 runner 的 `past` 上**临时 extend**(追加 L 个树节点 KV),取完 logits 后 **crop 还原**,不污染 `DynamicCache`。eagle2 的 `tree_verify_full.py` 注释已更新为这一表述。

**辅助模块** `tree_verify_parallel.py`(两包均有):

- `path_node_indices_root_to_leaf` — 根→叶节点索引
- `verify_leaf_path_against_parallel_logits` — 对单次 parallel forward 的 logits 做 greedy 路径比对
- 被 `_select_append_root_child` / `select_append_from_full_tree_logits` 复用

### 6. draft 网络(已对齐)

两包的 `draft_model.py` 结构完全一致(reuse embedding/lm_head + fc + 1 层 decoder + 跳过 layer0 input_layernorm);**官方 EAGLE checkpoint 两包通用**。EAGLE2 可用 `build_eagle2_draft` + `make_eagle2_draft_topk_fn` 直接喂给 `Eagle2Generator`。

---

## 两包已对齐的 Runner 能力

以下机制在 `Eagle1Generator` / `Eagle2Generator` 的 `runner.py` 中**完全一致**(详见 `eagle/eagle1.md` 调用栈):

| 机制 | 说明 |
|------|------|
| `refresh_target_each_node`(默认 True) | True:展开树时对每个内部节点重跑 target 取 hidden(接受率更高);False:只用根 hidden始终作为feature输入,建树时仅跑 draft |
| `root_hidden` 透传 | 从 prefill / append-replay 的 `output_hidden_states` 取末 token hidden,省掉每轮整段前缀重算 |
| `recovery_logits` 透传 | 从同一次前向取 `out_ids` 之后那一位的 target logits,供 extend verify 的 root 接受 |
| `n_rounds` / `n_accepted_tokens` | 诊断计数(accept/round = `n_accepted_tokens / n_rounds`) |
| append-replay | 每轮 verify 后把 append 重放进 target KV;`attention_mask` 必须覆盖 `[past + append]` 全长,否则 `recovery_logits` 错算 |

---

## 已知代码差异(非树展开核心)

| 位置 | 差异 |
|------|------|
| `hidden.py` 的 `_unwrap` | **eagle** 优先 `PreTrainedModel` 本身(保留 lm_head);**eagle2** 仍优先 `.model` 子模块(旧逻辑)。两包 **runner.py** 均已改为优先 `PreTrainedModel` 本身 |
| `eagle2/__init__.py` | 提到 `BaseModelForCausalLM.generate_eagle2`,但仓库内**尚无**该入口(EAGLE3 才有 `generate_eagle3`) |

---

## 文件级差异清单

| 文件 | EAGLE1 | EAGLE2 |
|------|--------|--------|
| `config.py` | `Eagle1Config` | `Eagle2TreeConfig`(+ `tree_expand_mode`) |
| `tree_draft.py` | `expand_draft_tree`(静态、无 cum_log_probs) | `expand_draft_tree_topk` + `expand_draft_tree_bfs` + `_compact_tree_by_score` + `_topk_log_probs` + `_FrontierNode` |
| `runner.py` | `Eagle1Generator` + `_verify_path_clean` | `Eagle2Generator` + `_verify_path_clean`(逻辑对齐) |
| `draft_model.py` | `Eagle1DraftModel` / `make_eagle1_draft_topk_fn`(logits topk) | `Eagle2DraftModel` / `make_eagle2_draft_topk_fn`(log_softmax topk);**checkpoint 通用** |
| `hidden.py` | `forward_target_last_hidden` | 同左;`_unwrap` 实现略有不同(见上) |
| `tree_verify_mask/full/attn/parallel.py` | 逻辑相同 | 逻辑相同(注释措辞略有差异) |
| `kernels/extend_attention.py` + `EXTEND_ATTENTION.md` | 有 | 有(两包各一份,内容基本相同) |
| 文档 | `eagle1.md`(完整调用栈) | 本文 + `kernels/EXTEND_ATTENTION.md`(**尚无**等价 standalone 调用栈文档) |

---

## 例子 / 测试 / 文档索引

### EAGLE1

| 类型 | 路径 | 说明 |
|------|------|------|
| 端到端 | `example/run_eagle1_llama3.py` | Llama-3-8B + 官方 EAGLE draft + perf.log;默认 `verify_mode=full_model_tree`,`refresh_target_each_node=False` |
| 端到端 | `example/run_eagle1_qwen25.py` | Qwen2.5 变体 |
| 冒烟 | `example/run_smoke_eagle1.py` | 轻量 CPU smoke |
| 测试 | `tests/test_eagle1_e2e_llama.py` | 无损 e2e(门控 `RUN_SPEC_E2E=1`) |
| 测试 | `tests/test_eagle_tree_draft.py` | 静态树展开单测 |
| 测试 | `tests/test_eagle_config.py` | 配置校验 |
| 文档 | `eagle1.md` | 完整函数调用栈 |

**EAGLE1 e2e 默认超参**:`num_steps=4`,`max_tree_nodes=24`。

### EAGLE2

| 类型 | 路径 | 说明 |
|------|------|------|
| 端到端 | `example/run_eagle2_llama3.py` | Llama-3-8B + 同一官方 EAGLE draft + perf.log;默认 `tree_expand_mode=cumulative`,`verify_mode=full_model_tree` |
| 冒烟 | `example/run_smoke_eagle2_draft.py` | 轻量 EAGLE draft + GPT-2 tiny |
| 冒烟 | `example/run_smoke_tree.py` | CPU mock topk_fn,无需 HF 下载;测树展开 + SDPA bias |
| 测试 | `tests/test_eagle2_draft_model.py` | draft 模型 + reference_paths 无损 |
| 测试 | `tests/test_eagle2_all.py` | 聚合 runner(树掩码 / 全树 / 累计剪枝 / verify attn) |
| 测试 | `tests/test_eagle2_tree_verify_mask.py` | 树掩码 |
| 测试 | `tests/test_eagle2_full_tree.py` | full_model_tree |
| 测试 | `tests/test_eagle2_cumulative_prune.py` | 累计概率剪枝 |
| 测试 | `tests/test_eagle2_verify_attn.py` | verify attention 后端 |
| 文档 | 本文 + `kernels/EXTEND_ATTENTION.md` | extend attention kernel 说明 |

**EAGLE2 e2e 默认超参**:`num_steps=5`,`max_tree_nodes=32`(比 EAGLE1 e2e 略大,便于体现动态剪枝优势)。
