# EAGLE3 相比 EAGLE2 的改动

本文对比本仓库的两个包:

- `spec_decoding.eagle2`(EAGLE2)
- `spec_decoding.eagle3_sgl`(EAGLE3,本目录)

两者共享「top-k draft 树 + extend 树掩码 verify + Runner 多轮调度」框架;**EAGLE3 的核心增量在 draft 侧**(多层 target 特征、EAGLE3 专用网络结构、自回归 KV draft),树展开与 verify 在工程上大量复用 EAGLE2 能力。

---

## 总结

> EAGLE3 = EAGLE2 的树投机 / verify 框架 + EAGLE3 的「多层 act 条件 draft + EAGLE3 midlayer(双路 norm + 2H qkv) + 自回归 draft KV」。
>
> - **EAGLE2 独有侧重**:单层 hidden 条件 + EAGLE1/EAGLE2 轻量 draft(`fc(2H→H)`)+ 累计概率 beam 树。
> - **EAGLE3 独有侧重**:低/中/高多层 hidden 拼接 + EAGLE3 draft创新(独立词表 / d2t / RoPE midlayer)+ 默认 draft AR模式建树。

**Draft 权重不通用**:EAGLE2 用 EAGLE1/EAGLE2 checkpoint(如 `yuhuili/EAGLE-LLaMA3-Instruct-8B`);EAGLE3 用 EAGLE3 checkpoint(如 `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`),网络结构与词表映射均不同。

---

## 共同点

### 树投机 + verify 框架

- **Top-k > 1 树形 draft**:每步保留 K 个分支,一次 verify 多条候选路径。
- **verify 模式**:`full_model_tree`(extend verify,推荐)与 `reference_paths`(逐叶干净前向,慢、作对照)。
- **verify 后端**:`verify_attn_backend ∈ {auto, eager, flash_attn, triton_tree}`。
- **extend verify 路径**(Runner 实际调用):
  ```text
  full_model_tree
    → full_tree_verify_extend(HF sdpa + 4D cross mask)
       或 full_tree_verify_triton(Triton extend kernel)
    → _select_append_root_child(parent→child 接受 + recovery_logits root 接受)
  ```
- **KV 语义**:在 runner 的 `past` 上临时 extend L 个树节点,取 logits 后 `crop` 还原,不污染 `DynamicCache`。
- **树掩码模块**:`tree_verify_mask` / `tree_verify_full` / `tree_verify_attn` / `tree_verify_parallel`。
- **Triton kernel**:`kernels/extend_attention.py` + `kernels/EXTEND_ATTENTION.md`。

---

## 核心差异速查

| 维度 | EAGLE2 (`eagle2`) | EAGLE3 (`eagle3_sgl`) |
|------|-------------------|---------------------------|
| **draft 条件特征** | target **单层** hidden(`hidden_layer_index`,默认 -1) | target **多层** act 拼接(`eagle_layers`,默认 `[2, L//2, L-3]`) |
| **取特征 API** | `hidden.forward_target_last_hidden` → `[B, H]`=last token hidden | `hidden.forward_target_eagle_acts` → `[B, n*H]`=low med high level feature of all existed tokens hidden |
| **draft 网络** | `fc(2H→H)` + concat(emb, hidden) + 1 层 decoder(跳过 layer0 input_layernorm) | `fc(n*H→H)` + **midlayer**(embed/hidden **分别 norm** 后拼 2H 进 qkv)+ RoPE |
| **lm_head / 词表** | **复用** target embedding + lm_head | 可选 **独立 draft 词表**(`draft_vocab_size` + `d2t` → target token id) |
| **draft checkpoint** | EAGLE1/EAGLE2 权重(与 EAGLE1 通用) | EAGLE3 专用权重(fc / midlayer / lm_head / d2t) |
| **默认建树方式** | 单 token `draft_topk_fn` 逐步展开 | **自回归 draft**(`autoregressive_draft=True`): `expand_draft_tree_ar`带 draft KV + RoPE, 每次draft forward时无需重算已有token的kv |
| **draft KV 维护** | 无(每节点单步 forward 或 refresh target hidden) | `_DraftKvState` + `prefill` / `extend_tokens` 维护增量的 draft KV + root logits + root hidden供下一波draft speculative使用,配合autoregressive_draft |
| **target act 缓存** | 每轮按需取 `root_hidden` | 增量维护 `acts_all`(只用在了一次prefill,以及 append 后 cat 新增 act) |
| **树展开函数** | `expand_draft_tree_topk` / `expand_draft_tree_bfs` | `expand_draft_tree_topk` / `expand_draft_tree`(static)/ `expand_draft_tree_ar` |
| **tree_expand_mode** | `cumulative` / `bfs` | `cumulative` / `static`(= EAGLE1 静态 BFS) |
| **配置类** | `Eagle2TreeConfig` | `Eagle3SglConfig`(+ `eagle_layers` / `resolve_eagle_layers`) |
| **生成器类** | `Eagle2Generator` | `Eagle3SglGenerator` |
| **generate_step 根条件** | `root_hidden: [B, H]` | `root_acts: [B, n*H]` + `acts_all` + `draft_state` |
| **性能 profiling** | 无 | `EAGLE3_PROFILE=1` + `_prof("stage:...")` |
| **e2e 模型** | Llama-3-8B + EAGLE-LLaMA3 draft | Llama-3.1-8B + EAGLE3-LLaMA3.1 draft |

### 核心差异描述

本节汇总讲解 EAGLE3 时最常见的几个概念问题:**draft state数据结构**、**为什么 draft 要 prefill/extend/step**、**EAGLE shift 错位匹配**、以及 **接受率为何 EAGLE3 更高**。

#### 总览:Target 与 Draft 调用栈

```text
generate() 每一轮:

  Target(大模型)                    Draft(小模型 EAGLE3)
  ─────────────────                  ───────────────────────
  target prefill                     draft prefill(仅首轮)
    → past(target KV)                → draft_state.kv(draft KV)
    → recovery_logits                  → root_logits(draft 猜下一 token)
    → acts_all(多层特征,原料)        → root_hidden(feature 递归起点)

                                    draft step() + hidden,logits,kv保存在draft_state便于传递

  target verify

  target replay append               draft extend_tokens(append)
    → 更新 past、acts_all              → 更新 draft_state
```

**关键结论**:Target prefill 和 draft prefill 是**两个不同模型、两套不同 KV**;`acts_all` 是从 target 提取后交给 draft 消费的「条件原料」。

---

#### 核心:如何理解eagle3新增的draft state这个数据结构？

首先,两包里 `past`(target KV)含义相同:

> 只保存 **已被 verify 接受、写进 `out_ids` 的真实前缀** 的 KV。

| | EAGLE2 | EAGLE3(AR 默认) |
|---|--------|-------------------|
| **Target KV (`past`)** | 只存已接受前缀 | 同左 |
| **Draft KV** | ❌ 无 | ✅ `draft_state.kv` |
| **建树时 draft 怎么跑** | 每节点 `draft_topk_fn` 冷启动 | `prefill/extend` + `step` 带 KV+RoPE |
| **Verify 后更新** | 只更新 target `past` + `root_hidden` | 同时更新 target `past` + `draft_state` |

EAGLE2 循环(简化):

```text
target prefill → past + root_hidden(target 末 token hidden)
每轮:draft_topk_fn 建树(无 draft KV)→ target verify(past 临时 extend,crop)
     → target replay 只写入接受的 append → 更新 root_hidden
```

EAGLE3 的 `draft_state` **不是**「上一个 draft token(可能被拒绝)的状态」,而是:

> Draft 小模型在处理完**当前已接受前缀**之后的状态(KV + 末步 logits/hidden),用来猜下一轮 speculative tokens。只有 verify 接受的 append 才会经 `extend_tokens` 写进 draft KV。draft state对齐了eagle3训练形态

Verify 阶段两边一致:在 `past` 上**临时** extend 整棵 draft 树 → 取 logits → **crop 还原**;rejected draft 不污染任何 permanent cache。

---

#### 为什么有 `forward`,还要 `prefill` / `extend` / `step`？

| 函数 | 场景 | 核心能力 |
|------|------|----------|
| **`forward`** | 非 AR 建树(`draft_topk_fn`) | 无 KV、无 RoPE 位置管理;每节点冷启动单步 forward |
| **`prefill`** | 首轮:对 prompt bootstrap | 因果 attention 建 draft KV;RoPE 0..S-1 |
| **`extend_tokens`** | 每轮 verify 后写 append | 在已有 draft KV 上批量 extend(比逐步 step 快) |
| **`step`** | `expand_draft_tree_ar` 建树 | 单 token + 指定 RoPE 位置 + KV fork(分支各自 extend) |

`forward` 做不到:`step` 里那种「首轮 cond 来自 target act(`n*H` 过 fc),后续 cond 来自 draft 自身 hidden 递归」的状态传递。

**一句话**:`forward` = 单步(调试 / 非 AR 建树);`prefill/extend/step` = 带 KV和hidden和logits 的自回归推理,对齐 EAGLE3 训练时的 AR 分布,接受率更高。

---

#### 为什么 target prefill 之后,还要 draft prefill？

| | Target prefill | Draft prefill |
|---|---------------------|---------------------|
| **模型** | 大模型(8B) | 小模型 EAGLE3 draft |
| **产物** | `past`、recovery_logits、`acts_all` | `draft_state`(draft KV、root_logits、root_hidden) |
| **用途** | verify + 提供 draft 条件原料(低中高三层hidden特征) | draft 侧自回归状态,供建树 step 用 |

不能省略 draft prefill 的原因:

- Target KV **不能**给 draft 建树以及自回归AR用(不同网络、不同权重)。
- 仅有 `acts_all` (低中高三层hidden特征)只是原料,draft 网络仍须**自己跑一遍**才能得到 draft KV 和 draft logits保存到draft state方便传递。

---

#### EAGLE shift:prefill 里为何 `ids = prefix_ids[:, 1:]`、`acts = acts_all[:, :-1, :]`？

EAGLE3 约定:**draft 处理 token `t_i` 时,条件来自 `act_{i-1}`**(target 读完 `t_{i-1}` 后的多层特征)。

```text
位置:     0      1      2      3     ...   P-1
token:   t0     t1     t2     t3    ...  t_{P-1}
act:    act0   act1   act2   act3   ...  act_{P-1}

draft 输入 token    条件 act
────────────────    ──────────
t1                  act0
t2                  act1
...
t_{P-1}             act_{P-2}
```

对应切片:

```python
ids   = prefix_ids[:, 1:]         # [t1, ..., t_{P-1}]
acts  = target_acts_all[:, :-1]   # [act0, ..., act_{P-2}]
```

`t0` 不参与 draft prefill:prompt 首 token 前没有可用的 target 条件。prefill 末态 `(root_logits, root_hidden, kv)` 表示 draft 已处理完 `t1..t_{P-1}`,站在 prefix 末尾准备猜下一个 token。

---

#### 接受率:EAGLE3 通常更高,主要因为什么？

**结论**:EAGLE3 AR 路径对接受率更友好,因为 draft **猜得更准**:

1. **Draft 持久自回归状态**(`prefill → extend → step`):RoPE 连续、KV 连续、feature 递归;EAGLE2 每节点冷启动 `draft_topk_fn`,与 EAGLE3 draft 训练分布不一致。
2. **更强的条件特征**:多层 act(即hidden) vs 单层 hidden;独立 draft 词表 + lm_head。
3. **可选 cumulative beam**:两包都有;draft 质量高时 beam 剪枝更有效。
4. **Target verify 两边一致**:接受率差异主要来自draft侧的质量,而非 target verify方式, target verify方式仅影响verify速度,从而影响整体性能。

> EAGLE2 与 EAGLE3 的 target KV 都只跟踪已接受的真实前缀;EAGLE3 额外维护 draft 小模型自己的 KV/state,且用多层 act + RoPE + feature 递归,draft 提案更接近真实 EAGLE3 推理,故接受率通常更高。

---

## 逐项详述

### 1. Draft 条件特征:单层 vs 多层(最核心)

**EAGLE2**(`eagle2/hidden.py`):

```text
forward_target_last_hidden(target, input_ids, hidden_layer_index=-1)
  → 取指定层(默认最后一层)末 token hidden [B, H]
  → draft: fc(concat(emb, h)) 其中 h 为 [B, H]
```

**EAGLE3**(`eagle3_sgl/hidden.py`):

```text
forward_target_eagle_acts(target, input_ids, eagle_layers=[2, L//2, L-3])
  → 取多层「进入该层之前的残差流」末 token 激活并拼接 [B, n*H]
  → draft: fc(n*H → H) 降维后再与 embedding 分别归一化、拼接进 midlayer
```

EAGLE3 用早/中/晚三层信息做 draft 条件,比 EAGLE2 单层末 hidden 信息更丰富,draft 猜得更准(接受率更高)。

### 2. Draft 网络结构

| 组件 | EAGLE2 `Eagle2DraftModel` | EAGLE3 `Eagle3SglDraftModel` |
|------|---------------------------|------------------------------|
| 特征融合 | `fc: Linear(2H, H)`,concat(emb, hidden) 后一次线性 | `fc: Linear(n*H, H)`,emb 与 hidden **分开**降维 |
| 首层 decoder | `_DraftDecoderLayer`,**跳过** input_layernorm | `_Eagle3MidLayer`,embed/hidden **分别** RMSNorm 后拼 2H |
| Attention 输入维 | H | **2H**(qkv 输入 = concat(norm(emb), norm(hidden))) |
| RoPE | 无(单 token S=1 时无影响) | **有**(prefill / step / extend 均带 RoPE) |
| 输出头 | 复用 target `lm_head` | 独立 `lm_head(draft_vocab)` + `hot_token_id = d2t + arange` 映射回 target id |
| AR 接口 | 无 | `prefill` / `step` / `extend` / `extend_tokens` / `topk_target_ids` |

### 3. 树展开策略

两包均支持 **累计 log 概率 beam**(`expand_draft_tree_topk`),逻辑与 EAGLE2 一致:

```text
cum_logp(child) = cum_logp(parent) + log p(child|parent)
每步 beam 保留 top-K 路径 → _compact_tree_by_score 全局剪枝
```

差异在**默认路径与额外模式**:

| 模式 | EAGLE2 | EAGLE3 |
|------|--------|------------|
| cumulative(默认) | `expand_draft_tree_topk` + 单 token topk_fn | 同左(非 AR 路径);AR 路径见下 |
| 静态 BFS | `tree_expand_mode="bfs"` → `expand_draft_tree_bfs` | `tree_expand_mode="static"` → `expand_draft_tree` |
| 自回归建树 | **无** | `autoregressive_draft=True` → `expand_draft_tree_ar`(draft KV + RoPE + feature 递归) |

EAGLE3 的 `expand_draft_tree_ar` 用 draft 的 `step()` 逐步展开静态 top-k 树,每步更新 draft hidden/KV,接受率更高。

### 4. Runner 调度差异

**EAGLE2 `generate()` 每轮**:

```text
target prefill -> acts_all + past_kv
for round in speculative rounds
  → draft fwd(..., root_acts, acts_all, draft_state)
    → expand_draft_tree_topk(draft_model, root_hidden, target_hidden)
  → verify 
  → append
  -> target replay 
```

**EAGLE3 `generate()` 每轮(默认 AR draft)**:

```text
target prefill -> acts_all + past_kv
draft prefill -> root token + root hidden + root logits -> DraftKvState
for round in speculative rounds
  → draft step(..., root_acts, acts_all, draft_state)
    → expand_draft_tree_ar(draft_model, root_logits, root_hidden, root_kv)
  → verify 
  → append
  -> target replay 
  → draft extend_tokens
```

EAGLE3 额外维护:

- **`acts_all`** `[1, P, n*H]`:全前缀各位置多层 act,append 后只 cat 新增段(避免每轮整段 target 重算)。
- **`_DraftKvState`**:draft KV + prefix_len + root_logits + root_hidden,避免每轮 draft prefill 整段 prefix。

### 5. 参数差异

**EAGLE2 `Eagle2TreeConfig`**:

- `hidden_layer_index: int = -1`
- `tree_expand_mode: cumulative | bfs`
- 默认 `topk=4, num_steps=3, max_tree_nodes=32`

**EAGLE3 `Eagle3SglConfig`**:

- `eagle_layers: Optional[List[int]]` + `resolve_eagle_layers(num_hidden_layers)`
- `tree_expand_mode: cumulative | static`
- 默认 `topk=8, num_steps=8, max_tree_nodes=64`

其余 `verify_mode` / `verify_attn_backend` 字段与白名单校验一致。


## 例子 / 测试 / 文档索引

### EAGLE2

| 类型 | 路径 |
|------|------|
| 端到端 | `example/run_eagle2_llama3.py`(Llama-3-8B + EAGLE draft) |
| 冒烟 | `example/run_smoke_eagle2_draft.py`、`run_smoke_tree.py` |
| 测试 | `tests/test_eagle2_*` |
| 对比文档 | `eagle2_vs_eagle1.md` |

### EAGLE3

| 类型 | 路径 |
|------|------|
| 端到端 | `example/run_eagle3_sgl_llama31.py`(Llama-3.1-8B + EAGLE3 draft) |
| 冒烟 | `example/run_smoke_eagle3_sgl.py` |
| 性能 | `example/profile_eagle3_sgl.py`、`time_stages_eagle3_sgl.py` |
| 基准 | `example/run_mtbench_eagle3_sgl.py` |
| 测试 | `tests/test_eagle3_sgl.py`、`test_eagle3_*` |
| 优化记录 | `tps_opti_path.md` |


---

## 关系图

```text
EAGLE1 (eagle)          EAGLE2 (eagle2)              EAGLE3 (eagle3_sgl)
─────────────────       ─────────────────            ───────────────────────
单层 hidden             单层 hidden                  多层 act(hidden) 拼接
fc(2H→H)                fc(2H→H)                     fc(n*H→H)
复用 lm_head            复用 lm_head                 可选的独立 draft_vocab + d2t
静态 BFS 树             + 累计 beam 树               + 静态AR 树(后续可改进为累积beam树)
                        (EAGLE2 核心增量)            + AR draft KV (EAGLE3 核心增量)
target verify           target verify                target verify
                        target replay kv             target replay kv and hidden states
                                                     draft extend update state(kv, hidden, logits) 
```
