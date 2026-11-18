# EAGLE3 draft prefill 待优化方案

## 目标

不改变 draft 建树和 target verify 的整体架构.

只利用 target prefill 已经产生的两个结果, 让 draft prefill 多向前推进一步:

- `recovery_logits`: target 对下一个 token 的预测.
- `acts_all` 的最后一个特征: target 最后一个位置的 hidden feature.

## 当前流程

假设 target 前缀为 `x0 ... x(P-1)`, 对应特征为 `f0 ... f(P-1)`.

target prefill 后已经得到:

```python
first_target_token = recovery_logits.argmax()  # 确定的 xP
last_target_feature = acts_all[:, -1:, :]       # f(P-1)
```

但当前 draft prefill 只消费到:

```text
token x(P-1) + feature f(P-2)
```

所以当前 draft 的 `root_logits` 仍然在预测 `xP`, 而这个 token 其实已经被 target 的 `recovery_logits` 确定了.

## 优化后的流程

### 1. 先正常执行 draft prefill

保持当前输入不变:

```text
(x1, f0), (x2, f1), ..., (x(P-1), f(P-2))
```

得到 draft 的初始 KV 和 hidden.

### 2. 直接消费 target 已经确定的 token

把下面这一对输入追加给 draft:

```text
first_target_token + last_target_feature
```

也就是:

```text
xP + f(P-1) -> draft -> 更新 draft KV 和 hidden
```

这里的 `first_target_token` 来自 `recovery_logits.argmax()`, 不是让 draft 再猜一次.

`last_target_feature` 是 target 的真实 hidden feature, 不是 draft 预测的 hidden.

### 3. 从下一个位置开始建树

追加后, draft 的 `root_logits` 预测的是 `x(P+1)`, 而不是 `xP`.

因此后续建树可以直接从 `x(P+1)` 的候选开始, 不再让 draft 为已经确定的 `xP` 重新提供根层候选.

## target verify 是否需要改变

target verify 仍然独立执行, 不需要和 draft prefill 混在一起.

但 target 必须先处理 `xP`, 才能得到用于检查 `x(P+1)` 的 logits. 可以让 `xP` 作为 verify 中的已确定根节点, 后续 draft 候选挂在它下面, 验证规则保持不变.

最终只有 `xP` 可以直接接受, 后面的 draft token 仍然必须逐个通过 target verify.

## 状态更新必须同步调整

这个方案会让 draft 比 `out_ids` 提前消费一个 token, 因此不能直接套用当前长度和更新逻辑:

- `draft_state.prefix_len` 需要反映 draft 实际消费到的位置.
- `draft_kv_length` 需要与已经追加的 draft KV 一致.
- verify 后更新 draft 时, 不能再次把 `xP` 写入 KV, 否则会产生重复 KV.
- `_update_draft_after_verify()` 需要跳过已经被 prefill 消费的 token, 只处理后续接受的 token.

首版可以只修改首次 prefill, 先验证正确性. 如果每轮更新都使用同样的提前消费方式, 也需要处理相同的状态对齐问题.

## 预期收益

- 避免 draft 为已经确定的 `xP` 再做无效根层猜测.
- draft 的下一步可以使用 target 真实特征, 减少对预测 hidden 的依赖.
- 可能减少根节点不匹配导致的整棵树浪费.

## 代价和风险

- 多一次 draft 单 token 计算.
- draft 状态与 `out_ids` 的对应关系更复杂, 必须重点检查 KV 是否重复或错位.
- target 仍需要计算 `xP` 之后的 logits, 所以不一定减少 target forward.
- 是否提升接受率和端到端速度需要实验确认.

## 验证指标

- 与 target 单独 greedy 的 lossless 一致性.
- draft KV 长度和 RoPE 位置是否连续且无重复.
- 平均接受 token 数 `accepted_tokens / rounds`.
- 端到端 tokens/s 和相对 HF greedy 的 speedup.
