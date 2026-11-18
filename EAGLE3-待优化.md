# EAGLE3 根节点待优化方案

## 背景

当前 `generate_step_payload()` 的顺序是:

```text
draft 建完整 top-k 树
-> target verify
-> 用 recovery_logits 检查根节点
-> 根节点不匹配时丢弃整棵树
```

greedy 解码下, target 的下一个 token 在建树前已经确定:

```python
target_root_token = recovery_logits.argmax()
```

因此可以提前利用这个结果, 减少无效建树和 verify.

## 方案一: 根节点不在 draft 候选中时提前返回

### 逻辑

1. 从 `recovery_logits` 算出 `target_root_token`.
2. 根据 `draft_state.root_logits` 得到 draft 根层 top-k 候选.
3. 如果 `target_root_token` 不在候选中, 直接返回 `[target_root_token]`.
4. 如果存在, 继续当前的完整建树和 verify 流程.

### 为什么结果不变

target 根 token 不在任何根层候选中, 意味着所有路径都会在第 0 个位置被拒绝. 当前 verify 最终也只会返回 `target_root_token`, 所以提前返回与原来的结果相同.

### 收益和代价

- 收益: 避免必然失败情况下的整棵 draft 树构建和 target 树 verify.
- 代价: 每轮多一次 draft 根层 top-k 和成员判断, 成本很小.
- 改动范围: 只改 `generate_step_payload()`, 不需要改树结构和 verify.

## 方案二: 强制使用 target 根 token, 再展开子树

### 逻辑

1. 直接把 `target_root_token` 作为唯一根节点, 不再使用 draft 根层 top-k 选择根节点.
2. 将这个 token 输入 `draft_model.step()`.
3. draft 基于自己的 `root_hidden` 和 KV, 继续预测根节点的子候选.
4. 按现有规则扩展后续树节点并进行 target verify.

第一轮的树形变化示例:

```text
当前: 多个 draft 根候选, 每个根候选再继续展开
改后: target 根 token 作为唯一根, 只展开它的后续候选
```

### 正确性

- 根节点来自 target greedy, 所以根节点本身一定正确.
- 后续 draft token 仍然逐层接受 target verify, 不能直接默认接受.
- 因此最终输出仍可与 target 单独 greedy 保持一致.

### 注意点

- target 根 token 仍需参与 verify forward, 因为需要它之后的 hidden 和 logits 来检查子节点.
- 第一版可以保留现有根节点比较逻辑, 此时比较必然相等, 改动更小.
- target 根 token 可能是 draft 不太擅长的路径, 后续接受长度不一定提升.
- 单一根节点减少了根层宽度, 但可以把树节点预算用于更深的后续候选.

### 需要实验确认

- 平均接受 token 数 `accepted_tokens / rounds`.
- draft 建树耗时.
- target verify 耗时.
- 端到端 tokens/s 和相对 HF greedy 的 speedup.
- 与 target 单独 greedy 的 lossless 一致性.

## 建议顺序

先实现方案一, 因为它改动小且结果路径基本不变. 再单独实验方案二, 确认接受率和端到端速度是否真的提升.
