# Transformers DynamicCache vs StaticCache

本文解析 transformers/src/transformers/cache_utils.py 中 DynamicCache 与 StaticCache 在数据结构、更新方式以及 Dynamic / Static 各自体现在哪里。

---

## 一、共同背景

两者都继承自 Cache，用于在自回归生成时保存各层的 Key / Value 状态，避免每步重复计算历史 token 的 attention。

每层 K/V 的逻辑形状都是：

```
[batch_size, num_heads, seq_len, head_dim]
```

模型 attention 层通过统一接口调用（以 Llama 为例）：

```python
if past_key_value is not None:
    # sin and cos are specific to RoPE models; cache_position needed for the static cache
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

差异主要在：tensor 是否预分配、seq_len 是否可变、update 如何写入。

---

## 二、数据结构对比

| 维度 | DynamicCache | StaticCache |
|------|--------------|-------------|
| 初始化 | 空 list，按需增长 | 按 config 一次性预分配所有层 |
| 每层 tensor 形状 | [B, H, seq_len, D]，seq_len 随生成增长 | [max_batch, H, max_cache_len, D]，固定不变 |
| 最大长度 | 无上限（get_max_cache_shape() → None） | 构造时指定 max_cache_len |
| batch 大小 | 随输入变化 | 构造时指定 max_batch_size |
| 内存管理 | 每步 torch.cat 可能重新分配 | 初始化后地址不变，原地写入 |
| compile 友好 | 否（is_compileable = False） | 是（is_compileable = True） |

### DynamicCache：动态增长的 list

```python
def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
    super().__init__()
    self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
    self.key_cache: List[torch.Tensor] = []
    self.value_cache: List[torch.Tensor] = []
```

- 开始时 key_cache / value_cache 为空
- 第一次访问某层时 append 新 tensor
- 后续每步在 seq 维（dim=-2） 上拼接，seq_len 不断变大

### StaticCache：固定大小的预分配 buffer

```python
cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
for idx in range(config.num_hidden_layers):
    new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
    new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
    if not is_torchdynamo_compiling():
        self.register_buffer(f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
        self.register_buffer(f"value_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
        new_layer_key_cache = getattr(self, f"key_cache_{idx}")
        new_layer_value_cache = getattr(self, f"value_cache_{idx}")
        torch._dynamo.mark_static_address(new_layer_key_cache)
        torch._dynamo.mark_static_address(new_layer_value_cache)
    self.key_cache.append(new_layer_key_cache)
    self.value_cache.append(new_layer_value_cache)
```

要点：

1. 构造时根据 config.num_hidden_layers 预分配全部层
2. 形状在 (max_batch_size, num_kv_heads, max_cache_len, head_dim) 上固定
3. 用 register_buffer + mark_static_address 保证 tensor 地址不变，便于 torch.compile / CUDA Graph

---

## 三、update 行为：Dynamic vs Static 的核心体现

### DynamicCache —— 追加拼接

```python
if key_states is not None:
    if len(self.key_cache) <= layer_idx:
        # 首次访问该层
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    elif len(self.key_cache[layer_idx]) == 0:
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
    else:
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
```

Dynamic 体现在：

- 容量动态：没有 max_cache_len，能一直变长
- 形状动态：每步 torch.cat，tensor 的 seq 维变大，可能触发新分配
- 层数动态：层可以选择性地append（支持 skip layer）
- 无需 cache_position：新 token 总是 append 到末尾

返回的是 拼接后的完整 K/V（历史 + 当前step）。

### StaticCache —— 原地写入固定槽位

```python
cache_position = cache_kwargs.get("cache_position")
k_out = self.key_cache[layer_idx]
v_out = self.value_cache[layer_idx]
key_states = key_states.to(k_out.dtype)
value_states = value_states.to(v_out.dtype)

if cache_position is None:
    k_out.copy_(key_states)
    v_out.copy_(value_states)
else:
    k_out.index_copy_(2, cache_position, key_states)
    v_out.index_copy_(2, cache_position, value_states)

return k_out, v_out
```

Static 体现在：

- 容量静态：max_cache_len 在构造时定死
- 形状静态：tensor 大小不变，只改内容
- 地址静态：index_copy_ / copy_ 原地写，不 new tensor
- 必须知道写入位置：依赖 cache_position（例如 prefill 写 [0,1,...,n-1]，decode 写 [n]）

返回的是 同一块预分配 buffer 的引用（k_out, v_out），不是新 tensor。

---

## 四、功能与使用场景

### DynamicCache —— 默认、灵活

> A cache that grows dynamically as more tokens are generated. This is the default for generative models.

- generate() 默认创建 DynamicCache()
- 用法简单：past_key_values = DynamicCache()，不用事先知道最大长度
- 支持 crop()、batch_split()、batch_repeat_interleave() 等生成策略
- 代价：torch.cat 带来分配与拷贝，不适合 compile

### StaticCache —— 为 compile 设计

> Static Cache class to be used with torch.compile(model).

- 需要提前规划：

```python
max_generated_length = prompt_len + 10
past_key_values = StaticCache(
    config=model.config,
    max_batch_size=1,
    max_cache_len=max_generated_length,
    device=model.device,
    dtype=model.dtype,
)
```

- 固定 shape → 固定 attention mask → 减少 torch.compile 重编译
- 提供 reset()：原地 zero_() 清空，复用同一块内存
- 超出 max_cache_len 无法继续写（需重新建 cache 或换更大 buffer）

---

## 五、概念对照图

```
DynamicCache                          StaticCache
─────────────────                     ─────────────────
初始化: []                             初始化: zeros[B, H, L_max, D] × num_layers

Prefill (len=5):                       Prefill (len=5):
  append → [B,H,5,D]                     index_copy_ 写入 slot 0..4

Decode step 6:                         Decode step 6:
  cat → [B,H,6,D]  ← 新 tensor           index_copy_ 写入 slot 5
                                        buffer 仍是 [B,H,L_max,D]

seq_len: 5 → 6 → 7 → ... (增长)         seq_len 维大小: 始终 L_max
内存: 可能每步 realloc                  内存: 一次分配，地址不变
compile: ✗                              compile: ✓
```

---

## 六、如何理解Dynamic和Static

| 含义 | DynamicCache | StaticCache |
|------|--------------|-------------|
| 序列长度 | 动态增长 | 固定上限 |
| 内存布局 | 按需分配、可能 realloc | 预分配、地址固定 |
| 写入方式 | torch.cat 追加 | index_copy_ 按索引写 |
| API 复杂度 | 低（开箱即用） | 高（需 max_len、cache_position） |
| 典型场景 | 普通 generate()、研究原型 | torch.compile、低延迟 serving |
| 子类 | OffloadedCache、QuantizedCache 等 | SlidingWindowCache（固定窗口 + 循环覆盖） |

一句话：

- Dynamic = cache 跟着生成长度一起长，实现简单，但 shape 总在变。
- Static = cache 大小事先定好，只往固定 buffer 里写，为编译器和 GPU 图优化服务。

---

## 七、和 attention 计算的配合

对 DynamicCache，update() 返回的 K/V 已是历史 + 当前的完整序列，attention 直接算即可。

对 StaticCache，buffer 大小是 max_cache_len，有效 token 由 cache_position 和 causal mask 限定；generate() 里还会为 StaticCache 预建固定形状的 4D mask，避免 compile 时 mask shape 变化：

```python
# generation/utils.py
# 6. Create 4D attention mask if we are using a `StaticCache`
# (important for performant compiled forward pass)
if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
    ...
```

---

## 八、与 spec decoding / eagle 优化的关系

- 调试与原型：通常用 DynamicCache，API 简单，不用提前规划 max_cache_len。
- torch.compile / 固定 batch serving：需要切到 StaticCache，并正确处理 cache_position 与 max_cache_len。

---

## 参考源码

- transformers/src/transformers/cache_utils.py — DynamicCache（约 L341）、StaticCache（约 L1060）
- transformers/src/transformers/models/llama/modeling_llama.py — attention 层 past_key_value.update() 调用
- transformers/src/transformers/generation/utils.py — generate() 默认 DynamicCache、StaticCache 4D mask 处理
