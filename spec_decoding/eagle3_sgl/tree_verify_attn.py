# SPDX-License-Identifier: Apache-2.0
"""
Verify attention backends(extend 路径):

- triton_tree (topk>1): SGLang extend_attention_fwd + custom_mask。
- eager/sdpa: HF forward + 4D 树 cross mask(full_tree_verify_extend)。
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from .tree_verify_mask import (
    TreeLayout,
    build_tree_cross_attn_bias_with_prefix,
    build_tree_cross_bool_mask_with_prefix,
)

_TRITON_SUPPORTED_ATTN = frozenset(
    {
        "LlamaAttention",
        "Qwen2Attention",
        "Qwen3Attention",
        "MistralAttention",
        "Gemma2Attention",
        "Qwen3VLTextAttention",
    }
)


def triton_extend_available() -> bool:
    """是否可用 Triton extend attention kernel(需 CUDA + triton + 本包 kernel)."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        from .kernels.extend_attention import extend_attention_fwd  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class TritonTreeVerifyMetadata:
    """Indptr / indices / bool mask for extend_attention_fwd (batch=1)."""

    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    max_len_extend: int
    extend_len: int


def build_triton_tree_verify_metadata(
    layout: TreeLayout,
    past_len: int,
    *,
    device: torch.device,
) -> TritonTreeVerifyMetadata:
    """为一次树 verify 构造 triton extend attention 所需的索引/掩码元数据(batch=1).

    记 L = 树节点数 len(layout), P = past_len 前缀长度, total_kv = P + L.
    kernel 把 K/V 分成两段处理:
      - stage-1: 前缀 KV(长 P), 所有树节点都可见, 不查掩码;
      - stage-2: 本次 extend 的 L 个树节点(在 K_Extend/V_Extend 里), 套树掩码(兄弟分支互不可见).

    返回各字段(shape / 含义):
      - qo_indptr   [0, L]: query 段偏移, 共 L 个树节点 query;
      - kv_indptr   [0, P]: 前缀 KV 段偏移, 只含 P 个前缀 token, 不含树节点;
      - kv_indices  arange(P): 前缀 KV 在 buffer 中的行号(连续 0..P-1);
      - custom_mask bool 二维 [L, total_kv], True=允许 attend;
                    前缀列全 True, 树段按祖先链; kernel 用 stride(0) 作行宽;
      - mask_indptr [0, L*total_kv]: custom_mask 的段偏移;
      - max_len_extend / extend_len = L.

    关键坑: kv_indptr 必须是 [0, P] 而非 [0, P+L]. 若误把树节点也算进前缀,
    stage-1 会在 skip_prefix 下无掩码地 attend 所有树节点 → 兄弟分支泄漏, verify 结果错乱.
    """
    l_n = len(layout)
    total_kv = int(past_len) + l_n
    # * 直接建 bool cross 掩码(True=允许), 保持二维 [tree_len, cache_len+tree_len]
    custom_mask = build_tree_cross_bool_mask_with_prefix(
        layout, past_len, device=device
    )
    # import pdb;pdb.set_trace()
    #* custom_mask -> [32, 44]
    qo_indptr = torch.tensor([0, l_n], dtype=torch.int32, device=device)
    # 前缀 KV 长度 = past_len(不含 extend)
    kv_indptr = torch.tensor([0, int(past_len)], dtype=torch.int32, device=device)
    kv_indices = torch.arange(int(past_len), dtype=torch.int32, device=device)
    # mask 行 stride = total_kv(前缀 + extend),与 kernel 的 cur_seq_len 一致
    mask_indptr = torch.tensor([0, l_n * total_kv], dtype=torch.int32, device=device)
    return TritonTreeVerifyMetadata(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        max_len_extend=l_n,
        extend_len=l_n,
    )


def _apply_rotary(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """给 q/k 施加 RoPE(用 decoder 传入的 (cos, sin));缺失则报错或原样返回."""
    if position_embeddings is not None:
        cos, sin = position_embeddings
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            return apply_rotary_pos_emb(query_states, key_states, cos, sin)
        except ImportError:
            pass
    rotary = getattr(module, "rotary_emb", None)
    if rotary is None:
        return query_states, key_states
    raise RuntimeError(
        "triton_tree verify needs position_embeddings from the decoder layer"
    )


def _layer_idx(module: nn.Module) -> int:
    """取该 attention 模块的层号(写 KV cache 时定位层);缺失则报错."""
    idx = getattr(module, "layer_idx", None)
    if idx is not None:
        return int(idx)
    raise AttributeError("self_attn.layer_idx is required for triton tree verify")


# 植入了triton custom mask attn的attn layer
def _triton_tree_attn_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Any = None,
    attention_mask: Any = None,
    past_key_values: Any = None,
    cache_position: Any = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from .kernels.extend_attention import extend_attention_fwd

    # transformers 5.x 把 attention 的 cache kwarg 从 past_key_value 改名为 past_key_values(复数);
    # 两种命名都兜底,否则前缀 KV 不会传进来 → 树节点只能 attend 自己 → verify 全错.
    past_key_value = past_key_values
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_value")

    ctx: TritonTreeVerifyMetadata = module._llmqrt_triton_meta
    bsz, q_len, _ = hidden_states.shape
    if bsz != 1:
        raise NotImplementedError("triton_tree verify supports batch_size=1 only")

    # transformers 5.x 把头数从 attention 模块挪到了 config,这里两种命名都兼容.
    # *---------parameter-----
    mcfg = getattr(module, "config", None)
    num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        num_heads = int(getattr(mcfg, "num_attention_heads"))
    num_kv = getattr(module, "num_key_value_heads", None)
    if num_kv is None:
        num_kv = int(getattr(mcfg, "num_key_value_heads", num_heads))
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        head_dim = module.q_proj.out_features // num_heads
    # *-----qkv linear
    query_states = (
        module.q_proj(hidden_states)
        .view(bsz, q_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    key_states = (
        module.k_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
    )
    value_states = (
        module.v_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
    )
    # Qwen3/Qwen3-VL 在 RoPE 之前有 per-head q_norm/k_norm, Llama 没有; 按存在性施加.
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    k_norm = getattr(module, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)

    # * [batch, seq_len, num_heads,  head_dim] -> [batch, num_heads, seq_len, head_dim]

    # *-----rope
    query_states, key_states = _apply_rotary(
        module, query_states, key_states, position_embeddings
    )

    # *-----cache---
    #! 实际上 past_key_value 是一个dynamic cache, 更新指定index layer的 cache
    if past_key_value is not None:
        cache_kwargs = {}
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        key_states, value_states = past_key_value.update(
            key_states, value_states, _layer_idx(module), cache_kwargs
        )

    # (Pdb) query_states.shape [batch, num_heads, seq_len, head_dim]
    # torch.Size([1, 12, 6, 64])
    # (Pdb) key_states.shape [batch, num_heads, seq_len, head_dim]
    # torch.Size([1, 12, 12, 64])

    # * 拍平成 kernel 布局: Q 全头, K/V kv 头(GQA 在 kernel 内); 末 L 行是本次 extend
    q = query_states.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()  # * [L, num_heads, head_dim]
    k_all = key_states.transpose(1, 2).reshape(-1, num_kv, head_dim).contiguous()  # * [P+L, num_kv_heads, head_dim]
    v_all = value_states.transpose(1, 2).reshape(-1, num_kv, head_dim).contiguous()  # * [P+L, num_kv_heads, head_dim]
    l_ext = ctx.extend_len  # * L
    k_extend = k_all[-l_ext:]  # * 取后面 L 行(本轮), [L, num_kv_heads, head_dim]
    v_extend = v_all[-l_ext:]  # * [L, num_kv_heads, head_dim]
    o = torch.empty_like(q)

    extend_attention_fwd(
        q_extend=q,  # * [L, num_heads, head_dim], 本次 L 个树节点的 Q
        k_extend=k_extend,  # * [L, num_kv_heads, head_dim], 本次 extend 的 K(k_all 末 L 行)
        v_extend=v_extend,  # * [L, num_kv_heads, head_dim], 本次 extend 的 V
        o_extend=o,  # * [L, num_heads, head_dim], 输出 buffer
        k_buffer=k_all,  # * [P+L, num_kv_heads, head_dim], 前缀+extend 整段 K
        v_buffer=v_all,  # * [P+L, num_kv_heads, head_dim], 前缀+extend 整段 V
        qo_indptr=ctx.qo_indptr,  # * [0, L], query 段偏移
        kv_indptr=ctx.kv_indptr,  # * [0, P], 前缀 KV 段偏移(不含树节点)
        kv_indices=ctx.kv_indices,  # * arange(P), 前缀 KV 行号
        custom_mask=ctx.custom_mask,  # * bool [L, P+L], 树掩码(True=允许)
        mask_indptr=ctx.mask_indptr,  # * [0, L*(P+L)], custom_mask 段偏移
        max_len_extend=ctx.max_len_extend,  # * L, grid 第三维 ceil(L/BLOCK_M)
    )

    # * o_proj 回 hidden_size; transformers 5.x 的 attention 返回 (attn_output, attn_weights) 两元组(KV cache 已就地更新).
    attn_output = o.reshape(bsz, q_len, num_heads * head_dim)
    attn_output = module.o_proj(attn_output)
    return attn_output, None


@contextmanager
def tree_verify_triton_attention_context(
    decoder: nn.Module,
    metadata: TritonTreeVerifyMetadata,
):
    """Patch each decoder self_attn.forward to call extend_attention_fwd."""
    originals: List[Tuple[nn.Module, Callable[..., Any]]] = []
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("decoder.layers not found for triton_tree verify")

    for layer in layers:
        attn = layer.self_attn
        if attn.__class__.__name__ not in _TRITON_SUPPORTED_ATTN:
            raise NotImplementedError(
                f"triton_tree verify unsupported attention: {attn.__class__.__name__}"
            )
        attn._llmqrt_triton_meta = metadata
        if not hasattr(attn, "_llmqrt_orig_forward"):
            attn._llmqrt_orig_forward = attn.forward
        # * 原来的forward都放在这里了
        originals.append((attn, attn._llmqrt_orig_forward))

        def _wrapped(
            self: nn.Module,
            hidden_states: torch.Tensor,
            *args: Any,
            **kw: Any,
        ) -> Any:
            return _triton_tree_attn_forward(self, hidden_states, *args, **kw)

        attn.forward = types.MethodType(_wrapped, attn)

    try:
        yield
    finally:
        # * 当程序 离开对应的`with tree_verify_triton_attention_context(...)` 代码块 时触发
        for attn, orig in originals:
            attn.forward = orig
            if hasattr(attn, "_llmqrt_triton_meta"):
                del attn._llmqrt_triton_meta
