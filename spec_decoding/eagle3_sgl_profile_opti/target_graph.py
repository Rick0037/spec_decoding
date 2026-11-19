# SPDX-License-Identifier: Apache-2.0
"""Step 5：target decode 的 CUDA graph（仅 verify 前向）。

设计：

- target KV 换成 连续 static 缓存（HF StaticCache），按 cache_position 用
  index_copy_ 原地写 → 没有 DynamicCache 的逐步 cat/realloc，shape 全程固定。
- verify 前向：对固定 L 个树节点跑一次 target，注意力用 固定 shape 的 4D 树 mask
  + 普通 SDPA（mask 宽度 padding 到 max_len），读整段 static KV。
- 因为输入/输出/KV 全是固定 shape 的持久 buffer，且写位置经由 cache_position
  驱动（index_copy_ 运行时读索引），整个 verify 前向可被 CUDA graph 捕获，
  每轮只 copy_ 新的 tokens/position/cache_position/mask 进静态 buffer 后 graph.replay()。

verify 写入的树 KV 落在 slots [cur_len, cur_len+L)；这些 slot 下一轮
被 replay/下一次 verify 覆盖，且 mask 保证永不被当作前缀读到。
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .tree_verify_full import _select_append_root_child
from .tree_verify_mask import (
    TreeVerifyLayout,
    _ancestor_self_matrix,
    layout_tree_position_ids,
)


def make_static_cache(config: Any, max_len: int, device: torch.device, dtype: torch.dtype):
    """构造 HF StaticCache"""
    from transformers import StaticCache

    for kwargs in (
        dict(config=config, max_batch_size=1, max_cache_len=max_len, device=device, dtype=dtype),
        dict(config=config, max_cache_len=max_len, device=device, dtype=dtype),
    ):
        try:
            return StaticCache(**kwargs)
        except TypeError:
            continue
    return StaticCache(config, 1, max_len, device, dtype)


def build_padded_tree_mask(
    layout: TreeVerifyLayout,
    cur_len: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[1,1,L,max_len] bias：前缀 [0:cur_len]=0，树段 [cur_len:cur_len+L] 按祖先矩阵，
    其余（未写/未来 slot）=-inf。可写进 out（graph 静态 buffer）以原地更新。"""
    L = len(layout)
    neg = torch.finfo(dtype).min
    A = _ancestor_self_matrix(layout.parent, L, device)  # [L,L] bool
    allow = torch.zeros((L, max_len), dtype=torch.bool, device=device)
    allow[:, :cur_len] = True
    allow[:, cur_len:cur_len + L] = A
    if out is None:
        out = torch.empty((1, 1, L, max_len), dtype=dtype, device=device)
    out.fill_(neg)
    out[0, 0][allow] = 0.0
    return out


def build_causal_replay_mask(
    cur_len: int, a_len: int, max_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """[1,1,A,max_len] bias：replay 第 j 个 token（位置 cur_len+j）causal attend
    [0:cur_len+j]，其余（未来/未写 slot）=-inf"""
    neg = torch.finfo(dtype).min
    rows = torch.arange(a_len, device=device).unsqueeze(1)
    cols = torch.arange(max_len, device=device).unsqueeze(0)
    allow = cols <= (cur_len + rows)  # [A, max_len] bool
    m = torch.full((1, 1, a_len, max_len), neg, dtype=dtype, device=device)
    m[0, 0][allow] = 0.0
    return m


class StaticGraphVerifier:
    """对 L=固定 的树 verify 前向做一次 CUDA graph 捕获，后续轮次仅 replay。

    L 与捕获时不同则回退回 eager 前向。
    """

    def __init__(
        self,
        target_m: torch.nn.Module,
        static_cache: Any,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        enable_graph: bool = True,
    ) -> None:
        self.target_m = target_m
        self.sc = static_cache
        self.max_len = int(max_len)
        self.device = device
        self.dtype = dtype
        self.enable_graph = enable_graph and torch.cuda.is_available()
        self._L: Optional[int] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        # 静态输入/输出 buffer（捕获后地址固定）
        self._tok_buf: Optional[torch.Tensor] = None
        self._pos_buf: Optional[torch.Tensor] = None
        self._cachepos_buf: Optional[torch.Tensor] = None
        self._mask_buf: Optional[torch.Tensor] = None
        self._logits_static: Optional[torch.Tensor] = None

    # --- 每轮把当轮的值写进静态输入 buffer ---
    def _fill_inputs(self, layout: TreeVerifyLayout, cur_len: int) -> None:
        L = len(layout)
        self._tok_buf[0].copy_(torch.tensor(layout.token_ids, device=self.device, dtype=torch.long))
        self._pos_buf[0].copy_(layout_tree_position_ids(layout, cur_len).to(self.device))
        self._cachepos_buf.copy_(torch.arange(cur_len, cur_len + L, device=self.device))
        build_padded_tree_mask(layout, cur_len, self.max_len, self.device, self.dtype, out=self._mask_buf)

    def _run_forward(self) -> torch.Tensor:
        out = self.target_m(
            self._tok_buf,
            past_key_values=self.sc,
            position_ids=self._pos_buf,
            cache_position=self._cachepos_buf,
            attention_mask=self._mask_buf,
            use_cache=True,
        )
        return out.logits

    def _capture(self, L: int, layout: TreeVerifyLayout, cur_len: int) -> None:
        self._L = L
        self._tok_buf = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._pos_buf = torch.zeros((1, L), dtype=torch.long, device=self.device)
        self._cachepos_buf = torch.zeros((L,), dtype=torch.long, device=self.device)
        self._mask_buf = torch.empty((1, 1, L, self.max_len), dtype=self.dtype, device=self.device)
        self._fill_inputs(layout, cur_len)
        if not self.enable_graph:
            return
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run_forward()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            logits = self._run_forward()
        self._logits_static = logits

    @torch.inference_mode()
    def verify(
        self,
        layout: TreeVerifyLayout,
        cur_len: int,
        recovery_logits: torch.Tensor,
        *,
        parent: List[int],
        token_ids: List[int],
        leaves: List[int],
    ) -> Tuple[List[int], int, int, List[int]]:
        L = len(layout)
        if self._L is None:
            self._capture(L, layout, cur_len)
            node_logits = (self._logits_static if self.enable_graph else self._run_forward())[0]
            return _select_append_root_child(node_logits, recovery_logits, parent, token_ids, leaves)
        if L != self._L or not self.enable_graph:
            # L 与捕获不同 → eager 前向
            return self._verify_eager(
                layout, cur_len, recovery_logits,
                parent=parent, token_ids=token_ids, leaves=leaves,
            )
        self._fill_inputs(layout, cur_len)
        self._graph.replay()
        node_logits = self._logits_static[0]
        return _select_append_root_child(node_logits, recovery_logits, parent, token_ids, leaves)

    @torch.inference_mode()
    def _verify_eager(self, layout, cur_len, recovery_logits, *, parent, token_ids, leaves):
        L = len(layout)
        tok = torch.tensor([layout.token_ids], device=self.device, dtype=torch.long)
        pos = layout_tree_position_ids(layout, cur_len).unsqueeze(0).to(self.device)
        cpos = torch.arange(cur_len, cur_len + L, device=self.device)
        mask = build_padded_tree_mask(layout, cur_len, self.max_len, self.device, self.dtype)
        out = self.target_m(tok, past_key_values=self.sc, position_ids=pos,
                            cache_position=cpos, attention_mask=mask, use_cache=True)
        node_logits = out.logits[0]
        return _select_append_root_child(node_logits, recovery_logits, parent, token_ids, leaves)
