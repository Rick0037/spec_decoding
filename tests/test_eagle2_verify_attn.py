# SPDX-License-Identifier: Apache-2.0
"""Metadata / backend resolution for tree verify attention kernels."""

import torch

from spec_decoding.eagle2.tree_verify_attn import (
    build_triton_tree_verify_metadata,
    resolve_verify_attn_backend,
    tree_custom_mask_bool,
)
from spec_decoding.eagle2.tree_verify_mask import TreeVerifyLayout


def test_triton_custom_mask_shape_batch1():
    layout = TreeVerifyLayout(
        token_ids=[10, 20, 30],
        parent=[-1, 0, 0],
        dfs_order=[0, 1, 2],
    )
    past_len = 5
    l_n = 3
    mask = tree_custom_mask_bool(layout, past_len, device=torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (l_n, past_len + l_n)
    meta = build_triton_tree_verify_metadata(
        layout, past_len, device=torch.device("cpu")
    )
    assert meta.max_len_extend == l_n
    # kv_indptr 只描述前缀 KV(长 past_len, 不含 extend 的 L 个树节点, 树节点在 K_Extend 里)
    assert meta.kv_indptr.tolist() == [0, past_len]
    assert meta.mask_indptr.tolist() == [0, l_n * (past_len + l_n)]


def test_resolve_auto_topk1_without_flash_falls_back_eager():
    b = resolve_verify_attn_backend("auto", topk=1, layout_len=3)
    assert b in ("flash_attn", "eager")


def test_resolve_auto_topk_gt1_without_triton_falls_back_eager():
    b = resolve_verify_attn_backend("auto", topk=4, layout_len=3)
    assert b in ("triton_tree", "eager")
