# SPDX-License-Identifier: Apache-2.0
"""
spec_decoding.eagle3_sgl —— EAGLE3 多层特征 + 自回归 draft + 树掩码 verify。

draft 条件特征 = target 低/中/高多层 act 拼接(eagle_layers),经 fc 融合;
draft 自回归建树(携带自己的 KV,并沿路径递归更新 hidden)。

外层调度:runner.Eagle3SglGenerator。
draft:draft_model.build_eagle3_sgl_draft(复用 target embedding/lm_head)。
"""

from __future__ import annotations

from typing import Any

from .config import Eagle3SglConfig
from .tree_draft import (
    TreeDraftResult,
    expand_draft_tree_ar,
    tree_draft_to_verify_layout,
)
from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
)

__all__ = [
    "Eagle3SglConfig",
    "Eagle3SglGenerator",
    "Eagle3SglDraftModel",
    "build_eagle3_sgl_draft",
    "load_eagle3_draft_weights",
    "TreeDraftResult",
    "TreeVerifyLayout",
    "build_tree_cross_attn_bias_with_prefix",
    "expand_draft_tree_ar",
    "forward_target_eagle_acts",
    "tree_draft_to_verify_layout",
    "triton_extend_available",
]


def __getattr__(name: str) -> Any:
    if name == "forward_target_eagle_acts":
        from .hidden import forward_target_eagle_acts

        return forward_target_eagle_acts
    if name == "Eagle3SglGenerator":
        from .runner import Eagle3SglGenerator

        return Eagle3SglGenerator
    if name in (
        "Eagle3SglDraftModel",
        "build_eagle3_sgl_draft",
        "load_eagle3_draft_weights",
    ):
        from . import draft_model as dm

        return getattr(dm, name)
    if name == "triton_extend_available":
        from .tree_verify_attn import triton_extend_available

        return triton_extend_available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
