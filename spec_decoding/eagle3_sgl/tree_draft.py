"""EAGLE3 自回归 draft 建树：携带 KV cache，并沿树路径递归更新 hidden。"""

from __future__ import annotations

from collections import deque
from typing import List

import torch

from .config import Eagle3SglConfig
from .tree_verify_mask import TreeLayout


def expand_draft_tree_ar(
    cfg: Eagle3SglConfig,
    draft_model,
    root_logits: torch.Tensor,
    root_hidden: torch.Tensor,
    root_kv,
    draft_kv_length: int,
    *,
    device: torch.device,
    rope_delta: int = 0,
) -> TreeLayout:
    """按 BFS 顺序构造 top-k 树，每个分支独立延续自己的 hidden 和 KV。"""
    topk = cfg.topk
    if topk < 1 or cfg.max_tree_nodes < 1:
        raise ValueError("invalid topk/max_tree_nodes")

    token_ids: List[int] = []
    parent: List[int] = []
    root_topk_ids, _ = draft_model.convert_to_target_topk_token_ids(root_logits, topk)
    root_tokens = [int(token) for token in root_topk_ids[0, :topk].tolist()]

    queue = deque()
    for token_id in root_tokens:
        if len(token_ids) >= cfg.max_tree_nodes:
            break
        node_index = len(token_ids)
        token_ids.append(token_id)
        parent.append(-1)
        queue.append((node_index, 1, token_id, root_hidden, root_kv))

    while queue and len(token_ids) < cfg.max_tree_nodes:
        (
            parent_index,
            node_depth,
            parent_token_id,
            parent_hidden,
            parent_kv,
        ) = queue.popleft()
        if node_depth >= cfg.num_steps:
            continue

        draft_input_ids = torch.tensor(
            [[parent_token_id]], device=device, dtype=torch.long
        )
        child_logits, child_hidden, child_kv = draft_model.step(
            draft_input_ids,
            parent_hidden,
            parent_kv,
            # *`node_depth` 从 1 开始, 要转成从 0 开始的偏移.
            draft_kv_length + node_depth - 1 + rope_delta,
        )
        child_topk_ids, _ = draft_model.convert_to_target_topk_token_ids(
            child_logits, topk
        )
        for child_token_id in [
            int(token) for token in child_topk_ids[0, :topk].tolist()
        ]:
            if len(token_ids) >= cfg.max_tree_nodes:
                break
            child_index = len(token_ids)
            token_ids.append(child_token_id)
            parent.append(parent_index)
            queue.append(
                (
                    child_index,
                    node_depth + 1,
                    child_token_id,
                    child_hidden,
                    child_kv,
                )
            )

    return TreeLayout(token_ids=token_ids, parent=parent)
