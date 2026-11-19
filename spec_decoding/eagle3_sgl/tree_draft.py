"""EAGLE3 自回归 draft 建树：携带 KV cache，并沿树路径递归更新 hidden。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from .config import Eagle3SglConfig
from .tree_verify_mask import TreeLayout


def _expand_draft_tree_bfs_ar(
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


def _topk_target_log_probs(
    draft_model, logits: torch.Tensor, topk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 target token id 和完整 draft 词表归一化后的 topk log 概率."""
    topk_ids, topk_scores = draft_model.convert_to_target_topk_token_ids(logits, topk)
    log_normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    topk_log_probabilities = topk_scores.float() - log_normalizer
    return topk_ids, topk_log_probabilities


@dataclass
class _FrontierNode:
    """一条活跃 draft 路径的 AR 状态."""

    tree_index: int
    token_id: int
    node_depth: int
    cumulative_log_probability: float
    hidden: torch.Tensor
    kv: Any


def _compact_tree_by_score(
    token_ids: List[int],
    parent: List[int],
    cumulative_log_probabilities: List[float],
    max_nodes: int,
) -> TreeLayout:
    """按累计分保留节点和祖先, 并保证最终节点数不超过 max_nodes."""
    if len(token_ids) <= max_nodes:
        return TreeLayout(token_ids=token_ids, parent=parent)

    ranked_node_indices = sorted(
        range(len(token_ids)),
        key=lambda node_index: cumulative_log_probabilities[node_index],
        reverse=True,
    )
    kept_node_indices: set[int] = set()

    for node_index in ranked_node_indices:
        # 一个高分节点必须连同全部祖先一起保留, 否则 parent 链会断.
        missing_ancestors = []
        ancestor_index = node_index
        while ancestor_index >= 0 and ancestor_index not in kept_node_indices:
            missing_ancestors.append(ancestor_index)
            ancestor_index = parent[ancestor_index]

        # 预算放不下整条缺失祖先链时跳过该候选, 继续尝试其他高分节点.
        if len(kept_node_indices) + len(missing_ancestors) <= max_nodes:
            kept_node_indices.update(missing_ancestors)
        if len(kept_node_indices) >= max_nodes:
            break

    ordered_old_indices = sorted(kept_node_indices)
    old_to_new_index = {
        old_index: new_index for new_index, old_index in enumerate(ordered_old_indices)
    }
    compacted_token_ids = [token_ids[old_index] for old_index in ordered_old_indices]
    compacted_parent = [
        old_to_new_index[parent[old_index]] if parent[old_index] >= 0 else -1
        for old_index in ordered_old_indices
    ]
    return TreeLayout(token_ids=compacted_token_ids, parent=compacted_parent)


def _expand_draft_tree_eagle2_ar(
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
    """EAGLE2 累计分数 beam + EAGLE3 AR hidden/KV.

    每层只让累计分最高的 topk 条路径继续扩展. 展开后再按分数剪到
    max_tree_nodes. verify 只看到最终 TreeLayout, 因此接受逻辑不变.
    """
    topk = cfg.topk
    if topk < 1 or cfg.max_tree_nodes < 1:
        raise ValueError("invalid topk/max_tree_nodes")
    beam_width = min(topk, cfg.max_tree_nodes)

    token_ids: List[int] = []
    parent: List[int] = []
    cumulative_log_probabilities: List[float] = []

    root_topk_ids, root_topk_log_probabilities = _topk_target_log_probs(
        draft_model, root_logits, beam_width
    )
    root_token_ids = root_topk_ids[0].tolist()
    root_log_probabilities = root_topk_log_probabilities[0].tolist()

    frontier: List[_FrontierNode] = []
    for token_id, log_probability in zip(root_token_ids, root_log_probabilities):
        node_index = len(token_ids)
        token_ids.append(int(token_id))
        parent.append(-1)
        cumulative_log_probabilities.append(float(log_probability))
        # 根层所有节点共享同一个 target 条件 hidden 和 draft KV.
        frontier.append(
            _FrontierNode(
                tree_index=node_index,
                token_id=int(token_id),
                node_depth=1,
                cumulative_log_probability=float(log_probability),
                hidden=root_hidden,
                kv=root_kv,
            )
        )

    for _ in range(1, cfg.num_steps):
        if not frontier:
            break
        candidates: List[_FrontierNode] = []

        for frontier_node in frontier:
            draft_input_ids = torch.tensor(
                [[frontier_node.token_id]], device=device, dtype=torch.long
            )
            child_logits, child_hidden, child_kv = draft_model.step(
                draft_input_ids,
                frontier_node.hidden,
                frontier_node.kv,
                draft_kv_length + frontier_node.node_depth - 1 + rope_delta,
            )
            child_topk_ids, child_topk_log_probabilities = _topk_target_log_probs(
                draft_model, child_logits, beam_width
            )
            child_token_ids = child_topk_ids[0].tolist()
            child_log_probabilities = child_topk_log_probabilities[0].tolist()

            for child_token_id, child_log_probability in zip(
                child_token_ids, child_log_probabilities
            ):
                child_index = len(token_ids)
                child_depth = frontier_node.node_depth + 1
                child_cumulative_log_probability = (
                    frontier_node.cumulative_log_probability
                    + float(child_log_probability)
                )
                token_ids.append(int(child_token_id))
                parent.append(frontier_node.tree_index)
                cumulative_log_probabilities.append(child_cumulative_log_probability)
                # 同一父节点的 topk children 共享父步产生的 hidden 和 KV.
                # 下一步再用各自 token 分化.
                candidates.append(
                    _FrontierNode(
                        tree_index=child_index,
                        token_id=int(child_token_id),
                        node_depth=child_depth,
                        cumulative_log_probability=child_cumulative_log_probability,
                        hidden=child_hidden,
                        kv=child_kv,
                    )
                )

        # 稳定排序: 分数相同时保留先创建的候选, 保证结果可复现.
        candidates.sort(
            key=lambda candidate: candidate.cumulative_log_probability, reverse=True
        )
        frontier = candidates[:beam_width]

    return _compact_tree_by_score(
        token_ids,
        parent,
        cumulative_log_probabilities,
        cfg.max_tree_nodes,
    )


def _expand_draft_tree_eagle2_batch_ar(
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
    """按层 batch 展开 EAGLE2 frontier, 每层只做一次 draft forward.

    候选 token, parent 和累计分数在 GPU 上逐层保存, 最后统一取回做树剪枝.
    """
    topk = cfg.topk
    if topk < 1 or cfg.max_tree_nodes < 1:
        raise ValueError("invalid topk/max_tree_nodes")
    beam_width = min(topk, cfg.max_tree_nodes)

    root_topk_ids, root_topk_log_probabilities = _topk_target_log_probs(
        draft_model, root_logits, beam_width
    )
    root_token_ids = root_topk_ids[0]
    root_log_probabilities = root_topk_log_probabilities[0]
    root_count = int(root_token_ids.shape[0])

    token_id_chunks = [root_token_ids]
    parent_chunks = [
        torch.full((root_count,), -1, dtype=torch.long, device=device)
    ]
    cumulative_score_chunks = [root_log_probabilities]

    # frontier 的每一行是一条活跃路径, batch 内路径互不影响.
    frontier_tokens = root_token_ids.view(root_count, 1)
    frontier_hidden = root_hidden.expand(root_count, -1).contiguous()
    root_key, root_value = root_kv
    frontier_key = root_key.expand(root_count, *root_key.shape[1:]).contiguous()
    frontier_value = root_value.expand(root_count, *root_value.shape[1:]).contiguous()
    frontier_scores = root_log_probabilities
    frontier_global_indices = torch.arange(root_count, device=device)

    total_candidates = root_count
    node_depth = 1

    for _ in range(1, cfg.num_steps):
        frontier_size = int(frontier_tokens.shape[0])
        if frontier_size == 0:
            break

        # 当前层所有活跃路径只做一次 draft forward.
        child_logits, child_hidden, child_kv = draft_model.step(
            frontier_tokens,
            frontier_hidden,
            (frontier_key, frontier_value),
            draft_kv_length + node_depth - 1 + rope_delta,
        )
        child_topk_ids, child_topk_log_probabilities = _topk_target_log_probs(
            draft_model, child_logits, beam_width
        )
        candidate_width = int(child_topk_ids.shape[1])
        candidate_count = frontier_size * candidate_width

        # [N,K] 按父节点优先展平, 与串行版的候选创建顺序一致.
        flat_candidate_scores = (
            frontier_scores.unsqueeze(1) + child_topk_log_probabilities
        ).reshape(-1)
        flat_candidate_token_ids = child_topk_ids.reshape(-1)
        flat_parent_global_indices = frontier_global_indices.repeat_interleave(
            candidate_width
        )

        # 所有候选都留给最终按分数剪枝, 不只保留进入下一层的 beam.
        token_id_chunks.append(flat_candidate_token_ids)
        parent_chunks.append(flat_parent_global_indices)
        cumulative_score_chunks.append(flat_candidate_scores)

        # 在整个 frontier 的 N*K 个候选中选下一层 beam, 不再逐父节点保留.
        selected_count = min(beam_width, candidate_count)
        selected_scores, selected_flat_indices = torch.topk(
            flat_candidate_scores, k=selected_count, dim=-1
        )
        selected_parent_rows = selected_flat_indices // candidate_width

        frontier_tokens = flat_candidate_token_ids.index_select(
            0, selected_flat_indices
        ).view(selected_count, 1)
        frontier_hidden = child_hidden.index_select(0, selected_parent_rows)
        frontier_key = child_kv[0].index_select(0, selected_parent_rows)
        frontier_value = child_kv[1].index_select(0, selected_parent_rows)
        frontier_scores = selected_scores
        frontier_global_indices = total_candidates + selected_flat_indices

        total_candidates += candidate_count
        node_depth += 1

    # 只在整棵树展开结束后取回数据, 不做逐节点 D2H.
    token_ids = torch.cat(token_id_chunks).tolist()
    parent = torch.cat(parent_chunks).tolist()
    cumulative_log_probabilities = torch.cat(cumulative_score_chunks).tolist()
    return _compact_tree_by_score(
        token_ids,
        parent,
        cumulative_log_probabilities,
        cfg.max_tree_nodes,
    )


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
    """按配置选择 EAGLE2 累计分数树或原 BFS 树."""
    if cfg.draft_tree_mode == "eagle2_batch":
        return _expand_draft_tree_eagle2_batch_ar(
            cfg,
            draft_model,
            root_logits,
            root_hidden,
            root_kv,
            draft_kv_length,
            device=device,
            rope_delta=rope_delta,
        )
    if cfg.draft_tree_mode == "eagle2":
        return _expand_draft_tree_eagle2_ar(
            cfg,
            draft_model,
            root_logits,
            root_hidden,
            root_kv,
            draft_kv_length,
            device=device,
            rope_delta=rope_delta,
        )
    return _expand_draft_tree_bfs_ar(
        cfg,
        draft_model,
        root_logits,
        root_hidden,
        root_kv,
        draft_kv_length,
        device=device,
        rope_delta=rope_delta,
    )
