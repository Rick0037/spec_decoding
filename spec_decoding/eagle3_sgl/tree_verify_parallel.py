# SPDX-License-Identifier: Apache-2.0
"""
树形 verify 的路径辅助函数。

被 tree_verify_full._select_append_root_child 复用:把叶子回溯成
根→叶的节点下标序列,用于逐位 greedy 比对。
"""

from __future__ import annotations

from typing import List


def path_node_indices_root_to_leaf(parent: List[int], leaf: int) -> List[int]:
    """从根到 leaf 的节点下标(含叶),根在 parent[i]==-1 的链上."""
    out: List[int] = []
    k = leaf
    while k >= 0:
        out.append(k)
        k = parent[k]
    out.reverse()  # 回溯是叶→根,反转成根→叶
    return out
