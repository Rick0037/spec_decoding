# SPDX-License-Identifier: Apache-2.0
"""
专用 Verify Attention(树形掩码,不是普通 causal)

一次 target forward 里若要把 多个树节点 当作序列上的多个位置一起算
self-attention,不能用标准下三角 mask:那样会允许 兄弟子树 互相 attend,
未选中的 draft 分支会污染当前位置的表示,verify 结果 不再等价于只沿
选中路径逐步算大模型。

EAGLE 的做法:位置 i 的 Q 只能 attend 树意义上从根到 i 路径上的 key
(含 i 自身);KV cache 里的历史前缀照常全局可见。

L 个 draft query 对前缀 + L 整段 K 的并行 SDPA,用
build_tree_cross_attn_bias_with_prefix([L, past+L])。

这里输出 SDPA additive bias(禁止处为 -inf),与
torch.nn.functional.scaled_dot_product_attention(..., attn_mask=...) 对接。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


# -----------------------------------------------------------------------------
# 摊平后节点 i 的语义:draft 树里的第 i 个待 verify 位置,parent[i] 指向
# 父节点下标(-1 为根).token_ids[i] 为该位置上的 draft token.与一次 verify
# forward 里新 token 段的第 i 列对齐后即可套树掩码.
# -----------------------------------------------------------------------------
@dataclass
class TreeLayout:
    """
    draft 树的「摊平」表示:把树拍成一个长度为 L 的节点数组,供一次 verify 前向使用。

    摊平规则:节点 i = 建树顺序(BFS)下的第 i 个候选 token。
      - token_ids[i] = 该节点的候选 token
      - parent[i]    = 该节点父节点的下标;-1 表示直接挂在已确认前缀之后(根的孩子)

    有了 parent 指针就能沿链找回每个节点的全部祖先——这正是树掩码唯一需要的
    信息:verify 时节点 i 只允许 attend 自己的祖先链(+前缀),不允许看到兄弟
    分支。至于节点在数组里按什么顺序排(BFS/DFS),掩码逻辑并不关心。

    """

    token_ids: List[int]
    """摊平后第 i 个 verify 位置上的 draft token id(与 parent[i] 同属节点 i)。"""
    parent: List[int]
    """parent[i] 为节点 i 的父节点下标;-1 表示根(整段序列中第一个节点)。"""

    def __len__(self) -> int:
        return len(self.token_ids)


def node_depth_from_root(parent: List[int], node: int) -> int:
    """Root-to-node depth (edge count); used as tree position_ids offset."""
    d = 0
    k = node
    while parent[k] >= 0:
        d += 1
        k = parent[k]
    return d


def layout_tree_position_ids(
    layout: TreeLayout, past_len: int, rope_delta: int = 0
) -> torch.LongTensor:
    """Absolute position_ids for batched tree verify, shape [L]."""
    depths = [node_depth_from_root(layout.parent, i) for i in range(len(layout))]
    return torch.tensor(depths, dtype=torch.long) + int(past_len) + int(rope_delta)


# 用于tree attention verify,attention_bias即表示attention_mask
def build_tree_cross_attn_bias_with_prefix(
    layout: TreeLayout,
    past_seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    并行 tree verify(SDPA)用的 cross 掩码:Q 只有树上 L 个 draft 位置,
    K/V 对应 整段 past_seq_len + L(前缀 token + 各树节点 token 与 x 对齐)。

    第 i 个 draft query(0 <= i < L)允许 attend 的 key 下标 j:

    - j < past_seq_len:前缀任意位置(整段 prompt 可见);
    - j >= past_seq_len:令 k = j - past_seq_len,仅当 k 是 i 的祖先或 k==i 时允许。

    返回 [L, past_seq_len + L],可直接广播到
    scaled_dot_product_attention 的 attn_mask(additive,禁止处为 -inf)。
    """
    #!  bias[i, j] 允许节点`i` attend 位置`j`
    n = len(layout)
    if n == 0:
        return torch.zeros(0, past_seq_len, device=device, dtype=dtype)

    neg = torch.finfo(dtype).min  # * 默认数值
    # * q -> draft_len, kv -> past + draft_len
    # (Pdb) bias.shape
    # torch.Size([6, 12])
    # ! [n, past_seq_len+n] -> (q, k) 相乘
    # * 每行来看 看之前的 past_seq_len 列都是ok的
    # TODO by ruixuan 计算复杂度可以优化 -> 已优化: 前缀列整块置 0, 树部分用祖先矩阵一次写入.
    # 之前在 GPU 上逐元素赋值, 每轮要 L*(past_len+L) 次 CUDA 写; 带图 prompt 长 547 时
    # 单这一步就要 400ms. 现在先在 CPU 算好整块掩码, 只做一次 H2D 拷贝.
    bias = torch.full((n, past_seq_len + n), neg, dtype=dtype)
    bias[:, :past_seq_len] = 0.0

    # * ancestor[i, k] = True 表示节点 k 在根->i 的路径上(含 i 自己).
    # 沿 parent 指针向上走, 只有 O(L*树深) 次纯 Python 操作, 不碰 GPU.
    ancestor = torch.eye(n, dtype=torch.bool)
    for i in range(n):
        k = layout.parent[i]
        while k >= 0:
            ancestor[i, k] = True
            k = layout.parent[k]
    # * 如果i 看向 j . 再path上就不mask掉
    tree_part = bias[:, past_seq_len:]
    tree_part[ancestor] = 0.0
    return bias.to(device)
