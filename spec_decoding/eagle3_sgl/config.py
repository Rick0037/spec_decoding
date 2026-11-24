# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 超参:多层 act 特征 + 自回归 draft 建树 + 树掩码 verify。

- draft 条件特征为 target 低/中/高多层 hidden 拼接(eagle_layers);
- topk/num_steps/max_tree_nodes 控制自回归 draft 树规模。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def default_eagle_layers(num_hidden_layers: int) -> List[int]:
    """默认 EAGLE3 特征层(早/中/晚三层)."""
    L = num_hidden_layers
    return [2, L // 2, L - 3]


@dataclass
class Eagle3SglConfig:
    """EAGLE3 树投机超参."""

    topk: int = 8
    """每层 draft 展开的分支数(>1 时形成树)。"""
    num_steps: int = 8
    """从根出发的 draft 深度。"""
    max_tree_nodes: int = 64
    """verify 节点预算. BFS 按创建顺序截断, EAGLE2 按累计分数剪枝。"""
    draft_tree_mode: str = "eagle2_batch"
    """draft 建树策略: eagle2_batch=按层 batch beam, bfs=原 BFS。"""
    preconsume_first_token: bool = True
    """draft prefill 预消费 target 已确定的首个 token + 末位 feature; 建树时它作为确定根节点."""
    skip_replay: bool = True
    """verify 顺带提交接受路径的 KV 和 feature, 省掉每轮 target replay."""
    decode_cuda_graph: bool = False
    """target 用 StaticCache, verify+hidden 前向走 CUDA graph(固定 L); 要求 skip_replay."""
    graph_capture: bool = True
    """decode_cuda_graph 开启时是否真正捕获 graph; False 则只用 StaticCache eager."""
    draft_cuda_graph: bool = False
    """把固定形状的 eagle2_batch draft tree 整体捕获为 CUDA Graph；依赖 decode_cuda_graph。"""
    graph_cache_margin: int = 16
    """StaticCache 容量余量: prompt + max_new_tokens + max_tree_nodes + 1 + margin."""
    eagle_layers: Optional[List[int]] = None
    """作为 draft 条件的 target 多层 hidden 下标;None 时用 default_eagle_layers。"""
    verify_attn_backend: str = "triton_tree"
    """
    树 verify 的 target attention 后端:

    - triton_tree(默认):Triton extend_attention_fwd + custom_mask。
    - eager:HF sdpa + 4D 树 cross mask。
    decode_cuda_graph 开启时强制 eager(graph 路径固定 SDPA)。
    """

    def __post_init__(self) -> None:
        """校验超参合法性(topk/num_steps >= 1,建树模式 / backend 取值在白名单内)."""
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.max_tree_nodes < 1:
            raise ValueError("max_tree_nodes must be >= 1")
        if self.draft_tree_mode not in ("eagle2_batch", "bfs"):
            raise ValueError("draft_tree_mode must be eagle2_batch or bfs")
        if self.decode_cuda_graph and not self.skip_replay:
            raise ValueError("decode_cuda_graph requires skip_replay")
        if self.draft_cuda_graph and not self.decode_cuda_graph:
            raise ValueError("draft_cuda_graph requires decode_cuda_graph")
        if self.draft_cuda_graph and not self.graph_capture:
            raise ValueError("draft_cuda_graph requires graph_capture")
        if self.draft_cuda_graph and self.draft_tree_mode != "eagle2_batch":
            raise ValueError("draft_cuda_graph currently requires eagle2_batch")
        if self.graph_cache_margin < 0:
            raise ValueError("graph_cache_margin must be >= 0")
        if self.verify_attn_backend not in ("eager", "triton_tree"):
            raise ValueError("verify_attn_backend must be eager or triton_tree")
        if self.decode_cuda_graph:
            self.verify_attn_backend = "eager"

    def resolve_eagle_layers(self, num_hidden_layers: int) -> List[int]:
        """返回配置的或默认的 eagle_layers(并校验下标在 [0, num_hidden_layers) 内)."""
        if self.eagle_layers is not None:
            for idx in self.eagle_layers:
                if idx < 0 or idx >= num_hidden_layers:
                    raise ValueError(
                        f"eagle_layers index {idx} out of range for num_hidden_layers={num_hidden_layers}"
                    )
            return list(self.eagle_layers)
        return default_eagle_layers(num_hidden_layers)
