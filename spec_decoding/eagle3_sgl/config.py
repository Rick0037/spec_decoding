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
    """draft 建树策略: eagle2=逐节点 beam, eagle2_batch=按层 batch beam, bfs=原 BFS。"""
    preconsume_first_token: bool = True
    """draft prefill 预消费 target 已确定的首个 token + 末位 feature; 建树时它作为确定根节点."""
    eagle_layers: Optional[List[int]] = None
    """作为 draft 条件的 target 多层 hidden 下标;None 时用 default_eagle_layers。"""
    verify_mode: str = "full_model_tree"
    """
    full_model_tree(推荐):一次 全层 target forward + 树掩码(tree_verify_full)。
    reference_paths:对每个叶子路径用标准 causal forward 做 greedy 校验(慢、作对照)。
    """
    verify_attn_backend: str = "auto"
    """
    full_model_tree 时 target attention 后端:

    - auto:topk==1 且可用 → flash_attn;topk>1 且 Triton 可用 → triton_tree;否则 eager。
    - flash_attn:HF flash_attention_2(链式因果,不叠树 mask)。
    - triton_tree:Triton extend_attention_fwd + custom_mask。
    - eager:target_m.forward + 树 mask patch + 临时 eager。
    """

    def __post_init__(self) -> None:
        """校验超参合法性(topk/num_steps >= 1,verify_mode / backend 取值在白名单内)."""
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.max_tree_nodes < 1:
            raise ValueError("max_tree_nodes must be >= 1")
        if self.draft_tree_mode not in ("eagle2", "eagle2_batch", "bfs"):
            raise ValueError("draft_tree_mode must be eagle2, eagle2_batch, or bfs")
        if self.verify_mode not in (
            "full_model_tree",
            "reference_paths",
        ):
            raise ValueError("verify_mode must be full_model_tree or reference_paths")
        if self.verify_attn_backend not in (
            "auto",
            "eager",
            "flash_attn",
            "triton_tree",
        ):
            raise ValueError(
                "verify_attn_backend must be auto, eager, flash_attn, or triton_tree"
            )

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
