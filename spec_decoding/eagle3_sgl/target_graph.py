# SPDX-License-Identifier: Apache-2.0
"""skip_replay 路径的 StaticCache + CUDA graph verify.

设计:
- target KV 换成 HF StaticCache, 预分配固定 shape, 原地写, 没有 DynamicCache 的逐层 cat.
- verify 前向: 树节点 padding 到固定 L + 树 mask padding 到 max_len + 普通 SDPA,
  带 output_hidden_states=True. 输入/输出/KV/hidden 全是持久 buffer,
  整个 verify 前向可被 CUDA graph 捕获, 每轮只 copy_ 新输入后 graph.replay().
- 提交(commit)在 graph 外 eager 执行: 把接受节点的 KV 行按路径顺序收进前缀槽位,
  再拨回写入指针. 接受长度 m 逐轮变化, 不适合进 graph; gather 是纯显存拷贝, 开销很小.

关键语义(transformers 5.20):
StaticLayer.update 忽略传入的 cache_position, 写入位置只由内部 cumulative_length
决定, 且 add_ 是原地操作(graph 安全). 所以每次 verify 前必须 rollback_static_cache
把写入指针拨回 cur_len, 否则上一轮的树 KV 会污染本轮前缀.

树 KV 落在 [cur_len, cur_len+L); commit 把接受的 m 行紧凑到 [cur_len, cur_len+m),
其余槽位被 mask 屏蔽且下一轮 verify 覆盖, 永远不会被当作前缀读到.
padding 节点只用于凑固定 L, 不进 leaves, 其 KV 槽位同样不会被提交.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .hidden import select_eagle_acts_from_hidden_states
from .tree_verify_full import _select_append_root_child
from .tree_verify_mask import TreeLayout, layout_tree_position_ids


def make_static_cache(config: Any, max_len: int, device: torch.device, dtype: torch.dtype):
    """构造 HF StaticCache, 兼容不同版本的签名."""
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


def rollback_static_cache(static_cache: Any, length: int) -> None:
    """把每层写入指针拨回 length. 必须在每次 verify 前调用."""
    for layer in static_cache.layers:
        layer.cumulative_length.fill_(int(length))


def _ancestor_self_matrix(parent: List[int], node_count: int) -> torch.Tensor:
    """[L,L] bool: ancestor[i,k]=True 表示 k 在根->i 路径上(含 i). CPU 上走 parent 链."""
    ancestor = torch.eye(node_count, dtype=torch.bool)
    for node_index in range(node_count):
        ancestor_index = parent[node_index]
        while ancestor_index >= 0:
            ancestor[node_index, ancestor_index] = True
            ancestor_index = parent[ancestor_index]
    return ancestor


def pad_tree_layout(layout: TreeLayout, target_node_count: int) -> TreeLayout:
    """把树 padding 到固定节点数. padding 节点 parent=-1, 不进 leaves, 不参与接受."""
    node_count = len(layout)
    if node_count >= target_node_count:
        return layout
    padding_count = target_node_count - node_count
    return TreeLayout(
        token_ids=layout.token_ids + [0] * padding_count,
        parent=layout.parent + [-1] * padding_count,
    )


def build_padded_tree_mask(
    layout: TreeLayout,
    cur_len: int,
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[1,1,L,max_len] bias: 前缀 [0:cur_len]=0, 树段按祖先矩阵, 其余 slot=-inf.

    可写进 out(graph 静态 buffer)以原地更新. padding 行只允许看前缀,
    避免整行 -inf 导致 SDPA 出 NaN; 它们的输出不参与接受.
    """
    node_count = len(layout)
    neg = torch.finfo(dtype).min
    ancestor = _ancestor_self_matrix(layout.parent, node_count).to(device)
    allow = torch.zeros((node_count, max_len), dtype=torch.bool, device=device)
    allow[:, :cur_len] = True
    allow[:, cur_len : cur_len + node_count] = ancestor
    if out is None:
        out = torch.empty((1, 1, node_count, max_len), dtype=dtype, device=device)
    out.fill_(neg)
    out[0, 0][allow] = 0.0
    return out


def commit_accepted_static_kv(
    static_cache: Any,
    cur_len: int,
    accepted_nodes: List[int],
    device: torch.device,
) -> None:
    """把接受路径的 KV 行按根→叶顺序紧凑到 [cur_len, cur_len+m), 并拨回写入指针.

    StaticCache 的 keys/values 是固定 buffer, 用 index_select 取出接受行(自动处理
    源/目标区间重叠), 再原地 copy_ 回前缀尾部. 接受节点下标沿路径递增(BFS 序父先于子),
    目标槽位不高于源槽位, 紧凑后 [cur_len+m, cur_len+L) 的残留槽位下一轮会被覆盖.
    """
    accepted_count = len(accepted_nodes)
    if accepted_count > 0:
        rows = torch.tensor(accepted_nodes, device=device, dtype=torch.long) + cur_len
        for layer in static_cache.layers:
            layer.keys[:, :, cur_len : cur_len + accepted_count].copy_(
                layer.keys.index_select(2, rows)
            )
            layer.values[:, :, cur_len : cur_len + accepted_count].copy_(
                layer.values.index_select(2, rows)
            )
    rollback_static_cache(static_cache, cur_len + accepted_count)


class StaticGraphCommitVerifier:
    """对固定 L 的树 verify+hidden 前向做一次 CUDA graph 捕获, 后续轮次仅 replay.

    树节点不足 L 时 padding 到 L; commit 在 graph 外 eager 执行(接受长度逐轮变化).
    position_ids 用 2D [1,L]: Llama 直接可用; Qwen3-VL 会自动 expand 成 4 轴
    (text + 3 轴 MRoPE), 文本 token 三轴相同, 值 = 树深度 + cur_len + rope_delta.
    """

    def __init__(
        self,
        target_m: torch.nn.Module,
        static_cache: Any,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        capture_tree_nodes: int,
        enable_graph: bool = True,
    ) -> None:
        self.target_m = target_m
        self.static_cache = static_cache
        self.max_len = int(max_len)
        self.device = device
        self.dtype = dtype
        self.capture_tree_nodes = int(capture_tree_nodes)
        self.enable_graph = enable_graph and torch.cuda.is_available()
        self._captured = False
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._token_buffer: Optional[torch.Tensor] = None
        self._position_buffer: Optional[torch.Tensor] = None
        self._cache_position_buffer: Optional[torch.Tensor] = None
        self._mask_buffer: Optional[torch.Tensor] = None
        self._static_logits: Optional[torch.Tensor] = None
        self._static_hidden: Optional[Tuple[torch.Tensor, ...]] = None

    def _fill_inputs(self, layout: TreeLayout, cur_len: int, rope_delta: int) -> None:
        node_count = len(layout)
        self._token_buffer[0].copy_(
            torch.tensor(layout.token_ids, device=self.device, dtype=torch.long)
        )
        self._position_buffer[0].copy_(
            layout_tree_position_ids(layout, cur_len, rope_delta).to(self.device)
        )
        self._cache_position_buffer.copy_(
            torch.arange(cur_len, cur_len + node_count, device=self.device)
        )
        build_padded_tree_mask(
            layout, cur_len, self.max_len, self.device, self.dtype, out=self._mask_buffer
        )
        rollback_static_cache(self.static_cache, cur_len)

    def _run_forward(self):
        return self.target_m(
            self._token_buffer,
            past_key_values=self.static_cache,
            position_ids=self._position_buffer,
            cache_position=self._cache_position_buffer,
            attention_mask=self._mask_buffer,
            use_cache=True,
            output_hidden_states=True,
        )

    def _capture(self, layout: TreeLayout, cur_len: int, rope_delta: int) -> None:
        node_count = len(layout)
        self._token_buffer = torch.zeros((1, node_count), dtype=torch.long, device=self.device)
        self._position_buffer = torch.zeros((1, node_count), dtype=torch.long, device=self.device)
        self._cache_position_buffer = torch.zeros(
            (node_count,), dtype=torch.long, device=self.device
        )
        self._mask_buffer = torch.empty(
            (1, 1, node_count, self.max_len), dtype=self.dtype, device=self.device
        )
        self._fill_inputs(layout, cur_len, rope_delta)
        # warmup: 触发 StaticCache 懒初始化与 kernel autotune;
        # 每次前向前都拨回写入指针, 保证 KV 落在正确槽位.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                rollback_static_cache(self.static_cache, cur_len)
                self._run_forward()
        torch.cuda.current_stream().wait_stream(side_stream)
        rollback_static_cache(self.static_cache, cur_len)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            output = self._run_forward()
        self._static_logits = output.logits
        self._static_hidden = output.hidden_states
        self._captured = True

    def _forward_eager(self, layout: TreeLayout, cur_len: int, rope_delta: int):
        node_count = len(layout)
        rollback_static_cache(self.static_cache, cur_len)
        token_ids = torch.tensor([layout.token_ids], device=self.device, dtype=torch.long)
        position_ids = (
            layout_tree_position_ids(layout, cur_len, rope_delta).unsqueeze(0).to(self.device)
        )
        cache_position = torch.arange(cur_len, cur_len + node_count, device=self.device)
        mask = build_padded_tree_mask(layout, cur_len, self.max_len, self.device, self.dtype)
        return self.target_m(
            token_ids,
            past_key_values=self.static_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )

    @torch.inference_mode()
    def verify_commit(
        self,
        layout: TreeLayout,
        cur_len: int,
        recovery_logits: Optional[torch.Tensor],
        *,
        leaves: List[int],
        rope_delta: int = 0,
        confirmed_root: bool = False,
        eagle_layers: List[int],
    ) -> Tuple[List[int], int, torch.Tensor]:
        """graph replay(或 eager 静态前向) + 接受路径 KV/feature 提交.

        返回 (append_list, proposed_draft_tokens, accepted_features [1, m, n*H]),
        与 full_tree_verify_commit 契约一致.
        """
        real_node_count = len(layout)
        padded_layout = pad_tree_layout(layout, self.capture_tree_nodes)
        if self.enable_graph:
            if not self._captured:
                self._capture(padded_layout, cur_len, rope_delta)
                # 捕获期间 kernel 只录制不执行, 输出 buffer 内容未定义,
                # 必须 replay 一次才拿到本轮真实 logits.
                rollback_static_cache(self.static_cache, cur_len)
                self._graph.replay()
            else:
                self._fill_inputs(padded_layout, cur_len, rope_delta)
                self._graph.replay()
            node_logits = self._static_logits[0, :real_node_count]
            hidden_states = self._static_hidden
        else:
            output = self._forward_eager(padded_layout, cur_len, rope_delta)
            node_logits = output.logits[0, :real_node_count]
            hidden_states = output.hidden_states
        append_list, proposed_draft_tokens, accepted_nodes = _select_append_root_child(
            node_logits,
            recovery_logits,
            layout.parent,
            layout.token_ids,
            leaves,
            confirmed_root=confirmed_root,
        )
        commit_accepted_static_kv(self.static_cache, cur_len, accepted_nodes, self.device)
        # hidden 只含本次 forward 的 L 个树节点(含 padding), 按接受节点下标取 feature.
        node_features = select_eagle_acts_from_hidden_states(
            hidden_states, eagle_layers, last_token_only=False
        )
        accepted_rows = torch.tensor(accepted_nodes, device=self.device, dtype=torch.long)
        accepted_features = node_features.index_select(1, accepted_rows)
        return append_list, proposed_draft_tokens, accepted_features