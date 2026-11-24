"""固定形状 EAGLE2 draft 树的 CUDA Graph 实现。"""

from __future__ import annotations

from typing import Optional

import torch

from .tree_draft import _compact_tree_by_score, _topk_target_log_probs
from .tree_verify_mask import TreeLayout


class StaticGraphDraftTree:
    """把完整的 eagle2_batch 建树展开捕获为一张 CUDA Graph。"""

    def __init__(self, draft_model, cfg, max_len, device):
        self.draft_model = draft_model
        self.cfg = cfg
        self.max_len = int(max_len)
        self.device = torch.device(device)
        self.beam = min(int(cfg.topk), int(cfg.max_tree_nodes))
        self.num_steps = int(cfg.num_steps)
        self.total_nodes = self.beam + (self.num_steps - 1) * self.beam * self.beam

        if cfg.draft_tree_mode != "eagle2_batch":
            raise ValueError("StaticGraphDraftTree only supports eagle2_batch")
        if self.device.type != "cuda":
            raise ValueError("StaticGraphDraftTree only supports CUDA")
        if self.max_len < 1 or self.beam < 1 or self.num_steps < 1:
            raise ValueError("invalid max_len/topk/max_tree_nodes/num_steps")

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False
        self._initialized = False

    def _check_inputs(
        self,
        root_logits: torch.Tensor,
        root_hidden: torch.Tensor,
        root_key: torch.Tensor,
        root_value: torch.Tensor,
        root_len: int,
    ) -> None:
        if root_logits.dim() != 2 or root_logits.shape[0] != 1:
            raise ValueError("root_logits must have shape [1, vocab]")
        if root_hidden.dim() != 2 or root_hidden.shape[0] != 1:
            raise ValueError("root_hidden must have shape [1, hidden]")
        if root_key.dim() != 4 or root_key.shape[0] != 1:
            raise ValueError("root key cache must have shape [1, nkv, len, head_dim]")
        if root_value.shape != root_key.shape:
            raise ValueError("root key/value cache shapes must match")
        if root_logits.shape[-1] < self.beam:
            raise ValueError("root vocabulary is smaller than the fixed beam")
        if root_len < 0 or root_len > root_key.shape[2]:
            raise ValueError("invalid draft_kv_length")
        if root_len + self.num_steps - 1 > self.max_len:
            raise ValueError("draft tree exceeds the static KV capacity")

        tensors = (root_logits, root_hidden, root_key, root_value)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("all draft tree inputs must be CUDA tensors")
        if any(tensor.device != root_logits.device for tensor in tensors[1:]):
            raise ValueError("all draft tree inputs must be on the same device")
        if self.device.index is not None and root_logits.device != self.device:
            raise ValueError("draft tree inputs are on a different CUDA device")

        if self._initialized:
            if (
                root_logits.shape != self._root_logits.shape
                or root_logits.dtype != self._root_logits.dtype
                or root_hidden.shape != self._root_hidden.shape
                or root_hidden.dtype != self._root_hidden.dtype
            ):
                raise ValueError("root input shapes/dtypes must remain fixed")
            key_buffer = self._key_buffers[0]
            if (
                root_key.shape[1] != key_buffer.shape[1]
                or root_key.shape[3] != key_buffer.shape[3]
                or root_key.dtype != key_buffer.dtype
                or root_value.dtype != self._value_buffers[0].dtype
                or root_logits.device != self.device
            ):
                raise ValueError("root KV shapes/dtypes/device must remain fixed")

    def _initialize(
        self,
        root_logits: torch.Tensor,
        root_hidden: torch.Tensor,
        root_key: torch.Tensor,
        root_value: torch.Tensor,
    ) -> None:
        self.device = root_logits.device
        # inv_freq 可能仍在 CPU；提前固定到 GPU，避免 RoPE 在捕获时做 H2D。
        midlayer = getattr(self.draft_model, "midlayer", None)
        inv_freq = getattr(midlayer, "inv_freq", None)
        if inv_freq is not None:
            midlayer.inv_freq = inv_freq.to(device=self.device, dtype=torch.float32)
        self._root_logits = torch.zeros_like(root_logits)
        self._root_hidden = torch.zeros_like(root_hidden)
        self._root_len = torch.zeros((), dtype=torch.long, device=self.device)
        self._rope_delta = torch.zeros((), dtype=torch.long, device=self.device)

        # 两份完整 frontier KV 交替作为源/目的，避免重排时源目重叠。
        kv_shape = (self.beam, root_key.shape[1], self.max_len, root_key.shape[3])
        self._key_buffers = tuple(
            torch.zeros(kv_shape, dtype=root_key.dtype, device=self.device)
            for _ in range(2)
        )
        self._value_buffers = tuple(
            torch.zeros(kv_shape, dtype=root_value.dtype, device=self.device)
            for _ in range(2)
        )

        self._token_buffer = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )
        self._parent_buffer = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )
        self._score_buffer = torch.zeros(
            self.total_nodes, dtype=torch.float32, device=self.device
        )
        self._root_parent = torch.full(
            (self.beam,), -1, dtype=torch.long, device=self.device
        )
        self._root_global_indices = torch.arange(self.beam, device=self.device)
        self._key_positions = torch.arange(self.max_len, device=self.device).view(
            1, 1, 1, self.max_len
        )
        self._key_mask = torch.zeros(
            (1, 1, 1, self.max_len), dtype=torch.bool, device=self.device
        )
        self._initialized = True

    def _fill_inputs(
        self,
        root_logits: torch.Tensor,
        root_hidden: torch.Tensor,
        root_key: torch.Tensor,
        root_value: torch.Tensor,
        root_len: int,
        rope_delta: int,
    ) -> None:
        """每次 warmup/capture/replay 前恢复根输入；buf0 会被上轮重排。"""
        self._root_logits.copy_(root_logits)
        self._root_hidden.copy_(root_hidden)
        self._root_len.fill_(root_len)
        self._rope_delta.fill_(rope_delta)
        for buffers, root in (
            (self._key_buffers, root_key),
            (self._value_buffers, root_value),
        ):
            prefix = root[:, :, :root_len].expand(self.beam, -1, -1, -1)
            buffers[0][:, :, :root_len].copy_(prefix)

    def _run_static(self) -> None:
        """所有形状和 Python 控制流固定；不得在捕获期间做 D2H 同步。"""
        root_ids, root_scores = _topk_target_log_probs(
            self.draft_model, self._root_logits, self.beam
        )
        self._token_buffer[: self.beam].copy_(root_ids[0])
        self._parent_buffer[: self.beam].copy_(self._root_parent)
        self._score_buffer[: self.beam].copy_(root_scores[0])

        frontier_tokens = self._token_buffer[: self.beam].view(self.beam, 1)
        frontier_hidden = self._root_hidden.expand(self.beam, -1).contiguous()
        frontier_scores = self._score_buffer[: self.beam]
        frontier_global_indices = self._root_global_indices
        source_index = 0

        for node_depth in range(1, self.num_steps):
            cache_position = (self._root_len + node_depth - 1).reshape(1)
            rope_position = (cache_position + self._rope_delta).reshape(1)
            self._key_mask.copy_(self._key_positions <= cache_position.view(1, 1, 1, 1))

            child_logits, child_hidden = self.draft_model.step_static(
                frontier_tokens,
                frontier_hidden,
                self._key_buffers[source_index],
                self._value_buffers[source_index],
                cache_position,
                rope_position,
                self._key_mask,
            )
            child_ids, child_log_probs = _topk_target_log_probs(
                self.draft_model, child_logits, self.beam
            )
            candidate_scores = (frontier_scores.unsqueeze(1) + child_log_probs).reshape(
                -1
            )
            candidate_ids = child_ids.reshape(-1)
            candidate_parents = frontier_global_indices.repeat_interleave(self.beam)

            # 每层保留全部候选供图外剪枝，同时用全局 top-beam 更新 frontier。
            candidate_offset = self.beam + (node_depth - 1) * self.beam * self.beam
            candidate_end = candidate_offset + self.beam * self.beam
            self._token_buffer[candidate_offset:candidate_end].copy_(candidate_ids)
            self._parent_buffer[candidate_offset:candidate_end].copy_(candidate_parents)
            self._score_buffer[candidate_offset:candidate_end].copy_(candidate_scores)

            selected_scores, selected_indices = torch.topk(
                candidate_scores, k=self.beam, dim=-1
            )
            selected_parent_rows = selected_indices // self.beam
            destination_index = 1 - source_index
            self._key_buffers[destination_index].copy_(
                self._key_buffers[source_index].index_select(0, selected_parent_rows)
            )
            self._value_buffers[destination_index].copy_(
                self._value_buffers[source_index].index_select(0, selected_parent_rows)
            )

            frontier_tokens = candidate_ids.index_select(0, selected_indices).view(
                self.beam, 1
            )
            frontier_hidden = child_hidden.index_select(0, selected_parent_rows)
            frontier_scores = selected_scores
            frontier_global_indices = candidate_offset + selected_indices
            source_index = destination_index

    def _capture(
        self,
        root_logits: torch.Tensor,
        root_hidden: torch.Tensor,
        root_key: torch.Tensor,
        root_value: torch.Tensor,
        root_len: int,
        rope_delta: int,
    ) -> None:
        # * 使用 side_stream  进行热身, 等待default stream
        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(torch.cuda.current_stream(self.device))
        # * 开始热身
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_inputs(
                    root_logits, root_hidden, root_key, root_value, root_len, rope_delta
                )
                self._run_static()
        # * 默认流等待热身结束
        torch.cuda.current_stream(self.device).wait_stream(side_stream)

        self._fill_inputs(
            root_logits, root_hidden, root_key, root_value, root_len, rope_delta
        )
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run_static()
        self._captured = True

        # capture 只录制不执行，必须重新填根输入并 replay 才得到本轮输出。
        self._fill_inputs(
            root_logits, root_hidden, root_key, root_value, root_len, rope_delta
        )
        self._graph.replay()

    @torch.inference_mode()
    def build_tree(
        self,
        root_logits: torch.Tensor,
        root_hidden: torch.Tensor,
        root_kv,
        draft_kv_length: int,
        rope_delta: int = 0,
    ) -> TreeLayout:
        root_key, root_value = root_kv
        root_len = int(draft_kv_length)
        rope_delta = int(rope_delta)
        self._check_inputs(root_logits, root_hidden, root_key, root_value, root_len)
        if not self._initialized:
            self._initialize(root_logits, root_hidden, root_key, root_value)

        if self._captured:
            self._fill_inputs(
                root_logits, root_hidden, root_key, root_value, root_len, rope_delta
            )
            self._graph.replay()
        else:
            self._capture(
                root_logits, root_hidden, root_key, root_value, root_len, rope_delta
            )

        # D2H 和变长树剪枝均留在图外，保持捕获内容固定。
        token_ids = self._token_buffer.tolist()
        parent = self._parent_buffer.tolist()
        scores = self._score_buffer.tolist()
        return _compact_tree_by_score(
            token_ids, parent, scores, int(self.cfg.max_tree_nodes)
        )
