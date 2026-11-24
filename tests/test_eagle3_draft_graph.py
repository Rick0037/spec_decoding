"""EAGLE3 draft tree 静态 KV / CUDA Graph 正确性测试。"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spec_decoding.eagle3_sgl.config import Eagle3SglConfig
from spec_decoding.eagle3_sgl.draft_model import (
    Eagle3SglDraftModel,
    _Eagle3MidLayer,
    _RMSNorm,
)
from spec_decoding.eagle3_sgl.tree_draft import expand_draft_tree_ar


def _tiny_draft(device: torch.device, dtype=torch.float16):
    hidden, vocab = 32, 64
    draft = Eagle3SglDraftModel.__new__(Eagle3SglDraftModel)
    nn.Module.__init__(draft)
    kwargs = {"device": device, "dtype": dtype}
    draft.n_acts = 1
    draft.hidden = hidden
    draft.draft_vocab_size = vocab
    draft._shared_embed = [nn.Embedding(vocab, hidden, **kwargs)]
    draft._shared_lmhead = None
    draft.hot_token_id = None
    draft.fc = nn.Identity()
    draft.midlayer = _Eagle3MidLayer(
        hidden,
        n_heads=4,
        intermediate=64,
        n_kv_heads=2,
        rope_base=10000.0,
        eps=1e-6,
        **kwargs,
    )
    draft.norm = _RMSNorm(hidden, eps=1e-6, **kwargs)
    draft.lm_head = nn.Linear(hidden, vocab, bias=False, **kwargs)
    with torch.no_grad():
        for parameter in draft.parameters():
            parameter.normal_(0, 0.05)
    return draft.eval()


def _root(draft, prefix_len: int, seed: int):
    device = draft.lm_head.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(
        0, draft.draft_vocab_size, (1, prefix_len), generator=generator, device=device
    )
    cond = torch.randn(
        1,
        prefix_len,
        draft.hidden,
        generator=generator,
        device=device,
        dtype=draft.lm_head.weight.dtype,
    ) * 0.1
    logits, hidden, kv = draft.prefill(ids, cond)
    return logits, hidden, kv


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
class DraftTreeGraphTest(unittest.TestCase):
    def test_static_step_matches_dynamic_kv_step(self):
        device = torch.device("cuda")
        draft = _tiny_draft(device)
        root_logits, root_hidden, root_kv = _root(draft, 6, 1)
        root_len = root_kv[0].shape[2]
        tokens = root_logits.topk(2, dim=-1).indices[0].view(2, 1)
        hidden = root_hidden.expand(2, -1).contiguous()
        past = tuple(
            value.expand(2, *value.shape[1:]).contiguous() for value in root_kv
        )
        expected_logits, expected_hidden, expected_kv = draft.step(
            tokens, hidden, past, root_len
        )

        max_len = 16
        static_kv = tuple(
            torch.zeros(
                2,
                value.shape[1],
                max_len,
                value.shape[3],
                device=device,
                dtype=value.dtype,
            )
            for value in past
        )
        for static, value in zip(static_kv, past):
            static[:, :, :root_len].copy_(value)
        cache_position = torch.tensor([root_len], device=device)
        key_mask = (
            torch.arange(max_len, device=device) <= root_len
        ).view(1, 1, 1, max_len)
        actual_norm_hidden, actual_hidden = draft.step_static(
            tokens,
            hidden,
            static_kv[0],
            static_kv[1],
            cache_position,
            cache_position,
            key_mask,
        )
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2)
        with torch.inference_mode():
            actual_logits = draft._lm_logits(actual_norm_hidden)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-2, atol=2e-2)
        for static, expected in zip(static_kv, expected_kv):
            torch.testing.assert_close(
                static[:, :, : root_len + 1], expected, rtol=2e-2, atol=2e-2
            )

    def test_whole_tree_graph_matches_eager_and_reuses_capture(self):
        from spec_decoding.eagle3_sgl.draft_graph import StaticGraphDraftTree

        device = torch.device("cuda")
        draft = _tiny_draft(device)
        config = Eagle3SglConfig(
            topk=2,
            num_steps=4,
            max_tree_nodes=8,
            draft_tree_mode="eagle2_batch",
        )
        graph = StaticGraphDraftTree(draft, config, max_len=32, device=device)

        for prefix_len, seed, rope_delta in ((6, 2, 0), (9, 3, 3)):
            root_logits, root_hidden, root_kv = _root(draft, prefix_len, seed)
            root_len = root_kv[0].shape[2]
            expected = expand_draft_tree_ar(
                config,
                draft,
                root_logits,
                root_hidden,
                root_kv,
                draft_kv_length=root_len,
                device=device,
                rope_delta=rope_delta,
            )
            actual = graph.build_tree(
                root_logits,
                root_hidden,
                root_kv,
                draft_kv_length=root_len,
                rope_delta=rope_delta,
            )
            self.assertEqual(actual.token_ids, expected.token_ids)
            self.assertEqual(actual.parent, expected.parent)

        self.assertTrue(graph._captured)
        captured_graph = graph._graph
        root_logits, root_hidden, root_kv = _root(draft, 7, 4)
        graph.build_tree(
            root_logits,
            root_hidden,
            root_kv,
            draft_kv_length=root_kv[0].shape[2],
        )
        self.assertIs(graph._graph, captured_graph)

    def test_svd_graph_matches_eager_svd_tree(self):
        from spec_decoding.eagle3_sgl import tree_draft as tree_draft_module
        from spec_decoding.eagle3_sgl.draft_graph import StaticGraphDraftTree
        from spec_decoding.eagle3_sgl.example.svd_draft.svd_head import (
            CandidateLogits,
            DraftHeadSwitcher,
            build_svd_head,
            candidate_topk_log_probs,
        )

        device = torch.device("cuda")
        draft = _tiny_draft(device)
        config = Eagle3SglConfig(
            topk=2,
            num_steps=4,
            max_tree_nodes=8,
            draft_tree_mode="eagle2_batch",
        )
        head = build_svd_head(draft.lm_head.weight, rank=16, num_candidates=32)
        graph = StaticGraphDraftTree(
            draft, config, max_len=32, device=device, svd_head=head
        )

        original_topk_log_probs = tree_draft_module._topk_target_log_probs

        def adaptive_topk_log_probs(draft_model, logits, topk):
            if isinstance(logits, CandidateLogits):
                return candidate_topk_log_probs(draft_model, logits, topk)
            return original_topk_log_probs(draft_model, logits, topk)

        tree_draft_module._topk_target_log_probs = adaptive_topk_log_probs
        try:
            with torch.inference_mode(), DraftHeadSwitcher(
                draft, head, native_candidates=True
            ) as switcher:
                captured_graph = None
                for prefix_len, seed, rope_delta in ((6, 2, 0), (9, 3, 3)):
                    # 根输入用 Full prefill: 图内根节点不做 SVD, eager 根也走完整 logits。
                    switcher.set_mode("full")
                    root_logits, root_hidden, root_kv = _root(draft, prefix_len, seed)
                    root_len = root_kv[0].shape[2]
                    switcher.set_mode("svd")
                    expected = expand_draft_tree_ar(
                        config,
                        draft,
                        root_logits,
                        root_hidden,
                        root_kv,
                        draft_kv_length=root_len,
                        device=device,
                        rope_delta=rope_delta,
                    )
                    actual = graph.build_tree(
                        root_logits,
                        root_hidden,
                        root_kv,
                        draft_kv_length=root_len,
                        rope_delta=rope_delta,
                    )
                    self.assertEqual(actual.token_ids, expected.token_ids)
                    self.assertEqual(actual.parent, expected.parent)
                    if captured_graph is not None:
                        self.assertIs(graph._graph, captured_graph)
                    captured_graph = graph._graph
        finally:
            tree_draft_module._topk_target_log_probs = original_topk_log_probs


if __name__ == "__main__":
    unittest.main(verbosity=2)
