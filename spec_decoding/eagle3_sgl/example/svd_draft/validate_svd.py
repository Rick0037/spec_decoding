"""SVD 代数、缓存、稀疏分布和原 Full 路径正确性验证。"""

from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn.functional as F

from .svd_head import (
    CandidateLogits,
    DraftHeadSwitcher,
    build_svd_head,
    candidate_topk_log_probs,
    evaluate_recall,
)


class SVDCorrectnessTest(unittest.TestCase):
    def _check_device(self, device: torch.device) -> None:
        generator = torch.Generator(device=device).manual_seed(7)
        vocab, hidden, true_rank = 96, 32, 4
        left = torch.randn(vocab, true_rank, generator=generator, device=device)
        right = torch.randn(true_rank, hidden, generator=generator, device=device)
        weight = (left @ right).contiguous()
        original = weight.clone()
        head = build_svd_head(
            weight,
            rank=true_rank,
            num_candidates=16,
            oversample=8,
            niter=2,
            seed=11,
        )
        rows = torch.randn(3, hidden, generator=generator, device=device)

        # A/B 形状、冻结状态、完整重构和原权重不变。
        self.assertEqual(head.a.shape, (vocab, true_rank))
        self.assertEqual(head.b.shape, (true_rank, hidden))
        self.assertFalse(head.a.requires_grad)
        self.assertFalse(head.b.requires_grad)
        self.assertIsNone(weight.grad)
        torch.testing.assert_close(weight, original, rtol=0, atol=0)
        reconstructed = head.a @ head.b
        relative_error = (reconstructed - weight).norm() / weight.norm()
        self.assertLess(float(relative_error.item()), 2e-4)
        self.assertGreater(head.stats.energy_retention, 0.999)

        # 第二阶段必须与独立计算的 W[C] @ h 一致，且支持多行输入。
        candidate_ids, candidate_logits = head.candidates(rows)
        self.assertEqual(candidate_ids.shape, (3, 16))
        self.assertEqual(candidate_logits.shape, (3, 16))
        expected = torch.einsum(
            "bkh,bh->bk", weight[candidate_ids].float(), rows.float()
        )
        torch.testing.assert_close(candidate_logits, expected, rtol=1e-5, atol=1e-5)

        # 候选外严格为 -inf；候选内保持精确分数，候选分布 q 与稀疏完整词表 q 一致。
        sparse = head.sparse_logits(rows)
        mask = torch.zeros_like(sparse, dtype=torch.bool)
        mask.scatter_(1, candidate_ids, True)
        self.assertTrue(torch.isinf(sparse[~mask]).all())
        self.assertTrue((sparse[~mask] < 0).all())
        torch.testing.assert_close(sparse.gather(1, candidate_ids), candidate_logits)
        full_probability = torch.softmax(sparse, dim=-1)
        candidate_probability = torch.softmax(candidate_logits, dim=-1)
        torch.testing.assert_close(
            full_probability.gather(1, candidate_ids), candidate_probability
        )
        self.assertEqual(int(full_probability.count_nonzero()), 3 * 16)

        # Full 路径保持原始 GEMM；SVD TopK 只使用候选内精确重打分。
        torch.testing.assert_close(head.full_logits(rows), F.linear(rows, weight))
        svd_ids, svd_scores = head.svd_topk(rows, 4)
        expected_scores, positions = torch.topk(candidate_logits, 4, dim=-1)
        self.assertTrue(torch.equal(svd_ids, candidate_ids.gather(1, positions)))
        torch.testing.assert_close(svd_scores, expected_scores)
        metrics = evaluate_recall(head, rows, topk=4)
        self.assertEqual(metrics["rows"], 3)
        self.assertLessEqual(metrics["max_rescore_abs_error"], 1e-5)

    def test_cpu_correctness(self) -> None:
        self._check_device(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_correctness(self) -> None:
        self._check_device(torch.device("cuda"))

    def test_cache_roundtrip_and_weight_binding(self) -> None:
        weight = torch.randn(64, 24)
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            first = build_svd_head(
                weight, rank=8, num_candidates=12, cache_dir=directory, seed=3
            )
            second = build_svd_head(
                weight, rank=8, num_candidates=12, cache_dir=directory, seed=3
            )
            self.assertFalse(first.stats.cache_hit)
            self.assertTrue(second.stats.cache_hit)
            self.assertEqual(first.stats.weight_sha256, second.stats.weight_sha256)
            torch.testing.assert_close(first.a, second.a)
            torch.testing.assert_close(first.b, second.b)

            changed = weight.clone()
            changed[0, 0] += 1
            third = build_svd_head(
                changed, rank=8, num_candidates=12, cache_dir=directory, seed=3
            )
            self.assertFalse(third.stats.cache_hit)
            self.assertNotEqual(first.stats.weight_sha256, third.stats.weight_sha256)

    def test_switcher_defaults_to_original_full_and_restores(self) -> None:
        class DummyDraft:
            def __init__(self, weight):
                self.lm_head = type("Head", (), {"weight": weight})()
                self.hot_token_id = None

            def _lm_logits(self, hidden):
                return F.linear(hidden, self.lm_head.weight)

            def convert_to_target_topk_token_ids(self, logits, k):
                scores, ids = torch.topk(logits, k, dim=-1)
                return ids, scores

        weight = torch.randn(64, 24)
        draft = DummyDraft(weight)
        head = build_svd_head(weight, rank=8, num_candidates=12)
        hidden = torch.randn(2, 24)
        with DraftHeadSwitcher(draft, head) as switcher:
            torch.testing.assert_close(draft._lm_logits(hidden), F.linear(hidden, weight))
            switcher.set_mode("svd")
            sparse = draft._lm_logits(hidden)
            self.assertEqual(int(torch.isfinite(sparse).sum()), 2 * 12)
        self.assertEqual(draft._lm_logits.__self__, draft)

    def test_native_candidate_tree_adapter(self) -> None:
        class DummyDraft:
            def __init__(self, weight):
                self.lm_head = type("Head", (), {"weight": weight})()
                self.hot_token_id = None

            def _lm_logits(self, hidden):
                return F.linear(hidden, self.lm_head.weight)

            def convert_to_target_topk_token_ids(self, logits, k):
                scores, ids = torch.topk(logits, k, dim=-1)
                return ids, scores

        weight = torch.randn(64, 24)
        draft = DummyDraft(weight)
        head = build_svd_head(weight, rank=8, num_candidates=12)
        hidden = torch.randn(2, 24)
        with DraftHeadSwitcher(draft, head, native_candidates=True) as switcher:
            switcher.set_mode("svd")
            candidates = draft._lm_logits(hidden)
            self.assertIsInstance(candidates, CandidateLogits)
            ids, scores = draft.convert_to_target_topk_token_ids(candidates, 4)
            expected_ids, expected_scores = head.svd_topk(hidden, 4)
            self.assertTrue(torch.equal(ids, expected_ids))
            torch.testing.assert_close(scores, expected_scores)

            _, log_probabilities = candidate_topk_log_probs(draft, candidates, 4)
            selected_scores, _ = torch.topk(candidates.logits, 4, dim=-1)
            expected_log_probabilities = selected_scores.float() - torch.logsumexp(
                candidates.logits.float(), dim=-1, keepdim=True
            )
            torch.testing.assert_close(log_probabilities, expected_log_probabilities)


if __name__ == "__main__":
    unittest.main(verbosity=2)
