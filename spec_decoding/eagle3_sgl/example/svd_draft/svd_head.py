"""无训练 SVD 候选 Head 低秩召回 + 原始 LM Head 精确重打分。"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


@dataclass
class SVDInitStats:
    rank: int
    oversample: int
    niter: int
    seed: int
    energy_retention: float
    reconstruction_error: float
    init_seconds: float
    weight_sha256: str
    weight_shape: tuple[int, int]
    weight_dtype: str
    source: str
    cache_hit: bool = False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weight_sha256(weight: torch.Tensor) -> str:
    """绑定缓存与原始权重内容；不复制 BF16 权重到 bytes。"""
    cpu = weight.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _randomized_svd(
    weight: torch.Tensor,
    rank: int,
    *,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """在 FP32 上做 randomized SVD，返回转为原权重 dtype 的 A、B。"""
    if weight.dim() != 2:
        raise ValueError(f"weight 必须是 [V,H]，得到 {tuple(weight.shape)}")
    vocab, hidden = weight.shape
    if not 0 < rank <= min(vocab, hidden):
        raise ValueError(f"rank={rank} 超出 weight={tuple(weight.shape)}")
    if oversample < 0 or niter < 0:
        raise ValueError("oversample/niter 必须非负")

    device = weight.device
    original_dtype = weight.dtype
    w = weight.float()
    q = min(rank + oversample, min(vocab, hidden))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    omega = torch.randn(
        hidden, q, generator=generator, device=device, dtype=torch.float32
    )

    previous_tf32 = None
    if device.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Y = W Omega，随后用 power iteration 提高尾部奇异值衰减较慢时的精度。
        basis, _ = torch.linalg.qr(w @ omega, mode="reduced")
        for _ in range(niter):
            z, _ = torch.linalg.qr(w.t() @ basis, mode="reduced")
            basis, _ = torch.linalg.qr(w @ z, mode="reduced")
        small = basis.t() @ w
        u_small, singular, vh = torch.linalg.svd(small, full_matrices=False)
        u = basis @ u_small
        a = (u[:, :rank] * singular[:rank]).to(original_dtype)
        b = vh[:rank].to(original_dtype)
        energy = float((singular[:rank].square().sum() / w.square().sum()).item())

        # 等距抽样检查 AB 对 W 的重构，不额外物化完整的 [V,H] 矩阵。
        sample_rows = min(256, vocab)
        rows = torch.linspace(0, vocab - 1, sample_rows, device=device).long()
        reconstructed = a[rows].float() @ b.float()
        reference = w[rows]
        error = float(
            (
                (reconstructed - reference).norm() / reference.norm().clamp_min(1e-12)
            ).item()
        )
        return a.detach(), b.detach(), energy, error
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def build_svd_head(
    weight: torch.Tensor,
    *,
    rank: int = 128,
    num_candidates: int = 128,
    oversample: int = 16,
    niter: int = 2,
    seed: int = 1234,
    cache_dir: Optional[str] = None,
    source: str = "",
) -> "SVDCandidateHead":
    """从原始权重生成或加载纯 SVD 因子；缓存键包含完整权重 SHA256。"""
    if not 0 < num_candidates <= weight.shape[0]:
        raise ValueError(
            f"num_candidates={num_candidates} 必须位于 [1,{weight.shape[0]}]"
        )
    digest = weight_sha256(weight)
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / (
            f"svd_{digest[:16]}_r{rank}_o{oversample}_n{niter}_s{seed}.pt"
        )
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            expected = {
                "weight_sha256": digest,
                "weight_shape": tuple(weight.shape),
                "weight_dtype": str(weight.dtype),
                "rank": rank,
                "oversample": oversample,
                "niter": niter,
                "seed": seed,
            }
            actual = {key: payload.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    f"SVD 缓存元数据不匹配: {cache_path}, {actual} != {expected}"
                )
            stats = SVDInitStats(
                rank=rank,
                oversample=oversample,
                niter=niter,
                seed=seed,
                energy_retention=float(payload["energy_retention"]),
                reconstruction_error=float(payload["reconstruction_error"]),
                init_seconds=0.0,
                weight_sha256=digest,
                weight_shape=tuple(weight.shape),
                weight_dtype=str(weight.dtype),
                source=source,
                cache_hit=True,
            )
            return SVDCandidateHead(
                weight,
                payload["a"].to(device=weight.device, dtype=weight.dtype),
                payload["b"].to(device=weight.device, dtype=weight.dtype),
                num_candidates,
                stats,
            )

    _sync(weight.device)
    start = time.perf_counter()
    a, b, energy, error = _randomized_svd(
        weight, rank, oversample=oversample, niter=niter, seed=seed
    )
    _sync(weight.device)
    elapsed = time.perf_counter() - start
    stats = SVDInitStats(
        rank=rank,
        oversample=oversample,
        niter=niter,
        seed=seed,
        energy_retention=energy,
        reconstruction_error=error,
        init_seconds=elapsed,
        weight_sha256=digest,
        weight_shape=tuple(weight.shape),
        weight_dtype=str(weight.dtype),
        source=source,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": a.cpu(),
            "b": b.cpu(),
            "energy_retention": energy,
            "reconstruction_error": error,
            "weight_sha256": digest,
            "weight_shape": tuple(weight.shape),
            "weight_dtype": str(weight.dtype),
            "rank": rank,
            "oversample": oversample,
            "niter": niter,
            "seed": seed,
        }
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, cache_path)
    return SVDCandidateHead(weight, a, b, num_candidates, stats)


class CandidateLogits(NamedTuple):
    """现有建树所需的最小输出：候选内精确 logits 与 draft token id。"""

    logits: torch.Tensor
    token_ids: torch.Tensor


class SVDCandidateHead:
    """保留原始 W；A/B 仅为冻结 Tensor，不是可训练 Parameter。"""

    def __init__(
        self,
        weight: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        num_candidates: int,
        stats: SVDInitStats,
    ) -> None:
        if a.shape != (weight.shape[0], b.shape[0]) or b.shape[1] != weight.shape[1]:
            raise ValueError(
                f"因子形状错误: W={tuple(weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        self.weight = weight
        self.a = a.detach()
        self.b = b.detach()
        self.num_candidates = num_candidates
        self.stats = stats
        self.a.requires_grad_(False)
        self.b.requires_grad_(False)

    @property
    def rank(self) -> int:
        return int(self.b.shape[0])

    @property
    def extra_bytes(self) -> int:
        return (
            self.a.numel() * self.a.element_size()
            + self.b.numel() * self.b.element_size()
        )

    def _check_hidden(self, hidden: torch.Tensor) -> None:
        if hidden.dim() != 2 or hidden.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"hidden 必须是 [N,{self.weight.shape[1]}]，得到 {tuple(hidden.shape)}"
            )
        if hidden.device != self.weight.device or hidden.dtype != self.weight.dtype:
            raise ValueError(
                f"hidden/weight 必须同 device/dtype: {hidden.device}/{hidden.dtype} vs "
                f"{self.weight.device}/{self.weight.dtype}"
            )

    @torch.inference_mode()
    def full_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        self._check_hidden(hidden)
        return F.linear(hidden, self.weight)

    @torch.inference_mode()
    def candidates(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回近似 Top-C 的全局 token id 与原始 W 的精确候选 logits。"""
        self._check_hidden(hidden)
        low_rank = F.linear(hidden, self.b)
        approximate = F.linear(low_rank, self.a)
        candidate_ids = torch.topk(
            approximate, self.num_candidates, dim=-1, sorted=False
        ).indices
        candidate_weight = self.weight[candidate_ids]
        exact_logits = torch.bmm(candidate_weight, hidden.unsqueeze(-1)).squeeze(-1)
        return candidate_ids, exact_logits

    @torch.inference_mode()
    def sparse_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """复用现有完整词表 TopK/LogSumExp：候选外置 -inf。"""
        candidate_ids, exact_logits = self.candidates(hidden)
        output = exact_logits.new_full(
            (hidden.shape[0], self.weight.shape[0]), float("-inf")
        )
        return output.scatter(1, candidate_ids, exact_logits)

    @torch.inference_mode()
    def full_topk(
        self, hidden: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, ids = torch.topk(self.full_logits(hidden), k, dim=-1)
        return ids, scores

    @torch.inference_mode()
    def svd_topk(
        self, hidden: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 < k <= self.num_candidates:
            raise ValueError(f"k={k} 必须位于 [1,{self.num_candidates}]")
        candidate_ids, candidate_logits = self.candidates(hidden)
        scores, positions = torch.topk(candidate_logits, k, dim=-1)
        return candidate_ids.gather(1, positions), scores


@torch.inference_mode()
def evaluate_recall(
    head: SVDCandidateHead, hidden: torch.Tensor, topk: int
) -> dict[str, float]:
    """在相同 hidden 上比较 Full 与 SVD；不计入任何测速区间。"""
    full_ids, _ = head.full_topk(hidden, topk)
    candidate_ids, candidate_logits = head.candidates(hidden)
    svd_ids, _ = head.svd_topk(hidden, topk)
    full_in_candidates = (full_ids.unsqueeze(2) == candidate_ids.unsqueeze(1)).any(
        dim=-1
    )
    final_overlap = (full_ids.unsqueeze(2) == svd_ids.unsqueeze(1)).any(dim=-1)
    expected = torch.einsum(
        "bkh,bh->bk", head.weight[candidate_ids].float(), hidden.float()
    ).to(hidden.dtype)
    max_rescore_error = float(
        (candidate_logits.float() - expected.float()).abs().max().item()
    )
    return {
        "rows": int(hidden.shape[0]),
        "candidate_top1_recall": float(full_in_candidates[:, 0].float().mean().item()),
        "candidate_topk_recall": float(full_in_candidates.float().mean().item()),
        "selected_top1_match": float(
            (full_ids[:, 0] == svd_ids[:, 0]).float().mean().item()
        ),
        "selected_topk_overlap": float(final_overlap.float().mean().item()),
        "max_rescore_abs_error": max_rescore_error,
    }


@torch.inference_mode()
def candidate_topk_log_probs(draft_model, candidates: CandidateLogits, topk: int):
    """与 tree_draft 相同口径，但只在 C 个精确候选上归一化 q。"""
    ids, scores = draft_model.convert_to_target_topk_token_ids(candidates, topk)
    normalizer = torch.logsumexp(candidates.logits.float(), dim=-1, keepdim=True)
    return ids, scores.float() - normalizer


class DraftHeadSwitcher:
    """实验期切换现有 draft 的 Head；full 调用原 bound method，退出可恢复。"""

    def __init__(
        self, draft_model, head: SVDCandidateHead, *, native_candidates: bool = False
    ) -> None:
        if getattr(draft_model, "lm_head", None) is None:
            raise ValueError("本实验要求 draft 使用独立 lm_head")
        if draft_model.lm_head.weight is not head.weight:
            raise ValueError("SVD head 与 draft lm_head 不是同一份原始权重")
        self.draft_model = draft_model
        self.head = head
        self.mode = "full"
        self.calls = 0
        self.recording = False
        self.recorded_hidden: list[torch.Tensor] = []
        self.native_candidates = native_candidates
        self._original = draft_model._lm_logits
        self._original_convert = draft_model.convert_to_target_topk_token_ids
        draft_model._lm_logits = self._logits
        draft_model.convert_to_target_topk_token_ids = self._convert

    def _convert(self, logits, k: int):
        if isinstance(logits, CandidateLogits):
            scores, positions = torch.topk(
                logits.logits, min(k, logits.logits.shape[-1]), dim=-1
            )
            ids = logits.token_ids.gather(1, positions)
            hot_token_id = getattr(self.draft_model, "hot_token_id", None)
            if hot_token_id is not None:
                ids = hot_token_id[ids]
            return ids, scores
        return self._original_convert(logits, k)

    def _logits(self, hidden: torch.Tensor):
        self.calls += 1
        if self.recording:
            self.recorded_hidden.append(hidden.detach().clone())
        if self.mode == "full":
            return self._original(hidden)
        if self.mode == "svd":
            if self.native_candidates:
                candidate_ids, exact_logits = self.head.candidates(hidden)
                return CandidateLogits(exact_logits, candidate_ids)
            return self.head.sparse_logits(hidden)
        if self.mode == "svd_sparse":
            return self.head.sparse_logits(hidden)
        raise ValueError(f"未知 lm_head_mode={self.mode!r}")

    def set_mode(self, mode: str) -> None:
        if mode not in ("full", "svd", "svd_sparse"):
            raise ValueError("mode 必须是 full、svd 或 svd_sparse")
        self.mode = mode

    def restore(self) -> None:
        self.draft_model._lm_logits = self._original
        self.draft_model.convert_to_target_topk_token_ids = self._original_convert

    def __enter__(self) -> "DraftHeadSwitcher":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.restore()


def measure_wall(fn, *, warmup: int, repeats: int) -> dict[str, float]:
    """每次包含 CPU launch，并在 GPU 上同步；返回毫秒统计。"""
    for _ in range(warmup):
        fn()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "p90_ms": ordered[p90_index],
        "repeats": repeats,
    }


def stats_json(stats: SVDInitStats) -> str:
    return json.dumps(asdict(stats), ensure_ascii=False, indent=2)
