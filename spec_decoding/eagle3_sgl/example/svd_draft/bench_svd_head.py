"""真实 Qwen3-VL EAGLE3 LM Head 的 Full/SVD 候选生成微基准。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .svd_head import build_svd_head, evaluate_recall, measure_wall, stats_json


DEFAULT_REPO = "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"


def resolve_weight_path(args) -> str:
    if args.weight_path:
        return args.weight_path
    from huggingface_hub import hf_hub_download

    # 严格只读本地缓存，绝不触发模型下载。
    return hf_hub_download(DEFAULT_REPO, "model.safetensors", local_files_only=True)


def load_weight(path: str, key: str, device: torch.device) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"{path} 中没有 {key}，可用键: {list(handle.keys())}")
        return handle.get_tensor(key).to(device)


def normalized_hidden(rows: int, hidden_size: int, weight: torch.Tensor, seed: int):
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    hidden = torch.randn(rows, hidden_size, generator=generator, device=weight.device)
    hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    return hidden.to(weight.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", help="本地 draft safetensors；默认读 HF 本地缓存")
    parser.add_argument("--weight-key", default="lm_head.weight")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--eval-rows", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cache-dir", default="/tmp/qwen3vl_svd_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", help="可选：把结果 JSON 写到指定路径")
    args = parser.parse_args()

    device = torch.device(args.device)
    weight_path = resolve_weight_path(args)
    weight = load_weight(weight_path, args.weight_key, device)
    print(f"[source] {weight_path}")
    print(f"[weight] {args.weight_key}: {tuple(weight.shape)} {weight.dtype} on {device}")
    print(f"[config] lm_head_mode=svd rank={args.rank} K={args.num_candidates} topk={args.topk}")

    head = build_svd_head(
        weight,
        rank=args.rank,
        num_candidates=args.num_candidates,
        oversample=args.oversample,
        niter=args.niter,
        seed=args.seed,
        cache_dir=None if args.no_cache else args.cache_dir,
        source=weight_path,
    )
    print("[init] " + stats_json(head.stats))
    print(
        f"[memory] full={weight.numel() * weight.element_size() / 2**20:.2f} MiB, "
        f"svd_extra={head.extra_bytes / 2**20:.2f} MiB"
    )

    # 随机 RMS 归一化 hidden 仅用于 Head 延迟和代数检查；召回率由 tree 脚本在真实 hidden 上另测。
    eval_hidden = normalized_hidden(args.eval_rows, weight.shape[1], weight, args.seed + 1)
    recall = evaluate_recall(head, eval_hidden, args.topk)
    print("[synthetic-correctness] " + json.dumps(recall, ensure_ascii=False))

    results = []
    for rows in args.rows:
        hidden = normalized_hidden(rows, weight.shape[1], weight, args.seed + rows)
        full = measure_wall(
            lambda: head.full_topk(hidden, args.topk),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        svd = measure_wall(
            lambda: head.svd_topk(hidden, args.topk),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result = {"rows": rows, "full": full, "svd": svd, "speedup": full["median_ms"] / svd["median_ms"]}
        results.append(result)

    print("\n| rows | Full median (ms) | SVD median (ms) | Full/SVD | SVD p90 (ms) |")
    print("|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['rows']} | {result['full']['median_ms']:.4f} | "
            f"{result['svd']['median_ms']:.4f} | {result['speedup']:.3f}x | "
            f"{result['svd']['p90_ms']:.4f} |"
        )
    print("\n注意：这是候选 Head 微基准，不是端到端 tokens/s；随机 hidden 的召回率不代表真实草稿分布。")

    if args.output:
        output = {
            "weight_path": weight_path,
            "weight_shape": tuple(weight.shape),
            "weight_dtype": str(weight.dtype),
            "rank": args.rank,
            "num_candidates": args.num_candidates,
            "topk": args.topk,
            "init": json.loads(stats_json(head.stats)),
            "extra_bytes": head.extra_bytes,
            "synthetic_correctness": recall,
            "timing": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
