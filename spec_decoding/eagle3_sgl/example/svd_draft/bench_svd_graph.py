"""图内 A/B：Full 与 SVD 候选 head 各捕获一张 draft tree CUDA Graph，比较建树延迟与树重合度。

根节点两边都用完整 root_logits；SVD 只替换 7 层子节点的 lm_head。
SVD 分解在计时外完成；测速区间只含 fill+replay+图外剪枝。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ...config import Eagle3SglConfig
from ...draft_graph import StaticGraphDraftTree
from ...draft_model import build_eagle3_sgl_draft
from ..bench_accept_tpot import (
    EXPLAIN_SUFFIX,
    FIXED_PROMPTS,
    load_target,
    load_vl_samples,
)
from .bench_svd_tree import (
    DEFAULT_VL_TSV,
    local_draft_file,
    make_draft_root,
    prepare_sample,
)
from .svd_head import build_svd_head, measure_wall, stats_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", default="vl", choices=["vl", "text"])
    parser.add_argument("--tsv", default=DEFAULT_VL_TSV, help="vl 模式数据集")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--explain", action="store_true", help="追加解释要求以拉长输出")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--preconsume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--cache-dir", default="/tmp/qwen3vl_svd_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", help="可选：把逐样本与汇总结果写入 JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("draft tree CUDA Graph 需要 CUDA")
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer, target, target_id, processor = load_target(args.target, device, dtype, "vl")
    text_config = getattr(target.config, "text_config", target.config)
    config = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        draft_tree_mode="eagle2_batch",
        preconsume_first_token=args.preconsume,
        verify_attn_backend="eager",
    )
    eagle_layers = config.resolve_eagle_layers(int(text_config.num_hidden_layers))

    draft_config_path = local_draft_file("config.json")
    draft_weights_path = local_draft_file("model.safetensors")
    with open(draft_config_path) as handle:
        draft_config = json.load(handle)
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=int(draft_config.get("draft_vocab_size", 32000)),
        draft_weights_path=draft_weights_path,
        device=device,
        dtype=dtype,
    )
    print(f"[source] target={target_id}")
    print(f"[source] draft={draft_weights_path}")

    if args.mode == "vl":
        samples = load_vl_samples(args.tsv, args.num_prompts)
        prompts = [
            question + (EXPLAIN_SUFFIX if args.explain else "")
            for question, _ in samples
        ]
    else:
        samples = None
        prompts = [
            prompt + (EXPLAIN_SUFFIX if args.explain else "")
            for prompt in FIXED_PROMPTS[: args.num_prompts]
        ]
    print(
        f"[config] mode={args.mode} num_prompts={len(prompts)} rank={args.rank} "
        f"K={args.num_candidates} topk={args.topk} steps={args.num_steps} "
        f"max_nodes={args.max_tree_nodes}"
    )

    head = build_svd_head(
        draft.lm_head.weight,
        rank=args.rank,
        num_candidates=args.num_candidates,
        oversample=args.oversample,
        niter=args.niter,
        seed=args.seed,
        cache_dir=None if args.no_cache else args.cache_dir,
        source=draft_weights_path,
    )
    print("[init] " + stats_json(head.stats))

    # 先收集全部真实根输入; 静态 KV 容量按最长前缀一次性定死。
    roots = []
    with torch.inference_mode():
        for index in range(len(prompts)):
            sample = prepare_sample(
                target, tokenizer, processor, samples, prompts, eagle_layers, device, index
            )
            root = make_draft_root(draft, sample, args.preconsume, device)
            roots.append((root, sample[4]))
            del sample
    max_len = max(root[3] for root, _ in roots) + args.num_steps + 8
    graph_full = StaticGraphDraftTree(draft, config, max_len, device)
    graph_svd = StaticGraphDraftTree(draft, config, max_len, device, svd_head=head)
    print(f"[graph] max_len={max_len} total_nodes={graph_full.total_nodes}")

    def build_tree(graph, root, rope_delta: int):
        root_logits, root_hidden, root_kv, prefix_length = root
        return graph.build_tree(
            root_logits,
            root_hidden,
            root_kv,
            draft_kv_length=prefix_length - 1,
            rope_delta=rope_delta,
        )

    per_sample = []
    for index, (root, rope_delta) in enumerate(roots):
        full_layout = build_tree(graph_full, root, rope_delta)
        svd_layout = build_tree(graph_svd, root, rope_delta)
        same_tree = (
            full_layout.token_ids == svd_layout.token_ids
            and full_layout.parent == svd_layout.parent
        )
        common = sum(
            (Counter(full_layout.token_ids) & Counter(svd_layout.token_ids)).values()
        )
        overlap = common / max(
            1, max(len(full_layout.token_ids), len(svd_layout.token_ids))
        )
        timings = {
            "full": measure_wall(
                lambda: build_tree(graph_full, root, rope_delta),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "svd": measure_wall(
                lambda: build_tree(graph_svd, root, rope_delta),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
        }
        per_sample.append(
            {
                "index": index,
                "prompt": prompts[index][:80],
                "prefix_length": root[3],
                "timings": timings,
                "exact_layout_match": same_tree,
                "token_multiset_overlap": overlap,
            }
        )
        print(
            f"[{index:02d}] prefix={root[3]} "
            f"full={timings['full']['median_ms']:.2f}ms "
            f"svd={timings['svd']['median_ms']:.2f}ms "
            f"overlap={overlap:.4f} exact={same_tree}"
        )

    summary = {
        "num_prompts": len(per_sample),
        "mean_full_median_ms": sum(
            item["timings"]["full"]["median_ms"] for item in per_sample
        ) / len(per_sample),
        "mean_svd_median_ms": sum(
            item["timings"]["svd"]["median_ms"] for item in per_sample
        ) / len(per_sample),
        "mean_token_multiset_overlap": sum(
            item["token_multiset_overlap"] for item in per_sample
        ) / len(per_sample),
        "exact_layout_match_count": sum(
            item["exact_layout_match"] for item in per_sample
        ),
    }
    summary["svd_speedup"] = summary["mean_full_median_ms"] / summary["mean_svd_median_ms"]

    print("\n==================== 图内 A/B 聚合 ====================")
    print(
        f"prompts={summary['num_prompts']} rank={args.rank} K={args.num_candidates}\n"
        f"mean_full={summary['mean_full_median_ms']:.2f}ms "
        f"mean_svd={summary['mean_svd_median_ms']:.2f}ms "
        f"speedup={summary['svd_speedup']:.3f}x\n"
        f"mean_overlap={summary['mean_token_multiset_overlap']:.4f} "
        f"tree_exact_match={summary['exact_layout_match_count']}/{summary['num_prompts']}"
    )
    print(
        "\n注意：测的是单轮稳态建树(fill+replay+图外剪枝)，不含 target verify/端到端 tokens/s；"
        "根节点两边都是完整 logits，SVD 只作用于子节点。"
    )

    if args.output:
        result = {
            "target": target_id,
            "draft_weights": draft_weights_path,
            "mode": args.mode,
            "tsv": args.tsv if args.mode == "vl" else None,
            "dtype": args.dtype,
            "rank": args.rank,
            "num_candidates": args.num_candidates,
            "topk": args.topk,
            "num_steps": args.num_steps,
            "max_tree_nodes": args.max_tree_nodes,
            "preconsume": args.preconsume,
            "max_len": max_len,
            "init": json.loads(stats_json(head.stats)),
            "summary": summary,
            "per_sample": per_sample,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
