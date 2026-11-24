"""在数据集(多条真实样本)上比较 Full/SVD 的现有 EAGLE3 草稿树生成与候选召回率。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ... import tree_draft as tree_draft_module
from ...config import Eagle3SglConfig
from ...draft_model import build_eagle3_sgl_draft
from ...hidden import select_eagle_acts_from_hidden_states
from ...tree_draft import expand_draft_tree_ar
from ..bench_accept_tpot import (
    EXPLAIN_SUFFIX,
    FIXED_PROMPTS,
    build_input_ids,
    build_vl_input,
    load_target,
    load_vl_samples,
)
from .svd_head import (
    CandidateLogits,
    DraftHeadSwitcher,
    build_svd_head,
    candidate_topk_log_probs,
    evaluate_recall,
    measure_wall,
    stats_json,
)


DEFAULT_REPO = "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"
DEFAULT_VL_TSV = "/root/autodl-tmp/LMUData/ChartQA_TEST.tsv"


def local_draft_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DEFAULT_REPO, filename, local_files_only=True)


def prepare_sample(target, tokenizer, processor, samples, prompts, eagle_layers, device, index):
    """target prefill 取真实 acts; 带图样本走 get_rope_index 的完整 MRoPE 路径."""
    if samples is not None:
        input_ids, multimodal = build_vl_input(
            processor, prompts[index], samples[index][1], device
        )
    else:
        input_ids, multimodal = build_input_ids(tokenizer, prompts[index], device), None

    rope_delta = 0
    draft_position_ids = None
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        if multimodal is not None:
            position_ids, rope_deltas = target.model.get_rope_index(
                input_ids,
                multimodal["mm_token_type_ids"],
                image_grid_thw=multimodal.get("image_grid_thw"),
                attention_mask=attention_mask,
            )
            rope_delta = int(rope_deltas.flatten()[0].item())
            draft_position_ids = position_ids
            outputs = target(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=True,
                **multimodal,
            )
        else:
            outputs = target(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        acts = select_eagle_acts_from_hidden_states(
            outputs.hidden_states, eagle_layers, last_token_only=False
        )
        recovery_logits = outputs.logits[0, -1].clone()
    del outputs
    return input_ids, acts, recovery_logits, draft_position_ids, rope_delta


def make_draft_root(draft, sample, preconsume: bool, device):
    """draft prefill(+可选预消费首 token) 得到建树根输入; 返回含前缀长度."""
    input_ids, acts, recovery_logits, draft_position_ids, rope_delta = sample
    root_logits, root_hidden, root_kv = draft.prefill(
        input_ids, acts, position_ids=draft_position_ids
    )
    prefix_length = int(input_ids.shape[1])
    if preconsume:
        first_token = int(recovery_logits.argmax().item())
        token_ids = torch.tensor([[first_token]], device=device, dtype=torch.long)
        root_logits, root_hidden, root_kv = draft.extend_tokens(
            token_ids,
            acts[:, -1:, :],
            root_kv,
            prefix_length - 1,
            rope_delta=rope_delta,
        )
        prefix_length += 1
    return root_logits, root_hidden, root_kv, prefix_length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mode", default="vl", choices=["vl", "text"])
    parser.add_argument("--tsv", default=DEFAULT_VL_TSV, help="vl 模式数据集")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--explain", action="store_true", help="追加解释要求以拉长输出")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--tree-mode", choices=["eagle2_batch", "bfs"], default="eagle2_batch")
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

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer, target, target_id, processor = load_target(args.target, device, dtype, "vl")
    text_config = getattr(target.config, "text_config", target.config)
    config = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        draft_tree_mode=args.tree_mode,
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

    # 数据集样本: vl 从 tsv 读图文, text 用固定 prompt 集; 顺序固定保证可复现.
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

    def make_root(mode: str, sample):
        switcher.set_mode(mode)
        return make_draft_root(draft, sample, args.preconsume, device)

    def build_tree(mode: str, root, rope_delta: int):
        root_logits, root_hidden, root_kv, prefix_length = root
        switcher.set_mode(mode)
        return expand_draft_tree_ar(
            config,
            draft,
            root_logits,
            root_hidden,
            root_kv,
            draft_kv_length=prefix_length - 1,
            device=device,
            rope_delta=rope_delta,
        )

    original_topk_log_probs = tree_draft_module._topk_target_log_probs

    def adaptive_topk_log_probs(draft_model, logits, topk):
        if isinstance(logits, CandidateLogits):
            return candidate_topk_log_probs(draft_model, logits, topk)
        return original_topk_log_probs(draft_model, logits, topk)

    per_sample = []
    tree_draft_module._topk_target_log_probs = adaptive_topk_log_probs
    try:
        with torch.inference_mode(), DraftHeadSwitcher(
            draft, head, native_candidates=True
        ) as switcher:
            for index in range(len(prompts)):
                sample = prepare_sample(
                    target, tokenizer, processor, samples, prompts, eagle_layers, device, index
                )
                rope_delta = sample[4]
                full_root = make_root("full", sample)
                sparse_root = make_root("svd_sparse", sample)
                svd_root = make_root("svd", sample)

                # 沿 Full 草稿轨迹记录真实 norm(hidden)，离线算召回，不污染测速.
                switcher.recording = True
                switcher.recorded_hidden.clear()
                full_layout = build_tree("full", full_root, rope_delta)
                switcher.recording = False
                real_hidden = torch.cat(switcher.recorded_hidden, dim=0)
                recall = evaluate_recall(head, real_hidden, args.topk)

                sparse_layout = build_tree("svd_sparse", sparse_root, rope_delta)
                svd_layout = build_tree("svd", svd_root, rope_delta)
                same_tree = (
                    full_layout.token_ids == svd_layout.token_ids
                    and full_layout.parent == svd_layout.parent
                )
                sparse_native_match = (
                    sparse_layout.token_ids == svd_layout.token_ids
                    and sparse_layout.parent == svd_layout.parent
                )
                common = sum(
                    (Counter(full_layout.token_ids) & Counter(svd_layout.token_ids)).values()
                )
                overlap = common / max(
                    1, max(len(full_layout.token_ids), len(svd_layout.token_ids))
                )

                timings = {
                    "full": measure_wall(
                        lambda: build_tree("full", full_root, rope_delta),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    ),
                    "svd_sparse": measure_wall(
                        lambda: build_tree("svd_sparse", sparse_root, rope_delta),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    ),
                    "svd_native": measure_wall(
                        lambda: build_tree("svd", svd_root, rope_delta),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    ),
                }
                per_sample.append(
                    {
                        "index": index,
                        "prompt": prompts[index][:80],
                        "rows": recall["rows"],
                        "recall": recall,
                        "timings": timings,
                        "exact_layout_match": same_tree,
                        "sparse_native_exact_match": sparse_native_match,
                        "token_multiset_overlap": overlap,
                    }
                )
                print(
                    f"[{index:02d}] rows={recall['rows']} "
                    f"top1_recall={recall['candidate_top1_recall']:.4f} "
                    f"topk_recall={recall['candidate_topk_recall']:.4f} "
                    f"full={timings['full']['median_ms']:.2f}ms "
                    f"native={timings['svd_native']['median_ms']:.2f}ms "
                    f"overlap={overlap:.4f}"
                )
                del sample, full_root, sparse_root, svd_root, real_hidden
                torch.cuda.empty_cache()
    finally:
        tree_draft_module._topk_target_log_probs = original_topk_log_probs

    # 召回按行数加权汇总; 延迟报各样本 median 的均值与总和口径的加速比.
    total_rows = sum(item["rows"] for item in per_sample)
    weighted = lambda key: sum(
        item["recall"][key] * item["rows"] for item in per_sample
    ) / max(1, total_rows)
    summary = {
        "num_prompts": len(per_sample),
        "total_rows": total_rows,
        "candidate_top1_recall": weighted("candidate_top1_recall"),
        "candidate_topk_recall": weighted("candidate_topk_recall"),
        "selected_top1_match": weighted("selected_top1_match"),
        "selected_topk_overlap": weighted("selected_topk_overlap"),
        "mean_token_multiset_overlap": sum(
            item["token_multiset_overlap"] for item in per_sample
        ) / len(per_sample),
        "exact_layout_match_count": sum(
            item["exact_layout_match"] for item in per_sample
        ),
        "sparse_native_all_match": all(
            item["sparse_native_exact_match"] for item in per_sample
        ),
    }
    for mode in ("full", "svd_sparse", "svd_native"):
        summary[f"mean_{mode}_median_ms"] = sum(
            item["timings"][mode]["median_ms"] for item in per_sample
        ) / len(per_sample)
    summary["native_speedup"] = (
        summary["mean_full_median_ms"] / summary["mean_svd_native_median_ms"]
    )
    summary["sparse_speedup"] = (
        summary["mean_full_median_ms"] / summary["mean_svd_sparse_median_ms"]
    )

    print("\n==================== 数据集聚合 ====================")
    print(
        f"prompts={summary['num_prompts']} hidden_rows={total_rows}\n"
        f"top1_recall={summary['candidate_top1_recall']:.4f} "
        f"topk_recall={summary['candidate_topk_recall']:.4f} "
        f"top1_match={summary['selected_top1_match']:.4f}\n"
        f"mean_full={summary['mean_full_median_ms']:.2f}ms "
        f"mean_sparse={summary['mean_svd_sparse_median_ms']:.2f}ms "
        f"mean_native={summary['mean_svd_native_median_ms']:.2f}ms\n"
        f"native_speedup={summary['native_speedup']:.3f}x "
        f"sparse_speedup={summary['sparse_speedup']:.3f}x "
        f"tree_exact_match={summary['exact_layout_match_count']}/{summary['num_prompts']}"
    )
    print(
        "\n注意：每条样本测一个稳态轮次的建树，不含 target verify/接受长度/端到端 tokens/s；"
        "召回按 hidden 行数加权。"
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
            "tree_mode": args.tree_mode,
            "preconsume": args.preconsume,
            "init": json.loads(stats_json(head.stats)),
            "summary": summary,
            "per_sample": per_sample,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
