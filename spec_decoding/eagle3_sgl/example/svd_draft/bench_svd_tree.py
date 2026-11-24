"""在真实 Qwen3-VL hidden 上比较 Full/SVD 的现有 EAGLE3 草稿树生成。"""

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
from ..bench_accept_tpot import build_input_ids, load_target
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


def local_draft_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DEFAULT_REPO, filename, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt", default="Explain what speculative decoding is, in two sentences.")
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cache-dir", default="/tmp/qwen3vl_svd_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", help="可选：把结果 JSON 写到指定路径")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer, target, target_id, _ = load_target(args.target, device, dtype, "vl")
    input_ids = build_input_ids(tokenizer, args.prompt, device)
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
    print(
        f"[config] mode=full|svd rank={args.rank} K={args.num_candidates} "
        f"topk={args.topk} steps={args.num_steps} max_nodes={args.max_tree_nodes}"
    )

    # target 只做一次 prefill 来取得真实条件特征，不属于草稿树计时段。
    with torch.inference_mode():
        outputs = target(input_ids, use_cache=True, output_hidden_states=True)
        acts = select_eagle_acts_from_hidden_states(
            outputs.hidden_states, eagle_layers, last_token_only=False
        )
        recovery_logits = outputs.logits[0, -1].clone()
        del outputs

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

    def make_root(mode: str):
        switcher.set_mode(mode)
        root_logits, root_hidden, root_kv = draft.prefill(input_ids, acts)
        prefix_length = int(input_ids.shape[1])
        if args.preconsume:
            first_token = int(recovery_logits.argmax().item())
            token_ids = torch.tensor([[first_token]], device=device, dtype=torch.long)
            root_logits, root_hidden, root_kv = draft.extend_tokens(
                token_ids,
                acts[:, -1:, :],
                root_kv,
                prefix_length - 1,
            )
            prefix_length += 1
        return root_logits, root_hidden, root_kv, prefix_length

    def build_tree(mode: str, root):
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
            full_root = make_root("full")
            sparse_root = make_root("svd_sparse")
            svd_root = make_root("svd")

            # 沿 Full 草稿轨迹记录真实 norm(hidden)，离线计算候选召回率，不污染测速。
            switcher.recording = True
            switcher.recorded_hidden.clear()
            full_layout = build_tree("full", full_root)
            switcher.recording = False
            real_hidden = torch.cat(switcher.recorded_hidden, dim=0)
            recall = evaluate_recall(head, real_hidden, args.topk)
            print("[real-hidden-correctness] " + json.dumps(recall, ensure_ascii=False))

            sparse_layout = build_tree("svd_sparse", sparse_root)
            svd_layout = build_tree("svd", svd_root)
            same_tree = (
                full_layout.token_ids == svd_layout.token_ids
                and full_layout.parent == svd_layout.parent
            )
            sparse_native_match = (
                sparse_layout.token_ids == svd_layout.token_ids
                and sparse_layout.parent == svd_layout.parent
            )
            full_counter = Counter(full_layout.token_ids)
            svd_counter = Counter(svd_layout.token_ids)
            common = sum((full_counter & svd_counter).values())
            token_overlap = common / max(
                1, max(len(full_layout.token_ids), len(svd_layout.token_ids))
            )

            timings = {
                "full": measure_wall(
                    lambda: build_tree("full", full_root),
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
                "svd_sparse": measure_wall(
                    lambda: build_tree("svd_sparse", sparse_root),
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
                "svd_native": measure_wall(
                    lambda: build_tree("svd", svd_root),
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
            }
    finally:
        tree_draft_module._topk_target_log_probs = original_topk_log_probs

    full_ms = timings["full"]["median_ms"]
    layouts = {
        "full": full_layout,
        "svd_sparse": sparse_layout,
        "svd_native": svd_layout,
    }
    print("\n| Draft tree mode | median (ms) | p90 (ms) | Full/mode | nodes |")
    print("|---|---:|---:|---:|---:|")
    for mode, timing in timings.items():
        print(
            f"| {mode} | {timing['median_ms']:.3f} | {timing['p90_ms']:.3f} | "
            f"{full_ms / timing['median_ms']:.3f}x | {len(layouts[mode].token_ids)} |"
        )
    print(
        f"[tree-output] full_native_exact_match={same_tree}, "
        f"sparse_native_exact_match={sparse_native_match}, "
        f"full_native_token_overlap={token_overlap:.4f}"
    )
    print(
        "\n注意：这里只测现有 draft 建树，不含 target prefill/verify、接受长度或端到端 tokens/s；"
        "layout 不同表示候选召回变化，不是目标模型正确性错误。"
    )

    if args.output:
        result = {
            "target": target_id,
            "draft_weights": draft_weights_path,
            "prompt": args.prompt,
            "dtype": args.dtype,
            "rank": args.rank,
            "num_candidates": args.num_candidates,
            "topk": args.topk,
            "num_steps": args.num_steps,
            "max_tree_nodes": args.max_tree_nodes,
            "tree_mode": args.tree_mode,
            "preconsume": args.preconsume,
            "init": json.loads(stats_json(head.stats)),
            "real_hidden_correctness": recall,
            "timings": timings,
            "native_speedup": full_ms / timings["svd_native"]["median_ms"],
            "sparse_speedup": full_ms / timings["svd_sparse"]["median_ms"],
            "full_nodes": len(full_layout.token_ids),
            "svd_nodes": len(svd_layout.token_ids),
            "exact_layout_match": same_tree,
            "sparse_native_exact_match": sparse_native_match,
            "token_multiset_overlap": token_overlap,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
