#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
无 profiler 干扰的分阶段真实墙钟测量:用 synchronize + perf_counter 给每个阶段计时,
回答哪个阶段最耗时。配合 profile_eagle3_sgl.py 的算子表(哪些 op 多余)一起看。

阶段:target_prefill / draft_prefill_init / draft_build_tree / target_verify /
      verify_meta / verify_forward / verify_select / target_replay / draft_extend。
支持 text / vl 两种预设。此处有逐阶段同步开销, 只用于定位; 正式 HF 对比用 bench。

跑法(Docker 容器内示例;将 <工程根> 换成挂载到容器内的仓库根)::

    cd <工程根> && PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl.example.time_stages_eagle3_sgl \\
        --max-new-tokens 128 --topk 8 --num-steps 8 --max-tree-nodes 64
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch

from spec_decoding.eagle3_sgl.example.bench_accept_tpot import (
    DEFAULT_VL_TSV,
    EXPLAIN_SUFFIX,
    FIXED_PROMPTS,
    MODE_PRESETS,
    build_input_ids,
    build_vl_input,
    load_target,
    load_vl_samples,
)


class StageTimer:
    """同步式分阶段计时;用栈避免父子阶段重复计时(子阶段时间从父阶段扣除)."""

    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self._stack = []  # (name, t_start, child_accum)

    def wrap(self, name, fn):
        def inner(*a, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._stack.append([name, t0, 0.0])
            try:
                return fn(*a, **k)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                nm, ts, child = self._stack.pop()
                dur = t1 - ts
                self.total[nm] += dur - child  # 自身净时间
                self.calls[nm] += 1
                if self._stack:
                    self._stack[-1][2] += dur  # 计入父阶段的子时间

        return inner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="text", choices=["text", "vl"])
    p.add_argument("--target", default=None)
    p.add_argument("--eagle3-draft", default=None)
    p.add_argument("--tsv", default=DEFAULT_VL_TSV)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--prompt", default=FIXED_PROMPTS[0])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-tokens", type=int, default=32)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--max-tree-nodes", type=int, default=64)
    p.add_argument(
        "--tree-mode", default="eagle2_batch", choices=["eagle2", "eagle2_batch", "bfs"]
    )
    p.add_argument("--verify-mode", default="full_model_tree")
    p.add_argument("--verify-attn-backend", default="triton_tree")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--report-out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
    )
    import spec_decoding.eagle3_sgl.runner as R

    preset = MODE_PRESETS[args.mode]
    target_id_arg = args.target or preset["target"]
    draft_id = args.eagle3_draft or preset["draft"]
    tok, target, target_id, processor = load_target(
        target_id_arg, device, dtype, args.mode
    )
    text_config = getattr(target.config, "text_config", target.config)
    n_layers = int(text_config.num_hidden_layers)
    draft_cfg = json.load(open(hf_hub_download(draft_id, "config.json")))
    draft_bin = hf_hub_download(draft_id, preset["draft_weight_file"])
    cfg = Eagle3SglConfig(
        topk=args.topk,
        num_steps=args.num_steps,
        max_tree_nodes=args.max_tree_nodes,
        draft_tree_mode=args.tree_mode,
        verify_mode=args.verify_mode,
        verify_attn_backend=args.verify_attn_backend,
    )
    eagle_layers = cfg.resolve_eagle_layers(n_layers)
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=int(draft_cfg.get("draft_vocab_size", 32000)),
        draft_weights_path=draft_bin,
        device=device,
        dtype=dtype,
    )
    gen = Eagle3SglGenerator(target, cfg, draft)

    if args.mode == "vl":
        question, image = load_vl_samples(args.tsv, 1)[0]
        prompt = question + (EXPLAIN_SUFFIX if args.explain else "")
        input_ids, multimodal_inputs = build_vl_input(
            processor, prompt, image, device
        )
    else:
        prompt = args.prompt + (EXPLAIN_SUFFIX if args.explain else "")
        input_ids = build_input_ids(tok, prompt, device)
        multimodal_inputs = None
    eos = tok.eos_token_id

    # warmup
    with torch.inference_mode():
        gen.generate(
            input_ids,
            max_new_tokens=args.warmup_tokens,
            eos_token_id=eos,
            multimodal_inputs=multimodal_inputs,
        )
    gen.n_rounds = 0
    gen.n_accepted_tokens = 0
    gen.n_proposed_draft_tokens = 0
    gen.n_tree_nodes = 0

    # 安装分阶段计时(包裹热点 callable)
    T = StageTimer()
    gen._build_tree = T.wrap("draft_build_tree", gen._build_tree)
    gen.draft_model.extend_tokens = T.wrap(
        "draft_extend", gen.draft_model.extend_tokens
    )
    gen.draft_model.prefill = T.wrap("draft_prefill_init", gen.draft_model.prefill)

    # target forward:区分 prefill(首次) / verify(内部) / replay(其余)
    _orig_tf = gen.target_m.forward
    state = {"in_verify_depth": 0, "first": True}

    # verify 包裹:计时 + 标记,使内部 target forward 归到 verify,不重复计为 replay
    import spec_decoding.eagle3_sgl.tree_verify_full as TVF

    def verify_mark(name, fn):
        base = T.wrap(name, fn)

        def inner(*a, **k):
            state["in_verify_depth"] += 1
            try:
                return base(*a, **k)
            finally:
                state["in_verify_depth"] -= 1

        return inner

    R.full_tree_verify_extend = verify_mark(
        "target_verify", TVF.full_tree_verify_extend
    )
    R.full_tree_verify_triton = verify_mark(
        "target_verify", TVF.full_tree_verify_triton
    )
    # 子阶段由 StageTimer 的栈从 target_verify 中扣除, 不会重复计数.
    TVF.build_triton_tree_verify_metadata = T.wrap(
        "verify_meta", TVF.build_triton_tree_verify_metadata
    )
    TVF._forward_target_tree_and_restore_cache = T.wrap(
        "verify_forward", TVF._forward_target_tree_and_restore_cache
    )
    TVF._select_append_root_child = T.wrap(
        "verify_select", TVF._select_append_root_child
    )

    def tf_wrap(*a, **k):
        if state["first"]:
            state["first"] = False
            nm = "target_prefill"
        elif state["in_verify_depth"] > 0:
            return _orig_tf(*a, **k)  # verify 内部,已由 verify 计时
        else:
            nm = "target_replay"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return _orig_tf(*a, **k)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            T.total[nm] += time.perf_counter() - t0
            T.calls[nm] += 1

    gen.target_m.forward = tf_wrap

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = gen.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos,
            multimodal_inputs=multimodal_inputs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n_new = int(out_ids.shape[1] - input_ids.shape[1])
    accept = gen.n_accepted_tokens / max(1, gen.n_rounds)

    order = [
        "target_prefill",
        "draft_prefill_init",
        "draft_build_tree",
        "target_verify",
        "verify_meta",
        "verify_forward",
        "verify_select",
        "target_replay",
        "draft_extend",
    ]
    measured = sum(T.total[k] for k in order)
    lines = [
        "\n==================== EAGLE3 分阶段墙钟(父阶段已扣除子阶段)====================",
        f"mode={args.mode} tree_mode={args.tree_mode} target={target_id} draft={draft_id}",
        f"new_tokens={n_new} rounds={gen.n_rounds} accept/round={accept:.2f} wall={wall:.2f}s spec_tok/s={n_new / wall:.2f}",
        f"{'stage':<24}{'calls':>8}{'total_ms':>12}{'%measured':>11}{'%wall':>9}{'ms/call':>10}",
    ]
    for k in order:
        ms = T.total[k] * 1e3
        c = T.calls[k]
        lines.append(
            f"{k:<24}{c:>8}{ms:>12.1f}{100 * T.total[k] / max(measured, 1e-9):>10.1f}%{100 * T.total[k] / wall:>8.1f}%{ms / max(c, 1):>10.2f}"
        )
    lines.append(
        f"{'(measured sum)':<24}{'':>8}{measured * 1e3:>12.1f}{100.0:>10.1f}%{100 * measured / wall:>8.1f}%"
    )
    lines.append(
        f"{'(unmeasured/python)':<24}{'':>8}{(wall - measured) * 1e3:>12.1f}{'':>11}{100 * (wall - measured) / wall:>8.1f}%"
    )
    report = "\n".join(lines)
    print(report)
    if args.report_out:
        with open(args.report_out, "w") as f:
            f.write(report + "\n")
        print(f"[time] report -> {args.report_out}")


if __name__ == "__main__":
    main()
