#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
torch.profiler 剖析 EAGLE3:定位单步投机里最耗时 / 多余的操作。

输出三部分:
  1) 分阶段(record_function "stage:*")device 时间占比:target_prefill / draft_prefill_init /
     draft_build_tree / target_verify / target_replay / draft_extend。
  2) Top-N 算子(按 self CUDA 时间)—— 看哪个 kernel 最贵。
  3) Top CPU 算子 / 同步点(aten::item / cuda*Synchronize / copy_ / repeat_interleave 等)——
     看哪些是调度 / 同步 / 多余 H2D-D2H开销。

必须在 import 包前设置 EAGLE3_SGL_PROFILE=1。支持 text / vl 两种预设;
本脚本只用于诊断, 正式 HF 速度对比请运行不带插桩的 bench_accept_tpot.py。

跑法(Docker 容器内示例;将 <工程根> 换成挂载到容器内的仓库根)::

    cd <工程根> && PYTHONPATH=. python3 -m spec_decoding.eagle3_sgl.example.profile_eagle3_sgl \\
        --max-new-tokens 128 --topk 8 --num-steps 8 --max-tree-nodes 64 \\
        --trace-out /tmp/eagle3_trace.json.gz
"""

from __future__ import annotations
from torch.profiler import ProfilerActivity, profile
import torch
import time

import argparse
import gzip
import json
import os
import shutil

os.environ.setdefault("EAGLE3_SGL_PROFILE", "1")  # 必须在 import runner 前

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


def _stage_table(prof) -> str:
    """汇总 stage 的 device 时间.

    PyTorch 2.14 会把一个 stage 分成 CPU / CUDA 两条同名事件. 正确的子 kernel
    时间在 CPU 事件的 device_time_total 里; CUDA 同名事件是另一条区间, 不能使用,
    否则既会重复计数, 又可能出现超过 100% 的占比.
    """
    rows = []
    for event in prof.key_averages():
        if not event.key.startswith("stage:") or event.device_type.name != "CPU":
            continue
        device_time = float(event.device_time_total or 0)
        cpu_time = float(event.cpu_time_total or 0)
        rows.append((event.key, event.count, device_time, cpu_time))
    rows.sort(key=lambda row: row[2], reverse=True)
    total_device_time = sum(row[2] for row in rows) or 1.0
    lines = [
        "\n================= 分阶段 device 时间(仅 CPU stage 事件,含子 kernel)=================",
        f"{'stage':<28}{'calls':>8}{'dev_ms':>12}{'%dev':>8}{'cpu_ms':>12}",
    ]
    for key, count, device_time, cpu_time in rows:
        lines.append(
            f"{key:<28}{count:>8}{device_time / 1e3:>12.2f}"
            f"{100 * device_time / total_device_time:>7.1f}%{cpu_time / 1e3:>12.2f}"
        )
    lines.append(
        f"{'(stage total)':<28}{'':>8}{total_device_time / 1e3:>12.2f}{100.0:>7.1f}%"
    )
    return "\n".join(lines)


def _counts_table(prof) -> str:
    """统计调度, 同步和拷贝事件, 用来判断是不是 CPU / launch bound."""
    wanted_events = [
        "cudaStreamSynchronize",
        "cudaMemcpyAsync",
        "Memcpy HtoD",
        "Memcpy DtoH",
        "aten::copy_",
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::cat",
        "cudaLaunchKernel",
    ]
    aggregated = {}
    for event in prof.key_averages():
        for wanted in wanted_events:
            if wanted.lower() in event.key.lower():
                cpu_time = float(getattr(event, "cpu_time_total", 0) or 0)
                cuda_time = float(event.self_device_time_total or 0)
                count, old_cpu, old_cuda = aggregated.get(wanted, (0, 0.0, 0.0))
                aggregated[wanted] = (
                    count + event.count,
                    old_cpu + cpu_time,
                    old_cuda + cuda_time,
                )
                break
    lines = [
        "\n================= 关键事件计数(调度 / 同步 / 拷贝)=================",
        f"{'event':<28}{'count':>12}{'cpu_ms':>12}{'cuda_ms':>12}",
    ]
    for wanted in wanted_events:
        if wanted in aggregated:
            count, cpu_time, cuda_time = aggregated[wanted]
            lines.append(
                f"{wanted:<28}{count:>12}{cpu_time / 1e3:>12.1f}{cuda_time / 1e3:>12.1f}"
            )
    return "\n".join(lines)


def _operator_table(prof, sort_by: str, row_limit: int) -> str:
    """Top 算子表不混入 stage 区间, 避免 stage 与子算子被重复归因."""
    from torch.autograd.profiler import EventList

    operator_events = EventList(
        [
            event
            for event in prof.key_averages()
            if not event.key.startswith("stage:")
        ]
    )
    return operator_events.table(sort_by=sort_by, row_limit=row_limit)


def main() -> None:
    p = argparse.ArgumentParser(description="torch.profiler for EAGLE3")
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
    p.add_argument(
        "--verify-mode",
        default="full_model_tree",
        choices=["reference_paths", "full_model_tree"],
    )
    p.add_argument(
        "--verify-attn-backend",
        default="triton_tree",
        choices=["auto", "eager", "flash_attn", "triton_tree"],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--trace-out", default="")
    p.add_argument("--report-out", default="")
    args = p.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
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
    import spec_decoding.eagle3_sgl.runner as runner_mod

    assert runner_mod._PROFILE, "EAGLE3_SGL_PROFILE 未启用(需在 import 前设置)"

    preset = MODE_PRESETS[args.mode]
    target_id_arg = args.target or preset["target"]
    draft_id = args.eagle3_draft or preset["draft"]
    tok, target, target_id, processor = load_target(
        target_id_arg, device, dtype, args.mode
    )
    text_config = getattr(target.config, "text_config", target.config)
    n_layers = int(text_config.num_hidden_layers)

    draft_cfg_path = hf_hub_download(draft_id, "config.json")
    draft_vocab_size = int(
        json.load(open(draft_cfg_path)).get("draft_vocab_size", 32000)
    )
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
    print(
        f"[profile] eagle_layers={eagle_layers} draft_vocab_size={draft_vocab_size}",
        flush=True,
    )

    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=draft_vocab_size,
        draft_weights_path=draft_bin,
        device=device,
        dtype=dtype,
    )
    gen = Eagle3SglGenerator(target, cfg, draft)
    if args.mode == "vl":
        question, image = load_vl_samples(args.tsv, 1)[0]
        prompt = question + (EXPLAIN_SUFFIX if args.explain else "")
        input_ids, multimodal_inputs = build_vl_input(processor, prompt, image, device)
    else:
        prompt = args.prompt + (EXPLAIN_SUFFIX if args.explain else "")
        input_ids = build_input_ids(tok, prompt, device)
        multimodal_inputs = None
    eos = tok.eos_token_id

    # 预热(含 CUDA graph / kernel autotune / 编译缓存),不计入 profile
    print("[profile] warmup ...", flush=True)
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"[profile] profiling generate(max_new_tokens={args.max_new_tokens}) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
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

    header = (
        f"\n==================== EAGLE3 torch.profiler ====================\n"
        f"mode={args.mode} target={target_id} draft={draft_id}\n"
        f"tree_mode={args.tree_mode} verify={args.verify_mode}({args.verify_attn_backend}) "
        f"topk={args.topk} steps={args.num_steps} nodes={args.max_tree_nodes} dtype={args.dtype}\n"
        f"new_tokens={n_new} rounds={gen.n_rounds} accept/round={accept:.2f} "
        f"wall={wall:.2f}s spec_tok/s={n_new / wall:.2f}"
    )
    print(header)

    stage = _stage_table(prof)
    counts = _counts_table(prof)
    print(stage)
    print(counts)

    by_self_cuda = _operator_table(
        prof, sort_by="self_cuda_time_total", row_limit=args.top_n
    )
    by_cpu = _operator_table(
        prof, sort_by="self_cpu_time_total", row_limit=args.top_n
    )
    print("\n================= Top 算子(self CUDA 时间)=================")
    print(by_self_cuda)
    print("\n================= Top 算子(self CPU 时间,看调度/同步)=================")
    print(by_cpu)

    if args.trace_out:
        # Chrome trace 文本体积很大, 默认只保留 gzip 压缩版.
        trace_path = os.path.abspath(args.trace_out)
        raw_trace_path = trace_path[:-3] if trace_path.endswith(".gz") else trace_path
        compressed_trace_path = (
            trace_path if trace_path.endswith(".gz") else trace_path + ".gz"
        )
        prof.export_chrome_trace(raw_trace_path)
        with open(raw_trace_path, "rb") as source, gzip.open(
            compressed_trace_path, "wb"
        ) as destination:
            shutil.copyfileobj(source, destination)
        os.remove(raw_trace_path)
        print(f"[profile] chrome trace(gzip) -> {compressed_trace_path}")

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(stage + "\n")
            f.write(counts + "\n")
            f.write("\n== Top self CUDA ==\n")
            f.write(by_self_cuda + "\n")
            f.write("\n== Top self CPU ==\n")
            f.write(by_cpu + "\n")
        print(f"[profile] report -> {os.path.abspath(args.report_out)}")


if __name__ == "__main__":
    main()
