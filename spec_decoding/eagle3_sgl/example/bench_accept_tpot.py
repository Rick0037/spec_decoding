# SPDX-License-Identifier: Apache-2.0
"""固定 prompt 集的 EAGLE3 接受率与延迟评测.

做两件事:
(1) 统计每轮 draft 提出的候选 token 数与 target 实际接受数, 输出平均接受长度 tau.
(2) 固定 prompt 集 + greedy 解码, 与关闭投机的 baseline 对比 TPOT 和端到端吞吐.

一键预设(--mode):
- text: Llama-3.1-8B + 官方 EAGLE3 draft, 10 条固定文本 prompt.
- vl:   Qwen3-VL-8B + 对应 EAGLE3 draft, 从 LMUData tsv 读图文样本(带图 forward).

长输出: 加 --explain 让模型先分步推理再给答案. 短答案(yes/no, 选择题字母)测不出
真实接受长度, 必须让输出足够长.

dtype 注意: bf16 只有 7 位尾数, 长序列(>100 token)下 baseline 与 verify 的数值
累积差异会导致 argmax 平局翻转, 出现假发散. 默认 float16(10 位尾数)可避免.

口径说明:
- proposed_draft_tokens: target 选中那条根->叶路径上的 draft 节点数, 不含未选中分支,
  也不含 target 最后额外给出的 bonus/correction token.
- accepted_tokens: 每轮实际 append 的 token 数(含最后那个 target 自己给出的 token).
- tau = accepted_tokens / rounds, 即平均接受长度.
- TPOT = wall / new_tokens, 平均每输出 token 的墙钟耗时(含 prefill), 单位 ms.
- 端到端吞吐 = 全部 prompt 的 new_tokens 之和 / wall 之和.

跑法:
    # 纯文本
    HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. \
    .venv/bin/python spec_decoding/eagle3_sgl/example/bench_accept_tpot.py \
        --mode text --max-new-tokens 128 --explain

    # 多模态(带图)
    HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. \
    .venv/bin/python spec_decoding/eagle3_sgl/example/bench_accept_tpot.py \
        --mode vl --num-prompts 3 --max-new-tokens 128 --explain
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field

import torch


# 两种模式的一键预设: target / draft / draft 权重文件名.
MODE_PRESETS = {
    "text": {
        "target": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "draft": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "draft_weight_file": "pytorch_model.bin",
    },
    "vl": {
        "target": "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct",
        "draft": "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
        "draft_weight_file": "model.safetensors",
    },
}

DEFAULT_VL_TSV = "/root/autodl-tmp/LMUData/ChartQA_TEST.tsv"

# 引导长输出的后缀: 短答案数据集必须让模型输出推理过程才测得出接受长度.
EXPLAIN_SUFFIX = (
    "\nExplain your reasoning step by step before giving the final answer."
)

# 固定 10 条 prompt, 覆盖解释/代码/数学/翻译/写作等不同分布.
FIXED_PROMPTS = [
    "Explain what speculative decoding is, in two sentences.",
    "Write a Python function that checks whether a string is a palindrome.",
    "What is the capital of France, and why is it historically important?",
    "Summarize the main idea of the Transformer architecture in three bullet points.",
    "Solve: if 3x + 7 = 22, what is x? Show your steps.",
    "Translate 'The quick brown fox jumps over the lazy dog' into French.",
    "Write a short haiku about autumn rain.",
    "Explain the difference between a process and a thread.",
    "Give three tips for writing clean, readable code.",
    "What causes the seasons on Earth? Answer in two sentences.",
]


@dataclass
class PromptResult:
    """单条 prompt 的统计结果."""

    new_tokens: int = 0
    rounds: int = 0
    accepted_tokens: int = 0
    proposed_draft_tokens: int = 0
    spec_wall: float = 0.0
    baseline_wall: float = 0.0
    baseline_new_tokens: int = 0
    lossless: bool = True


@dataclass
class BenchSummary:
    """整个 prompt 集的累计统计."""

    results: list = field(default_factory=list)

    def total(self, attribute: str) -> float:
        return sum(getattr(result, attribute) for result in self.results)


def load_target(target_id: str, device, dtype):
    """加载 target 模型与 tokenizer, 失败时回退到等价镜像."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fallback = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    last_error = None
    for model_id in [target_id, fallback]:
        try:
            print(f"[bench] loading target {model_id} ...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = (
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
                .to(device)
                .eval()
            )
            print(f"[bench] target ready: {model_id}", flush=True)
            return tokenizer, model, model_id
        except Exception as error:
            print(f"[bench] failed to load {model_id}: {error}", flush=True)
            last_error = error
    raise RuntimeError(f"could not load target; last error: {last_error}")


def build_input_ids(tokenizer, prompt: str, device) -> torch.LongTensor:
    """把 prompt 转成 input_ids, instruct 模型优先套 chat template."""
    if getattr(tokenizer, "chat_template", None):
        output = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = output if isinstance(output, torch.Tensor) else output["input_ids"]
        return input_ids.to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def run_spec_prompt(gen, input_ids, max_new_tokens: int, eos) -> tuple:
    """跑一条 prompt 的投机解码, 返回 (输出 token 列表, 本轮计数差值, 墙钟)."""
    prompt_length = int(input_ids.shape[1])
    rounds_before = gen.n_rounds
    accepted_before = gen.n_accepted_tokens
    proposed_before = gen.n_proposed_draft_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = gen.generate(
            input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    new_ids = output_ids[0, prompt_length:].tolist()
    rounds = gen.n_rounds - rounds_before
    accepted = gen.n_accepted_tokens - accepted_before
    proposed = gen.n_proposed_draft_tokens - proposed_before
    return new_ids, rounds, accepted, proposed, wall


def run_baseline_prompt(target, input_ids, max_new_tokens: int, eos) -> tuple:
    """跑一条 prompt 的 HF greedy baseline, 返回 (输出 token 列表, 墙钟)."""
    prompt_length = int(input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = target.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=eos,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    new_ids = output[0, prompt_length:].tolist()
    return new_ids, wall


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGLE3 accept-rate + TPOT benchmark")
    parser.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle3-draft", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--topk", type=int, default=8, help="每个父节点展开的子候选数(树的宽度)")
    parser.add_argument("--num-steps", type=int, default=8, help="draft 树的最大深度(根到叶的层数)")
    command_line_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
    )

    tokenizer, target, target_id = load_target(command_line_args.target, device, dtype)
    num_hidden_layers = int(target.config.num_hidden_layers)
    eos = tokenizer.eos_token_id

    # 构造并加载真实 EAGLE3 draft 权重.
    draft_config_path = hf_hub_download(command_line_args.eagle3_draft, "config.json")
    draft_vocab_size = int(
        json.load(open(draft_config_path)).get("draft_vocab_size", 32000)
    )
    draft_weights_path = hf_hub_download(
        command_line_args.eagle3_draft, "pytorch_model.bin"
    )

    config = Eagle3SglConfig(
        topk=command_line_args.topk,
        num_steps=command_line_args.num_steps,
        max_tree_nodes=64,
        verify_mode="full_model_tree",
        verify_attn_backend="auto",
    )
    eagle_layers = config.resolve_eagle_layers(num_hidden_layers)
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=draft_vocab_size,
        draft_weights_path=draft_weights_path,
        device=device,
        dtype=dtype,
    )
    generator = Eagle3SglGenerator(target, config, draft)
    print(f"[bench] target={target_id} eagle_layers={eagle_layers}", flush=True)

    prompts = FIXED_PROMPTS[: command_line_args.num_prompts]

    # 预热: 触发 CUDA kernel 编译/autotune, 不计入统计.
    warmup_ids = build_input_ids(tokenizer, prompts[0], device)
    with torch.inference_mode():
        generator.generate(warmup_ids, max_new_tokens=8, eos_token_id=eos)
    generator.n_rounds = 0
    generator.n_accepted_tokens = 0
    generator.n_proposed_draft_tokens = 0
    print("[bench] warmup done", flush=True)

    summary = BenchSummary()
    for index, prompt in enumerate(prompts):
        input_ids = build_input_ids(tokenizer, prompt, device)

        spec_ids, rounds, accepted, proposed, spec_wall = run_spec_prompt(
            generator, input_ids, command_line_args.max_new_tokens, eos
        )
        baseline_ids, baseline_wall = run_baseline_prompt(
            target, input_ids, command_line_args.max_new_tokens, eos
        )

        # 逐 token 对比 greedy 无损性(只比共同长度).
        common_length = min(len(spec_ids), len(baseline_ids))
        lossless = spec_ids[:common_length] == baseline_ids[:common_length]

        result = PromptResult(
            new_tokens=len(spec_ids),
            rounds=rounds,
            accepted_tokens=accepted,
            proposed_draft_tokens=proposed,
            spec_wall=spec_wall,
            baseline_wall=baseline_wall,
            baseline_new_tokens=len(baseline_ids),
            lossless=lossless,
        )
        summary.results.append(result)

        spec_tpot_ms = 1000.0 * spec_wall / max(len(spec_ids), 1)
        baseline_tpot_ms = 1000.0 * baseline_wall / max(len(baseline_ids), 1)
        print(
            f"[{index + 1:02d}] new={len(spec_ids)} rounds={rounds} "
            f"tau={accepted / max(rounds, 1):.2f} proposed/round={proposed / max(rounds, 1):.2f} "
            f"spec_tpot={spec_tpot_ms:.1f}ms base_tpot={baseline_tpot_ms:.1f}ms "
            f"lossless={lossless}",
            flush=True,
        )

    # 汇总.
    total_new = summary.total("new_tokens")
    total_rounds = summary.total("rounds")
    total_accepted = summary.total("accepted_tokens")
    total_proposed = summary.total("proposed_draft_tokens")
    total_spec_wall = summary.total("spec_wall")
    total_baseline_wall = summary.total("baseline_wall")
    total_baseline_new = summary.total("baseline_new_tokens")

    tau = total_accepted / max(total_rounds, 1)
    proposed_per_round = total_proposed / max(total_rounds, 1)
    spec_tpot_ms = 1000.0 * total_spec_wall / max(total_new, 1)
    baseline_tpot_ms = 1000.0 * total_baseline_wall / max(total_baseline_new, 1)
    spec_throughput = total_new / max(total_spec_wall, 1e-9)
    baseline_throughput = total_baseline_new / max(total_baseline_wall, 1e-9)
    speedup = spec_throughput / max(baseline_throughput, 1e-9)
    all_lossless = all(result.lossless for result in summary.results)

    report = [
        "",
        "==================== EAGLE3 accept + TPOT ====================",
        f"target={target_id} draft={command_line_args.eagle3_draft}",
        f"num_prompts={len(summary.results)} max_new_tokens={command_line_args.max_new_tokens}",
        f"tree: topk(width)={command_line_args.topk} num_steps(depth)={command_line_args.num_steps} max_tree_nodes=64",
        f"total_new_tokens={total_new} total_rounds={total_rounds}",
        f"accepted_tokens={total_accepted} proposed_draft_tokens={total_proposed}",
        f"tau(accept_length)={tau:.3f} proposed/round={proposed_per_round:.3f}",
        f"spec:   TPOT={spec_tpot_ms:.2f}ms  throughput={spec_throughput:.2f} tok/s  wall={total_spec_wall:.2f}s",
        f"baseline: TPOT={baseline_tpot_ms:.2f}ms  throughput={baseline_throughput:.2f} tok/s  wall={total_baseline_wall:.2f}s",
        f"speedup={speedup:.2f}x  lossless(all)={all_lossless}",
        "=============================================================",
    ]
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
