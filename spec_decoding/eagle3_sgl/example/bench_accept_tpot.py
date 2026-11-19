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
        --mode vl --num-prompts 10 --max-new-tokens 128 --explain

verify 默认走 triton_tree kernel, 需要退回 HF sdpa 时加 --verify-backend eager.
本脚本不开启 profile, 用于干净的 HF 速度对比. 诊断耗时占比使用:
- time_stages_eagle3_sgl.py: 分阶段墙钟和 verify 内部拆解.
- profile_eagle3_sgl.py: torch.profiler 算子, 同步, 拷贝和 launch 统计.
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
EXPLAIN_SUFFIX = "\nExplain your reasoning step by step before giving the final answer."

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
    tree_nodes: int = 0
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


def load_target(target_id: str, device, dtype, mode: str):
    """加载 target 模型与 tokenizer. 文本模式失败时回退到等价镜像, VL 模式不回退."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    if mode == "vl":
        print(f"[bench] loading target {target_id} ...", flush=True)
        processor = AutoProcessor.from_pretrained(target_id)
        model = (
            AutoModelForImageTextToText.from_pretrained(target_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        print(f"[bench] target ready: {target_id}", flush=True)
        return processor.tokenizer, model, target_id, processor

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
            return tokenizer, model, model_id, None
        except Exception as error:
            print(f"[bench] failed to load {model_id}: {error}", flush=True)
            last_error = error
    raise RuntimeError(f"could not load target; last error: {last_error}")


def load_vl_samples(tsv_path: str, num_samples: int):
    """读 VLMEvalKit 格式 tsv 前 num_samples 条, 返回 (question, PIL.Image).

    兼容两种图片存储: image_path 指向真实文件, 或 image 列存 base64.
    base64 图片很大, 需要提高 csv 字段上限.
    """
    import base64
    import io
    import os
    import sys

    from PIL import Image

    csv.field_size_limit(sys.maxsize)
    samples = []
    with open(tsv_path, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            image_path = row.get("image_path") or ""
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                encoded = row.get("image") or ""
                if not encoded:
                    continue
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            samples.append((row["question"], image))
            if len(samples) >= num_samples:
                break
    return samples


def build_vl_input(processor, question: str, image, device) -> tuple:
    """把图文样本转成 (input_ids, multimodal_inputs)."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    multimodal_inputs = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "mm_token_type_ids": inputs["mm_token_type_ids"],
    }
    return inputs["input_ids"], multimodal_inputs


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


def run_spec_prompt(
    gen, input_ids, max_new_tokens: int, eos, multimodal_inputs=None
) -> tuple:
    """跑一条 prompt 的投机解码, 返回 (输出 token 列表, 本轮计数差值, 墙钟)."""
    prompt_length = int(input_ids.shape[1])
    rounds_before = gen.n_rounds
    accepted_before = gen.n_accepted_tokens
    proposed_before = gen.n_proposed_draft_tokens
    tree_nodes_before = gen.n_tree_nodes

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = gen.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            multimodal_inputs=multimodal_inputs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    new_ids = output_ids[0, prompt_length:].tolist()
    rounds = gen.n_rounds - rounds_before
    accepted = gen.n_accepted_tokens - accepted_before
    proposed = gen.n_proposed_draft_tokens - proposed_before
    tree_nodes = gen.n_tree_nodes - tree_nodes_before
    return new_ids, rounds, accepted, proposed, tree_nodes, wall


def run_baseline_prompt(
    target, input_ids, max_new_tokens: int, eos, multimodal_inputs=None
) -> tuple:
    """跑一条 prompt 的 HF greedy baseline, 返回 (输出 token 列表, 墙钟).

    带 KV cache 的正常 generate, 与投机解码公平对比 TPOT.
    """
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
            **(multimodal_inputs or {}),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    new_ids = output[0, prompt_length:].tolist()
    return new_ids, wall


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGLE3 accept-rate + TPOT benchmark")
    parser.add_argument(
        "--mode",
        default="text",
        choices=["text", "vl"],
        help="一键预设: text=Llama3.1+固定文本 prompt, vl=Qwen3-VL+tsv 图文样本",
    )
    parser.add_argument("--target", default=None, help="覆盖预设的 target 模型")
    parser.add_argument("--eagle3-draft", default=None, help="覆盖预设的 draft 模型")
    parser.add_argument(
        "--tsv", default=DEFAULT_VL_TSV, help="vl 模式的图文数据集(VLMEvalKit tsv)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="追加解释推理的要求以拉长输出; 短答案测不出真实接受长度",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="bf16 尾数只有 7 位, 长输出下可能出现 argmax 平局翻转导致假发散",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument(
        "--topk", type=int, default=8, help="每个父节点展开的子候选数(树的宽度)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=8, help="draft 树的最大深度(根到叶的层数)"
    )
    parser.add_argument(
        "--max-tree-nodes",
        type=int,
        default=64,
        help="verify 节点预算; BFS 按创建顺序截断, EAGLE2 展开后按累计分数剪枝",
    )
    parser.add_argument(
        "--tree-mode",
        default="eagle2_batch",
        choices=["eagle2", "eagle2_batch", "bfs"],
        help="eagle2=逐节点 beam, eagle2_batch=按层 batch beam, bfs=原 BFS",
    )
    parser.add_argument(
        "--preconsume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draft prefill 预消费 target 确定的首 token + 末位 feature; --no-preconsume 关闭",
    )
    parser.add_argument(
        "--skip-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify 顺带提交接受路径 KV/feature, 省掉每轮 target replay forward; --no-skip-replay 回退对照",
    )
    parser.add_argument("--verify-backend", default="triton_tree")
    parser.add_argument(
        "--verify-mode",
        default="full_model_tree",
        choices=["full_model_tree", "reference_paths"],
        help="reference_paths 用逐路径 causal forward 做对照, 慢但可排除树掩码/位置的嫌疑",
    )
    command_line_args = parser.parse_args()

    mode = command_line_args.mode
    preset = MODE_PRESETS[mode]
    target_id_arg = command_line_args.target or preset["target"]
    draft_id = command_line_args.eagle3_draft or preset["draft"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[command_line_args.dtype]

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
    )

    tokenizer, target, target_id, processor = load_target(
        target_id_arg, device, dtype, mode
    )
    # Qwen3-VL 的层数等维度在 text_config 里, 不在顶层 config.
    text_config = getattr(target.config, "text_config", target.config)
    num_hidden_layers = int(text_config.num_hidden_layers)
    eos = tokenizer.eos_token_id

    # 构造并加载真实 EAGLE3 draft 权重.
    draft_config_path = hf_hub_download(draft_id, "config.json")
    draft_vocab_size = int(
        json.load(open(draft_config_path)).get("draft_vocab_size", 32000)
    )
    draft_weights_path = hf_hub_download(draft_id, preset["draft_weight_file"])

    config = Eagle3SglConfig(
        topk=command_line_args.topk,
        num_steps=command_line_args.num_steps,
        max_tree_nodes=command_line_args.max_tree_nodes,
        draft_tree_mode=command_line_args.tree_mode,
        preconsume_first_token=command_line_args.preconsume,
        skip_replay=command_line_args.skip_replay,
        verify_mode=command_line_args.verify_mode,
        verify_attn_backend=command_line_args.verify_backend,
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

    # 统一成 (input_ids, multimodal_inputs) 列表, 文本模式的 multimodal_inputs 为 None.
    samples = []
    if mode == "vl":
        for question, image in load_vl_samples(
            command_line_args.tsv, command_line_args.num_prompts
        ):
            prompt_text = question + (
                EXPLAIN_SUFFIX if command_line_args.explain else ""
            )
            samples.append(build_vl_input(processor, prompt_text, image, device))
    else:
        for prompt in FIXED_PROMPTS[: command_line_args.num_prompts]:
            prompt_text = prompt + (EXPLAIN_SUFFIX if command_line_args.explain else "")
            samples.append((build_input_ids(tokenizer, prompt_text, device), None))

    # 预热: 触发 CUDA kernel 编译/autotune, 不计入统计.
    warmup_ids, warmup_mm = samples[0]
    with torch.inference_mode():
        generator.generate(
            warmup_ids,
            max_new_tokens=8,
            eos_token_id=eos,
            multimodal_inputs=warmup_mm,
        )
    # HF baseline 也单独预热, 避免第一次 generate 的 cache 分配和冷启动计入对比.
    with torch.inference_mode():
        target.generate(
            warmup_ids,
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=eos,
            **(warmup_mm or {}),
        )
    generator.n_rounds = 0
    generator.n_accepted_tokens = 0
    generator.n_proposed_draft_tokens = 0
    generator.n_tree_nodes = 0
    print("[bench] warmup done (spec + HF baseline)", flush=True)

    summary = BenchSummary()
    for index, (input_ids, multimodal_inputs) in enumerate(samples):
        spec_ids, rounds, accepted, proposed, tree_nodes, spec_wall = run_spec_prompt(
            generator,
            input_ids,
            command_line_args.max_new_tokens,
            eos,
            multimodal_inputs,
        )
        baseline_ids, baseline_wall = run_baseline_prompt(
            target,
            input_ids,
            command_line_args.max_new_tokens,
            eos,
            multimodal_inputs,
        )

        # 逐 token 对比 greedy 无损性. 只比共同长度, 且遇到 EOS 就截断:
        # EOS 之后的内容没有语义, 不该判为发散.
        common_length = min(len(spec_ids), len(baseline_ids))
        for position, token in enumerate(spec_ids[:common_length]):
            if token == eos:
                common_length = position
                break
        lossless = spec_ids[:common_length] == baseline_ids[:common_length]

        result = PromptResult(
            new_tokens=len(spec_ids),
            rounds=rounds,
            accepted_tokens=accepted,
            proposed_draft_tokens=proposed,
            tree_nodes=tree_nodes,
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
        if not lossless:
            # 打印首个分叉位置与两侧文本, 便于区分逻辑 bug 和数值精度导致的 argmax 翻转.
            for position, (spec_token, base_token) in enumerate(
                zip(spec_ids[:common_length], baseline_ids[:common_length])
            ):
                if spec_token != base_token:
                    print(
                        f"     [diverge] at {position}: "
                        f"spec={spec_token}({tokenizer.decode([spec_token])!r}) "
                        f"base={base_token}({tokenizer.decode([base_token])!r})",
                        flush=True,
                    )
                    print(
                        f"     spec text: {tokenizer.decode(spec_ids[:common_length])!r}",
                        flush=True,
                    )
                    print(
                        f"     base text: {tokenizer.decode(baseline_ids[:common_length])!r}",
                        flush=True,
                    )
                    # 用干净整段 forward 复算分叉点的 logits: top1/top2 差值极小说明是
                    # argmax 平局被数值噪声翻转, 差值很大才说明 verify 逻辑错位.
                    diverge_prefix = torch.cat(
                        [
                            input_ids,
                            torch.tensor(
                                [baseline_ids[:position]],
                                device=device,
                                dtype=input_ids.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    forward_kwargs = {}
                    if multimodal_inputs is not None:
                        mm_token_type_ids = multimodal_inputs["mm_token_type_ids"]
                        pad_length = diverge_prefix.shape[1] - mm_token_type_ids.shape[1]
                        forward_kwargs = {
                            "pixel_values": multimodal_inputs["pixel_values"],
                            "image_grid_thw": multimodal_inputs["image_grid_thw"],
                            "mm_token_type_ids": torch.cat(
                                [
                                    mm_token_type_ids,
                                    mm_token_type_ids.new_zeros((1, pad_length)),
                                ],
                                dim=1,
                            ),
                        }
                    with torch.inference_mode():
                        diverge_logits = (
                            target(diverge_prefix, use_cache=False, **forward_kwargs)
                            .logits[0, -1]
                            .float()
                        )
                    top_values, top_indices = torch.topk(diverge_logits, 5)
                    gap = float(top_values[0] - top_values[1])
                    print(
                        f"     logits gap(top1-top2)={gap:.6f} top5="
                        + ", ".join(
                            f"{int(token)}({tokenizer.decode([int(token)])!r}):{float(value):.4f}"
                            for value, token in zip(top_values, top_indices)
                        ),
                        flush=True,
                    )
                    break

    # 汇总.
    total_new = summary.total("new_tokens")
    total_rounds = summary.total("rounds")
    total_accepted = summary.total("accepted_tokens")
    total_proposed = summary.total("proposed_draft_tokens")
    total_tree_nodes = summary.total("tree_nodes")
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
        f"mode={mode} dtype={command_line_args.dtype} explain={command_line_args.explain}",
        f"target={target_id} draft={draft_id}",
        f"num_prompts={len(summary.results)} max_new_tokens={command_line_args.max_new_tokens}",
        f"tree: mode={command_line_args.tree_mode} preconsume={command_line_args.preconsume} skip_replay={command_line_args.skip_replay} topk(width)={command_line_args.topk} num_steps(depth)={command_line_args.num_steps} max_tree_nodes={command_line_args.max_tree_nodes}",
        f"total_new_tokens={total_new} total_rounds={total_rounds}",
        f"accepted_tokens={total_accepted} proposed_draft_tokens={total_proposed}",
        f"tau(accept_length)={tau:.3f} proposed/round={proposed_per_round:.3f}",
        f"tree_nodes/round={total_tree_nodes / max(total_rounds, 1):.2f} (verify 实际算了这么多节点)",
        f"ms/round={1000.0 * total_spec_wall / max(total_rounds, 1):.2f}",
        f"spec:   TPOT={spec_tpot_ms:.2f}ms  throughput={spec_throughput:.2f} tok/s  wall={total_spec_wall:.2f}s",
        f"baseline: TPOT={baseline_tpot_ms:.2f}ms  throughput={baseline_throughput:.2f} tok/s  wall={total_baseline_wall:.2f}s",
        f"speedup={speedup:.2f}x  lossless(all)={all_lossless}",
        "=============================================================",
    ]
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
