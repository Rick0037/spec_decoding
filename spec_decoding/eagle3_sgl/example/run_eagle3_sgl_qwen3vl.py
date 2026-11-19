# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B + EAGLE3 纯文本端到端.

target: 本地 Qwen3-VL-8B-Instruct(多模态模型, 但这里只用纯文本输入, 不走 vision).
draft:  taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3(公开 EAGLE3 权重, 含 fc/midlayer/lm_head/d2t).

文本-only 时 Qwen3-VL 的 MRoPE 退化为标准 RoPE(三轴 position 相同), 已用数值对拍确认
与 draft 内部 rope 完全一致(cos/sin 差异 0.0). 因此 draft 只需复用同一 rope_theta.

verify 默认走 eager(HF sdpa + 4D 树 mask), 避开 triton kernel 对层路径的假设.

跑法(镜像联网取 draft 权重):
    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=. \
    .venv/bin/python spec_decoding/eagle3_sgl/example/run_eagle3_sgl_qwen3vl.py \
        --max-new-tokens 48 --prompt "Explain speculative decoding in one sentence."
"""

from __future__ import annotations

import argparse
import time

import torch


def load_target(target_path: str, device, dtype):
    """加载 Qwen3-VL target; 纯文本下用其语言模型部分."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    model = None
    last_error = None
    # Qwen3-VL 是 image-text-to-text 架构, 优先用对应 Auto 类, 失败回退 CausalLM.
    for auto_cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers

            auto_cls = getattr(transformers, auto_cls_name)
            model = (
                auto_cls.from_pretrained(target_path, torch_dtype=dtype)
                .to(device)
                .eval()
            )
            print(f"[qwen3vl] loaded with {auto_cls_name}", flush=True)
            break
        except Exception as error:
            print(f"[qwen3vl] {auto_cls_name} failed: {error}", flush=True)
            last_error = error
    if model is None:
        raise RuntimeError(f"could not load target; last error: {last_error}")
    return tokenizer, model


def build_input_ids(tokenizer, prompt: str, device):
    """纯文本输入转 input_ids, 优先套 chat template."""
    if getattr(tokenizer, "chat_template", None):
        output = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = output if isinstance(output, torch.Tensor) else output["input_ids"]
        return input_ids.to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGLE3 on Qwen3-VL-8B (text-only)")
    parser.add_argument(
        "--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--eagle3-draft", default="taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"
    )
    parser.add_argument(
        "--prompt", default="Explain what speculative decoding is, in two sentences."
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--verify-backend", default="eager")
    command_line_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
    )

    print(f"[qwen3vl] loading target {command_line_args.target} ...", flush=True)
    tokenizer, target = load_target(command_line_args.target, device, dtype)
    text_config = getattr(target.config, "text_config", target.config)
    num_hidden_layers = int(text_config.num_hidden_layers)
    eos = tokenizer.eos_token_id

    # 下载并加载 EAGLE3 draft 权重(model.safetensors).
    draft_weights_path = hf_hub_download(
        command_line_args.eagle3_draft, "model.safetensors"
    )
    config = Eagle3SglConfig(
        topk=8,
        num_steps=8,
        max_tree_nodes=64,
        verify_attn_backend=command_line_args.verify_backend,
    )
    eagle_layers = config.resolve_eagle_layers(num_hidden_layers)
    print(
        f"[qwen3vl] num_layers={num_hidden_layers} eagle_layers={eagle_layers}",
        flush=True,
    )

    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=32000,
        draft_weights_path=draft_weights_path,
        device=device,
        dtype=dtype,
    )
    generator = Eagle3SglGenerator(target, config, draft)

    input_ids = build_input_ids(tokenizer, command_line_args.prompt, device)
    prompt_length = int(input_ids.shape[1])

    # 投机解码.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = generator.generate(
        input_ids, max_new_tokens=command_line_args.max_new_tokens, eos_token_id=eos
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    spec_wall = time.perf_counter() - start
    spec_new_ids = output_ids[0, prompt_length:].tolist()

    # 干净整段前向 greedy baseline: 每步 use_cache=False 重算整段, 与 verify 的 target
    # forward 同一数值源, 避免 bf16 下 KV cache 累积造成的 argmax 平局翻转误判.
    with torch.inference_mode():
        current_ids = input_ids.clone()
        baseline_new_ids = []
        for _ in range(command_line_args.max_new_tokens):
            next_token = int(target(current_ids, use_cache=False).logits[0, -1].argmax())
            baseline_new_ids.append(next_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)], dim=1
            )
            if eos is not None and next_token == eos:
                break

    common_length = min(len(spec_new_ids), len(baseline_new_ids))
    lossless = spec_new_ids[:common_length] == baseline_new_ids[:common_length]
    if not lossless:
        for index, (spec_token, base_token) in enumerate(
            zip(spec_new_ids, baseline_new_ids)
        ):
            if spec_token != base_token:
                print(
                    f"[diverge] first at {index}: spec={spec_token} "
                    f"({tokenizer.decode([spec_token])!r}) base={base_token} "
                    f"({tokenizer.decode([base_token])!r})"
                )
                break
    accept_per_round = generator.n_accepted_tokens / max(1, generator.n_rounds)
    proposed_per_round = generator.n_proposed_draft_tokens / max(1, generator.n_rounds)
    spec_tokens_per_second = len(spec_new_ids) / max(spec_wall, 1e-9)

    print("\n==================== Qwen3-VL EAGLE3 ====================")
    print(
        f"new_tokens={len(spec_new_ids)} rounds={generator.n_rounds} "
        f"accept/round={accept_per_round:.2f} proposed/round={proposed_per_round:.2f}"
    )
    print(f"wall={spec_wall:.2f}s spec_tok/s={spec_tokens_per_second:.2f} lossless={lossless}")
    print("[greedy]    ", tokenizer.decode(baseline_new_ids[:common_length], skip_special_tokens=True))
    print("[eagle3_sgl]", tokenizer.decode(spec_new_ids[:common_length], skip_special_tokens=True))
    print("========================================================")
    assert lossless, "Qwen3-VL EAGLE3 output diverged from greedy baseline"
    print("[qwen3vl-e2e] OK (lossless)")


if __name__ == "__main__":
    main()
