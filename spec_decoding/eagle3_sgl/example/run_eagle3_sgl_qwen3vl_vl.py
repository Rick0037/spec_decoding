# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B + EAGLE3 真多模态(带图)端到端.

target: 本地 Qwen3-VL-8B-Instruct, 走完整多模态 forward(vision embedding + MRoPE).
draft:  taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3.

与纯文本的区别:
- prefill 传入 pixel_values/image_grid_thw/mm_token_type_ids, target 自动算 MRoPE.
- draft prefill 用 get_rope_index 的三维 position(图像段是三维网格).
- 新生成的都是文本 token, 三轴 position 相同, 只在一维位置上加 rope_delta 偏移.

verify 走 eager(HF sdpa + 4D 树 mask). baseline 用带图干净整段 greedy 做同源 lossless 校验.

跑法:
    HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=. \
    .venv/bin/python spec_decoding/eagle3_sgl/example/run_eagle3_sgl_qwen3vl_vl.py --num-samples 3
"""

from __future__ import annotations

import argparse
import csv

import torch
from PIL import Image


def load_vl_samples(tsv_path: str, num_samples: int):
    """读 VLMEvalKit 格式 tsv 前 num_samples 条, 返回 (question, answer, PIL.Image).

    兼容两种图片存储: image_path 指向真实文件, 或 image 列存 base64.
    这些数据集的 base64 图片很大, 需要提高 csv 字段上限.
    """
    import base64
    import io
    import os
    import sys

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
            samples.append((row["question"], row.get("answer", ""), image))
            if len(samples) >= num_samples:
                break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGLE3 on Qwen3-VL (multimodal)")
    parser.add_argument(
        "--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument(
        "--eagle3-draft", default="taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"
    )
    parser.add_argument("--tsv", default="/root/autodl-tmp/LMUData/POPE_local.tsv")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--verify-backend", default="eager")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="要求模型解释推理过程而非只给答案; 短答案数据集测不出接受长度, 需长输出",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="float32 用于排查长序列 argmax 平局翻转(显存翻倍)",
    )
    command_line_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[command_line_args.dtype]

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from spec_decoding.eagle3_sgl import (
        Eagle3SglConfig,
        Eagle3SglGenerator,
        build_eagle3_sgl_draft,
    )

    print(f"[vl] loading target {command_line_args.target} ...", flush=True)
    processor = AutoProcessor.from_pretrained(command_line_args.target)
    target = (
        AutoModelForImageTextToText.from_pretrained(
            command_line_args.target, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    text_config = getattr(target.config, "text_config", target.config)
    num_hidden_layers = int(text_config.num_hidden_layers)

    draft_weights_path = hf_hub_download(
        command_line_args.eagle3_draft, "model.safetensors"
    )
    config = Eagle3SglConfig(
        topk=8,
        num_steps=8,
        max_tree_nodes=64,
        verify_mode="full_model_tree",
        verify_attn_backend=command_line_args.verify_backend,
    )
    eagle_layers = config.resolve_eagle_layers(num_hidden_layers)
    draft = build_eagle3_sgl_draft(
        target,
        eagle_layers,
        draft_vocab_size=32000,
        draft_weights_path=draft_weights_path,
        device=device,
        dtype=dtype,
    )
    generator = Eagle3SglGenerator(target, config, draft)
    print(
        f"[vl] num_layers={num_hidden_layers} eagle_layers={eagle_layers}", flush=True
    )

    samples = load_vl_samples(command_line_args.tsv, command_line_args.num_samples)
    all_lossless = True
    total_rounds = 0
    total_accepted = 0

    for index, (question, truth, image) in enumerate(samples):
        # 计数器是实例级累计, 记录快照以算出单条样本的统计.
        rounds_before = generator.n_rounds
        accepted_before = generator.n_accepted_tokens
        prompt_text = question
        if command_line_args.explain:
            prompt_text = (
                f"{question}\nExplain your reasoning step by step before giving "
                "the final answer."
            )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_length = int(input_ids.shape[1])
        multimodal_inputs = {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
        }

        # 投机解码(带图).
        output_ids = generator.generate(
            input_ids,
            max_new_tokens=command_line_args.max_new_tokens,
            multimodal_inputs=multimodal_inputs,
        )
        spec_new_ids = output_ids[0, prompt_length:].tolist()

        # 带图干净整段 greedy baseline(与 verify 同源).
        with torch.inference_mode():
            current_ids = input_ids.clone()
            current_mm = multimodal_inputs["mm_token_type_ids"].clone()
            baseline_new_ids = []
            for _ in range(command_line_args.max_new_tokens):
                next_token = int(
                    target(
                        current_ids,
                        pixel_values=multimodal_inputs["pixel_values"],
                        image_grid_thw=multimodal_inputs["image_grid_thw"],
                        mm_token_type_ids=current_mm,
                        use_cache=False,
                    )
                    .logits[0, -1]
                    .argmax()
                )
                baseline_new_ids.append(next_token)
                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_token]], device=device)], dim=1
                )
                current_mm = torch.cat(
                    [
                        current_mm,
                        torch.zeros((1, 1), dtype=current_mm.dtype, device=device),
                    ],
                    dim=1,
                )

        common_length = min(len(spec_new_ids), len(baseline_new_ids))
        # 出现 EOS 后只比较到 EOS 之前: EOS 之后的内容没有语义, 不该判为发散.
        eos_token_id = getattr(processor, "tokenizer", processor).eos_token_id
        for position, token in enumerate(spec_new_ids[:common_length]):
            if token == eos_token_id:
                common_length = position
                break
        lossless = spec_new_ids[:common_length] == baseline_new_ids[:common_length]
        all_lossless = all_lossless and lossless
        sample_rounds = generator.n_rounds - rounds_before
        sample_accepted = generator.n_accepted_tokens - accepted_before
        total_rounds += sample_rounds
        total_accepted += sample_accepted
        accept = sample_accepted / max(1, sample_rounds)

        print(f"\n--- sample {index} ---")
        print(f"question: {question}")
        print(f"truth:    {truth}")
        print(
            f"spec:     {processor.decode(spec_new_ids, skip_special_tokens=True).strip()}"
        )
        print(
            f"baseline: {processor.decode(baseline_new_ids, skip_special_tokens=True).strip()}"
        )
        print(
            f"lossless={lossless} new={len(spec_new_ids)} "
            f"rounds={sample_rounds} accept/round={accept:.2f}"
        )
        if not lossless:
            for pos, (spec_token, base_token) in enumerate(
                zip(spec_new_ids, baseline_new_ids)
            ):
                if spec_token != base_token:
                    print(f"[diverge] at {pos}: spec={spec_token} base={base_token}")
                    break

    print(f"\n[vl] lossless(all)={all_lossless}")
    print(
        f"[vl] total rounds={total_rounds} accepted={total_accepted} "
        f"accept/round={total_accepted / max(1, total_rounds):.2f}"
    )
    assert all_lossless, "multimodal EAGLE3 diverged from baseline"
    print("[vl-e2e] OK (lossless)")


if __name__ == "__main__":
    main()
