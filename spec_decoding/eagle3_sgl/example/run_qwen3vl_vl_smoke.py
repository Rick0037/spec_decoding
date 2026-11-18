# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL 真多模态输入冒烟(第 1 步: 只验证带图 target forward + 干净 greedy).

不接 draft. 目的是确认图像管道(image processor + vision embedding + MRoPE 位置)
能正常出图描述, 并建立后续投机解码对比用的 ground truth.

数据: VLMEvalKit 格式 tsv(question/answer/image_path 列) + 本地图像文件.
默认用 POPE_local.tsv(yes/no 短答案, 生成 token 少, 便于快速验证).

跑法:
    HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=. \
    .venv/bin/python spec_decoding/eagle3_sgl/example/run_qwen3vl_vl_smoke.py --num-samples 3
"""

from __future__ import annotations

import argparse
import csv

import torch
from PIL import Image


def load_pope_samples(tsv_path: str, num_samples: int):
    """读 POPE tsv 的前 num_samples 条, 返回 (question, answer, image_path) 列表."""
    samples = []
    with open(tsv_path, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            samples.append((row["question"], row["answer"], row["image_path"]))
            if len(samples) >= num_samples:
                break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL multimodal input smoke")
    parser.add_argument(
        "--target", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--tsv", default="/root/autodl-tmp/LMUData/POPE_local.tsv")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    command_line_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[vl-smoke] loading {command_line_args.target} ...", flush=True)
    processor = AutoProcessor.from_pretrained(command_line_args.target)
    target = (
        AutoModelForImageTextToText.from_pretrained(
            command_line_args.target, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )

    samples = load_pope_samples(command_line_args.tsv, command_line_args.num_samples)
    print(f"[vl-smoke] {len(samples)} samples", flush=True)

    for index, (question, answer, image_path) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")
        # Qwen3-VL chat message: 图像占位 + 文本问题.
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

        with torch.inference_mode():
            output = target.generate(
                **inputs,
                max_new_tokens=command_line_args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        model_answer = processor.decode(generated_ids, skip_special_tokens=True)

        print(f"\n--- sample {index} ---")
        print(f"image:    {image_path}")
        print(f"question: {question}")
        print(f"truth:    {answer}")
        print(f"model:    {model_answer.strip()}")

    print("\n[vl-smoke] OK (multimodal forward works)")


if __name__ == "__main__":
    main()
