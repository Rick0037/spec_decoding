#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
真·EAGLE2 端到端(Llama-2-7B base):本地 target + 官方 EAGLE draft,性能写入 perf.log。

EAGLE2 与 EAGLE1 共用同一份 draft checkpoint(结构一致:fc(2H→H) + 1 层 decoder
+ 复用 target embedding/lm_head);EAGLE2 只在推理期改进树展开:累计 log 概率
beam + 按分数全局剪枝(tree_expand_mode=cumulative),节点利用率高于静态树。

模型:
- target:/root/autodl-tmp/code/LlmQT/Llama-2-7b-ms(base 版,非 chat)。
- draft :yuhuili/EAGLE-llama2-chat-7B(chat 版训练的 draft;配 base 结构兼容,
  接受率会比配 chat 版略低——base 续写分布与 draft 学到的对话分布有偏差)。

跑法(单卡 24GB,bf16 下 7B≈14GB;树/verify 配置已固定):
    HF_HOME=/root/autodl-tmp/hf_cache PYTHONPATH=. python3 -m spec_decoding.eagle2.example.run_eagle2_llama2
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def _load_target(target_id: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eagle2-llama2] loading target {target_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = (
        AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    print(f"[eagle2-llama2] target ready: {target_id}", flush=True)
    return tokenizer, model, target_id


def _first_divergence(reference_tokens, other_tokens):
    """返回 (共同前缀长度, 首个不一致位置);位置为 -1 表示完全一致。"""
    common_length = min(len(reference_tokens), len(other_tokens))
    divergence_index = next(
        (
            index
            for index in range(common_length)
            if reference_tokens[index] != other_tokens[index]
        ),
        -1,
    )
    return common_length, divergence_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Real EAGLE2 on Llama-2-7B (base)")
    parser.add_argument("--target", default="/root/autodl-tmp/code/LlmQT/Llama-2-7b-ms")
    parser.add_argument("--eagle-draft", default="yuhuili/EAGLE-llama2-chat-7B")
    parser.add_argument(
        "--prompt",
        default="Speculative decoding accelerates LLM inference by",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    command_line_args = parser.parse_args()

    # EAGLE2 官方推荐 4/5/32; 这里放大到 8/8/128, 让 verify 的 L 超过 BLOCK_M=64
    topk, num_steps, max_tree_nodes = 4, 5, 32
    # topk, num_steps, max_tree_nodes = 8, 8, 128
    tree_expand_mode, verify_mode, verify_attn_backend = (
        "cumulative",
        "full_model_tree",
        "auto",
    )
    perf_log_path = "perf.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle2 import (
        Eagle2TreeConfig,
        Eagle2Generator,
        build_eagle2_draft,
        make_eagle2_draft_topk_fn,
    )

    tokenizer, target, target_repo = _load_target(
        command_line_args.target, device, dtype
    )

    # 官方 EAGLE draft 权重(单 bin);与 EAGLE1 同一份 checkpoint,结构通用
    draft_weights_path = hf_hub_download(
        command_line_args.eagle_draft, "pytorch_model.bin"
    )

    draft_model = build_eagle2_draft(
        target,
        num_layers=1,
        draft_weights_path=draft_weights_path,
        device=device,
        dtype=dtype,
    )
    # 把 draft model 包装成 topk 回调
    draft_topk_fn = make_eagle2_draft_topk_fn(draft_model, topk=topk)

    generator_config = Eagle2TreeConfig(
        topk=topk,  # 每个节点展开的候选数
        num_steps=num_steps,  # 树的最大深度
        max_tree_nodes=max_tree_nodes,  # 整棵树最多保留的节点数
        tree_expand_mode=tree_expand_mode,  # cumulative(eagle2) 或 static(eagle1)
        verify_mode=verify_mode,  # 一次树状 verify 或逐路径多次 verify
        verify_attn_backend=verify_attn_backend,  # verify 用的 attention 后端
    )
    # 大模型上不逐节点刷新 target(只在根取一次 hidden),避免 O(树节点) 次 target 前向
    generator = Eagle2Generator(
        target, generator_config, draft_topk_fn, refresh_target_each_node=False
    )

    # base 模型无对话格式,直接纯编码 prompt
    input_ids = tokenizer(command_line_args.prompt, return_tensors="pt")[
        "input_ids"
    ].to(device)
    prefix_length = int(input_ids.shape[1])
    eos_token_id = tokenizer.eos_token_id

    # 贪心基线:干净整段前向(use_cache=False)逐步 argmax,与 verify 数值同源
    with torch.inference_mode():
        greedy_input_ids = input_ids.clone()
        greedy_generated_tokens = []
        for _ in range(command_line_args.max_new_tokens):
            next_token_id = int(
                target(greedy_input_ids, use_cache=False).logits[0, -1].argmax()
            )
            greedy_generated_tokens.append(next_token_id)
            greedy_input_ids = torch.cat(
                [greedy_input_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1,
            )
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    speculative_start_time = time.time()
    speculative_output_ids = generator.generate(
        input_ids,
        max_new_tokens=command_line_args.max_new_tokens,
        eos_token_id=eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    speculative_elapsed_seconds = time.time() - speculative_start_time
    speculative_generated_tokens = speculative_output_ids[0, prefix_length:].tolist()

    num_generated_tokens = len(speculative_generated_tokens)
    accepted_tokens_per_round = generator.n_accepted_tokens / max(1, generator.n_rounds)

    # HF 标准贪心(use_cache=True)作为耗时基线
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_start_time = time.time()
    with torch.inference_mode():
        baseline_output_ids = target.generate(
            input_ids,
            max_new_tokens=command_line_args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_elapsed_seconds = time.time() - baseline_start_time
    baseline_generated_tokens = baseline_output_ids[0, prefix_length:].tolist()
    baseline_num_generated_tokens = len(baseline_generated_tokens)

    hf_common_length, hf_divergence_index = _first_divergence(
        speculative_generated_tokens, baseline_generated_tokens
    )
    greedy_common_length, greedy_divergence_index = _first_divergence(
        speculative_generated_tokens, greedy_generated_tokens
    )
    is_lossless = greedy_divergence_index == -1  # 与干净贪心一致即为无损

    report = (
        f"[eagle2] target={target_repo} draft={command_line_args.eagle_draft} "
        f"verify={verify_mode}(backend={verify_attn_backend}) tree={tree_expand_mode} "
        f"topk={topk} num_steps={num_steps} max_nodes={max_tree_nodes} dtype=bfloat16\n"
        f"  new_tokens={num_generated_tokens}  rounds={generator.n_rounds}  "
        f"accept/round={accepted_tokens_per_round:.2f}\n"
        f"  wall={speculative_elapsed_seconds:.2f}s  "
        f"spec_tok/s={num_generated_tokens / speculative_elapsed_seconds:.2f}\n"
        f"  lossless(vs clean-greedy)={is_lossless}  "
        f"[vs clean: first_div={greedy_divergence_index}/{greedy_common_length}; "
        f"vs HF cached: first_div={hf_divergence_index}/{hf_common_length}]\n"
        f"  baseline(HF greedy) new_tokens={baseline_num_generated_tokens}  "
        f"wall={baseline_elapsed_seconds:.2f}s  "
        f"base_tok/s={baseline_num_generated_tokens / baseline_elapsed_seconds:.2f}  "
        f"speedup={(num_generated_tokens/speculative_elapsed_seconds)/(baseline_num_generated_tokens/baseline_elapsed_seconds):.2f}x\n"
    )
    num_compare_tokens = min(len(greedy_generated_tokens), num_generated_tokens)
    print("\n==================== EAGLE2 (Llama-2-7B base) ====================")
    print(report, end="")
    print(
        "[greedy]",
        tokenizer.decode(
            torch.tensor(greedy_generated_tokens[:num_compare_tokens]),
            skip_special_tokens=True,
        ),
    )
    print(
        "[eagle2]",
        tokenizer.decode(
            torch.tensor(speculative_generated_tokens[:num_compare_tokens]),
            skip_special_tokens=True,
        ),
    )
    print("===================================================================")

    with open(perf_log_path, "a") as log_file:
        log_file.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        log_file.write(report)
        log_file.write("\n")
    print(f"[eagle2-llama2] perf appended to {os.path.abspath(perf_log_path)}")
    if is_lossless:
        print("[eagle2-llama2] OK (lossless vs clean greedy)")
    else:
        print(
            f"[eagle2-llama2] FAIL: diverged from clean greedy at idx "
            f"{greedy_divergence_index} "
            f"(accept/round={accepted_tokens_per_round:.2f})"
        )


if __name__ == "__main__":
    main()
