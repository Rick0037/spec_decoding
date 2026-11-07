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
    tok = AutoTokenizer.from_pretrained(target_id)
    model = (
        AutoModelForCausalLM.from_pretrained(target_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    print(f"[eagle2-llama2] target ready: {target_id}", flush=True)
    return tok, model, target_id


def main() -> None:
    p = argparse.ArgumentParser(description="Real EAGLE2 on Llama-2-7B (base)")
    p.add_argument("--target", default="/root/autodl-tmp/code/LlmQT/Llama-2-7b-ms")
    p.add_argument("--eagle-draft", default="yuhuili/EAGLE-llama2-chat-7B")
    p.add_argument(
        "--prompt",
        default="Speculative decoding accelerates LLM inference by",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    # EAGLE2 官方推荐形态:cumulative 动态树,num_steps=5 / 32 节点
    # (比 EAGLE1 的 4/24 略大,体现动态剪枝的节点利用率优势)
    topk, num_steps, max_tree_nodes = 4, 5, 32
    tree_expand_mode, verify_mode, verify_attn_backend = (
        "cumulative",
        "full_model_tree",
        "auto",
    )
    perf_log = "perf.log"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    from huggingface_hub import hf_hub_download

    from spec_decoding.eagle2 import (
        Eagle2TreeConfig,
        Eagle2Generator,
        build_eagle2_draft,
        make_eagle2_draft_topk_fn,
    )

    tok, target, target_used = _load_target(args.target, device, dtype)

    # 官方 EAGLE draft 权重(单 bin);与 EAGLE1 同一份 checkpoint,结构通用
    draft_bin = hf_hub_download(args.eagle_draft, "pytorch_model.bin")
    draft = build_eagle2_draft(
        target, num_layers=1, draft_weights_path=draft_bin, device=device, dtype=dtype
    )
    draft_topk_fn = make_eagle2_draft_topk_fn(draft, topk=topk)

    cfg = Eagle2TreeConfig(
        topk=topk,
        num_steps=num_steps,
        max_tree_nodes=max_tree_nodes,
        tree_expand_mode=tree_expand_mode,
        verify_mode=verify_mode,
        verify_attn_backend=verify_attn_backend,
    )
    # 大模型上不逐节点刷新 target(只在根取一次 hidden),避免 O(树节点) 次 target 前向
    gen = Eagle2Generator(target, cfg, draft_topk_fn, refresh_target_each_node=False)

    # base 模型无对话格式,直接纯编码 prompt
    input_ids = tok(args.prompt, return_tensors="pt")["input_ids"].to(device)
    plen = int(input_ids.shape[1])
    eos = tok.eos_token_id

    # 贪心基线:干净整段前向(use_cache=False)逐步 argmax,与 verify 数值同源
    with torch.inference_mode():
        cur = input_ids.clone()
        greedy_ids = []
        for _ in range(args.max_new_tokens):
            nt = int(target(cur, use_cache=False).logits[0, -1].argmax())
            greedy_ids.append(nt)
            cur = torch.cat([cur, torch.tensor([[nt]], device=device)], dim=1)
            if eos is not None and nt == eos:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out_ids = gen.generate(
        input_ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    spec_cont = out_ids[0, plen:].tolist()

    n_new = len(spec_cont)
    accept_per_round = gen.n_accepted_tokens / max(1, gen.n_rounds)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.inference_mode():
        bg = target.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=eos,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    base_dt = time.time() - t1
    base_cont = bg[0, plen:].tolist()
    base_new = len(base_cont)

    def _first_div(a, b):
        mm = min(len(a), len(b))
        d = next((i for i in range(mm) if a[i] != b[i]), -1)
        return mm, d

    m_hf, div_hf = _first_div(spec_cont, base_cont)
    m_cl, div_cl = _first_div(spec_cont, greedy_ids)
    lossless = div_cl == -1  # 与干净贪心一致即为无损

    rec = (
        f"[eagle2] target={target_used} draft={args.eagle_draft} "
        f"verify={verify_mode}(backend={verify_attn_backend}) tree={tree_expand_mode} "
        f"topk={topk} num_steps={num_steps} max_nodes={max_tree_nodes} dtype=bfloat16\n"
        f"  new_tokens={n_new}  rounds={gen.n_rounds}  accept/round={accept_per_round:.2f}\n"
        f"  wall={dt:.2f}s  spec_tok/s={n_new / dt:.2f}\n"
        f"  lossless(vs clean-greedy)={lossless}  [vs clean: first_div={div_cl}/{m_cl}; "
        f"vs HF cached: first_div={div_hf}/{m_hf}]\n"
        f"  baseline(HF greedy) new_tokens={base_new}  wall={base_dt:.2f}s  "
        f"base_tok/s={base_new / base_dt:.2f}  speedup={(n_new/dt)/(base_new/base_dt):.2f}x\n"
    )
    mshow = min(len(greedy_ids), n_new)
    print("\n==================== EAGLE2 (Llama-2-7B base) ====================")
    print(rec, end="")
    print(
        "[greedy]",
        tok.decode(torch.tensor(greedy_ids[:mshow]), skip_special_tokens=True),
    )
    print(
        "[eagle2]",
        tok.decode(torch.tensor(spec_cont[:mshow]), skip_special_tokens=True),
    )
    print("===================================================================")

    with open(perf_log, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        f.write(rec)
        f.write("\n")
    print(f"[eagle2-llama2] perf appended to {os.path.abspath(perf_log)}")
    if lossless:
        print("[eagle2-llama2] OK (lossless vs clean greedy)")
    else:
        print(
            f"[eagle2-llama2] FAIL: diverged from clean greedy at idx {div_cl} "
            f"(accept/round={accept_per_round:.2f})"
        )


if __name__ == "__main__":
    main()
