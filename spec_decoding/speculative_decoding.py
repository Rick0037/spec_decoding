# SPDX-License-Identifier: Apache-2.0
"""
Greedy speculative decoding(线性链 / topk=1)。

核心思想
--------
1. 两阶段分工:Draft 用小模型快速「猜」后续若干个 token;Target 用大
   模型只做一次并行前向去「验」这些猜测是否与真实分布一致。算力从「每步
   只跑大模型」转为「多猜 + 少次大模型」,从而降低 decode 延迟。

2. 验证即真值:在 greedy 下,对每个候选位置比较
   argmax(target_logits) 与 draft 提案;一致则接受,不一致则在该
   位置采用 target 的 argmax(等价于无投机时大模型自己会采的 token),
   因而与「不用投机、逐步 greedy」分布一致(无损)。

PS: 这套“不一致就直接用 target 的” 只在 greedy 下成立 。
如果是采样(temperature / top-p) draft token 和 target 的采样结果不一致时不能直接替换


3. Bonus 步:若 r 个 draft token 全部被 target 接受,说明大模型在
   这 r 步上与 draft 完全对齐;此时还需再采 第 r+1 个 token(由 verify
   forward 在最后一个新位置上的 logits 给出),否则会少生成一步。

4. 范围:本文件只实现 topk=1 的链式 greedy 投机(draft 只吃 token、
   验证一条路径);topk>1 的树形候选需要树状 mask 与多路 verify,不在本文件内。

target / draft 须为 HF CausalLM,词表需一致。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def _unwrap_pretrain_model(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    """Require a raw HF CausalLM (*ForCausalLM) that produces .logits."""
    if isinstance(m, PreTrainedModel):
        return m
    raise TypeError("target/draft must be HF PreTrainedModel")


@dataclass
class SpeculativeConfig:
    """
    parameter
    """

    num_draft_tokens: int = 4
    eos_token_id: Optional[int] = None  #! 结束符号对应的token id
    pad_token_id: Optional[int] = None  #! padding 对应的token id
    topk: int = 1  # topk == 1 Greedy

    def __post_init__(self) -> None:
        if self.topk != 1:
            raise ValueError(
                "chain speculative decode only supports topk=1; "
                "tree speculation requires tree attention."
            )


@torch.inference_mode()
def speculative_greedy_decode(
    target: Union[PreTrainedModel, nn.Module],
    draft: Union[PreTrainedModel, nn.Module],
    input_ids: torch.LongTensor,  # * [1, seq]
    *,
    max_new_tokens: int,
    config: SpeculativeConfig,
) -> torch.LongTensor:
    """
    投机解码主循环。
    外层 while 每一轮等价于一次「draft_forward → verify」:
    先扩展候选序列,再由 target并行核对;
    接受的 token 写回out_ids,
    并刷新两边的 KV(此处 draft 采用整段重算以对齐前缀,属正确性优先实现)。
    """
    target_m = _unwrap_pretrain_model(target)
    draft_m = _unwrap_pretrain_model(draft)
    device = input_ids.device
    # * no use
    num_draft_tokens = config.num_draft_tokens
    eos = config.eos_token_id
    # * [batch_size, seq_lens]
    out_ids = input_ids.clone()
    new_count = 0
    # * 如果没有完成所有output的时候就一直投机, 直到投机结束

    while new_count < max_new_tokens:
        if eos is not None and out_ids[0, -1].item() == eos:
            break
        # ---------- Phase A — Draft(快路径)----------
        # 在当前完整的prompt上跑 draft,logits[:, -1, :] 即下一个 token分布;
        # 再自回归 r-1 步得到链式候选 d_0..d_{r-1}.draft 仅影响速度,其质量好坏
        # 都会被下面的 target verify 纠正,故 draft 侧 KV 实现不影响最终正确性.
        att_full = torch.ones_like(out_ids, device=device)
        # * prefill token
        pref_d = draft_m(out_ids, attention_mask=att_full, use_cache=True)
        # ! cur_past_d 实际上是一个引用, 这种dynamic cache
        cur_past_d = pref_d.past_key_values
        # * logits.shape=[batch, seq_len, vocab_size]
        # * pref_d -> seqlen 维度的最后一个, 在这个维度上取 dim= -1
        draft_tokens: list[int] = [int(pref_d.logits[:, -1, :].argmax(dim=-1).item())]

        # * draft_tokens 逐渐append, 这里取最后一个
        next_input_token = torch.tensor(
            [[draft_tokens[-1]]], device=device, dtype=torch.long
        )

        for _ in range(num_draft_tokens - 1):
            if eos is not None and draft_tokens[-1] == eos:
                break
            step = draft_m(
                next_input_token,
                past_key_values=cur_past_d,  # decoding需要kv cache提速
                use_cache=True,
                attention_mask=torch.ones(1, 1, device=device, dtype=torch.long),
            )
            nxt = int(step.logits[:, -1, :].argmax(dim=-1).item())
            draft_tokens.append(nxt)
            # * 这里才有第一个输入才加进去
            cur_past_d = step.past_key_values
            next_input_token = torch.tensor([[nxt]], device=device, dtype=torch.long)

        g = len(draft_tokens)
        # ---------- Phase B — Verify(真路径,一次并行,无共享增量 KV)----------
        # forward prefill + g 个候选;
        # use_cache=False 避免 HF DynamicCache的原地修改:
        # 若把同一个 cache 先喂 verify 再喂 draft 会把(含被拒绝的)draft KV 追加进去,后续步骤就建立在被污染的 KV 上导致输出错乱.
        # 整段前向里第 prefix_len-1+j 个位置的 logits 即 d_j 所在位置的 target greedy:
        #   - d_0 的真值 = logits[prefix_len-1](吃完真实前缀后的预测);
        #   - 全accept 时 bonus = logits[prefix_len-1+g](最后一格).

        # * draft_batch.shape = [1, g]
        draft_batch = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        # * 本次 out_ids.shape =  [batch_size, seq_lens]
        prefix_len = int(out_ids.shape[1])
        full_tokens = torch.cat([out_ids, draft_batch], dim=1)
        output_full = target_m(
            full_tokens,
            attention_mask=torch.ones_like(full_tokens, device=device),
            # 避免 HF DynamicCache 的原地修改追加没有被接受的draft kv,use_cache=false的代价是target 每轮 O(L) 重算,没有 KV 复用的加速
            use_cache=False,
        )
        # * output_full.shape -> [1, prefix_len + g, vocab]
        logits_full = output_full.logits[0]  # [prefix_len + g, vocab]
        target_preds = []
        for j in range(g):
            target_preds.append(logits_full[prefix_len - 1 + j].argmax(dim=-1).item())

        accept_len = 0
        for j in range(g):
            if target_preds[j] == draft_tokens[j]:
                accept_len += 1
            else:
                break
        # * 当 target_preds[j] == draft_tokens[j] 时候接受
        # ---------- Phase C — Accept / 纠正(保证与无投机 greedy 同分布)----------
        if accept_len == g:
            # 全中:再取 bonus(吃完 d_{g-1} 后那一格的 target greedy).
            # *最后一个
            bonus = int(logits_full[prefix_len - 1 + g].argmax(dim=-1).item())
            append = draft_tokens + [bonus]
        else:
            # 首错位 accept_len:接受前 accept_len 个 draft,该位改用 target greedy.
            append = draft_tokens[:accept_len] + [target_preds[accept_len]]

        out_ids = torch.cat(
            [out_ids, torch.tensor([append], device=device, dtype=torch.long)],
            dim=1,  # * 在sequence上进行append
        )
        new_count += len(append)

        if eos is not None and any(t == eos for t in append):
            break

    return out_ids


def speculative_greedy_decode_from_strings(
    target: Union[PreTrainedModel, nn.Module],
    draft: Union[PreTrainedModel, nn.Module],
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    config: Optional[SpeculativeConfig] = None,
) -> Tuple[torch.LongTensor, str]:
    """封装 encode → speculative_greedy_decode → decode;便于脚本演示."""
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(_unwrap_pretrain_model(target).parameters()).device
    input_ids = enc["input_ids"].to(device)
    if config is None:
        config = SpeculativeConfig()
    if (
        config.eos_token_id is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        # 与 HF generate 行为对齐:未显式配置时用 tokenizer 的 eos,便于早停
        config = replace(config, eos_token_id=tokenizer.eos_token_id)
    out = speculative_greedy_decode(
        target, draft, input_ids, max_new_tokens=max_new_tokens, config=config
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return out, text
