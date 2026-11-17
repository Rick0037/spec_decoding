# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 外层调度:target 多层 act 条件 draft + 树 + verify → 多步生成。

典型调用栈(generate() 一轮 decode):

    generate()
      ├─ target prefill → past, recovery_logits, acts_all
      ├─ draft_model.prefill() → draft_state(AR 路径)
      └─ loop:
           generate_step_payload()
             ├─ _build_tree() → expand_draft_tree_ar
             └─ verify:
                  full_tree_verify_extend|triton  (verify_mode=full_model_tree)
                  或 _verify_path_clean * 叶子数   (verify_mode=reference_paths)
           target replay append → 更新 past / recovery_logits / acts_all
           draft_model.extend_tokens() → 更新 draft_state(AR 路径)

verify_mode=full_model_tree(默认):extend verify,复用前缀 KV,树掩码一次前向 L 节点。
verify_mode=reference_paths:逐叶干净前向,实现简单、便于对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import Eagle3SglConfig
from .hidden import select_eagle_acts_from_hidden_states
from .tree_draft import (
    TreeDraftResult,
    expand_draft_tree_ar,
    tree_draft_to_verify_layout,
)
from .tree_verify_attn import triton_extend_available
from .tree_verify_full import (
    full_tree_verify_extend,
    full_tree_verify_triton,
)


def _unwrap(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    if isinstance(m, PreTrainedModel):
        return m
    inner = getattr(m, "model", None)
    if inner is not None and isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("target must be HF PreTrainedModel or BaseModelForCausalLM")


def _num_hidden_layers(m: Union[PreTrainedModel, nn.Module]) -> int:
    """从 target config 推断 decoder 层数(兼容 num_hidden_layers / n_layer)."""
    cfg = getattr(_unwrap(m), "config", None)
    for attr in ("num_hidden_layers", "n_layer"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise ValueError("cannot infer num_hidden_layers from target config")


def _leaf_indices(parent: List[int]) -> List[int]:
    """返回树里所有叶子节点的下标(没有任何子节点的节点)."""
    n = len(parent)
    has_ch = [False] * n
    for i in range(n):
        p = parent[i]
        if p >= 0:
            has_ch[p] = True  # 标记 p 有孩子
    return [i for i in range(n) if not has_ch[i]]


def _path_tokens_to_leaf(parent: List[int], tok: List[int], leaf: int) -> List[int]:
    """从某叶子沿 parent 指针回溯到根,返回 根→叶 顺序的 token 列表."""
    out: List[int] = []
    k = leaf
    while k >= 0:
        out.append(tok[k])
        k = parent[k]
    out.reverse()  # 回溯是叶→根,反转成根→叶
    return out


# -----------------------------------------------------------------------------
# Verify(参考实现, 串行verify):用 一次干净整段前向(无共享增量 KV)核对一条 draft 路径.
# 位置对齐:logits[P-1+j] 是 path[j] 所在位置的 target greedy 预测:
#   - path[0] 的真值 = logits[P-1];
#   - 全中时 bonus = logits[P-1+len(path)].
# 用 use_cache=False 避免 HF DynamicCache 原地修改(把被拒绝的续写留进缓存).
# -----------------------------------------------------------------------------
@torch.inference_mode()
def _verify_path_clean(
    target_m: PreTrainedModel,
    full_prefix_ids: torch.LongTensor,
    path_tokens: List[int],
    device: torch.device,
) -> Tuple[int, int]:
    """对 prefix + path 做一次干净前向,返回 (match_len, next_token)。

    Parameters
    ----------
    target_m
        完整 HF CausalLM(带 lm_head)。
    full_prefix_ids
        当前已接受前缀 [1, P](prompt + 此前各轮 append)。
    path_tokens
        某条 draft 叶路径上的 token 列表(根→叶);空列表表示无前向 path,只取 greedy 下一 token。
    device
        计算设备。

    Returns
    -------
    match_len
        path 上连续被 target greedy 接受的长度。
    next_token
        全中时 = bonus(logits[P-1+len(path)] 的 argmax);否则 = 首个不匹配处 target 修正 token。
    """
    if not path_tokens:
        out = target_m(
            full_prefix_ids,
            attention_mask=torch.ones_like(full_prefix_ids, device=device),
            use_cache=False,
        )
        return 0, int(out.logits[0, -1, :].argmax(dim=-1).item())

    P = int(full_prefix_ids.shape[1])
    # 整段 [prefix | path] 一次 forward;不写入 runner 的 past(use_cache=False)
    full = torch.cat(
        [full_prefix_ids, torch.tensor([path_tokens], device=device, dtype=torch.long)],
        dim=1,
    )
    logits = target_m(
        full,
        attention_mask=torch.ones_like(full, device=device),
        use_cache=False,
    ).logits[0]
    # logits[P-1+j] = 在已见 prefix + path[:j]条件下对 path[j] 的预测
    mlen = 0
    corr: Optional[int] = None
    for j, t in enumerate(path_tokens):
        g = int(logits[P - 1 + j].argmax(dim=-1).item())
        if g == t:
            mlen += 1
        else:
            corr = g
            break
    if mlen == len(path_tokens):
        nxt = int(logits[P - 1 + len(path_tokens)].argmax(dim=-1).item())  # bonus
    else:
        nxt = int(corr)
    return mlen, nxt


@dataclass
class _DraftKvState:
    """保存 draft 的增量状态，供下一轮建树复用。"""

    kv: Any
    """draft 注意力层的 (key, value) 缓存；序列长度为 prefix_len - 1。"""
    prefix_len: int
    """当前已接受序列长度，与 out_ids.shape[1] 一致。"""
    root_logits: torch.Tensor
    """末位置的 draft logits [1, draft_vocab_size]，用于选择根层候选。"""
    root_hidden: torch.Tensor
    """末位置的 draft hidden [1, hidden_size]，作为各分支递归展开的初始条件。"""


# -----------------------------------------------------------------------------
# eagle3投机编排:draft 树 + target verify + 每轮投机后draft状态更新
# -----------------------------------------------------------------------------
class Eagle3SglGenerator:
    """
    EAGLE3 多层 act 条件 AR draft + top-k 树 + 树掩码 verify。

    draft 使用自己的 KV cache, 并沿树路径递归更新 hidden。
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        cfg: Eagle3SglConfig,
        draft_model,
    ) -> None:
        """保存 target、AR draft 和多层特征配置。"""
        self.target = target
        self.target_m = _unwrap(target)
        self.cfg = cfg
        self.draft_model = draft_model
        # EAGLE3 多层特征下标(draft 条件来自这些层的末 token hidden 拼接)
        self.eagle_layers: List[int] = cfg.resolve_eagle_layers(
            _num_hidden_layers(target)
        )
        # 诊断计数:投机采样的轮数,采纳 的token 数(accept/round = n_accepted_tokens / n_rounds)
        self.n_rounds = 0
        self.n_accepted_tokens = 0

    def _build_tree(
        self,
        prefix_ids: torch.LongTensor,
        draft_state: _DraftKvState,
    ) -> TreeDraftResult:
        """用 AR draft 的 KV 和递归 hidden 构造 top-k 树。"""
        if draft_state.prefix_len != int(prefix_ids.shape[1]):
            raise ValueError(
                f"draft_state.prefix_len {draft_state.prefix_len} != "
                f"prefix len {prefix_ids.shape[1]}"
            )
        return expand_draft_tree_ar(
            self.cfg,
            self.draft_model,
            draft_state.root_logits,
            draft_state.root_hidden,
            draft_state.kv,
            int(prefix_ids.shape[1]) - 1,
            device=prefix_ids.device,
        )

    # ------------------------------------------------------------------
    # 单轮:prefix + past → 建树 → verify → 得到本步要追加的 tokens
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_step_payload(
        self,
        full_prefix_ids: torch.LongTensor,
        past_before: Any,
        recovery_logits: Optional[torch.Tensor] = None,
        draft_state: Optional[_DraftKvState] = None,
    ) -> torch.LongTensor:
        """单轮投机，返回本轮采纳的 token [1, A]。

        Parameters
        ----------
        full_prefix_ids
            当前已接受序列 [1, P](prompt + 此前各轮 append)。
        past_before
            target 对 full_prefix_ids 的 KV cache;extend verify 在其上临时 extend 后 crop。
        recovery_logits
            前缀末位预测下一 token 的 logits [V];full_model_tree 必填(根层 parent→child 接受)。
        draft_state
            增量维护的 draft KV + root 输出,避免每轮 prefill 整段 prefix。
        """
        device = full_prefix_ids.device
        # ① AR 建树
        draft_res = self._build_tree(full_prefix_ids, draft_state)
        layout = tree_draft_to_verify_layout(draft_res)

        past_len = int(full_prefix_ids.shape[1])
        leaves = _leaf_indices(draft_res.parent)
        # 退化:树只有空或无叶节点时,等价于投机解码失败,直接 target生成下一 token
        if not leaves:
            _, nxt = _verify_path_clean(self.target_m, full_prefix_ids, [], device)
            return torch.tensor([[nxt]], device=device, dtype=torch.long)
        if self.cfg.verify_mode == "full_model_tree":
            if recovery_logits is None:
                raise ValueError("full_model_tree verify requires recovery_logits")
            # 复用前缀 KV 的 extend verify(只前向 L 个树节点,不重算前缀);
            # backend=triton_tree(或 auto+topk>1+triton 可用)→ Triton 树注意力 kernel;否则走HF自带的torch sdpa.
            backend = self.cfg.verify_attn_backend
            # auto 选 triton 还需 target 真在 GPU 上:triton_extend_available 只看
            # 机器有无 CUDA,不看张量设备,CPU 模型会误入 triton 路径。
            target_on_cuda = next(self.target_m.parameters()).device.type == "cuda"
            use_triton = backend == "triton_tree" or (
                backend == "auto"
                and self.cfg.topk > 1
                and target_on_cuda
                and triton_extend_available()
            )
            verify_fn = (
                full_tree_verify_triton if use_triton else full_tree_verify_extend
            )
            append_list = verify_fn(
                self.target_m,
                layout,
                past_before,
                past_len,
                recovery_logits,
                leaves=leaves,
                device=device,
            )
            return torch.tensor([append_list], device=device, dtype=torch.long)

        # reference_paths:每条叶路径一次完整 causal forward(串行verify),选 match 最长的叶(慢,易对照)
        best_m = -1
        best_next = 0
        best_leaf = 0
        for leaf in leaves:
            path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, leaf)
            mlen, nxt = _verify_path_clean(self.target_m, full_prefix_ids, path, device)
            # 接受数相同时保留排在前面的路径(与 extend verify 一致)
            if mlen > best_m:
                best_m, best_next, best_leaf = mlen, nxt, leaf

        path = _path_tokens_to_leaf(draft_res.parent, draft_res.token_ids, best_leaf)
        append_list = path[:best_m] + [best_next]
        return torch.tensor([append_list], device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # 多轮自回归:每轮 = generate_step_payload + 用 target 重放 append 以对齐 KV.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """多轮外层循环:prefill → 反复 generate_step_payload + target/draft replay。

        Parameters
        ----------
        input_ids
            Prompt [1, P0]。
        max_new_tokens
            最多新生成 token 数(按 append 总 token 计,非轮数)。
        eos_token_id
            若给定,前缀末或 append 中出现该 id 则提前停止。

        Returns
        -------
        out_ids
            prompt + 全部已采纳 token [1, P0 + total_new]。
        """
        device = input_ids.device
        out_ids = input_ids.clone()
        new_tok = 0
        # ② Target prefill:KV + 末位 recovery_logits + 全前缀 act
        att = torch.ones_like(out_ids, device=device)
        out0 = self.target_m(
            out_ids, attention_mask=att, use_cache=True, output_hidden_states=True
        )
        past = out0.past_key_values
        # * recovery_logits.shape -> [vob_size]
        recovery_logits = out0.logits[0, -1]
        #! 为什么 last_token_only = false 是因为实际上需要全部的feature 需要进去算kv cache
        acts_all = select_eagle_acts_from_hidden_states(
            out0.hidden_states, self.eagle_layers, last_token_only=False
        )
        # *------first init------
        # * acts_all `[B, S, n_acts*H]`，本脚本是 `[1, prompt_len, 3*4096]`
        # （Llama-3.1-8B，`eagle_layers=[2, 16, 29]`）。
        # ③ AR draft prefill,建立首轮使用的 KV 和递归 hidden
        # * d_logits  跟target 一样得token
        # * d_hidden 上面这个token得hidden size
        # * d_kv 仅仅有 prompt
        d_logits, d_hidden, d_kv = self.draft_model.prefill(out_ids, acts_all)
        # * prefill得初始状态
        draft_state = _DraftKvState(
            kv=d_kv,
            prefix_len=int(out_ids.shape[1]),
            root_logits=d_logits,
            root_hidden=d_hidden,
        )

        while new_tok < max_new_tokens:
            if eos_token_id is not None and out_ids[0, -1].item() == eos_token_id:
                break  # 已到 EOS,停止
            # ④ 单轮:AR 建树 → verify → append
            """
                [in]
                out_ids 首字输入
                past 历史kv cache
                recovery_logits target吐出来得首字
                draft_state draftm model 的首字结论
                [out]
                append 实际接受的字段
            """
            append = self.generate_step_payload(
                out_ids, past, recovery_logits, draft_state
            )
            # 把 append 重放进 target KV.带 past 时 attention_mask 必须覆盖 [past + append] 全长,
            # 否则 append 不会 attend 到前缀 → out_sync.logits(recovery_logits)错.
            # (extend verify 在 past 上临时 extend 后 crop,不污染 past;故 past 始终 = out_ids 的 KV.)
            full_attn = torch.ones(
                (1, int(past.get_seq_length()) + int(append.shape[1])),
                device=device,
                dtype=torch.long,
            )
            # * 实际上这里的 out_sync 的hidden state 就仅仅又三个接收的token的hidden state了
            #! 如果这里是全部接收的情况下, 最后一个bonus token 必须补算的, 如果是中间接收的话, 这部分有冗余的
            out_sync = self.target_m(
                append,
                past_key_values=past,
                use_cache=True,
                attention_mask=full_attn,
                output_hidden_states=True,
            )
            past = out_sync.past_key_values
            recovery_logits = out_sync.logits[0, -1]
            # * 只计算本轮新增加的 hidden states
            append_acts = select_eagle_acts_from_hidden_states(
                out_sync.hidden_states, self.eagle_layers, last_token_only=False
            )
            # * 把本轮接受位置的特征拼到完整特征序列末尾
            # import pdb

            # pdb.set_trace()
            acts_all = torch.cat([acts_all, append_acts], dim=1)
            # token t 使用 t-1 位置的特征，所以切片两端都减 1
            p_old = draft_state.prefix_len
            p_new = p_old + int(append.shape[1])
            #! p_new - 1 的切片不包括 p_new - 1, 直到 p_new - 2 就停止了, 这里搞了一波错位
            #! append token 下标：21、22、23
            #! 对应 feature 下标:20、21、22
            next_time_send_features = acts_all[:, p_old - 1 : p_new - 1, :]
            """
            接受路径的临时 KV 理论上可以复用, 
            但里面使用的是未被 target 校正的预测 hidden, 直接提交可能累积误差.
            所以这里选择对接受 token 重新计算.
            """
            d_logits, d_hidden, d_kv = self.draft_model.extend_tokens(
                append,
                next_time_send_features,
                draft_state.kv,
                p_old - 1,
            )
            draft_state = _DraftKvState(
                kv=d_kv,
                prefix_len=p_new,
                root_logits=d_logits,
                root_hidden=d_hidden,
            )
            # *--- 这里才最终接收
            out_ids = torch.cat([out_ids, append], dim=1)
            new_tok += int(append.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(append.numel())
            if eos_token_id is not None and (append == eos_token_id).any():
                break
        return out_ids
