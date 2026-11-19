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

import contextlib
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .config import Eagle3SglConfig
from .hidden import select_eagle_acts_from_hidden_states
from .tree_draft import expand_draft_tree_ar
from .tree_verify_mask import TreeLayout
from .tree_verify_attn import triton_extend_available
from .tree_verify_full import (
    full_tree_verify_commit,
    full_tree_verify_extend,
    full_tree_verify_triton,
)


# 只有诊断脚本显式打开环境变量时才打 stage 标签, 正常 bench 保持零插桩.
_PROFILE = os.environ.get("EAGLE3_SGL_PROFILE", "0") == "1"


def _prof(name: str):
    if _PROFILE:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


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
    # 多模态模型(如 Qwen3-VL)的层数在 config.text_config, 纯文本模型直接在 config.
    cfg = getattr(cfg, "text_config", cfg)
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
    """draft 已消费的 target 位置数. 首轮预消费时比 out_ids 长度大 1, 否则一致."""
    root_logits: torch.Tensor
    """末位置的 draft logits [1, draft_vocab_size]，用于选择根层候选。"""
    root_hidden: torch.Tensor
    """末位置的 draft hidden [1, hidden_size]，作为各分支递归展开的初始条件。"""
    confirmed_root_token: Optional[int] = None
    """target 已确定且 draft 已消费的根 token(首轮 preconsume 或 skip_replay 稳态轮的 pending)."""


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
        # target 选中路径上的 draft 候选 token 总数, 不统计未选中的分支.
        self.n_proposed_draft_tokens = 0
        # 每轮实际送进 verify 的树节点总数(含未选中分支); verify 成本随这个数涨.
        self.n_tree_nodes = 0
        # 多模态时的 MRoPE 位置偏移(get_rope_index 的 rope_deltas); 纯文本恒为 0.
        self._rope_delta = 0

    def _build_tree(
        self,
        prefix_ids: torch.LongTensor,
        draft_state: _DraftKvState,
    ) -> TreeLayout:
        """用 AR draft 的 KV 和递归 hidden 构造 top-k 树."""
        prefix_length = int(prefix_ids.shape[1])
        expected_draft_prefix_length = prefix_length
        if draft_state.confirmed_root_token is not None:
            # 首轮预消费: draft 比 out_ids 多消费了一个确定 token.
            expected_draft_prefix_length += 1
        if draft_state.prefix_len != expected_draft_prefix_length:
            raise ValueError(
                f"draft_state.prefix_len {draft_state.prefix_len} != "
                f"expected draft prefix len {expected_draft_prefix_length}"
            )
        # * target 的 21 个 token, 对应 draft 的`21-1=20` 条 KV.
        # * 所以传进来的`draft_kv_length=20` 实际表示 draft 已有 KV 数量.
        # 首轮预消费 target 根 token 后, draft_kv_length 会额外增加 1.
        draft_layout = expand_draft_tree_ar(
            self.cfg,
            self.draft_model,
            draft_state.root_logits,
            draft_state.root_hidden,
            draft_state.kv,
            draft_kv_length=draft_state.prefix_len - 1,
            device=prefix_ids.device,
            rope_delta=self._rope_delta,
        )
        if draft_state.confirmed_root_token is None:
            return draft_layout
        # 把已确定的 target 根 token 放在最前面, draft 候选全部作为它的后代.
        # verify 的 root 预测就是 recovery_logits.argmax, 该节点必被接受.
        return TreeLayout(
            token_ids=[draft_state.confirmed_root_token] + draft_layout.token_ids,
            parent=[-1] + [index + 1 for index in draft_layout.parent],
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
        """单轮投机，返回本轮采纳的 token [batch_size == 1, accpect_len]

        Parameters
        ----------
        full_prefix_ids
            当前已接受序列 [batch_size == 1, accpect_len] (prompt + 此前各轮 append)。
        past_before
            target 对 full_prefix_ids 的 KV cache;extend verify 在其上临时 extend 后 crop。
        recovery_logits
            前缀末位预测下一 token 的 logits [V];full_model_tree 必填(根层 parent→child 接受)。
        draft_state
            增量维护的 draft KV + root 输出,避免每轮 prefill 整段 prefix。
        """
        device = full_prefix_ids.device
        # ① AR 建树
        with _prof("stage:draft_build_tree"):
            layout = self._build_tree(full_prefix_ids, draft_state)
        self.n_tree_nodes += len(layout)

        past_len = int(full_prefix_ids.shape[1])
        # * 找到所有的叶子节点
        leaves = _leaf_indices(layout.parent)
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
            # import pdb

            # pdb.set_trace()
            # * layout 还没有进行剪枝, 非常大
            with _prof("stage:target_verify"):
                append_list, proposed_draft_tokens = verify_fn(
                    self.target_m,
                    layout,
                    past_before,  # * 实际 past kv cache的数据结构
                    past_len,
                    recovery_logits,
                    leaves=leaves,
                    device=device,
                    rope_delta=self._rope_delta,
                )
            if draft_state.confirmed_root_token is not None:
                # 确定根节点不是 draft 提议, 不计入 proposed 统计.
                proposed_draft_tokens -= 1
            self.n_proposed_draft_tokens += proposed_draft_tokens
            return torch.tensor([append_list], device=device, dtype=torch.long)

        # reference_paths:每条叶路径一次完整 causal forward(串行verify),选 match 最长的叶(慢,易对照)
        best_m = -1
        best_next = 0
        best_leaf = 0
        for leaf in leaves:
            path = _path_tokens_to_leaf(layout.parent, layout.token_ids, leaf)
            with _prof("stage:target_verify"):
                mlen, nxt = _verify_path_clean(
                    self.target_m, full_prefix_ids, path, device
                )
            # 接受数相同时保留排在前面的路径(与 extend verify 一致)
            if mlen > best_m:
                best_m, best_next, best_leaf = mlen, nxt, leaf

        path = _path_tokens_to_leaf(layout.parent, layout.token_ids, best_leaf)
        append_list = path[:best_m] + [best_next]
        proposed_draft_tokens = len(path)
        if draft_state.confirmed_root_token is not None:
            proposed_draft_tokens -= 1
        self.n_proposed_draft_tokens += proposed_draft_tokens
        return torch.tensor([append_list], device=device, dtype=torch.long)

    def _prefill_target_and_draft(
        self,
        input_ids: torch.LongTensor,
        multimodal_inputs: Optional[dict] = None,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, _DraftKvState]:
        """初始化 target 和 draft, 返回第一轮需要的状态.

        multimodal_inputs: 多模态时传 pixel_values/image_grid_thw/mm_token_type_ids,
        纯文本为 None.
        """
        # ② Target prefill:KV + 末位 recovery_logits + 全前缀 act
        prefill_attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        draft_position_ids = None
        if multimodal_inputs is not None:
            # 多模态: 先用 get_rope_index 算三维 position 和 rope_delta, 并传给 target.
            position_ids, rope_deltas = self.target_m.model.get_rope_index(
                input_ids,
                multimodal_inputs["mm_token_type_ids"],
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                attention_mask=prefill_attention_mask,
            )
            self._rope_delta = int(rope_deltas.flatten()[0].item())
            draft_position_ids = position_ids
            with _prof("stage:target_prefill"):
                prefill_outputs = self.target_m(
                    input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    **multimodal_inputs,
                )
        else:
            self._rope_delta = 0
            with _prof("stage:target_prefill"):
                prefill_outputs = self.target_m(
                    input_ids,
                    attention_mask=prefill_attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                )
        past = prefill_outputs.past_key_values
        # * recovery_logits.shape -> [vob_size]
        recovery_logits = prefill_outputs.logits[0, -1]
        #! 为什么 last_token_only = false 是因为实际上需要全部的feature 需要进去算kv cache
        acts_all = select_eagle_acts_from_hidden_states(
            prefill_outputs.hidden_states, self.eagle_layers, last_token_only=False
        )
        draft_state = self._init_draft_state(
            input_ids, acts_all, recovery_logits, draft_position_ids
        )
        return past, recovery_logits, acts_all, draft_state

    def _init_draft_state(
        self,
        input_ids: torch.LongTensor,
        acts_all: torch.Tensor,
        recovery_logits: torch.Tensor,
        draft_position_ids: Optional[torch.Tensor],
    ) -> _DraftKvState:
        """draft prefill + 可选预消费, 建立首轮 draft 状态. eager/static 两条路径共用."""
        # *------first init------
        # * acts_all `[B, S, n_acts*H]`，本脚本是 `[1, prompt_len, 3*4096]`
        # （Llama-3.1-8B，`eagle_layers=[2, 16, 29]`）。
        # ③ AR draft prefill,建立首轮使用的 KV 和递归 hidden
        # * draft_logits 跟target 一样得token
        # * draft_hidden 上面这个token得hidden size
        # * draft_kv 仅仅有 prompt
        confirmed_root_token = None
        draft_prefix_length = int(input_ids.shape[1])
        with _prof("stage:draft_prefill_init"):
            draft_logits, draft_hidden, draft_kv = self.draft_model.prefill(
                input_ids, acts_all, position_ids=draft_position_ids
            )
            if self.cfg.preconsume_first_token:
                # target 已确定首个新 token, 用真实末位特征让 draft 提前消费它.
                # 输入对 = (xP, f(P-1)), draft RoPE 位置 = P-1 (+ rope_delta).
                first_target_token = int(recovery_logits.argmax(dim=-1).item())
                first_target_token_ids = torch.tensor(
                    [[first_target_token]], device=input_ids.device, dtype=torch.long
                )
                last_target_features = acts_all[:, -1:, :]
                draft_logits, draft_hidden, draft_kv = self.draft_model.extend_tokens(
                    first_target_token_ids,
                    last_target_features,
                    draft_kv,
                    int(input_ids.shape[1]) - 1,
                    rope_delta=self._rope_delta,
                )
                confirmed_root_token = first_target_token
                draft_prefix_length += 1
        # * prefill得初始状态
        return _DraftKvState(
            kv=draft_kv,
            prefix_len=draft_prefix_length,
            root_logits=draft_logits,
            root_hidden=draft_hidden,
            confirmed_root_token=confirmed_root_token,
        )

    def _update_target_after_verify(
        self,
        append: torch.LongTensor,
        past: Any,
        acts_all: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        """把接受的 token 写入 target KV, 并更新下一轮需要的输出."""
        # 把 append 重放进 target KV.带 past 时 attention_mask 必须覆盖 [past + append] 全长,
        # 否则 append 不会 attend 到前缀 → accepted_target_outputs.logits(recovery_logits)错.
        # (extend verify 在 past 上临时 extend 后 crop,不污染 past;故 past 始终 = out_ids 的 KV.)
        accepted_attention_mask = torch.ones(
            (1, int(past.get_seq_length()) + int(append.shape[1])),
            device=append.device,
            dtype=torch.long,
        )
        # * 实际上这里的 accepted_target_outputs 的hidden state 就仅仅又三个接收的token的hidden state了
        #! 如果这里是全部接收的情况下, 最后一个bonus token 必须补算的, 如果是中间接收的话, 这部分有冗余的
        # Qwen3-VL 必须显式传 position_ids: 不传时它会用全长 attention_mask 的 cumsum
        # 推 position, 而输入只有 append, 长度对不上会 reshape 错乱 / 位置错位.
        # 其他架构(GPT2/Llama)的 forward 不接受 position_ids, 只在多模态时传.
        forward_kwargs = {}
        if self._rope_delta != 0:
            append_length = int(append.shape[1])
            forward_kwargs["position_ids"] = (
                torch.arange(
                    int(past.get_seq_length()),
                    int(past.get_seq_length()) + append_length,
                    device=append.device,
                    dtype=torch.long,
                ).view(1, 1, -1)
                + self._rope_delta
            )
        with _prof("stage:target_replay"):
            accepted_target_outputs = self.target_m(
                append,
                past_key_values=past,
                use_cache=True,
                attention_mask=accepted_attention_mask,
                output_hidden_states=True,
                **forward_kwargs,
            )
        past = accepted_target_outputs.past_key_values
        recovery_logits = accepted_target_outputs.logits[0, -1]
        # * 只计算本轮新增加的 hidden states
        accepted_features = select_eagle_acts_from_hidden_states(
            accepted_target_outputs.hidden_states,
            self.eagle_layers,
            last_token_only=False,
        )
        # * 把本轮接受位置的特征拼到完整特征序列末尾
        # import pdb

        # pdb.set_trace()
        acts_all = torch.cat([acts_all, accepted_features], dim=1)
        return past, recovery_logits, acts_all

    def _update_draft_after_verify(
        self,
        append: torch.LongTensor,
        acts_all: torch.Tensor,
        draft_state: _DraftKvState,
    ) -> _DraftKvState:
        """用接受 token 和 target 特征更新 draft 状态."""
        tokens_to_extend = append
        if draft_state.confirmed_root_token is not None:
            # 首轮根 token 已在 prefill 时写入 draft KV, 这里不能重复写入.
            if int(append[0, 0].item()) != draft_state.confirmed_root_token:
                raise ValueError("first append token != confirmed draft root token")
            tokens_to_extend = append[:, 1:]
        # token t 使用 t-1 位置的特征，所以切片两端都减 1
        old_prefix_length = draft_state.prefix_len
        new_prefix_length = old_prefix_length + int(tokens_to_extend.shape[1])
        if int(tokens_to_extend.shape[1]) == 0:
            # 首轮只接受了确定根 token: draft 已消费它, 状态无需再动.
            return _DraftKvState(
                kv=draft_state.kv,
                prefix_len=new_prefix_length,
                root_logits=draft_state.root_logits,
                root_hidden=draft_state.root_hidden,
            )
        #! new_prefix_length - 1 的切片不包括 new_prefix_length - 1, 直到 new_prefix_length - 2 就停止了, 这里搞了一波错位
        #! append token 下标：21、22、23
        #! 对应 feature 下标:20、21、22
        draft_input_features = acts_all[
            :, old_prefix_length - 1 : new_prefix_length - 1, :
        ]
        """
        接受路径的临时 KV 理论上可以复用, 
        但里面使用的是未被 target 校正的预测 hidden, 直接提交可能累积误差.
        所以这里选择对接受 token 重新计算.
        """
        with _prof("stage:draft_extend"):
            draft_logits, draft_hidden, draft_kv = self.draft_model.extend_tokens(
                tokens_to_extend,
                draft_input_features,
                draft_state.kv,
                old_prefix_length - 1,
                rope_delta=self._rope_delta,
            )
        return _DraftKvState(
            kv=draft_kv,
            prefix_len=new_prefix_length,
            root_logits=draft_logits,
            root_hidden=draft_hidden,
        )

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
        multimodal_inputs: Optional[dict] = None,
    ) -> torch.LongTensor:
        """
        generate 主流程, prefill target -> prefill draft
        """
        if self.cfg.decode_cuda_graph:
            return self._generate_static_graph(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                multimodal_inputs=multimodal_inputs,
            )
        if self.cfg.skip_replay:
            return self._generate_no_replay(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                multimodal_inputs=multimodal_inputs,
            )
        out_ids = input_ids.clone()
        num_new_tokens = 0
        # * 1. traget 进行 prefill 生产 feature, draft 进行 prefill, 生成 draft state
        (
            past,
            recovery_logits,
            acts_all,
            draft_state,
        ) = self._prefill_target_and_draft(out_ids, multimodal_inputs=multimodal_inputs)

        while num_new_tokens < max_new_tokens:
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
            # * 2. 根据draft state 进行投机解码, 验证并返回最终接收的部分
            append = self.generate_step_payload(
                out_ids, past, recovery_logits, draft_state
            )
            # * 3. target 更新kv cache
            past, recovery_logits, acts_all = self._update_target_after_verify(
                append, past, acts_all
            )
            # * 4. draft 更新kv cache & draft state
            draft_state = self._update_draft_after_verify(append, acts_all, draft_state)
            # *--- 这里才最终接收
            out_ids = torch.cat([out_ids, append], dim=1)
            num_new_tokens += int(append.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(append.numel())
            if eos_token_id is not None and (append == eos_token_id).any():
                break
        return out_ids

    # ------------------------------------------------------------------
    # skip_replay 主流程:verify 顺带提交接受路径的 KV 和 feature, 不做 target replay.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_no_replay(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        multimodal_inputs: Optional[dict] = None,
    ) -> torch.LongTensor:
        """无 replay 生成:每轮只有一次 target forward(verify 即提交).

        稳态不变量:out_ids 末位 pending 是已确定但尚未被 target forward 的 token;
        past 和 acts_all 覆盖 out_ids[:-1], draft 已消费全部 out_ids.
        每轮把 pending 作为确定根建树(必被接受), verify 提交接受路径的 KV/feature,
        bonus token 成为下一轮的 pending, 从而省掉 replay forward.
        """
        device = input_ids.device
        out_ids = input_ids.clone()
        num_new_tokens = 0
        past, recovery_logits, acts_all, draft_state = self._prefill_target_and_draft(
            out_ids, multimodal_inputs=multimodal_inputs
        )
        # 与 generate_step_payload 相同的 backend 选择逻辑.
        backend = self.cfg.verify_attn_backend
        target_on_cuda = next(self.target_m.parameters()).device.type == "cuda"
        use_triton = backend == "triton_tree" or (
            backend == "auto"
            and self.cfg.topk > 1
            and target_on_cuda
            and triton_extend_available()
        )
        pending_in_out_ids = False
        while num_new_tokens < max_new_tokens:
            last_token = int(out_ids[0, -1].item())
            if eos_token_id is not None and last_token == eos_token_id:
                break  # pending 可能已是 EOS, 无需再 forward
            if pending_in_out_ids:
                # 稳态轮: pending 视作 draft 已预消费的确定根, 与首轮 preconsume 状态对齐.
                # forward 前缀是 out_ids[:-1], draft.prefix_len 恰好比它大 1.
                draft_state.confirmed_root_token = last_token
                forwarded_prefix = out_ids[:, :-1]
                recovery_logits = None
            else:
                forwarded_prefix = out_ids
            with _prof("stage:draft_build_tree"):
                layout = self._build_tree(forwarded_prefix, draft_state)
            self.n_tree_nodes += len(layout)
            leaves = _leaf_indices(layout.parent)
            confirmed_root = draft_state.confirmed_root_token is not None
            with _prof("stage:target_verify_commit"):
                (
                    append_list,
                    proposed_draft_tokens,
                    accepted_features,
                ) = full_tree_verify_commit(
                    self.target_m,
                    layout,
                    past,
                    int(forwarded_prefix.shape[1]),
                    recovery_logits,
                    leaves=leaves,
                    device=device,
                    eagle_layers=self.eagle_layers,
                    rope_delta=self._rope_delta,
                    confirmed_root=confirmed_root,
                    use_triton=use_triton,
                )
            if confirmed_root:
                # 确定根节点不是 draft 提议, 不计入 proposed 统计.
                proposed_draft_tokens -= 1
            self.n_proposed_draft_tokens += proposed_draft_tokens
            append = torch.tensor([append_list], device=device, dtype=torch.long)
            # 接受节点的 feature 直接并入 acts_all, 无需 replay 重算.
            acts_all = torch.cat([acts_all, accepted_features], dim=1)
            draft_state = self._update_draft_after_verify(append, acts_all, draft_state)
            if pending_in_out_ids:
                new_tokens = append[:, 1:]  # pending 已在 out_ids 里, 不重复追加
            else:
                new_tokens = append
            out_ids = torch.cat([out_ids, new_tokens], dim=1)
            num_new_tokens += int(new_tokens.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(new_tokens.numel())
            pending_in_out_ids = True
            if eos_token_id is not None and (new_tokens == eos_token_id).any():
                break
        return out_ids

    # ------------------------------------------------------------------
    # StaticCache + verify CUDA graph 路径: 在 skip_replay 基础上, 每轮唯一的
    # target forward(verify+hidden)走捕获好的 graph replay, commit 仍 eager.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_static_graph(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        multimodal_inputs: Optional[dict] = None,
    ) -> torch.LongTensor:
        """无 replay + StaticCache + verify CUDA graph 生成.

        稳态不变量与 _generate_no_replay 相同; 区别是 target KV 用固定容量
        StaticCache, verify 前向 padding 到固定 L 后 graph replay, 省掉逐 kernel
        launch 开销. 写入指针由 rollback_static_cache 显式控制.
        """
        from .target_graph import (
            StaticGraphCommitVerifier,
            make_static_cache,
        )

        device = input_ids.device
        dtype = next(self.target_m.parameters()).dtype
        out_ids = input_ids.clone()
        prompt_length = int(out_ids.shape[1])
        # 稳态树 = 1 个 pending 确定根 + 至多 max_tree_nodes 个 draft 节点.
        capture_tree_nodes = int(self.cfg.max_tree_nodes) + 1
        # 容量: 前缀 + 新 token + verify 临时树 slot + 余量.
        required_length = (
            prompt_length
            + max_new_tokens
            + capture_tree_nodes
            + int(self.cfg.graph_cache_margin)
        )
        # static cache + graph 跨 generate() 复用, 避免每次重新分配和捕获.
        sg = getattr(self, "_sg_state", None)
        if (
            sg is None
            or sg["max_len"] < required_length
            or sg["capture_tree_nodes"] != capture_tree_nodes
        ):
            static_cache = make_static_cache(
                self.target_m.config, required_length, device, dtype
            )
            verifier = StaticGraphCommitVerifier(
                self.target_m,
                static_cache,
                required_length,
                device,
                dtype,
                capture_tree_nodes=capture_tree_nodes,
                enable_graph=bool(self.cfg.graph_capture),
            )
            self._sg_state = {
                "sc": static_cache,
                "verifier": verifier,
                "max_len": required_length,
                "capture_tree_nodes": capture_tree_nodes,
            }
        else:
            static_cache = sg["sc"]
            verifier = sg["verifier"]
            static_cache.reset()

        # ② target prefill: 写入 static cache, 取末位 logits + 全前缀 act.
        prefill_attention_mask = torch.ones_like(input_ids, device=device)
        draft_position_ids = None
        prefill_kwargs = {}
        if multimodal_inputs is not None:
            position_ids, rope_deltas = self.target_m.model.get_rope_index(
                input_ids,
                multimodal_inputs["mm_token_type_ids"],
                image_grid_thw=multimodal_inputs.get("image_grid_thw"),
                attention_mask=prefill_attention_mask,
            )
            self._rope_delta = int(rope_deltas.flatten()[0].item())
            draft_position_ids = position_ids
            prefill_kwargs["position_ids"] = position_ids
            prefill_kwargs.update(multimodal_inputs)
        else:
            self._rope_delta = 0
            # 2D 位置: Llama/GPT2 直接用; Qwen3-VL 自动 expand 成 4 轴, 文本三轴相同.
            prefill_kwargs["position_ids"] = torch.arange(
                prompt_length, device=device
            ).unsqueeze(0)
        with _prof("stage:target_prefill"):
            prefill_outputs = self.target_m(
                out_ids,
                past_key_values=static_cache,
                cache_position=torch.arange(prompt_length, device=device),
                attention_mask=prefill_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                **prefill_kwargs,
            )
        recovery_logits = prefill_outputs.logits[0, -1]
        acts_all = select_eagle_acts_from_hidden_states(
            prefill_outputs.hidden_states, self.eagle_layers, last_token_only=False
        )
        # ③ draft prefill + 可选预消费, 与 eager 路径共用.
        draft_state = self._init_draft_state(
            out_ids, acts_all, recovery_logits, draft_position_ids
        )

        cur_len = prompt_length
        num_new_tokens = 0
        pending_in_out_ids = False
        while num_new_tokens < max_new_tokens:
            last_token = int(out_ids[0, -1].item())
            if eos_token_id is not None and last_token == eos_token_id:
                break  # pending 可能已是 EOS, 无需再 forward
            if pending_in_out_ids:
                # 稳态轮: pending 视作 draft 已预消费的确定根, 与首轮 preconsume 对齐.
                draft_state.confirmed_root_token = last_token
                forwarded_prefix = out_ids[:, :-1]
                recovery_logits = None
            else:
                forwarded_prefix = out_ids
            with _prof("stage:draft_build_tree"):
                layout = self._build_tree(forwarded_prefix, draft_state)
            self.n_tree_nodes += len(layout)
            leaves = _leaf_indices(layout.parent)
            confirmed_root = draft_state.confirmed_root_token is not None
            with _prof("stage:target_verify_commit"):
                (
                    append_list,
                    proposed_draft_tokens,
                    accepted_features,
                ) = verifier.verify_commit(
                    layout,
                    cur_len,
                    recovery_logits,
                    leaves=leaves,
                    rope_delta=self._rope_delta,
                    confirmed_root=confirmed_root,
                    eagle_layers=self.eagle_layers,
                )
            if confirmed_root:
                # 确定根节点不是 draft 提议, 不计入 proposed 统计.
                proposed_draft_tokens -= 1
            self.n_proposed_draft_tokens += proposed_draft_tokens
            append = torch.tensor([append_list], device=device, dtype=torch.long)
            # verify 写入的树 KV 已被 commit 紧凑成接受段, 写入指针拨到 cur_len+m.
            cur_len += len(append_list) - 1
            acts_all = torch.cat([acts_all, accepted_features], dim=1)
            draft_state = self._update_draft_after_verify(append, acts_all, draft_state)
            if pending_in_out_ids:
                new_tokens = append[:, 1:]  # pending 已在 out_ids 里, 不重复追加
            else:
                new_tokens = append
            out_ids = torch.cat([out_ids, new_tokens], dim=1)
            num_new_tokens += int(new_tokens.numel())
            self.n_rounds += 1
            self.n_accepted_tokens += int(new_tokens.numel())
            pending_in_out_ids = True
            if eos_token_id is not None and (new_tokens == eos_token_id).any():
                break
        return out_ids
