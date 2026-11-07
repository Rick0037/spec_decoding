# SPDX-License-Identifier: Apache-2.0
"""
全层 Target 树形 Verify(extend 路径)

复用前缀 KV cache,只前向 draft 树的 L 个节点(不重算前缀),每层
attention 对新 token 段施加树掩码(兄弟分支不可互看),再按 parent→child
规则做 greedy 接受。

两条入口
--------
- full_tree_verify_extend:HF sdpa + [1,1,L,past+L] 4D 树 cross mask。
- full_tree_verify_triton:SGLang extend_attention_fwd + custom_mask。

verify 跑在 cache 副本/临时 extend 后 crop,不污染 runner 的 past;
root 接受用 recovery_logits(out_ids 之后那一位的 logits)。
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .tree_verify_mask import (
    TreeVerifyLayout,
    build_tree_cross_attn_bias_with_prefix,
    layout_tree_position_ids,
)
from .tree_verify_attn import (
    build_triton_tree_verify_metadata,
    tree_verify_triton_attention_context,
)
from .tree_verify_parallel import path_node_indices_root_to_leaf


def _decoder_stack(target_m: PreTrainedModel) -> nn.Module:
    if hasattr(target_m, "model") and isinstance(getattr(target_m, "model"), nn.Module):
        return target_m.model
    raise TypeError(
        "full_model_tree verify expects a HF CausalLM with .model (e.g. Llama/Qwen)"
    )


# -----------------------------------------------------------------------------
# 复用前缀 KV cache 的extendverify(不重算前缀,只前向 L 个树节点).
# HF sdpa + 4D 树 cross mask,或 triton extend kernel;前缀 KV 来自 runner 的 past,
# 在 cache 副本上跑(不污染 past).root 接受用 recovery_logits(out_ids 之后那一位 logits).
# 正确的 parent→child 接受:节点 i 的输出预测其子节点(避免 select_append 的 off-by-one).
# -----------------------------------------------------------------------------
def _crop_cache_back_to(past_key_values: Any, target_len: int) -> None:
    """把 KV cache 截回 target_len。

    transformers 5.20+ 的 DynamicCache.crop 只接受负数(删除 N 个),旧的
    “正数=目标长度”会直接抛 ValueError;4.x 则相反,只认正数。这里优先负数,
    截完校验长度,不生效再退回正数。
    """
    cur = int(past_key_values.get_seq_length())
    if cur <= target_len:
        return
    try:
        past_key_values.crop(-(cur - target_len))
    except (TypeError, ValueError):
        past_key_values.crop(target_len)
    if int(past_key_values.get_seq_length()) != target_len:
        past_key_values.crop(target_len)


def _forward_and_extend_dynamic_kv_then_crop(
    target_m, tree_tokens, past_key_values, *, position_ids, attention_mask
):
    """在 runner 的 past 上临时 extend(追加 L 个树节点 KV)跑一次 forward,取完 logits 后 crop 还原。

    比深拷贝整份 cache 便宜得多(只多存/截掉 L 个位置的 KV);KV-replay 用全长 mask 后 past 不被污染。
    """
    before = int(past_key_values.get_seq_length())
    # * 实际推理
    out = target_m(
        tree_tokens,  # [batch_size, seq_len]
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        attention_mask=attention_mask,
    )
    # out.logits -> [batch_size, seq_len, voc_size]
    logits0 = out.logits[0]  # [seq_len, voc_size]
    # 去掉本次 verify 追加的树 KV,还原成 out_ids 的 KV
    _crop_cache_back_to(past_key_values, before)
    return logits0


def _select_append_root_child(
    all_node_logits: torch.Tensor,
    root_logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
) -> Tuple[List[int], int, int, List[int]]:
    """在所有根→叶路径中选接受最长的一条,返回本轮要 append 的 token."""
    best_accepted_len, best_next_token, best_leaf = -1, 0, 0
    for leaf in leaves:
        accepted_len, next_token = _match_one_leaf_path(
            all_node_logits, root_logits, parent, token_ids, leaf
        )
        if accepted_len > best_accepted_len:  # 接受数相同时,保留排在前面的路径
            best_accepted_len, best_next_token, best_leaf = (
                accepted_len,
                next_token,
                leaf,
            )

    best_path_nodes = path_node_indices_root_to_leaf(parent, best_leaf)
    best_path_tokens = [token_ids[i] for i in best_path_nodes]
    append_list = best_path_tokens[:best_accepted_len] + [best_next_token]
    return append_list, best_accepted_len, int(best_leaf), best_path_tokens


def _match_one_leaf_path(
    all_node_logits: torch.Tensor,
    root_logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaf: int,
) -> Tuple[int, int]:
    """对单条根→叶路径逐位 greedy 比对,返回 (接受个数, 下一位 token).

    对齐规则:路径上第 pos 个节点提案的 token,由它父节点的输出 logits 预测——
    pos=0 没有父(直接挂前缀后),用 root_logits(前缀末位的预测);
    pos>0 用 all_node_logits[父]。首个不一致即停,下一位给 target 的修正;
    全中则下一位是叶节点自己的 argmax(bonus)。
    """
    path_nodes = path_node_indices_root_to_leaf(parent, leaf)
    for pos, node_idx in enumerate(path_nodes):
        # 预测源:根节点用 root_logits,其余用父节点的输出 logits
        source_logits = (
            root_logits if pos == 0 else all_node_logits[path_nodes[pos - 1]]
        )
        target_token = int(source_logits.argmax(dim=-1).item())
        if target_token != token_ids[node_idx]:
            return pos, target_token  # 接受前 pos 个,第 pos+1 位用 target 修正
    return len(path_nodes), int(all_node_logits[path_nodes[-1]].argmax(dim=-1).item())


@torch.inference_mode()
def full_tree_verify_extend(
    target_m: PreTrainedModel,
    layout: TreeVerifyLayout,
    past_key_values: Any,
    past_len: int,
    recovery_logits: torch.Tensor,
    *,
    leaves: List[int],
    device: torch.device,
) -> Tuple[List[int], int, int, List[int]]:
    """复用前缀 KV 的 extend 树掩码 verify(HF sdpa + 4D 树 cross mask):只前向 L 个树节点。

    在 cache 副本上跑(不污染 runner 的 past);root 接受用 recovery_logits。无损(与贪心一致)。
    """
    L = len(layout)
    dtype = next(target_m.parameters()).dtype
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    cross = build_tree_cross_attn_bias_with_prefix(
        layout, past_len, device=device, dtype=dtype
    ).view(1, 1, L, past_len + L)
    pos = layout_tree_position_ids(layout, past_len).unsqueeze(0).to(device)
    node_logits = _forward_and_extend_dynamic_kv_then_crop(
        target_m, tree_tokens, past_key_values, position_ids=pos, attention_mask=cross
    )  # [L, V]
    return _select_append_root_child(
        node_logits, recovery_logits, layout.parent, layout.token_ids, leaves
    )


@torch.inference_mode()
def full_tree_verify_triton(
    target_m: PreTrainedModel,
    layout: TreeVerifyLayout,
    past_key_values: Any,
    past_len: int,
    root_logits: torch.Tensor,
    *,
    leaves: List[int],
    device: torch.device,
) -> Tuple[List[int], int, int, List[int]]:
    """extend verify 的 triton 版:
    复用前缀 KV,用 SGLang extend_attention_fwd + 树 custom_mask。
    数值与 full_tree_verify_extend 等价;
    同样在 cache 副本上跑、root 接受用 root_logits
    """
    L = len(layout)  # draft_lens
    # (Pdb) layout
    # TreeVerifyLayout(token_ids=[23557, 18160, 18160, 23557, 23557, 18160],
    # parent=[-1, -1, 0, 0, 1, 1], dfs_order=[0, 1, 2, 3, 4, 5])
    # import pdb

    # pdb.set_trace()
    # [batch_size, seq_len]
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    # * past_len = 6, layout -> [0, 0, 1, 1, 1, 1]
    # * pos->[6,6,7,7,7,7]
    pos = layout_tree_position_ids(layout, past_len).unsqueeze(0).to(device)
    # * 创建
    meta = build_triton_tree_verify_metadata(layout, past_len, device=device)
    decoder = _decoder_stack(target_m)
    # * decoder -> embedding + decoder layer (remove LM head)
    # ! with tree_verify_triton_attention_context 实际上就是使用 trition kernel 去替换linear中的
    # * 每次都需要重新建树, 重新替换attention之后验证, 然后 再替换回去
    with tree_verify_triton_attention_context(decoder, meta):
        all_node_logits = _forward_and_extend_dynamic_kv_then_crop(
            target_m,
            tree_tokens,
            past_key_values,  # * 从runner 中就一直是引用传递
            position_ids=pos,
            attention_mask=torch.ones(1, L, device=device, dtype=torch.long),
        )
        # [L, V]
    # * 一次验证的逻辑, 贪心, 没有对比draft 以及 target的logits, 直接one-hot对比输出
    # * node_logits == logit[0] -> [seq_len, vob_size]
    return _select_append_root_child(
        all_node_logits, root_logits, layout.parent, layout.token_ids, leaves
    )
