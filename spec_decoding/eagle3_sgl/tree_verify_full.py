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

verify 临时扩展 cache,取完 logits 后 crop 还原,不污染 runner 的 past;
root 接受用 recovery_logits(out_ids 之后那一位的 logits)。
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .tree_verify_mask import (
    TreeLayout,
    build_tree_cross_attn_bias_with_prefix,
    layout_tree_position_ids,
)
from .tree_verify_attn import (
    build_triton_tree_verify_metadata,
    tree_verify_triton_attention_context,
)


def path_node_indices_root_to_leaf(parent: List[int], leaf: int) -> List[int]:
    """从根到 leaf 的节点下标(含叶)。"""
    path: List[int] = []
    node = leaf
    while node >= 0:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def _decoder_stack(target_m: PreTrainedModel) -> nn.Module:
    inner = getattr(target_m, "model", None)
    if not isinstance(inner, nn.Module):
        raise TypeError(
            "full_model_tree verify expects a HF CausalLM with .model (e.g. Llama/Qwen)"
        )
    # 多模态模型(如 Qwen3-VL)的 decoder 层在 model.language_model.layers.
    language_model = getattr(inner, "language_model", None)
    if getattr(inner, "layers", None) is None and isinstance(language_model, nn.Module):
        return language_model
    return inner


# -----------------------------------------------------------------------------
# 复用前缀 KV cache 的extendverify(不重算前缀,只前向 L 个树节点).
# HF sdpa + 4D 树 cross mask,或 triton extend kernel;前缀 KV 来自 runner 的 past,
# 临时扩展 cache,取完 logits 后 crop 还原(不污染 past).root 接受用 recovery_logits.
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


def _forward_target_tree_and_restore_cache(
    target_m, tree_tokens, past_key_values, *, position_ids, attention_mask
):
    """
    使用forward 进行验证, 并使用past key value进行加速, 加速之后又截取回去, 因为在验证阶段,
    实际不应该加入到cache中
    """
    before = int(past_key_values.get_seq_length())

    out = target_m(
        tree_tokens,  # [batch_size, seq_len]
        past_key_values=past_key_values,
        use_cache=True,
        attention_mask=attention_mask,
        **position_ids,
    )
    # out.logits -> [batch_size, seq_len, voc_size]
    logits0 = out.logits[0]  # [seq_len, voc_size]
    # 去掉本次 verify 追加的树 KV,还原成 out_ids 的 KV
    _crop_cache_back_to(past_key_values, before)
    return logits0


def _select_append_root_child(
    tree_node_next_token_logits: torch.Tensor,
    root_logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaves: List[int],
) -> Tuple[List[int], int]:
    """
    第一次 prefill:
    输出 = target(["The", "cat"])
    输出[0] = 看完 "The" 的预测
    输出[1] = 看完 "cat" 的预测  # 保存为 recovery_logits

    第二次 verify, 输入只有 draft token:
    输出 = target(["sat", "mat"], past_key_values="The cat 的 KV")
    tree_node_next_token_logits[0] = 看完 "sat" 的预测
    tree_node_next_token_logits[1] = 看完 "mat" 的预测

    root_logits          -> 检查第 0 个 draft token
    tree_node_next_token_logits[0]   -> 检查第 1 个 draft token
    tree_node_next_token_logits[1]   -> 检查第 2 个 draft token
    tree_node_next_token_logits[2]   -> 检查第 3 个 draft token
    """
    best_accepted_len, best_next_token, best_leaf = -1, 0, 0
    for leaf in leaves:
        accepted_len, next_token = _match_one_leaf_path(
            tree_node_next_token_logits, root_logits, parent, token_ids, leaf
        )
        if accepted_len > best_accepted_len:  # 接受数相同时,保留排在前面的路径
            best_accepted_len, best_next_token, best_leaf = (
                accepted_len,
                next_token,
                leaf,
            )

    best_path_nodes = path_node_indices_root_to_leaf(parent, best_leaf)
    best_path_tokens = [token_ids[i] for i in best_path_nodes]
    # ! 接收长度的token + 一个 target的 next new token
    append_list = best_path_tokens[:best_accepted_len] + [best_next_token]
    # 同时返回选中路径的 draft 候选数, 不统计其他分支.
    return append_list, len(best_path_nodes)


def _match_one_leaf_path(
    tree_node_next_token_logits: torch.Tensor,
    root_logits: torch.Tensor,
    parent: List[int],
    token_ids: List[int],
    leaf: int,
) -> Tuple[int, int]:
    """对单条根→叶路径逐位 greedy 比对,返回 (接受个数, 下一位 token).

    对齐规则:路径上第 pos 个节点提案的 token,由它父节点的输出 logits 预测——
    pos=0 没有父(直接挂前缀后),用 root_logits(前缀末位的预测);
    pos>0 用 tree_node_next_token_logits[父]。首个不一致即停,下一位给 target 的修正;
    全中则下一位是叶节点自己的 argmax(bonus)。
    """
    path_nodes = path_node_indices_root_to_leaf(parent, leaf)
    for pos, node_idx in enumerate(path_nodes):
        #! 预测源:根节点用 root_logits,其余用父节点的输出 logits
        source_logits = (
            root_logits
            if pos == 0
            else tree_node_next_token_logits[path_nodes[pos - 1]]
        )
        target_token = int(source_logits.argmax(dim=-1).item())
        if target_token != token_ids[node_idx]:
            return pos, target_token  # 接受前 pos 个,第 pos+1 位用 target 修正
    next_token_logits = tree_node_next_token_logits[path_nodes[-1]]
    return len(path_nodes), int(next_token_logits.argmax(dim=-1).item())


@torch.inference_mode()
def full_tree_verify_extend(
    target_m: PreTrainedModel,
    layout: TreeLayout,
    past_key_values: Any,
    past_len: int,
    recovery_logits: torch.Tensor,
    *,
    leaves: List[int],
    device: torch.device,
    rope_delta: int = 0,
) -> Tuple[List[int], int]:
    """复用前缀 KV 的 extend 树掩码 verify(HF sdpa + 4D 树 cross mask):只前向 L 个树节点。

    临时扩展 cache,取完 logits 后 crop 还原;root 接受用 recovery_logits。无损(与贪心一致)。
    """
    L = len(layout)
    dtype = next(target_m.parameters()).dtype
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    cross = build_tree_cross_attn_bias_with_prefix(
        layout, past_len, device=device, dtype=dtype
    ).view(1, 1, L, past_len + L)
    pos = layout_tree_position_ids(layout, past_len, rope_delta).unsqueeze(0).to(device)
    position_kwargs = {}
    if rope_delta != 0:
        # 多模态才需要显式 position_ids; GPT2/Llama 的 forward 不接受该参数.
        position_kwargs["position_ids"] = pos
    node_logits = _forward_target_tree_and_restore_cache(
        target_m,
        tree_tokens,
        past_key_values,
        position_ids=position_kwargs,
        attention_mask=cross,
    )  # [L, V]
    return _select_append_root_child(
        node_logits, recovery_logits, layout.parent, layout.token_ids, leaves
    )


@torch.inference_mode()
def full_tree_verify_triton(
    target_m: PreTrainedModel,
    layout: TreeLayout,
    past_key_values: Any,
    past_len: int,
    root_logits: torch.Tensor,
    *,
    leaves: List[int],
    device: torch.device,
    rope_delta: int = 0,
) -> Tuple[List[int], int]:
    """extend verify 的 triton 版:
    复用前缀 KV,用 SGLang extend_attention_fwd + 树 custom_mask。
    数值与 full_tree_verify_extend 等价;
    同样临时扩展 cache 并 crop 还原;root 接受用 root_logits
    """
    L = len(layout)  # draft_lens
    # (Pdb) layout
    # TreeLayout(token_ids=[23557, 18160, 18160, 23557, 23557, 18160],
    # parent=[-1, -1, 0, 0, 1, 1])
    # import pdb

    # pdb.set_trace()
    # [batch_size, seq_len]
    tree_tokens = torch.tensor([layout.token_ids], device=device, dtype=torch.long)
    # * past_len = 6, layout -> [0, 0, 1, 1, 1, 1]
    # * pos->[6,6,7,7,7,7]
    pos = layout_tree_position_ids(layout, past_len, rope_delta).unsqueeze(0).to(device)
    position_kwargs = {}
    if rope_delta != 0:
        position_kwargs["position_ids"] = pos
    # * 创建
    meta = build_triton_tree_verify_metadata(layout, past_len, device=device)
    decoder = _decoder_stack(target_m)
    # * decoder -> embedding + decoder layer (remove LM head)
    # ! with tree_verify_triton_attention_context 实际上就是使用 trition kernel 去替换linear中的
    # * 每次都需要重新建树, 重新替换attention之后验证, 然后 再替换回去
    with tree_verify_triton_attention_context(decoder, meta):
        tree_node_next_token_logits = _forward_target_tree_and_restore_cache(
            target_m,
            tree_tokens,
            past_key_values,  # * 从runner 中就一直是引用传递, 先增加
            position_ids=position_kwargs,
            attention_mask=torch.ones(1, L, device=device, dtype=torch.long),
        )
        # [L, V]
    # * 一次验证的逻辑, 贪心, 没有对比draft 以及 target的logits, 直接one-hot对比输出
    # * node_logits == logit[0] -> [seq_len, vob_size]
    return _select_append_root_child(
        tree_node_next_token_logits,
        root_logits,
        layout.parent,
        layout.token_ids,
        leaves,
    )
