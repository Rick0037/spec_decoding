"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
TRITON_INTERPRET=1
"""

import torch
import triton
import triton.language as tl

_is_cuda = torch.cuda.is_available()
CUDA_CAPABILITY = (0, 0)
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


def _get_block_sizes_for_extend_attention(head_dim: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        head_dim: Q/K/V head dimension

    Returns:
        tuple: (BLOCK_M, BLOCK_DMODEL, BLOCK_N, num_warps)
    """
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware (CUDA)
    if _is_cuda and CUDA_CAPABILITY[0] == 12:
        # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
        if head_dim <= 128:
            BLOCK_M, BLOCK_N = (64, 128)
        elif head_dim <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 32)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 9:
        # Hopper architecture (H100, etc.)
        if head_dim <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
        # Ampere architecture (A100, etc.)
        # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
            if head_dim <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif head_dim <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        else:
            if head_dim <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif head_dim <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
    else:
        # Older architectures / CPU fallback sizing
        BLOCK_M, BLOCK_N = (64, 64) if head_dim <= 128 else (32, 32)

    num_warps = 4 if head_dim <= 64 else 8

    return BLOCK_M, BLOCK_DMODEL, BLOCK_N, num_warps


# * 树 verify 的 attention kernel. tree_len = 本轮树节点数, cache_len = 已缓存前缀长度.
# * 一个 program 算 BLOCK_M 行 query, key 分两段扫:
# *   stage 1: 前缀 cache_len 个 KV, 全部可见, 行号从 kv_indices 取;
# *   stage 2: 本轮 tree_len 个树节点, 只有祖先可见, 查 custom_mask.
# * 两段共用 online softmax 累加器, 不存 [tree_len, cache_len+tree_len] 的注意力矩阵.
@triton.jit
def _fwd_kernel(
    Q_Extend,  # * [tree_len, q_head, head_dim] 没办法 batchsize 以及 tree得长度都混到tree len中了
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    HEAD_DIM: tl.constexpr,
    # * qk = q @ k^T 的 M/K/N 分块: M=query 行, K=head_dim(补齐到 2 的幂), N=key 列
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    # 对应seq_len得维度, 遍历q来处理
    q_block_idx = tl.program_id(2)
    # kv head index
    kv_head_idx = q_head_idx // kv_group_num

    # * indptr 存每条序列在扁平张量里的分界; batch=1 时 qo=[0,tree_len], kv=[0,cache_len],
    # ! 全部都在第一维度 + 相关得索引就可以找到对应得数值

    # 不同得batch上得数值是不一样得, 对于另外两个维度一样得q 来说是相同得
    extend_start_row = tl.load(qo_indptr + batch_idx)
    extend_len = tl.load(qo_indptr + batch_idx + 1) - extend_start_row
    prefix_start_row = tl.load(kv_indptr + batch_idx)
    prefix_len = tl.load(kv_indptr + batch_idx + 1) - prefix_start_row
    total_kv_len = prefix_len + extend_len
    mask_segment_start = tl.load(mask_indptr + batch_idx)

    # * 2d 情况下, 看到得一些东西
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    q_mask = (q_offsets[:, None] < extend_len) & (d_offsets[None, :] < HEAD_DIM)

    q_ptr = (
        Q_Extend
        + (extend_start_row + q_offsets[:, None]) * stride_qbs
        + q_head_idx * stride_qh
        + d_offsets[None, :]
    )

    # * 终于加载上来了
    q_val = tl.load(
        q_ptr,
        mask=q_mask,
        other=0.0,
    )

    # stage 1: compute scores with prefix
    k_offsets = tl.arange(0, BLOCK_N)

    output_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    softmax_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    running_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for k_start in range(0, prefix_len, BLOCK_N):
        prefix_k_mask = (k_start + k_offsets) < prefix_len

        # * [BLOCK_N] int32: 本块前缀 token 在 KV buffer 的物理行号, 不是 K/V 数据
        # prefix_kv_rows-> [128, 0]
        prefix_kv_rows = tl.load(
            # * kv_indices -> [0, 1, 2, 3, 4, 5, 6, ..., 11]
            kv_indices + prefix_start_row + k_start + k_offsets,
            mask=prefix_k_mask,
            other=0,
        )

        # 前缀段全可见, 不查 custom_mask
        k_mask = (prefix_k_mask[:, None]) & (d_offsets[None, :] < HEAD_DIM)

        # load k in transposed way
        k_ptr = K_Buffer + (
            prefix_kv_rows[:, None] * stride_buf_kbs
            + kv_head_idx * stride_buf_kh
            + d_offsets[None, :]
        )

        k_val = tl.load(
            k_ptr,
            mask=k_mask,
            other=0.0,
        )

        current_tile_score = tl.dot(q_val, tl.trans(k_val))
        current_tile_score *= sm_scale

        # * q 对应prefix 全部都可以见到
        score_mask = (q_offsets[:, None] < extend_len) & prefix_k_mask[None, :]
        current_tile_score = tl.where(score_mask, current_tile_score, float("-inf"))

        tile_row_max = tl.max(current_tile_score, axis=1)
        new_running_max = tl.maximum(tile_row_max, running_max)

        rescale_factor = tl.exp(running_max - new_running_max)
        prob_tile = tl.exp(current_tile_score - new_running_max[:, None])
        # * 已扫过 key 的未归一化注意力权重之和; 最大值变了, 旧的要缩到新口径
        softmax_sum = softmax_sum * rescale_factor + tl.sum(prob_tile, axis=1)

        v_ptr = V_Buffer + (
            prefix_kv_rows[:, None] * stride_buf_vbs
            + kv_head_idx * stride_buf_vh
            + d_offsets[None, :]
        )
        v = tl.load(
            v_ptr,
            mask=k_mask,
            other=0.0,
        )
        prob_tile = prob_tile.to(v.dtype)
        output_acc = output_acc * rescale_factor[:, None] + tl.dot(prob_tile, v)

        running_max = new_running_max

    # stage 2: compute the triangle part
    for k_start in range(0, extend_len, BLOCK_N):
        tree_k_mask = (k_start + k_offsets) < extend_len

        # * 树掩码: 只有祖先可见; load 的 other=0 已把无效行列置 0
        tree_mask_ptr = (
            mask_ptr
            + mask_segment_start
            + q_offsets[:, None] * total_kv_len
            + prefix_len
            + k_start
            + k_offsets[None, :]
        )
        score_mask = tl.load(
            tree_mask_ptr,
            mask=(q_offsets[:, None] < extend_len) & tree_k_mask[None, :],
            other=0,
        )

        skip_tile = tl.max(tl.max(score_mask.to(tl.int32), axis=1), axis=0) == 0

        if not skip_tile:
            k_mask = (tree_k_mask[:, None]) & (d_offsets[None, :] < HEAD_DIM)

            k_ptr = K_Extend + (
                (extend_start_row + k_start + k_offsets[:, None]) * stride_kbs
                + kv_head_idx * stride_kh
                + d_offsets[None, :]
            )
            k_val = tl.load(
                k_ptr,
                mask=k_mask,
                other=0.0,
            )

            current_tile_score = tl.dot(q_val, tl.trans(k_val))
            current_tile_score *= sm_scale

            current_tile_score = tl.where(score_mask, current_tile_score, float("-inf"))

            tile_row_max = tl.max(current_tile_score, axis=1)
            new_running_max = tl.maximum(tile_row_max, running_max)

            rescale_factor = tl.exp(running_max - new_running_max)
            prob_tile = tl.exp(current_tile_score - new_running_max[:, None])
            softmax_sum = softmax_sum * rescale_factor + tl.sum(prob_tile, axis=1)

            v_ptr = V_Extend + (
                (extend_start_row + k_start + k_offsets[:, None]) * stride_vbs
                + kv_head_idx * stride_vh
                + d_offsets[None, :]
            )
            v = tl.load(
                v_ptr,
                mask=k_mask,
                other=0.0,
            )
            prob_tile = prob_tile.to(v.dtype)
            output_acc = output_acc * rescale_factor[:, None] + tl.dot(prob_tile, v)

            running_max = new_running_max

    o_ptr = (
        O_Extend
        + (extend_start_row + q_offsets[:, None]) * stride_obs
        + q_head_idx * stride_oh
        + d_offsets[None, :]
    )
    tl.store(
        o_ptr,
        output_acc / softmax_sum[:, None],
        mask=q_mask,
    )


# * 多 batch 走 varlen 打包: 各序列在第 0 维首尾相接, 无 batch 维, 边界查 indptr.
def extend_attention_fwd(
    q_extend,  # * [tree_len, num_heads, head_dim]
    k_extend,  # * [tree_len, num_kv_heads, head_dim]
    v_extend,  # * [tree_len, num_kv_heads, head_dim]
    o_extend,  # * [tree_len, num_heads, head_dim], 输出 buffer
    k_buffer,  # * [cache_len+tree_len, num_kv_heads, head_dim], 前缀+extend 整段 K
    v_buffer,  # * [cache_len+tree_len, num_kv_heads, head_dim]
    qo_indptr,  # * [batch+1], query 段偏移, batch=1 时 [0, tree_len]
    kv_indptr,  # * [batch+1], 前缀 KV 段偏移, batch=1 时 [0, cache_len]
    kv_indices,  # * [cache_len], 前缀 KV 行号
    custom_mask,  # * bool 扁平 tree_len*(cache_len+tree_len), 只有树内方阵段被读
    mask_indptr,  # * [batch+1], custom_mask 段偏移, batch=1 时 [0, tree_len*(cache_len+tree_len)]
    max_len_extend,  # * int, 最长 extend 行数, 决定 grid 第三维
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    head_dim = q_extend.shape[-1]

    # Get block sizes and configuration
    # * BLOCK_DMODEL: head_dim 补齐到 2 的幂; BLOCK_M: 每个 program 算多少行 query;
    # * BLOCK_N: 内层循环步长, 每次算 [BLOCK_M, BLOCK_N] 的 qk 小块;
    # * num_warps: 每个 program 的 warp 数, 线程数 = num_warps * 32
    BLOCK_M, BLOCK_DMODEL, BLOCK_N, num_warps = _get_block_sizes_for_extend_attention(
        head_dim
    )
    # *  BLOCK_M, BLOCK_N = (64, 128), num_warps = 4

    sm_scale = 1.0 / (head_dim**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]  # GQA, group_q -> k, v

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    # * 实际这里 batch size 就是小丑,
    num_stages = 1

    # 没办法, 长度和 batchsize 都混到一个维度里面了
    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
