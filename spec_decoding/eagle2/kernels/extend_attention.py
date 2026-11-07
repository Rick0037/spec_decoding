"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

_is_cuda = torch.cuda.is_available()
CUDA_CAPABILITY = (0, 0)
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    BLOCK_DMODEL = triton.next_power_of_2(Lq)
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware (CUDA)
    if _is_cuda and CUDA_CAPABILITY[0] == 12:
        # sm120 workstation Blackwell architecture (RTX Pro 6000) has a much smaller shared memory size (100K)
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (64, 128)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 32)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 9:
        # Hopper architecture (H100, etc.)
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif _is_cuda and CUDA_CAPABILITY[0] >= 8:
        # Ampere architecture (A100, etc.)
        # sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
        if CUDA_CAPABILITY[1] == 9 or CUDA_CAPABILITY[1] == 6:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        else:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
    else:
        # Older architectures / CPU fallback sizing
        BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

    num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


# Triton kernel:extend(prefill+KV cache)注意力前向主体,支持 custom_mask(树掩码).
# 两段结构:stage 1 扫前缀 KV(分页,经 kv_indices gather),stage 2 扫本次 extend
# 的新 token(连续内存,套 custom_mask)。两段共用同一组 acc/deno/e_max,
# online softmax 边扫边归并,不物化 [L, past+L] 注意力矩阵。
@triton.jit
def _fwd_kernel(
    Q_Extend,
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
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        # 前缀默认全可见;SKIP_PREFIX_CUSTOM_MASK=True(树 verify 的用法)时
        # custom_mask 只描述树内 L×L 段,前缀段不查 mask。
        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # load k in transposed way
            offs_buf_k = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            qk = tl.dot(q.to(k.dtype), k)
            qk *= sm_scale

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_buf_v = (
                offs_kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        SKIP_TILE = False
        if USE_CUSTOM_MASK:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)

            qk *= sm_scale

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd(
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
    is_causal,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    skip_prefix_custom_mask=True,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    # Get block sizes and configuration
    BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

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
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
