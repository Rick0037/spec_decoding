# SPDX-License-Identifier: Apache-2.0
"""
EAGLE3 轻量 draft 网络(对齐官方 EAGLE3 llama_eagle3 结构)。

相对 EAGLE1 的轻量 draft,这里实现了 EAGLE3 的2处关键点:

1. 多层 target hidden 特征:fc = Linear(n*H, H) 先把 target 低/中/高 n 层 hidden 拼接
   (n*H)降到 H(与 token embedding 分开),n = len(eagle_layers)。
2. 独立 draft 词表 + d2t 映射 + 不共享 lm_head(可选):给定 draft_vocab_size 时,
   draft 用自己的 lm_head: Linear(H, draft_vocab_size),输出在draft 词表上;
   再用 hot_token_id = d2t + arange 把 draft id 映射回 target id。未给定则回退复用
   target 的 lm_head(draft id 即 target id)。

注意: draft有独立此表的意义:target 的词表很大(如 128k)。但实际生成时,绝大多数概率质量集中在一小
撮高频 token 上。EAGLE3 的观察是:draft 只需要在这热门子集(hot tokens,比如几万个)上猜就够了。
2.1 draft 每步的 lm_head 矩阵乘和 softmax 维度从 V_target 降到 V_draft,draft 本来就追求轻量快速
2.2 draft 的输出头Lm_head专门为猜得准训练,与 target 的 lm_head 解耦
于是给 draft 一个更小的独立词表 + 独立 lm_head,代价是 draft 只能提议这子集内的 token,但因为它们覆
盖了主要概率质量,接受率影响很小;而且即使 draft 提议不在子集、或猜错,最终都由 target verify 兜底,不影响正确性。

3. midlayer:对 token embedding 与(降维后的)hidden 分别归一化(input_layernorm /
   hidden_norm),拼接成 2H 再进 qkv(q/k/v_proj 输入维度 = 2H)。

draft 权重随机初始化时接受率低,但 target verify 保证输出无损。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


def _unwrap(m: Union[PreTrainedModel, nn.Module]) -> PreTrainedModel:
    if isinstance(m, PreTrainedModel):
        return m
    inner = getattr(m, "model", None)
    if inner is not None and isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError("target must be HF PreTrainedModel or BaseModelForCausalLM")


class _RMSNorm(nn.Module):
    """RMSNorm"""

    def __init__(self, dim: int, eps: float = 1e-6, **fk) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, **fk))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out = t.float() * cos + _rotate_half(t.float()) * sin
    return out.to(t.dtype)


class _Eagle3MidLayer(nn.Module):
    """EAGLE3 首层(midlayer):

    对 embeds 与 hidden 分别归一化后拼成 2H 喂 self-attn(qkv 输入 = 2H,
    q/o 输出 H,k/v 输出 n_kv_heads*head_dim,即 GQA),再过 post-norm + SwiGLU。
    残差流取自传入的 hidden(降维后的多层 act)。
    """

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        intermediate: int,
        n_kv_heads: int = None,
        rope_base: float = 10000.0,
        eps: float = 1e-6,
        mrope_section: Optional[List[int]] = None,
        **fk,
    ) -> None:
        super().__init__()
        # 多模态 MRoPE 的频率分段(如 Qwen3-VL [24,20,20]); None 表示标准一维 RoPE.
        self.mrope_section = mrope_section
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = hidden // n_heads
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_dim = hidden
        self.kv_dim = kv_dim
        inv = 1.0 / (
            rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv, persistent=False)
        self.input_layernorm = _RMSNorm(hidden, eps=eps, **fk)  # 归一化 token embedding
        # 归一化(降维后的)多层 act —— EAGLE3 特有
        self.hidden_norm = _RMSNorm(hidden, eps=eps, **fk)
        self.post_attention_layernorm = _RMSNorm(hidden, eps=eps, **fk)
        # qkv 输入维度 = 2H(embed 与 hidden 拼接);q/o 输出 H,k/v 输出 kv_dim(GQA)
        # * 偷偷降维度了
        # q/k/v 合并为一次 GEMM, 输出按 [q, k, v] 切分; gate/up 同理合并。
        self.qkv_proj = nn.Linear(2 * hidden, hidden + 2 * kv_dim, bias=False, **fk)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, **fk)
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False, **fk)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, **fk)

    def _qkv(self, x: torch.Tensor, b: int, seq_len: int):
        """合并 qkv GEMM 后按 [q, k, v] 切分, 转成 (b, heads, seq, head_dim)。"""
        q, k, v = self.qkv_proj(x).split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def _mrope_recomposition(self, freq: torch.Tensor) -> torch.Tensor:
        """按 mrope_section 交错重组三轴频率, 对齐 Qwen3-VL 的 MRoPE.

        freq: (3, bs, seq, head_dim//2). 文本-only 三轴相同, 此操作恒等.
        返回 (bs, seq, head_dim).
        """
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

    def _rope_cos_sin(self, positions: torch.Tensor, device, dtype) -> tuple:
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        if positions.dim() == 3:
            # 多模态: positions (3, bs, seq), 图像段是三维网格, 文本段三轴相同.
            freqs = (
                positions.to(device=device, dtype=torch.float32)[..., None] * inv_freq
            )
            cos = self._mrope_recomposition(freqs.cos())
            sin = self._mrope_recomposition(freqs.sin())
            return cos.to(dtype), sin.to(dtype)
        # 文本: positions 一维 [seq], 标准 RoPE(cat 布局).
        pos = positions.to(device=device, dtype=torch.float32).view(-1)
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def prefill(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ):
        # * `prefill` : 一次算多个位置, 需要因果 mask.
        b, S, _ = embeds.shape
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q, k, v = self._qkv(x, b, S)
        cos, sin = self._rope_cos_sin(positions, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        new_kv = (k, v)
        # import pdb

        # pdb.set_trace()
        # SDPA 原生 GQA: 不再把 KV 物理 repeat 到 q head 数
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o = o.transpose(1, 2).contiguous().view(b, S, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self._mlp(h)
        return residual + mlp, new_kv

    def extend(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        past_kv,
        start_position: int,
        rope_delta: int = 0,
    ):
        """在 past_kv 上批量 extend A 个新 token(一次 SDPA,比逐步 step 快)."""
        # * 一次算多个新 token, 需要手动因果 mask.
        b, A, _ = embeds.shape
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q, k, v = self._qkv(x, b, A)
        positions = torch.arange(
            start_position, start_position + A, device=x.device, dtype=torch.long
        ) + int(rope_delta)
        cos, sin = self._rope_cos_sin(positions, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            # KV 沿 seq 维拼接;下面用显式 mask 保证新 Q 段对整段 K 的因果性可见性
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        past_len = int(k.shape[2]) - A
        # 显式 mask:第 i 个新 token 的 Q 只能 attend 前 past_len+i+1 个 K
        # * 这里是并行的进行kv 的更新, 但是实际上, draft的时候是串行生成的, 需要加mask 和串行生成一样才行
        # * 核心原因是: 虽然一次并行处理 A 个接受的 token, 但结果必须和一个个串行输入完全一样 .
        idx_q = torch.arange(A, device=x.device).view(A, 1)
        idx_k = torch.arange(k.shape[2], device=x.device).view(1, -1)
        allowed = idx_k <= (past_len + idx_q)
        attn_mask = torch.zeros((A, k.shape[2]), device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        o = o.transpose(1, 2).contiguous().view(b, A, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self._mlp(h)
        return residual + mlp, new_kv

    def step(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        past_kv,
        position: int,
    ):
        # * 只算当前一个位置, 全部现有 KV 都能看, 不需要 mask.
        b = embeds.shape[0]
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q, k, v = self._qkv(x, b, 1)
        cos, sin = self._rope_cos_sin(
            torch.tensor([position], device=x.device), x.device, x.dtype
        )
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            # 树 BFS 每分支每节点的1个step:在已有 KV 上追加当前 token 的 (k,v)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        o = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        o = o.transpose(1, 2).contiguous().view(b, 1, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self._mlp(h)
        return residual + mlp, new_kv

    def step_static(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        rope_position: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """固定容量 KV 的单步版本;原地写 cache，供 draft tree CUDA Graph 使用。"""
        b = embeds.shape[0]
        residual = hidden
        e = self.input_layernorm(embeds)
        hn = self.hidden_norm(hidden)
        x = torch.cat([e, hn], dim=-1)
        q, k, v = self._qkv(x, b, 1)
        cos, sin = self._rope_cos_sin(rope_position, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # cache_position 是 GPU tensor，图重放时读取新值，不需要重新 capture。
        key_cache.index_copy_(2, cache_position, k)
        value_cache.index_copy_(2, cache_position, v)
        o = F.scaled_dot_product_attention(
            q, key_cache, value_cache, attn_mask=key_mask, enable_gqa=True
        )
        o = o.transpose(1, 2).contiguous().view(b, 1, self.n_heads * self.head_dim)
        hidden = residual + self.o_proj(o)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        mlp = self._mlp(h)
        return residual + mlp


class Eagle3SglDraftModel(nn.Module):
    """
    EAGLE3 draft(官方 EAGLE3 midlayer 结构):

    forward(token_ids[B,S], target_acts[B, n*H]) -> logits[B, V](取末位置)。
    V = draft_vocab_size(给定时,draft 自己的词表)或 target vocab(回退复用 lm_head)。
    """

    def __init__(
        self,
        target: Union[PreTrainedModel, nn.Module],
        eagle_layers: Sequence[int],
        *,
        draft_vocab_size: Optional[int] = None,
        d2t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """构造 EAGLE3 draft 网络。

        Parameters
        ----------
        target
            HF CausalLM;复用其 embedding(及可选 lm_head)。
        eagle_layers
            eagle3 act目标layer下标;n_acts = len(eagle_layers),决定 fc 输入维 n*H。
        draft_vocab_size
            独立 draft 词表大小;给定则建 lm_head(H, draft_vocab_size) + hot_token_id。
        d2t
            draft id → target id 偏移;hot_token_id[i] = d2t[i] + i。
        device, dtype
            新建参数(fc/midlayer/lm_head)的设备与 dtype。
        """
        super().__init__()
        tm = _unwrap(target)
        # 多模态模型(如 Qwen3-VL)的语言维度在 config.text_config, 纯文本模型直接在 config.
        cfg = getattr(tm.config, "text_config", tm.config)
        hidden = int(cfg.hidden_size)
        n_heads = int(getattr(cfg, "num_attention_heads", max(1, hidden // 64)))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        # * mlp middle dim
        intermediate = int(getattr(cfg, "intermediate_size", 4 * hidden))
        # rope_theta 新版放在 rope_parameters/rope_scaling, 旧版是顶层 rope_theta 属性.
        rope_params = (
            getattr(cfg, "rope_parameters", None)
            or getattr(cfg, "rope_scaling", None)
            or {}
        )
        rope_base = float(
            rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        )
        eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
        n_acts = len(list(eagle_layers))
        self.n_acts = n_acts
        self.hidden = hidden

        embed = tm.get_input_embeddings()
        # 复用 target embedding.
        self._shared_embed = [embed]

        if device is None:
            device = embed.weight.device
        if dtype is None:
            dtype = embed.weight.dtype
        fk = {"device": device, "dtype": dtype}

        # (1) 多层 aux 特征:fc 把 n 层拼接的 act(n*H)降到 H(与 embedding 分开).
        self.fc = nn.Linear(n_acts * hidden, hidden, bias=False, **fk)
        # (2) midlayer:embed/hidden 分别归一化后拼 2H 进 qkv(支持 GQA).
        self.midlayer = _Eagle3MidLayer(
            hidden,
            n_heads,
            intermediate,
            n_kv_heads=n_kv_heads,
            rope_base=rope_base,
            eps=eps,
            mrope_section=rope_params.get("mrope_section"),
            **fk,
        )
        self.norm = _RMSNorm(hidden, eps=eps, **fk)

        # (3) 独立 draft 词表 + d2t 映射 + 不共享 lm_head.
        self.draft_vocab_size = draft_vocab_size
        if draft_vocab_size is not None:
            self.lm_head = nn.Linear(
                hidden, draft_vocab_size, bias=False, **fk
            )  # draft 自己的输出头
            if d2t is None:
                d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
            # hot_token_id[draft_id] = 该 draft id 对应的 target vocab id
            hot = d2t.to(torch.long) + torch.arange(draft_vocab_size, dtype=torch.long)
            # * hot.shape -> [32000], hot_token_id.shape[32000]
            # * 加载权重之后 通过index 取 实际在target词表中得token ids
            self.register_buffer("hot_token_id", hot.to(device=device))
            self._shared_lmhead = None
        else:
            self.lm_head = None
            self.hot_token_id = None
            self._shared_lmhead = [tm.get_output_embeddings()]

    @property
    def embed(self) -> nn.Module:
        return self._shared_embed[0]

    def _lm_logits(self, x: torch.Tensor) -> torch.Tensor:
        """末 hidden → logits;基于独立词表或回退到 target lm_head / tied embed."""
        if self.lm_head is not None:
            return self.lm_head(x)  # draft 词表 [B, V_draft]
        lmh = self._shared_lmhead[0]
        if lmh is not None:
            return lmh(x)
        return F.linear(x, self.embed.weight)  # tied embedding

    @torch.inference_mode()
    def step(
        self,
        token_id: torch.LongTensor,
        cond: torch.Tensor,
        past_kv,
        position: int,
    ):
        """自回归单 token 步(expand_draft_tree_ar 建树用)。

        Parameters
        ----------
        token_id
            当前步输入 token [B, 1]。
        cond
            draft 条件 hidden [B, H] 或 [B, 1, n*H];若末维为 n*H 则过 fc。
        past_kv
            已有 draft KV (k, v) 或 None。
        position
            RoPE 绝对位置(树内 = draft_kv_length + depth - 1)。

        Returns
        -------
        logits, hidden, new_kv
            末步 logits [B, V]、更新后的 cond hidden [B, H]、extend 后的 KV。
        """
        embeds = self.embed(token_id)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond.shape[-1] == self.n_acts * self.hidden:
            reduced = self.fc(cond.to(embeds.dtype))  # 首轮 cond 仍是 n*H act
        else:
            # 后续步 cond 已是 H(feature 递归)
            reduced = cond.to(embeds.dtype)
        h, new_kv = self.midlayer.step(embeds, reduced, past_kv, position)
        # 末位置 norm + lm_head → 本步对下一 token的分布;h[:, -1] 作为下一步 cond(feature 递归)
        logits = self._lm_logits(self.norm(h)[:, -1, :])
        return logits, h[:, -1, :], new_kv

    @torch.inference_mode()
    def step_static(
        self,
        token_id: torch.LongTensor,
        cond: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        rope_position: torch.Tensor,
        key_mask: torch.Tensor,
    ):
        """draft tree graph 的单步接口;KV 由调用方的固定 buffer 持有。

        返回 (norm_hidden, feature_hidden);lm_head 由调用方施加(Full 或 SVD 候选 head)。
        """
        embeds = self.embed(token_id)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if cond.shape[-1] == self.n_acts * self.hidden:
            reduced = self.fc(cond.to(embeds.dtype))
        else:
            reduced = cond.to(embeds.dtype)
        h = self.midlayer.step_static(
            embeds,
            reduced,
            key_cache,
            value_cache,
            cache_position,
            rope_position,
            key_mask,
        )
        return self.norm(h)[:, -1, :], h[:, -1, :]

    @torch.inference_mode()
    def prefill(
        self,
        prefix_ids: torch.LongTensor,
        target_acts_all: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        #! draft model 也需要自己得kv cache
        """对 prompt做 draft prefill。

        Parameters
        ----------
        prefix_ids
            完整 prompt [1, P]。
        target_acts_all
            各位置多层 act [1, P, n*H](与 prefix_ids 等长)。

        Returns
        -------
        root_logits, root_hidden, kv
            末步 logits [1, V]、末 hidden [1, H]、draft KV(长 P-1)。

        EAGLE shift:draft 消费 prefix_ids[:,1:] 与 acts_all[:,:-1],
        即位置 t 的 draft 输入 act 来自 target 在 t-1 处的多层特征。
        """
        # EAGLE shift:draft 不消费 prompt 首 token
        # import pdb

        # pdb.set_trace()
        ids = prefix_ids[:, 1:]  # * 丢掉第一个token
        # 位置 t 的 draft 条件 = target 在 t-1 的多层 act
        # * (Pdb) target_acts_all.shape
        # *  torch.Size([1, 21, 12288])
        acts = target_acts_all[:, :-1, :]  # * 去掉最后一个
        embeds = self.embed(ids)
        S = embeds.shape[1]
        if S == 0:
            raise ValueError(
                "prefix too short for EAGLE3 draft prefill (need len >= 2)"
            )
        # * torch.Size([1, 21, 12288])-> torch.Size([1, 21, 4096])
        reduced = self.fc(acts.to(embeds.dtype))
        if position_ids is not None:
            # 多模态: position_ids 形状 (3, bs, seq), 第三维取前 S 个(对应 act 0..S-1).
            positions = position_ids[:, :, :S].to(embeds.device)
        else:
            positions = torch.arange(S, device=embeds.device)  # RoPE 0..S-1
        # 整段因果 self-attn,
        # 输出末 hidden + 长度为prefill tokens的 KV(供后续 extend;KV 条数 = prefill token 数)
        h, kv = self.midlayer.prefill(embeds, reduced, positions)
        # (Pdb) kv[0].shape
        # torch.Size([1, 8, 20, 128])
        last = self.norm(h)[:, -1, :]
        return self._lm_logits(last), h[:, -1, :], kv

    @torch.inference_mode()
    def extend_tokens(
        self,
        token_ids: torch.LongTensor,
        target_acts: torch.Tensor,
        past_kv,
        start_position: int,
        rope_delta: int = 0,
    ):
        """把本轮接受的 token 写入 draft, 并更新 KV 缓存.

        token_ids: 本轮接受的 token [batch_size, accept_len].
        target_acts: 对应的 target 特征 [batch_size, accept_len, n*H];
        TODO: 注意现在实际错位使用的token使用前一位置feature.
        past_kv: 之前保存的 draft KV.
        start_position: 首个 token 的 RoPE 位置, 等于 p_old - 1,
        rope_delta: 多模态时的 MRoPE 位置偏移(文本为 0).

        返回下一个 token 的 logits, 末位置 hidden 和更新后的 KV.
        """
        # A = int(token_ids.shape[1])
        # if A == 0:
        #     raise ValueError("extend_tokens requires at least one token")
        # if target_acts.shape[1] != A:
        #     raise ValueError(
        #         f"target_acts seq {target_acts.shape[1]} != token_ids seq {A}"
        #     )
        embeds = self.embed(token_ids)
        reduced = self.fc(target_acts.to(embeds.dtype))
        # 批量更新 KV等draft state
        h, kv = self.midlayer.extend(
            embeds, reduced, past_kv, int(start_position), rope_delta=rope_delta
        )
        last = self.norm(h)[:, -1, :]
        return self._lm_logits(last), h[:, -1, :], kv

    @torch.inference_mode()
    def convert_to_target_topk_token_ids(self, logits: torch.Tensor, k: int):
        """logits top-k;独立词表时经 hot_token_id 映射回 target id。

        Parameters
        ----------
        logits
            [B, V_draft] 或 [B, V_target]。
        k
            保留候选数(不超过词表大小)。

        Returns
        -------
        idx, scores
            target 词表空间的 token id 与对应分数。
        """
        k = min(k, logits.shape[-1])
        scores, idx = torch.topk(logits, k=k, dim=-1)
        if self.hot_token_id is not None:
            idx = self.hot_token_id[idx]  # draft_id → target_id
        return idx, scores


def build_eagle3_sgl_draft(
    target: Union[PreTrainedModel, nn.Module],
    eagle_layers: Sequence[int],
    *,
    # `*` 后面的参数必须写参数名传递
    draft_vocab_size: Optional[int] = None,
    d2t: Optional[torch.Tensor] = None,
    draft_weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Eagle3SglDraftModel:
    """构造 EAGLE3 draft。

    权重来源由 draft_weights_path 控制:
    - draft_weights_path=None(默认):随机初始化(结构对齐官方 EAGLE3,可跑通但接受率低,
      正确性仍由 target verify 保证);
    - draft_weights_path=<dir 或 .safetensors/.bin/.pt>:加载 fine-tune 好的 EAGLE3 draft
      权重(fc / midlayer / lm_head / d2t),加载真实 checkpoint 可获得实际接受率。

    词表相关:
    - 不传 draft_vocab_size:复用 target lm_head(draft id 即 target id)。
    - 传 draft_vocab_size(+d2t) 或从 checkpoint 加载 d2t:独立 draft 词表 + lm_head,
      topk 后用 d2t 映射回 target id。加载真实 ckpt 时,draft_vocab_size 应与 ckpt 的
      draft 词表大小一致(= len(d2t))。
    """
    draft_model = Eagle3SglDraftModel(
        target,
        eagle_layers,
        draft_vocab_size=draft_vocab_size,
        d2t=d2t,
        device=device,
        dtype=dtype,
    )
    if draft_weights_path is not None:
        load_eagle3_draft_weights(draft_model, draft_weights_path)
    return draft_model


def _read_state_dict(path: str) -> dict:
    """从目录 / .safetensors / .bin / .pt 读出一个扁平的 name->tensor 状态字典."""
    import glob
    import os

    files: List[str]
    if os.path.isdir(path):
        files = (
            sorted(glob.glob(os.path.join(path, "*.safetensors")))
            or sorted(glob.glob(os.path.join(path, "*.bin")))
            or sorted(glob.glob(os.path.join(path, "*.pt")))
        )
        if not files:
            raise FileNotFoundError(
                f"no weight file (*.safetensors/*.bin/*.pt) under {path}"
            )
    else:
        files = [path]

    sd: dict = {}
    for f in files:
        # 目录下可能多个 shard:合并进同一 state_dict
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd.update(load_file(f))
        else:
            obj = torch.load(f, map_location="cpu", weights_only=True)
            sd.update(obj.get("state_dict", obj) if isinstance(obj, dict) else obj)
    return sd


def _translate_eagle3_name(name: str) -> str:
    """把官方 EAGLE3 checkpoint 的参数名翻译到本模块的命名。

    - 去掉可能的 model. 前缀;
    - 去掉 self_attn. 段(如 midlayer.self_attn.q_proj → midlayer.q_proj);
    - 去掉 mlp. 段(如 midlayer.mlp.gate_proj → midlayer.gate_proj)。
    """
    if name.startswith("model."):
        name = name[len("model.") :]
    return name.replace("self_attn.", "").replace("mlp.", "")


def load_eagle3_draft_weights(
    draft_model: Eagle3SglDraftModel,
    path: str,
    *,
    strict: bool = False,
) -> dict:
    """把 fine-tune 好的 EAGLE3 draft 权重加载进 draft_model。

    处理:名称翻译(对齐官方 checkpoint 命名)、跳过 embed_tokens(复用 target)、
    由 d2t 计算 hot_token_id、按 my-layout 加载 fc/midlayer/norm/lm_head。

    Returns: {"loaded": [...], "missing": [...], "unexpected": [...]}。
    """
    src = _read_state_dict(path)
    local: dict = {}
    d2t = None
    for k, v in src.items():
        base = k.split(".")[-1]
        if base == "d2t":
            d2t = v
            continue
        if "t2d" in k or "embed_tokens" in k:
            continue  # t2d 不需要;embed 复用 target
        # 键名对齐本模块 Parameter 名(如去掉 self_attn. 前缀)
        local[_translate_eagle3_name(k)] = v

    # 由 d2t 设 hot_token_id(draft id -> target id)
    if d2t is not None and getattr(draft_model, "hot_token_id", None) is not None:
        hot = d2t.to(torch.long) + torch.arange(d2t.shape[0], dtype=torch.long)
        local["hot_token_id"] = hot.to(draft_model.hot_token_id.device)

    # q/k/v 与 gate/up 分别 cat 成合并 GEMM 的权重(模块内只有 qkv_proj/gate_up_proj)
    fused: dict = {}
    for fused_name, part_names in (
        (
            "midlayer.qkv_proj.weight",
            ("midlayer.q_proj.weight", "midlayer.k_proj.weight", "midlayer.v_proj.weight"),
        ),
        (
            "midlayer.gate_up_proj.weight",
            ("midlayer.gate_proj.weight", "midlayer.up_proj.weight"),
        ),
    ):
        if all(part in local for part in part_names):
            fused[fused_name] = torch.cat([local.pop(part) for part in part_names], dim=0)
    local.update(fused)

    res = draft_model.load_state_dict(local, strict=strict)
    unexpected = set(res.unexpected_keys)
    return {
        "loaded": [k for k in local if k not in unexpected],
        "missing": list(res.missing_keys),
        "unexpected": list(res.unexpected_keys),
    }
