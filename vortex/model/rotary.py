# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
# from vortex.ops.embedding.rotary import apply_rotary # Removed


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0, # Not directly used by apply_rotary_emb_torch
        cu_seqlens: Optional[torch.Tensor] = None, # Not used by apply_rotary_emb_torch
        max_seqlen: Optional[int] = None, # Not used by apply_rotary_emb_torch
    ):
        # The apply_rotary_emb_torch function handles offsets by having them pre-applied to cos/sin.
        # It also doesn't support cu_seqlens or max_seqlen directly.
        # Inplace is handled by assignment if necessary in the calling code.
        out = apply_rotary_emb_torch(x, cos, sin, interleaved)
        # We only need to save what's necessary for the backward pass with apply_rotary_emb_torch.
        # seqlen_offsets, cu_seqlens, max_seqlen are not used by the PyTorch version of backward.
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        # ctx.inplace = inplace # Not directly relevant for apply_rotary_emb_torch backward
        return out # apply_rotary_emb_torch is not inplace by default

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        # The backward pass for rotary embeddings involves applying the rotary embedding
        # with conjugated rotations, which means using -sin.
        # The apply_rotary_emb_torch function can be reused with -sin.
        # Cloning `do` might still be necessary depending on its contiguity,
        # but apply_rotary_emb_torch itself doesn't have an inplace option.
        # if not ctx.interleaved: # Removed inplace check
        #     do = do.clone() # Keep cloning for safety or remove if not needed
        dx = apply_rotary_emb_torch(do, cos, -sin, ctx.interleaved)
        return dx, None, None, None, None, None, None, None # Match signature of forward


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0, # Not directly used by apply_rotary_emb_torch
        num_heads_q: Union[int] = None,
    ):
        # apply_rotary_emb_torch is not inplace. Results need to be assigned back.
        # seqlen_offsets are assumed to be handled by pre-calculating cos/sin.
        q_slice_end = num_heads_q if qkv.dim() == 4 and num_heads_q is not None else None

        if cos_k is None and sin_k is None: # and qkv.is_contiguous(): # Contiguity check less critical for non-inplace
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
                qk_out = apply_rotary_emb_torch(qk, cos, sin, interleaved)
                qkv[:, :, :2] = qk_out.reshape(batch, seqlen, 2, nheads, headdim)
            else: # qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk_slice = qkv[:, :, : num_heads_q + num_heads_k]
                qk_out = apply_rotary_emb_torch(qk_slice, cos, sin, interleaved)
                qkv[:, :, : num_heads_q + num_heads_k] = qk_out
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
                q_out = apply_rotary_emb_torch(q, cos, sin, interleaved)
                k_out = apply_rotary_emb_torch(k, cos_k, sin_k, interleaved)
                qkv[:, :, 0], qkv[:, :, 1] = q_out, k_out
            else: # qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q_slice = qkv[:, :, :num_heads_q]
                k_slice = qkv[:, :, num_heads_q : num_heads_q + num_heads_k]

                q_out = apply_rotary_emb_torch(q_slice, cos, sin, interleaved)
                k_out = apply_rotary_emb_torch(k_slice, cos_k, sin_k, interleaved)

                qkv[:, :, :num_heads_q] = q_out
                qkv[:, :, num_heads_q : num_heads_q + num_heads_k] = k_out

        # Save tensors for backward. seqlen_offsets not needed for apply_rotary_emb_torch's backward.
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q # Store num_heads_q for reshaping in backward if needed
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        # Apply inverse rotation using -sin.
        # apply_rotary_emb_torch is not inplace.
        sin_neg = -sin
        sin_k_neg = -sin_k if sin_k is not None else None

        if cos_k is None and sin_k is None: # and dqkv.is_contiguous(): # Contiguity check
            if dqkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = dqkv.shape
                assert three == 3
                dqk_slice = dqkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
                dqk_out = apply_rotary_emb_torch(dqk_slice, cos, sin_neg, ctx.interleaved)
                dqkv[:, :, :2] = dqk_out.reshape(batch, seqlen, 2, nheads, headdim)
            else: # dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dqk_slice = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
                dqk_out = apply_rotary_emb_torch(dqk_slice, cos, sin_neg, ctx.interleaved)
                dqkv[:, :, : ctx.num_heads_q + num_heads_k] = dqk_out
        else:
            cos_k = cos if cos_k is None else cos_k # Should be saved if used
            # sin_k_neg was already prepared
            if dqkv.dim() == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
                dq_out = apply_rotary_emb_torch(dq, cos, sin_neg, ctx.interleaved)
                dk_out = apply_rotary_emb_torch(dk, cos_k, sin_k_neg, ctx.interleaved)
                dqkv[:, :, 0], dqkv[:, :, 1] = dq_out, dk_out
            else: # dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dq_slice = dqkv[:, :, :ctx.num_heads_q]
                dk_slice = dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k]

                dq_out = apply_rotary_emb_torch(dq_slice, cos, sin_neg, ctx.interleaved)
                dk_out = apply_rotary_emb_torch(dk_slice, cos_k, sin_k_neg, ctx.interleaved)
                dqkv[:, :, :ctx.num_heads_q] = dq_out
                dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k] = dk_out

        return dqkv, None, None, None, None, None, None, None # Match signature of forward


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q)


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv,
        cos,
        sin,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0, # Not directly used by apply_rotary_emb_torch
    ):
        # apply_rotary_emb_torch is not inplace. Result needs to be assigned back.
        # seqlen_offsets are assumed to be handled by pre-calculating cos/sin.
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k_slice = kv[:, :, 0]
        k_out = apply_rotary_emb_torch(k_slice, cos, sin, interleaved)
        kv[:, :, 0] = k_out

        # Save tensors for backward. seqlen_offsets not needed for apply_rotary_emb_torch's backward.
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        cos, sin = ctx.saved_tensors
        # Apply inverse rotation using -sin.
        sin_neg = -sin
        dk_slice = dkv[:, :, 0]
        dk_out = apply_rotary_emb_torch(dk_slice, cos, sin_neg, ctx.interleaved)
        dkv[:, :, 0] = dk_out
        return dkv, None, None, None, None # Match signature of forward


# apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply # ApplyRotaryEmbKV_ still used directly


def apply_rotary_emb_kv_( # This function becomes a simple wrapper if ApplyRotaryEmbKV_ is used directly
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0, # Passed to ApplyRotaryEmbKV_
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv
