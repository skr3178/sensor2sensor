"""Self- and cross-attention blocks for the LiDAR U-Net.

Both blocks operate on token sequences `[B, N, C]`. The U-Net is responsible
for flattening spatial maps `[B, C, H, W] -> [B, H*W, C]` before calling
and reshaping after.

Cross-attention adds 2D sinusoidal positional encoding (DiT / MMDiT style)
to Q and K *after projection*, in the per-head dim space. V is left untouched
so it carries pure content. The pos-enc is a fixed sin/cos buffer keyed by
(H, W, dim, device, dtype) — 0 params, ~0 FLOPs at runtime after the first
forward.

Why post-projection: it keeps the KV input channels (10 = 4 image + 6 raymap)
intact through the to_k/to_v linears, then lifts to q_dim where there's
plenty of bandwidth to express row/col position. Pre-projection pos-enc on
10-channel KV would either waste channels or collide with the conditioning
signal.

Reference: `Reference_code/diffusers/src/diffusers/models/embeddings.py`
(`get_2d_sincos_pos_embed_from_grid`, lines 288–318).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def disable_cross_attn_pos_enc():
    """Temporarily disable Fix #1 cross-attention pos-enc.

    Used for ablations / fair eval of pre-Fix#1 checkpoints. The original
    state is restored even if the body raises.
    """
    # Imported here to avoid pre-binding before the class is defined.
    prev = CrossAttention._USE_POS_ENC
    CrossAttention._USE_POS_ENC = False
    try:
        yield
    finally:
        CrossAttention._USE_POS_ENC = prev


def _build_2d_sincos_pos_embed(
    embed_dim: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 2D sin/cos positional embedding of shape `[H*W, embed_dim]`.

    `embed_dim` is split evenly between H and W. Each half must be divisible
    by 2 (sin/cos pair). If `embed_dim` is odd the last channel is left zero.
    """
    assert embed_dim >= 4, f"embed_dim {embed_dim} too small for 2D sincos"
    half = embed_dim // 2
    # round each half down to nearest even number of channels (so sin/cos pair fits)
    dim_h = (half // 2) * 2
    dim_w = (half // 2) * 2
    used = dim_h + dim_w
    leftover = embed_dim - used  # 0, 2, or so — zero-padded at the end

    # 1D sincos for H positions in `dim_h` channels.
    pos_h = torch.arange(H, device=device, dtype=torch.float32)
    pos_w = torch.arange(W, device=device, dtype=torch.float32)
    omega_h = torch.arange(dim_h // 2, device=device, dtype=torch.float32)
    omega_h = 1.0 / (10000.0 ** (omega_h / (dim_h / 2.0)))
    omega_w = torch.arange(dim_w // 2, device=device, dtype=torch.float32)
    omega_w = 1.0 / (10000.0 ** (omega_w / (dim_w / 2.0)))

    out_h = torch.outer(pos_h, omega_h)                            # [H, dim_h/2]
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # [H, dim_h]

    out_w = torch.outer(pos_w, omega_w)                            # [W, dim_w/2]
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # [W, dim_w]

    # Broadcast to grid: [H, W, dim_h] and [H, W, dim_w], then concat along channel.
    emb_h_grid = emb_h.unsqueeze(1).expand(H, W, dim_h)            # [H, W, dim_h]
    emb_w_grid = emb_w.unsqueeze(0).expand(H, W, dim_w)            # [H, W, dim_w]
    emb = torch.cat([emb_h_grid, emb_w_grid], dim=-1)              # [H, W, used]

    if leftover > 0:
        pad = torch.zeros(H, W, leftover, device=device, dtype=torch.float32)
        emb = torch.cat([emb, pad], dim=-1)                        # [H, W, embed_dim]

    emb = emb.reshape(H * W, embed_dim).to(dtype)
    return emb


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention. Q from `q_dim`, K/V from `kv_dim`.

    Optional `q_pos` and `k_pos` are additive positional embeddings applied
    AFTER the to_q / to_k projections (DiT / MMDiT style — pos info enters
    the attention logits via Q and K, V stays content-only).

    No flash-attn dep; uses torch's scaled_dot_product_attention.
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 8):
        super().__init__()
        assert q_dim % num_heads == 0, f"q_dim {q_dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, q_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, q_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, q_dim, bias=False)
        self.to_out = nn.Linear(q_dim, q_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_pos: torch.Tensor | None = None,
        k_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q     : [B, N_q,  q_dim]
        kv    : [B, N_kv, kv_dim]
        q_pos : [1, N_q,  q_dim]  optional additive pos-embed in q_dim space
        k_pos : [1, N_kv, q_dim]  optional additive pos-embed in q_dim space
        ->    : [B, N_q,  q_dim]
        """
        B, N_q, _ = q.shape
        N_kv = kv.shape[1]

        Q = self.to_q(q)                       # [B, N_q, q_dim]
        K = self.to_k(kv)                      # [B, N_kv, q_dim]
        V = self.to_v(kv)                      # [B, N_kv, q_dim]
        if q_pos is not None:
            Q = Q + q_pos
        if k_pos is not None:
            K = K + k_pos

        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # shape now: [B, heads, N, head_dim]

        out = F.scaled_dot_product_attention(Q, K, V)         # [B, heads, N_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention on a spatial feature map `[B, C, H, W]`."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.attn = MultiHeadAttention(q_dim=channels, kv_dim=channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)            # [B, H*W, C]
        h = self.attn(h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h                                # residual


class CrossAttention(nn.Module):
    """Cross-attention from spatial feature map to a pre-pooled KV context map.

    Both inputs are `[B, C, H, W]` but Q and KV may have different (H,W) and
    different channel counts.

    Adds 2D sin/cos positional embedding to Q and K AFTER projection (so the
    pos-enc lives in q_dim space, not the small kv_dim space). The embeddings
    are computed lazily and cached by (H, W, dim, device, dtype) to keep
    runtime cost at 0 after the first forward per shape.
    """

    # Class-level toggle for ablations / fair back-compat eval. When False,
    # forward() skips computing & adding pos_embed — model behaves like the
    # pre-Fix#1 (bag-of-features) cross-attention. Use the
    # `disable_cross_attn_pos_enc()` context manager (defined below) rather
    # than touching this directly so the global flag is always restored.
    _USE_POS_ENC: bool = True

    def __init__(self, q_channels: int, kv_channels: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.GroupNorm(num_groups=min(32, q_channels), num_channels=q_channels)
        # KV context is pre-pooled to a fixed grid; keep a simple LayerNorm on the channel dim.
        self.norm_kv = nn.LayerNorm(kv_channels)
        self.attn = MultiHeadAttention(q_dim=q_channels, kv_dim=kv_channels, num_heads=num_heads)

        self.q_channels = q_channels
        # In-memory cache of pos-embed tensors, keyed by (H, W, dim, device, dtype).
        # Buffers (rather than nn.Parameter) so they're not part of state_dict and
        # the back-compat path is clean. ~32 KiB per shape entry — trivial.
        self._pos_cache: dict[tuple[int, int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def _get_pos_embed(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (H, W, self.q_channels, device, dtype)
        cached = self._pos_cache.get(key)
        if cached is None:
            cached = _build_2d_sincos_pos_embed(self.q_channels, H, W, device, dtype)
            cached = cached.unsqueeze(0)              # [1, H*W, q_channels]
            self._pos_cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, C_q,  H_q,  W_q]   query feature map
        kv : [B, C_kv, H_kv, W_kv]  pre-pooled KV context (e.g. 8x64)
        """
        B, C, H, W = x.shape
        H_kv, W_kv = kv.shape[2], kv.shape[3]

        h = self.norm_q(x).flatten(2).transpose(1, 2)         # [B, H*W, C]
        kv_seq = kv.flatten(2).transpose(1, 2)                # [B, H_kv*W_kv, C_kv]
        kv_seq = self.norm_kv(kv_seq)

        if CrossAttention._USE_POS_ENC:
            q_pos = self._get_pos_embed(H,    W,    x.device, h.dtype)         # [1, H*W, C_q]
            k_pos = self._get_pos_embed(H_kv, W_kv, x.device, kv_seq.dtype)    # [1, H_kv*W_kv, C_q]
            h = self.attn(h, kv_seq, q_pos=q_pos, k_pos=k_pos)
        else:
            h = self.attn(h, kv_seq)  # ablation / back-compat: bag-of-features
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h                                          # residual
