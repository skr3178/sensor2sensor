"""Self- and cross-attention blocks for the LiDAR U-Net.

Both blocks operate on token sequences `[B, N, C]`. The U-Net is responsible
for flattening spatial maps `[B, C, H, W] -> [B, H*W, C]` before calling
and reshaping after.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention. Q from `q_dim`, K/V from `kv_dim`.

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

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q  : [B, N_q, q_dim]
        kv : [B, N_kv, kv_dim]
        ->   [B, N_q, q_dim]
        """
        B, N_q, _ = q.shape
        N_kv = kv.shape[1]

        Q = self.to_q(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.to_k(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.to_v(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
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
    """

    def __init__(self, q_channels: int, kv_channels: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.GroupNorm(num_groups=min(32, q_channels), num_channels=q_channels)
        # KV context is pre-pooled to a fixed grid; keep a simple LayerNorm on the channel dim.
        self.norm_kv = nn.LayerNorm(kv_channels)
        self.attn = MultiHeadAttention(q_dim=q_channels, kv_dim=kv_channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, C_q,  H_q,  W_q]   query feature map
        kv : [B, C_kv, H_kv, W_kv]  pre-pooled KV context (e.g. 8x64)
        """
        B, C, H, W = x.shape
        h = self.norm_q(x).flatten(2).transpose(1, 2)         # [B, H*W, C]
        kv_seq = kv.flatten(2).transpose(1, 2)                # [B, H_kv*W_kv, C_kv]
        kv_seq = self.norm_kv(kv_seq)
        h = self.attn(h, kv_seq)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h                                          # residual


class CrossSensorSelfAttn(nn.Module):
    """Paper §3.2.3 — symmetric self-attention over `[image_tokens; lidar_tokens]`.

    Drop-in replacement for `CrossAttention`: same `(q_channels, kv_channels)`
    constructor signature and same `(x, kv)` forward signature, so it can be
    swapped in at any UNet level without changing the surrounding plumbing.

    Mechanics (per UNet block i, paper notation):
      * T_L^i  = LiDAR tokens  [B, K_L, d_i]      — from x flattened
      * T_C^i  = camera tokens [B, K_C, d_i]      — from kv flattened + projected to d_i
        (paper assumes a parallel image-side U-Net with matching d_i. We don't
         have one; instead we project the fused camera KV channel-wise via a
         per-instance `kv_to_d` linear, which is the necessary adapter so the
         shared-self-attention math still works.)
      * T_U^i  = [T_L^i ; T_C^i] in R^{(K_L + K_C) x d_i}
      * Self-attention with shared QKV is computed over T_U^i.
      * Output: keep only the LiDAR-token slice, reshape to [B, C, H, W], add
        residual. Camera tokens are not propagated back into x — they're frozen
        conditioning at this point in the network.

    A zero-init output projection (`out_proj`) keeps the block at exact identity
    at initialization, so a freshly built Scope C UNet matches its Scope B sibling
    on the first forward pass.

    Args:
        q_channels:  d_i (LiDAR feature width at this UNet level).
        kv_channels: camera-KV channel width (10 in our setup).
        num_heads:   attention heads. Must divide `q_channels`.
    """

    def __init__(self, q_channels: int, kv_channels: int, num_heads: int = 8):
        super().__init__()
        assert q_channels % num_heads == 0, (
            f"q_channels {q_channels} must be divisible by num_heads {num_heads}"
        )
        self.norm_q  = nn.GroupNorm(num_groups=min(32, q_channels), num_channels=q_channels)
        self.norm_kv = nn.LayerNorm(kv_channels)
        # Per-level adapter: lift camera tokens from `kv_channels` to `q_channels`
        # so both modalities share `d_i` as the paper assumes.
        self.kv_to_d = nn.Linear(kv_channels, q_channels)
        self.norm_u  = nn.LayerNorm(q_channels)
        # Shared QKV self-attention over the concatenated token sequence.
        self.attn = MultiHeadAttention(q_dim=q_channels, kv_dim=q_channels, num_heads=num_heads)
        self.out_proj = nn.Linear(q_channels, q_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x  : [B, C_q,  H_q,  W_q]   LiDAR feature map at this UNet level.
        kv : [B, C_kv, H_kv, W_kv]  fused camera KV (post-CrossViewFusion + token-concat).
        """
        B, C, H, W = x.shape
        N_q = H * W

        # LiDAR tokens at d_i.
        q_seq = self.norm_q(x).flatten(2).transpose(1, 2)        # [B, N_q, C]

        # Camera tokens lifted to d_i.
        kv_seq = kv.flatten(2).transpose(1, 2)                   # [B, N_kv, C_kv]
        kv_seq = self.kv_to_d(self.norm_kv(kv_seq))              # [B, N_kv, C]

        # Unified sequence; one self-attention over both modalities.
        united = torch.cat([q_seq, kv_seq], dim=1)               # [B, N_q + N_kv, C]
        united = self.norm_u(united)
        united = self.attn(united, united)                       # shared QKV self-attention

        # Keep the LiDAR-token slice; output projection is zero-init.
        h = self.out_proj(united[:, :N_q])                       # [B, N_q, C]
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h                                              # residual
