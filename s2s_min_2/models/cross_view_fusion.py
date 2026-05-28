"""Cross-view fusion over the 6 NuScenes camera KV grids.

After per-camera SD-VAE encode + raymap + adaptive_avg_pool, each view contributes
a `[B, C=10, H=8, W=64]` KV grid. We stack the 6 views (yielding `[B, V=6, C, H, W]`)
and flatten the view + spatial axes into a single token sequence of length `V*H*W`,
then run a stack of standard pre-norm transformer self-attention blocks that share
projections across views. The result is the same shape, but now each token can attend
to tokens from every other view — the "flatten-concat-selfattn-split" pattern from
the X-Drive paper applied at the input side of the LiDAR U-Net.

Because the per-view channel count (10) doesn't divide nicely by typical head counts,
the module up-projects to `hidden_dim` (default 64), attends there, and projects back
to 10. The output projection is zero-initialised so the module starts as identity
(input is added back via a residual), giving a stable starting point for training.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SelfAttnBlock(nn.Module):
    """Pre-norm transformer block operating in `hidden_dim`."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(q_dim=hidden_dim, kv_dim=hidden_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = _MLP(hidden_dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossViewFusion(nn.Module):
    """Mix V=6 camera KV grids by shared-projection self-attention over `V*H*W` tokens.

    Args:
        channels:    per-view channel count (= kv_context.channels = 10).
        hidden_dim:  attention width after the in-projection. Must be divisible by `num_heads`.
        num_layers:  number of transformer blocks.
        num_heads:   attention heads in each block.
        mlp_ratio:   MLP expansion factor inside each block.

    Forward:
        x: [B, V, channels, H, W]
        ->  [B, V, channels, H, W] (same shape)
    """

    def __init__(
        self,
        channels: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        )
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.in_norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.blocks = nn.ModuleList(
            [_SelfAttnBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, channels)
        # Zero-init the output projection so the module starts as identity:
        # out_proj(...) = 0, then residual adds the original x back unchanged.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = x.shape
        assert C == self.channels, f"expected channels={self.channels}, got {C}"

        # [B, V, C, H, W] -> [B, V*H*W, C]
        seq = x.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)

        h = self.in_proj(self.in_norm(seq))                # [B, V*H*W, hidden_dim]
        for block in self.blocks:
            h = block(h)
        h = self.out_proj(self.out_norm(h))                # [B, V*H*W, C]

        out = seq + h                                       # residual at C-dim
        return out.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()
