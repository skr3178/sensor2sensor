"""Diffusion timestep conditioning.

Two pieces:
  1. `timestep_embedding(t, dim)` — canonical sinusoidal embedding from the
     OpenAI ADM codebase (verbatim port of guided-diffusion/nn.py:103).
  2. `TimestepMLP` — the standard 2-layer projection (Linear → SiLU → Linear)
     that takes the raw sinusoidal embedding to the dim used by per-block FiLM
     injection. Matches the `self.time_embed` pattern in guided-diffusion's
     UNetModel.__init__.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    Verbatim port of OpenAI guided-diffusion `nn.py:timestep_embedding`. Identical
    function appears in CompVis SD, MVDream, LiDAR-Diffusion, RangeLDM, and X-Drive.
    Same sinusoidal pattern as Transformer positional encoding (Vaswani 2017).

    Args:
        timesteps: 1-D tensor `[B]` of timestep indices (typically int but float ok).
        dim:       output embedding dimension. Must be even (we pad if odd).
        max_period: controls the lowest frequency. 10000 is the canonical value.

    Returns:
        `[B, dim]` float tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # zero-pad an odd dim
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepMLP(nn.Module):
    """Project the raw sinusoidal embedding into a richer FiLM-conditioning vector.

    Standard pattern: `Linear(in → 4·in) → SiLU → Linear(4·in → out)`. Inline in
    most diffusion U-Nets as `self.time_embed`.

    Args:
        in_dim:  raw sinusoidal embedding dim (typically equals U-Net stem channels).
        out_dim: per-block FiLM conditioning dim (typically `4 × in_dim`).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb_sin: torch.Tensor) -> torch.Tensor:
        """t_emb_sin: [B, in_dim]  ->  [B, out_dim]."""
        return self.net(t_emb_sin)
