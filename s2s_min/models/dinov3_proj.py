"""Trainable DINOv3 conditioning projection (Option B).

Frozen per-channel standardization (mean/std from the cache manifest) followed by a learned
1×1 conv 384→4. The 1×1 conv commutes with the later bilinear upsample to 32×56, so we project
on the cheap 14×24 patch grid and upsample the 4-channel result in the training loop.

The standardization stats are stored as buffers so they travel with the checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DINOv3Proj(nn.Module):
    def __init__(self, feat_mean, feat_std, in_ch: int = 384, out_ch: int = 4):
        super().__init__()
        mean = torch.as_tensor(feat_mean, dtype=torch.float32).view(1, in_ch, 1, 1)
        std = torch.as_tensor(feat_std, dtype=torch.float32).view(1, in_ch, 1, 1).clamp(min=1e-6)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 384, h, w] (any spatial size). Returns [B, out_ch, h, w]."""
        x = (x - self.mean) / self.std
        return self.conv(x)
