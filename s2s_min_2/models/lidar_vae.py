"""LiDAR range-image VAE (M1).

Compresses a 3-channel range image (range, intensity, validity) to a
`latent_channels`-channel latent at 4x lower spatial resolution on both axes,
and decodes back. Fully convolutional → the spatial dims are inherited from
the input; only the channel widths are baked in.

    encode:  [B, 3, H, W]      -> mu, logvar each [B, lc, H/4, W/4]
    decode:  [B, lc, H/4, W/4] -> recon [B, 3, H, W] in [0, 1]

Concrete shapes per dataset (default `latent_channels=8`):

                          input               latent
    nuScenes HDL-32E   [B, 3,  32, 1024]   [B, 8,  8,  256]
    Waymo TOP (cropped) [B, 3,  64, 2048]   [B, 8, 16, 512]

The annotations in `encode()` / `decode()` use the symbolic shapes — the older
nuScenes-specific numbers live in the dataset-specific docs.

Trained from scratch in M1, then frozen for M2/M3/M4.

Shape / channel / loss spec: s2s_min/docs/models.md sections 2.1-2.2.
Implemented through models.md sec 2.4 steps 2-5. Step 6 (loss function) is
the next file to land.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention
from .blocks import CircularConv2d, Downsample2d, ResBlock, Upsample2d


class LiDARVAE(nn.Module):
    """Range-image VAE. See module docstring for shapes."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        base_channels: int = 32,
        num_res_blocks: int = 2,
        groupnorm_groups: int = 32,
        num_attn_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Channel widths at each resolution. Spatial dims shown are
        # nuScenes (H=32 W=1024) / Waymo TOP (H=64 W=2048).
        ch1 = base_channels          # 32  @ H     x W
        ch2 = base_channels * 2      # 64  @ H/2   x W/2
        ch3 = base_channels * 4      # 128 @ H/4   x W/4

        # ============================ encoder ============================
        self.enc_stem = CircularConv2d(in_channels, ch1, kernel_size=3)

        self.enc_stage1 = nn.ModuleList(
            [ResBlock(ch1, ch1, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )
        self.enc_down1 = Downsample2d(ch1, ch2)

        self.enc_stage2 = nn.ModuleList(
            [ResBlock(ch2, ch2, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )
        self.enc_down2 = Downsample2d(ch2, ch3)

        self.enc_bottleneck = nn.ModuleList(
            [ResBlock(ch3, ch3, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )
        self.enc_bottleneck_attn = SelfAttention(ch3, num_heads=num_attn_heads)

        self.enc_head_norm = nn.GroupNorm(min(groupnorm_groups, ch3), ch3)
        # k=1 conv emits both mu and logvar concatenated on the channel axis.
        self.enc_head_conv = CircularConv2d(ch3, 2 * latent_channels, kernel_size=1)

        # ============================ decoder ============================
        self.dec_stem = CircularConv2d(latent_channels, ch3, kernel_size=3)

        self.dec_bottleneck_attn = SelfAttention(ch3, num_heads=num_attn_heads)
        self.dec_bottleneck = nn.ModuleList(
            [ResBlock(ch3, ch3, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )

        # Upsample preserves channels; a separate conv handles the channel reduction.
        self.dec_up2 = Upsample2d(ch3)
        self.dec_ch_down2 = CircularConv2d(ch3, ch2, kernel_size=3)
        self.dec_stage2 = nn.ModuleList(
            [ResBlock(ch2, ch2, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )

        self.dec_up1 = Upsample2d(ch2)
        self.dec_ch_down1 = CircularConv2d(ch2, ch1, kernel_size=3)
        self.dec_stage1 = nn.ModuleList(
            [ResBlock(ch1, ch1, groups=groupnorm_groups) for _ in range(num_res_blocks)]
        )

        self.dec_head_norm = nn.GroupNorm(min(groupnorm_groups, ch1), ch1)
        self.dec_head_conv = CircularConv2d(ch1, in_channels, kernel_size=3)

        # Zero-init head so a fresh decoder outputs 0 -> sigmoid -> 0.5 per channel.
        # Acts as a stable starting point for the recon loss (matches the dataset's
        # midpoint better than random predictions).
        nn.init.zeros_(self.dec_head_conv.conv.weight)
        nn.init.zeros_(self.dec_head_conv.conv.bias)

    # ----- encoder -------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a normalized range image to posterior parameters (mu, logvar).

        Args:
            x: range image in [0, 1], shape [B, 3, H, W].
        Returns:
            mu, logvar: each [B, latent_channels, H/4, W/4].
        """
        h = self.enc_stem(x)                                # [B,  ch1, H,   W  ]

        for blk in self.enc_stage1:                         # [B,  ch1, H,   W  ]
            h = blk(h)
        h = self.enc_down1(h)                               # [B,  ch2, H/2, W/2]

        for blk in self.enc_stage2:                         # [B,  ch2, H/2, W/2]
            h = blk(h)
        h = self.enc_down2(h)                               # [B,  ch3, H/4, W/4]

        for blk in self.enc_bottleneck:                     # [B,  ch3, H/4, W/4]
            h = blk(h)
        h = self.enc_bottleneck_attn(h)                     # [B,  ch3, H/4, W/4]

        h = F.silu(self.enc_head_norm(h))
        h = self.enc_head_conv(h)                           # [B, 2·lc, H/4, W/4]

        mu, logvar = h.chunk(2, dim=1)                      # each [B, lc, H/4, W/4]
        return mu, logvar

    # ----- decoder -------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent to a reconstructed range image in [0, 1].

        Args:
            z: latent tensor [B, latent_channels, H/4, W/4].
        Returns:
            x_hat: [B, 3, H, W], per-channel sigmoid -> [0, 1].
        """
        h = self.dec_stem(z)                                # [B, ch3, H/4, W/4]

        h = self.dec_bottleneck_attn(h)                     # [B, ch3, H/4, W/4]
        for blk in self.dec_bottleneck:                     # [B, ch3, H/4, W/4]
            h = blk(h)

        h = self.dec_up2(h)                                 # [B, ch3, H/2, W/2]
        h = self.dec_ch_down2(h)                            # [B, ch2, H/2, W/2]
        for blk in self.dec_stage2:                         # [B, ch2, H/2, W/2]
            h = blk(h)

        h = self.dec_up1(h)                                 # [B, ch2, H,   W  ]
        h = self.dec_ch_down1(h)                            # [B, ch1, H,   W  ]
        for blk in self.dec_stage1:                         # [B, ch1, H,   W  ]
            h = blk(h)

        h = F.silu(self.dec_head_norm(h))
        h = self.dec_head_conv(h)                           # [B,   3, H,   W  ]

        return torch.sigmoid(h)                             # [0, 1] per channel

    # ----- sampling + forward -------------------------------------------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps during training; z = mu at eval."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(mu)
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
