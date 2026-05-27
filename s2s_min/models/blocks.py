"""Building blocks for the LiDAR U-Net: circular conv, ResBlock, W-only downsample."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularConv2d(nn.Module):
    """Conv2d with circular padding on the W (azimuth) axis, zero padding on H.

    The LiDAR range image wraps at 0°/360° on its W axis, so a standard zero-padded
    conv corrupts the seam. H (elevation) is non-periodic.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual padding: zero on H, circular on W.
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")  # W (left, right)
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="constant", value=0.0)  # H (top, bottom)
        return self.conv(x)


class ResBlock(nn.Module):
    """Pre-norm ResNet block: GN -> SiLU -> CircConv -> GN -> SiLU -> CircConv (zero-init).

    AdaLN-Zero timestep modulation will be added in M0; for the M-1 shape test we
    leave the block free of timestep input.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        g_in = min(groups, in_ch)
        g_out = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = CircularConv2d(in_ch, out_ch, kernel_size=3)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = CircularConv2d(out_ch, out_ch, kernel_size=3)
        nn.init.zeros_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class DownsampleW(nn.Module):
    """Stride-2 conv on W axis only. Preserves H, halves W. Circular pad on W."""

    def __init__(self, channels: int):
        super().__init__()
        # Asymmetric stride: (1, 2). Circular pad on W is done manually.
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 2), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="constant", value=0.0)
        return self.conv(x)


class UpsampleW(nn.Module):
    """Nearest-neighbour upsample by 2 on W axis only, followed by circular conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CircularConv2d(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1.0, 2.0), mode="nearest")
        return self.conv(x)


class Downsample2d(nn.Module):
    """Stride-2 circular-aware conv on BOTH H and W. May change channel count.

    Used by the LiDAR VAE encoder (the diffusion U-Net uses `DownsampleW` instead,
    because its input H=8 is already too small to pool further).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = CircularConv2d(in_ch, out_ch, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    """Nearest-neighbour ×2 on BOTH H and W, followed by circular-pad conv.

    Channels preserved by this block. Channel reduction happens in a separate
    conv after the upsample (see the VAE decoder).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CircularConv2d(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderLevel(nn.Module):
    """One encoder level: N x (ResBlock + SelfAttn + CrossAttn), then DownsampleW.

    Used in M-1 as the unit under shape-test.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kv_channels: int,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        do_downsample: bool = True,
    ):
        from .attention import SelfAttention, CrossAttention

        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch))
            self.self_attns.append(SelfAttention(out_ch, num_heads=num_heads))
            self.cross_attns.append(CrossAttention(out_ch, kv_channels, num_heads=num_heads))
        self.downsample = DownsampleW(out_ch) if do_downsample else nn.Identity()

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for res, sa, ca in zip(self.res_blocks, self.self_attns, self.cross_attns):
            x = res(x)
            x = sa(x)
            x = ca(x, kv)
        return self.downsample(x)
