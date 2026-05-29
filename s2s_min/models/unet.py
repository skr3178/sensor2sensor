"""LiDAR diffusion U-Net (denoiser backbone) — trained in M3.

Operates on the LiDAR VAE latent. Conditions on a pre-pooled image + raymap
context via cross-attention. Standard latent-diffusion U-Net family
(SD / OpenAI ADM / RangeLDM lineage), with two LiDAR-specific adaptations:

  1. Circular padding on the W (azimuth) axis at every conv.
  2. W-only downsampling — H stays at 8 throughout because the input latent's
     H is already small after the LiDAR VAE's 4× spatial compression.

Topology (committed): stem → 2 encoder levels → bottleneck → 2 decoder levels → head.
Full spec: see s2s_min/docs/lidar-unet.md §1.

Build references:
  * OpenAI guided-diffusion `unet.py` (TimestepEmbedSequential pattern, UNetModel skeleton)
  * LiDAR-Diffusion `model_lidm.py` (circular + anisotropic adapter patterns)
  * Our own `s2s_min/models/{attention,blocks,timestep}.py`
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, SelfAttention
from .blocks import CircularConv2d, EncoderLevel, ResBlock, UpsampleW
from .timestep import TimestepMLP, timestep_embedding


# ────────────────────────────────────────────────────────────────────────────────
# Dispatch glue — port of guided-diffusion `TimestepEmbedSequential` extended for
# our (x, kv) cross-attention signature.
# ────────────────────────────────────────────────────────────────────────────────


class TimestepEmbedSequential(nn.Sequential):
    """Sequential that routes `t_emb` to ResBlocks and `kv` to CrossAttention.

    Plain modules receive only `x`. ResBlock receives `(x, t_emb)`. CrossAttention
    receives `(x, kv)`. SelfAttention and conv layers receive just `x`.
    """

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb=t_emb)
            elif isinstance(layer, CrossAttention):
                x = layer(x, kv)
            else:
                x = layer(x)
        return x


# ────────────────────────────────────────────────────────────────────────────────
# Bottleneck — `N × [ResBlock + SelfAttn + CrossAttn]` at the deepest level
# (no down/up sample). Mirrors guided-diffusion's UNetModel.middle_block pattern.
# ────────────────────────────────────────────────────────────────────────────────


class Bottleneck(nn.Module):
    """Bottleneck stage: N triplets of (ResBlock + SelfAttn + CrossAttn) at fixed spatial size.

    Args:
        in_ch:          incoming channels (first ResBlock's input).
        out_ch:         output channels (all subsequent ops run at this width).
        kv_channels:    channels of the cross-attention KV context.
        num_res_blocks: how many triplets.
        num_heads:      attention head count.
        t_emb_dim:      FiLM conditioning dim passed to each ResBlock.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kv_channels: int,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        t_emb_dim: int | None = None,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(in_ch if i == 0 else out_ch, out_ch, t_emb_dim=t_emb_dim)
            )
            self.self_attns.append(SelfAttention(out_ch, num_heads=num_heads))
            self.cross_attns.append(CrossAttention(out_ch, kv_channels, num_heads=num_heads))

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for res, sa, ca in zip(self.res_blocks, self.self_attns, self.cross_attns):
            x = res(x, t_emb=t_emb)
            x = sa(x)
            x = ca(x, kv)
        return x


# ────────────────────────────────────────────────────────────────────────────────
# DecoderLevel — UpsampleW → skip-concat → N triplets of (ResBlock + SelfAttn + CrossAttn).
# Mirror of EncoderLevel with up-first instead of down-last.
# ────────────────────────────────────────────────────────────────────────────────


class DecoderLevel(nn.Module):
    """One decoder level: UpsampleW → cat(skip) → N triplets.

    Args:
        in_ch:          incoming channels (BEFORE skip-concat).
        skip_ch:        channels of the encoder skip feature concatenated in.
        out_ch:         output channels (all blocks at this width).
        kv_channels:    channels of the cross-attention KV context.
        num_res_blocks: how many triplets.
        num_heads:      attention head count.
        do_upsample:    if True, prepend an UpsampleW. Default True.
        t_emb_dim:      FiLM conditioning dim passed to each ResBlock.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        kv_channels: int,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        do_upsample: bool = True,
        t_emb_dim: int | None = None,
    ):
        super().__init__()
        self.upsample = UpsampleW(in_ch) if do_upsample else nn.Identity()
        # After upsample + cat(skip), the first ResBlock sees `in_ch + skip_ch`.
        cat_ch = in_ch + skip_ch
        self.res_blocks = nn.ModuleList()
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(cat_ch if i == 0 else out_ch, out_ch, t_emb_dim=t_emb_dim)
            )
            self.self_attns.append(SelfAttention(out_ch, num_heads=num_heads))
            self.cross_attns.append(CrossAttention(out_ch, kv_channels, num_heads=num_heads))

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        kv: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for res, sa, ca in zip(self.res_blocks, self.self_attns, self.cross_attns):
            x = res(x, t_emb=t_emb)
            x = sa(x)
            x = ca(x, kv)
        return x


# ────────────────────────────────────────────────────────────────────────────────
# LiDARUNet — the main assembly.
# ────────────────────────────────────────────────────────────────────────────────


class LiDARUNet(nn.Module):
    """Conditional diffusion U-Net over the LiDAR VAE latent.

    Topology (defaults match docs/lidar-unet.md §1):

        stem  → Enc L0 (8×256)  → DownW → Enc L1 (8×128) → DownW
              → Bottleneck (8×64)
              → UpW → cat(skip_1) → Dec L1 (8×128) → UpW → cat(skip_0) → Dec L0 (8×256)
              → head

    `level_channels` is variable-length: `N = len(level_channels)` encoder/decoder
    pairs (the last `level_channels` entry sets the bottleneck width). N=3 reproduces
    the original 3-stage default; N=4 is the paper-match config (160, 320, 640, 1024).
    See `Unet_scaled.md` for the data-flow diagram of both shapes.

    Args:
        in_channels:        latent channels in (default 8 — matches LiDAR VAE).
        out_channels:       latent channels out (default 8 — predicts v or eps).
        stem_channels:      output of the stem conv (default 96).
        level_channels:     channels per level (default (96, 192, 384) → 3 stages).
                            Last entry = bottleneck width; everything before = encoder
                            level widths. N stages = (N-1) encoders + bottleneck + (N-1) decoders.
        num_res_blocks:     ResBlock count per level (default 2).
        kv_channels:        channels of the pre-pooled KV context (default 10 = 4 image + 6 raymap).
        num_heads:          attention head count (default 8).
        t_emb_in_dim:       raw sinusoidal embedding dim (default = stem_channels).
        t_emb_dim:          per-block FiLM conditioning dim (default = 4 × t_emb_in_dim).
        groupnorm_groups:   max GroupNorm group count, clamped per layer (default 32).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        stem_channels: int = 96,
        level_channels: tuple[int, ...] = (96, 192, 384),
        num_res_blocks: int = 2,
        kv_channels: int = 10,
        num_heads: int = 8,
        t_emb_in_dim: int | None = None,
        t_emb_dim: int | None = None,
        groupnorm_groups: int = 32,
    ):
        super().__init__()
        level_channels = tuple(level_channels)
        n_levels = len(level_channels)
        assert n_levels >= 2, f"need at least 2 levels (1 encoder + bottleneck), got {n_levels}"

        # Time conditioning dims.
        self.t_emb_in_dim = t_emb_in_dim or stem_channels
        self.t_emb_dim = t_emb_dim or 4 * self.t_emb_in_dim
        self.time_mlp = TimestepMLP(self.t_emb_in_dim, self.t_emb_dim)

        # Stem.
        self.stem = CircularConv2d(in_channels, stem_channels, kernel_size=3)

        # Encoders: one per non-bottleneck level, in shallow-to-deep order.
        # encoders[i].in_ch = stem_channels (i=0) else level_channels[i-1]
        # encoders[i].out_ch = level_channels[i]
        # Construction order matches the original (enc_l0 before enc_l1) so
        # seeded random init is preserved for the N=3 default.
        self.encoders = nn.ModuleList([
            EncoderLevel(
                in_ch=(stem_channels if i == 0 else level_channels[i - 1]),
                out_ch=level_channels[i],
                kv_channels=kv_channels,
                num_res_blocks=num_res_blocks,
                num_heads=num_heads,
                do_downsample=True,
                t_emb_dim=self.t_emb_dim,
                return_skip=True,
            )
            for i in range(n_levels - 1)
        ])

        # Bottleneck (the last level): no down/upsample, deepest channel width.
        self.bottleneck = Bottleneck(
            in_ch=level_channels[-2],
            out_ch=level_channels[-1],
            kv_channels=kv_channels,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,
            t_emb_dim=self.t_emb_dim,
        )

        # Decoders: mirror of encoders in deep-to-shallow order so decoders[0]
        # consumes the bottleneck output, decoders[1] consumes decoders[0]'s output, etc.
        # decoders[i] sees in_ch = level_channels[-1 - i]  (prev stage output),
        # skip_ch = level_channels[-2 - i] (matching encoder's skip),
        # out_ch  = level_channels[-2 - i] (match next decoder's expected in_ch).
        # Construction order: dec_l1 first, dec_l0 second — matches original.
        self.decoders = nn.ModuleList([
            DecoderLevel(
                in_ch=level_channels[-1 - i],
                skip_ch=level_channels[-2 - i],
                out_ch=level_channels[-2 - i],
                kv_channels=kv_channels,
                num_res_blocks=num_res_blocks,
                num_heads=num_heads,
                do_upsample=True,
                t_emb_dim=self.t_emb_dim,
            )
            for i in range(n_levels - 1)
        ])

        # Head: GN → SiLU → CircConv. Zero-init the final conv so the fresh
        # U-Net predicts ε ≈ 0 on the first forward, giving a stable starting loss.
        ch_finest = level_channels[0]
        self.head_norm = nn.GroupNorm(min(groupnorm_groups, ch_finest), ch_finest)
        self.head_conv = CircularConv2d(ch_finest, out_channels, kernel_size=3)
        nn.init.zeros_(self.head_conv.conv.weight)
        nn.init.zeros_(self.head_conv.conv.bias)

    # ---- back-compat: translate old 3-stage state_dict keys (enc_l0/enc_l1/dec_l0/dec_l1) ----
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Translate legacy hardcoded-3-stage checkpoints (`enc_l0.*`, etc.) to
        the new N-stage key namespace (`encoders.0.*`, etc.) on load.

        Called automatically by `nn.Module.load_state_dict`. Old keys are renamed
        in-place inside `state_dict` before the parent class processes them.
        """
        # Legacy 3-stage U-Net stored encoders/decoders under hardcoded names:
        #   enc_l0 → encoders.0,  enc_l1 → encoders.1
        #   dec_l1 → decoders.0,  dec_l0 → decoders.1   (decoders list is deepest-first)
        legacy_map = {
            "enc_l0": "encoders.0",
            "enc_l1": "encoders.1",
            "dec_l1": "decoders.0",
            "dec_l0": "decoders.1",
        }
        # Build a snapshot of keys to rename (mutating state_dict while iterating is bad).
        renames = []
        for k in state_dict:
            if not k.startswith(prefix):
                continue
            tail = k[len(prefix):]
            for old, new in legacy_map.items():
                if tail.startswith(old + "."):
                    renames.append((k, prefix + new + tail[len(old):]))
                    break
        for old_k, new_k in renames:
            state_dict[new_k] = state_dict.pop(old_k)

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def forward(
        self,
        z_noisy: torch.Tensor,
        t: torch.Tensor,
        kv_context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise / v from a noised LiDAR latent.

        Args:
            z_noisy:    [B, in_channels, H_lat, W_lat] noised latent, e.g. [B, 8, 8, 256].
            t:          [B] diffusion timesteps (int or float).
            kv_context: [B, kv_channels, H_kv, W_kv] pre-pooled image+raymap, e.g. [B, 10, 8, 64].

        Returns:
            [B, out_channels, H_lat, W_lat] predicted noise (or v under v-prediction).
        """
        # Timestep embedding (computed once, broadcast through every ResBlock via FiLM).
        t_emb = timestep_embedding(t, self.t_emb_in_dim)
        t_emb = self.time_mlp(t_emb)

        # Stem.
        x = self.stem(z_noisy)

        # Encoder, collecting skip features (shallow → deep).
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x, kv_context, t_emb=t_emb)
            skips.append(skip)

        # Bottleneck.
        x = self.bottleneck(x, kv_context, t_emb=t_emb)

        # Decoder with skip-concat (deep → shallow; consume skips in reverse).
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip, kv_context, t_emb=t_emb)

        # Head.
        x = self.head_conv(F.silu(self.head_norm(x)))
        return x


def count_params(module: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
