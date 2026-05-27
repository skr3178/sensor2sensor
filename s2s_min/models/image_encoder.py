"""Frozen SD 1.5 VAE encoder — image conditioning for the LiDAR U-Net.

Loads from the project-local checkpoint at `s2s_min/checkpoints/sd15_vae/`
(populated by `scripts/download_image_vae.py`). Encoder-only — we never call
`.decode()` in the pipeline.

Pattern verified in `scripts/visualize_image_vae.py` (4-sample CAM_FRONT eyeball);
this module refactors the same load + encode into a reusable `nn.Module`.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

DEFAULT_LOCAL_DIR = "s2s_min/checkpoints/sd15_vae"


class FrozenSDVAEEncoder(nn.Module):
    """Frozen Stable Diffusion 1.5 VAE encoder.

    Args:
        local_dir:  path to the local SD VAE checkpoint directory. Must contain
                    `vae/config.json` and `vae/diffusion_pytorch_model.fp16.safetensors`.
        variant:    diffusers weight variant (default "fp16").
        dtype:      torch dtype for the loaded weights (default fp16).

    Forward:
        rgb: [B, 3, H, W] RGB in **[-1, 1]** (SD convention, NOT [0, 1]).
             H, W must be multiples of 8 (we don't enforce).
        returns: [B, 4, H/8, W/8] latent, **pre-multiplied by `scaling_factor`**
                 (0.18215 for SD 1.5) so downstream consumers don't need to remember.
    """

    def __init__(
        self,
        local_dir: str | Path = DEFAULT_LOCAL_DIR,
        variant: str = "fp16",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            str(local_dir),
            subfolder="vae",
            variant=variant,
            torch_dtype=dtype,
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        # Cache the scaling factor as a buffer so it moves with .to(device).
        self.register_buffer(
            "scaling_factor",
            torch.tensor(self.vae.config.scaling_factor, dtype=dtype),
            persistent=False,
        )
        self.latent_channels: int = self.vae.config.latent_channels  # 4 for SD 1.5
        self.dtype_ = dtype

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB to scaled latent. Eats input in fp16, returns fp32 by default."""
        latent = self.vae.encode(rgb.to(self.dtype_)).latent_dist.mean
        latent = latent * self.scaling_factor
        return latent.to(torch.float32)
