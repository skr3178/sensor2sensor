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
# Pre-import safetensors.torch so diffusers can resolve `safetensors.torch.load_file`
# without us depending on its lazy submodule loading (which fails on some installs).
import safetensors.torch  # noqa: F401
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

    @torch.no_grad()
    def encode_views(self, rgb_views: torch.Tensor) -> torch.Tensor:
        """Encode a multi-view RGB tensor by flattening views into the batch dim.

        Args:
            rgb_views: [B, V, 3, H, W] in [-1, 1] (SD convention).

        Returns:
            [B, V, 4, H/8, W/8] scaled latent in fp32. Single VAE call for all
            B*V views; the view axis is preserved on output.
        """
        assert rgb_views.dim() == 5, f"expected [B,V,3,H,W], got {tuple(rgb_views.shape)}"
        B, V, C, H, W = rgb_views.shape
        latent = self.forward(rgb_views.reshape(B * V, C, H, W))
        Cz, Hz, Wz = latent.shape[1:]
        return latent.view(B, V, Cz, Hz, Wz)
