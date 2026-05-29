"""Dataset for the DINOv3-conditioning variant (Option B: SD-VAE replaced by DINOv3).

Joins two caches by sample_token:
  - latent cache   (cache_latents.py): provides `raymap` [6,32,56] and `mu` [8,8,256]
  - DINOv3 cache   (cache_dinov3.py):   provides `feat`  [384,14,24] (ViT-S/16 patch grid, f16)

Returns dicts with `dinov3`, `raymap`, `mu`, `sample_token`. The training loop projects
`dinov3` 384→4 (learned Conv1x1) and upsamples to 32×56 in place of the old `image_latent`,
keeping the KV context at 10 channels (4 + raymap 6) — so the U-Net is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedDINOv3Dataset(Dataset):
    def __init__(
        self,
        latent_cache_dir: str | Path,
        dinov3_cache_dir: str | Path,
        sample_tokens: Iterable[str] | None = None,
    ):
        self.lat = Path(latent_cache_dir)
        self.d3 = Path(dinov3_cache_dir)
        for d in (self.lat, self.d3):
            if not d.is_dir():
                raise FileNotFoundError(f"cache dir {d} not found")
        lat = {p.stem for p in self.lat.glob("*.npz")}
        d3 = {p.stem for p in self.d3.glob("*.npz")}
        toks = lat & d3
        if sample_tokens is not None:
            toks &= set(sample_tokens)
        if not toks:
            raise RuntimeError(
                f"no overlapping tokens between {self.lat} ({len(lat)}) and {self.d3} ({len(d3)})"
            )
        self.tokens = sorted(toks)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        tk = self.tokens[idx]
        lat = np.load(self.lat / f"{tk}.npz")
        d3 = np.load(self.d3 / f"{tk}.npz")
        return {
            "dinov3": torch.from_numpy(d3["feat"].astype(np.float32)),   # [384,14,24]
            "raymap": torch.from_numpy(lat["raymap"]),                   # [6,32,56]
            "mu":     torch.from_numpy(lat["mu"]),                       # [8,8,256]
            "sample_token": tk,
        }
