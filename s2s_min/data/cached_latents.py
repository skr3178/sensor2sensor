"""Dataset over the pre-encoded latent cache (produced by `train/cache_latents.py`).

Returns dicts with `image_latent`, `raymap`, `mu`, and `sample_token`. M3's
training loop and M4's inference loop both consume this dataset.

No SD VAE or LiDAR VAE is loaded by this Dataset — that's the whole point of
the cache (free up ~250 MB of frozen-encoder VRAM during M3 training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedLatentsDataset(Dataset):
    """Reads `{sample_token}.npz` files written by `train/cache_latents.py`.

    Args:
        cache_dir:     directory containing the .npz files. Files named
                       `MANIFEST.json` or anything not matching `*.npz` are ignored.
        sample_tokens: optional iterable of sample tokens to restrict to.
                       If None (default), include every .npz in `cache_dir`.

    Each `__getitem__` returns a dict with float32 tensors:
        image_latent : [4, 32, 56]
        raymap       : [6, 32, 56]
        mu           : [8, 8, 256]    (LiDAR VAE posterior mean; the diffusion target)
        sample_token : str            (for logging / debugging — not for the model)

    `logvar` is loaded only if the cache was written with `--save-logvar`.
    """

    def __init__(
        self,
        cache_dir: str | Path = "s2s_min/out/cached_latents",
        sample_tokens: Iterable[str] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"cache_dir {self.cache_dir} not found — did you run cache_latents.py?"
            )
        all_paths = sorted(self.cache_dir.glob("*.npz"))
        if sample_tokens is not None:
            wanted = set(sample_tokens)
            all_paths = [p for p in all_paths if p.stem in wanted]
        if len(all_paths) == 0:
            raise RuntimeError(f"no .npz files in {self.cache_dir} (after token filter)")
        self.paths = all_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        npz = np.load(path)
        item = {
            "image_latent": torch.from_numpy(npz["image_latent"]),
            "raymap":       torch.from_numpy(npz["raymap"]),
            "mu":           torch.from_numpy(npz["mu"]),
            "sample_token": str(npz["sample_token"]),
        }
        if "logvar" in npz.files:
            item["logvar"] = torch.from_numpy(npz["logvar"])
        return item
