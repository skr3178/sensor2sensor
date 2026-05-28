"""M2 — Pre-encode all subset latents to disk (multi-camera variant).

Walks the 10-scene subset once. For every keyframe sample that has all 7 sensors
(LIDAR_TOP + 6 surround cameras in `data.nuscenes_mini_paired.CAMERA_ORDER`):

  - Encode 6 RGB views through the frozen SD 1.5 VAE → `img_latents [6, 4, 32, 56]`
  - Build a raymap per camera on the 32x56 grid       → `raymaps     [6, 6, 32, 56]`
  - Encode LIDAR_TOP range image through the frozen LiDAR VAE → `mu  [8, 8, 256]`

Saves each sample as `{sample_token}.npz` under `out/cached_latents/`. M3's
training loop reads from this cache directly — neither encoder is loaded during
diffusion training, freeing VRAM and saving ~60 ms per step.

The metadata walk + per-camera intrinsics/extrinsics handling lives in
`data.nuscenes_mini_paired`. This script is just the "run both encoders and dump
the result" wrapper.

Run:
    PYTHONPATH=. python train/cache_latents.py
Output:
    out/cached_latents/<sample_token>.npz   (~0.8 MB per sample, 6 cams)
    out/cached_latents/MANIFEST.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.nuscenes_mini_paired import (
    CAMERA_ORDER,
    NuScenesPairedKeyframes,
    load_subset_tokens,
)
from models.image_encoder import FrozenSDVAEEncoder
from models.lidar_vae import LiDARVAE
from models.raymap import build_raymap

REPO_ROOT      = Path(__file__).resolve().parents[1]
NUSCENES_ROOT  = REPO_ROOT / "data" / "nuscenes_root"
SUBSET_TOKENS  = REPO_ROOT / "out" / "subset_scene_tokens.txt"
LIDAR_VAE_CKPT = REPO_ROOT / "out" / "lidar_vae_ema.pt"
IMG_VAE_DIR    = REPO_ROOT / "checkpoints" / "sd15_vae"
OUT_DIR        = REPO_ROOT / "out" / "cached_latents"
H_LAT, W_LAT   = 32, 56                         # SD VAE latent grid (256x448 / 8)
SD_DOWNSAMPLE  = 8
V              = len(CAMERA_ORDER)              # 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-logvar", action="store_true",
                        help="Also cache LiDAR VAE logvar (enables future sampled-latent variant).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only cache the first N samples (for debugging).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode samples whose .npz already exists. Default: skip.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- dataset ----
    scene_tokens = load_subset_tokens(SUBSET_TOKENS)
    ds = NuScenesPairedKeyframes(NUSCENES_ROOT, scene_tokens=scene_tokens)
    n_total = len(ds)
    if args.limit is not None:
        n_total = min(n_total, args.limit)
    print(f"paired samples to cache: {n_total}  ({V} cameras: {list(CAMERA_ORDER)})")

    # ---- load encoders (frozen) ----
    print("loading encoders ...")
    img_enc = FrozenSDVAEEncoder(local_dir=IMG_VAE_DIR).to(device)

    lidar_ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs_keys = set(inspect.signature(LiDARVAE.__init__).parameters)
    arch_kwargs = {k: v for k, v in lidar_ckpt["config"].items() if k in arch_kwargs_keys}
    lidar_vae = LiDARVAE(**arch_kwargs).to(device).eval()
    lidar_vae.load_state_dict(lidar_ckpt["state_dict"])
    lidar_vae.requires_grad_(False)
    print(f"  SD VAE step:    pretrained")
    print(f"  LiDAR VAE step: {lidar_ckpt.get('step', '?')}  ({arch_kwargs})")

    # ---- per-sample loop ----
    mu_stats, img_stats = [], []
    n_skipped = n_failed = n_written = 0
    t0 = time.time()
    for i in range(n_total):
        item = ds[i]
        sample_token = item["sample_token"]
        out_path = OUT_DIR / f"{sample_token}.npz"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        try:
            # Image side: encode all 6 cameras in one VAE call.
            cams = item["cams"].unsqueeze(0).to(device)              # [1, 6, 3, 256, 448]
            img_latents = img_enc.encode_views(cams)[0].cpu().numpy().astype(np.float32)
            # img_latents: [6, 4, 32, 56]

            # Per-camera raymap: flatten V into the build_raymap batch dim.
            K = item["cam_K"].to(device)                             # [6, 3, 3]
            T = item["cam_T_cam2ego"].to(device)                     # [6, 4, 4]
            raymaps_t = build_raymap(K, T, H_LAT, W_LAT, downsample=SD_DOWNSAMPLE)
            raymaps = raymaps_t.cpu().numpy().astype(np.float32)     # [6, 6, 32, 56]

            # LiDAR side.
            range_img = item["range_image"].unsqueeze(0).to(device)  # [1, 3, 32, 1024]
            with torch.no_grad():
                mu, logvar = lidar_vae.encode(range_img)
            mu_np     = mu[0].cpu().numpy().astype(np.float32)       # [8, 8, 256]
            logvar_np = logvar[0].cpu().numpy().astype(np.float32)   # [8, 8, 256]

            payload: dict[str, np.ndarray] = {
                "img_latents":  img_latents,
                "raymaps":      raymaps,
                "mu":           mu_np,
                "sample_token": np.array(sample_token),
                "camera_order": np.array(list(CAMERA_ORDER)),
            }
            if args.save_logvar:
                payload["logvar"] = logvar_np

            np.savez_compressed(out_path, **payload)
            n_written += 1
            mu_stats.append((mu_np.mean(), mu_np.std()))
            img_stats.append((img_latents.mean(), img_latents.std()))

        except Exception as e:
            print(f"  [FAIL] {sample_token}: {e}")
            n_failed += 1
            continue

        if (i + 1) % 50 == 0 or i + 1 == n_total:
            elapsed = time.time() - t0
            rate = (n_written + n_skipped) / max(elapsed, 1e-6)
            print(f"  {i+1:>4}/{n_total}  written={n_written} skipped={n_skipped} failed={n_failed}  "
                  f"({rate:.1f} samples/s)")

    # ---- manifest ----
    elapsed = time.time() - t0
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz"))
    manifest = {
        "n_paired_samples_in_subset": n_total,
        "n_written_this_run": n_written,
        "n_skipped_existing": n_skipped,
        "n_failed": n_failed,
        "save_logvar": args.save_logvar,
        "wall_time_seconds": round(elapsed, 1),
        "total_cache_bytes": total_bytes,
        "total_cache_mb": round(total_bytes / 1e6, 2),
        "camera_order": list(CAMERA_ORDER),
        "tensor_shapes": {
            "img_latents": [V, 4, H_LAT, W_LAT],
            "raymaps":     [V, 6, H_LAT, W_LAT],
            "mu":          [8, 8, 256],
            "logvar":      [8, 8, 256] if args.save_logvar else None,
        },
        "stats": {
            "mu_mean":  float(np.mean([s[0] for s in mu_stats]))  if mu_stats  else None,
            "mu_std":   float(np.mean([s[1] for s in mu_stats]))  if mu_stats  else None,
            "img_mean": float(np.mean([s[0] for s in img_stats])) if img_stats else None,
            "img_std":  float(np.mean([s[1] for s in img_stats])) if img_stats else None,
        },
        "lidar_vae_ckpt": str(LIDAR_VAE_CKPT),
        "lidar_vae_step": int(lidar_ckpt.get("step", -1)),
        "image_encoder_local_dir": str(IMG_VAE_DIR),
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
