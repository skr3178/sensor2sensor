"""End-to-end smoke test for the LiDAR VAE on a real Waymo sample.

Mirrors `encoder_smoke_test.py` but reads from the Waymo Open Dataset v2.0.1
parquet layout produced by `scripts/download_waymo_samples.sh`.

What this does:
    1. Opens the first lidar parquet under {waymo_root}/{split}/lidar/.
    2. Decodes one TOP-laser frame into a [4, 64, 2048] range image in [0, 1]
       (channels = range, intensity, elongation, validity).
    3. Pushes one batch (B=1) through a random-init LiDARVAE, then decodes.
    4. Prints shape / range / fill-rate stats for input, latent, recon.

Run from the repo root:
    python s2s_min/scripts/encoder_smoke_test_waymo.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S_DIR = REPO_ROOT / "s2s_min"
sys.path.insert(0, str(S2S_DIR))

import numpy as np
import torch

from data.waymo import WaymoLidarTopKeyframes
from data.waymo_range_image import (
    H_DEFAULT,
    W_DEFAULT,
    LASER_NAME_TOP,
    RANGE_MAX_M,
    INTENSITY_MAX,
)
from models.lidar_vae import LiDARVAE


def summarize_array(name: str, a: "np.ndarray | torch.Tensor") -> None:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    finite = np.isfinite(a)
    n_nan = int((~finite).sum())
    print(
        f"  {name:<32} shape={tuple(a.shape)!s:<22} "
        f"dtype={str(a.dtype):<10} "
        f"min={a[finite].min():+.4f} max={a[finite].max():+.4f} "
        f"mean={a[finite].mean():+.4f} std={a[finite].std():.4f}"
        + (f"  [NaN/Inf={n_nan}]" if n_nan else "")
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--waymo_root", type=Path,
                   default=S2S_DIR / "data" / "waymo",
                   help="Waymo v2 root (contains training/ and validation/).")
    p.add_argument("--split", default="training", choices=["training", "validation"])
    p.add_argument("--idx", type=int, default=0,
                   help="Which TOP-laser keyframe to load (0-based).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 70)
    print("LiDAR VAE ENCODER — REAL Waymo SAMPLE SMOKE TEST")
    print("=" * 70)
    print(f"  device       : {args.device}")
    print(f"  waymo_root   : {args.waymo_root}")
    print(f"  split        : {args.split}")

    # 1. Build dataset (does a cheap metadata scan; no range images decoded yet).
    t0 = time.perf_counter()
    ds = WaymoLidarTopKeyframes(args.waymo_root, split=args.split, return_dict=True)
    print(f"\n[1] Dataset scan ({(time.perf_counter()-t0)*1000:.1f} ms)")
    print(f"  total TOP-laser keyframes : {len(ds)}")
    print(f"  segments under {args.split:<10}: {len(ds.files)}")

    if args.idx >= len(ds):
        raise IndexError(f"--idx {args.idx} >= dataset size {len(ds)}")

    # 2. Decode one frame (cold; this triggers full parquet read for that file).
    t0 = time.perf_counter()
    item = ds[args.idx]
    decode_ms = (time.perf_counter() - t0) * 1000
    x = item["range_image"].numpy()
    print(f"\n[2] Decode keyframe {args.idx} ({decode_ms:.1f} ms, cold)")
    print(f"  segment      : {item['segment']}")
    print(f"  timestamp_us : {item['timestamp']}")
    print(f"  output shape : {x.shape}  (expected (4, {H_DEFAULT}, {W_DEFAULT}))")
    summarize_array("range channel     [0]", x[0])
    summarize_array("intensity channel [1]", x[1])
    summarize_array("elongation channel[2]", x[2])
    summarize_array("validity channel  [3]", x[3])
    fill = float(x[3].mean()) * 100
    print(f"  cell fill rate              {fill:.1f}%")

    # Second read from the same file should be near-instant (LRU cache hit).
    t0 = time.perf_counter()
    _ = ds[args.idx + 1] if args.idx + 1 < len(ds) else ds[args.idx]
    print(f"  warm read of next frame    {(time.perf_counter()-t0)*1000:.1f} ms")

    # 3. Encode through random-init VAE.
    device = torch.device(args.device)
    xb = torch.from_numpy(x).unsqueeze(0).to(device)

    vae = LiDARVAE(in_channels=4, latent_channels=8, base_channels=32).to(device).eval()
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"\n[3] Encode through random-init LiDARVAE ({n_params/1e6:.2f} M params)")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        mu, logvar = vae.encode(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  encode time                 {(time.perf_counter()-t0)*1000:.1f} ms")
    summarize_array("input  x", xb)
    summarize_array("latent mu", mu)
    summarize_array("latent logvar", logvar)
    print(f"  implied sigma stats         "
          f"min={(0.5*logvar).exp().min().item():.4f}  "
          f"max={(0.5*logvar).exp().max().item():.4f}  "
          f"mean={(0.5*logvar).exp().mean().item():.4f}")

    # 4. Decode back.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        x_hat = vae.decode(mu)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"\n[4] Decode mu -> reconstructed range image "
          f"({(time.perf_counter()-t0)*1000:.1f} ms)")
    summarize_array("recon x_hat", x_hat)
    print("  expected at init: every channel ≈ 0.5 (zero-init decoder head + sigmoid)")

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  peak CUDA memory            {mem_mb:.0f} MiB  (full forward, B=1)")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — full VAE runs end-to-end on a real Waymo sample.")
    print("=" * 70)


if __name__ == "__main__":
    main()
