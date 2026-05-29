"""End-to-end smoke test for the LiDAR VAE (encoder + decoder) on a real nuScenes sample.

What this does:
    1. Reads one `.pcd.bin` LiDAR_TOP keyframe directly from disk (via the
       `nuscenes/` symlink at the repo root).
    2. Converts it to a `[3, 32, 1024]` range image in [0, 1].
    3. Pushes one batch (B=1) through the (random-init) encoder, then
       decodes back to a range image.
    4. Prints input / latent / reconstruction stats so we can confirm shapes,
       ranges, that nothing NaNs out, and that the zero-init decoder gives
       a uniform ~0.5 output (the trained target replaces this).

No training, no loss. This is a plumbing check before M1 training begins.

Run from the repo root:
    python s2s_min/scripts/encoder_smoke_test.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `models.*` / `data.*` importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
S2S_DIR = REPO_ROOT / "s2s_min"
sys.path.insert(0, str(S2S_DIR))

import numpy as np
import torch

from data.range_image import (
    H_DEFAULT,
    W_DEFAULT,
    load_nuscenes_lidar_bin,
    point_cloud_to_range_image,
)
from models.lidar_vae import LiDARVAE


def find_one_lidar_bin(nuscenes_root: Path) -> Path:
    """Return the first `.pcd.bin` under `samples/LIDAR_TOP/`. No nuScenes-devkit needed."""
    lidar_dir = nuscenes_root / "samples" / "LIDAR_TOP"
    if not lidar_dir.is_dir():
        raise FileNotFoundError(
            f"LIDAR_TOP samples dir not found at {lidar_dir}. "
            f"Make sure `nuscenes/` symlink at the repo root points at the dataset."
        )
    # iterdir() order is filesystem-dependent; sorted() makes the smoke test reproducible.
    for p in sorted(lidar_dir.iterdir()):
        if p.suffix == ".bin":
            return p
    raise FileNotFoundError(f"No .pcd.bin files in {lidar_dir}")


def summarize_array(name: str, a: np.ndarray | torch.Tensor) -> None:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    finite = np.isfinite(a)
    n_nan = int((~finite).sum())
    print(
        f"  {name:<28} shape={tuple(a.shape)!s:<22} "
        f"dtype={str(a.dtype):<10} "
        f"min={a[finite].min():+.4f} max={a[finite].max():+.4f} "
        f"mean={a[finite].mean():+.4f} std={a[finite].std():.4f}"
        + (f"  [NaN/Inf={n_nan}]" if n_nan else "")
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nuscenes_root", type=Path,
                   default=REPO_ROOT / "nuscenes",
                   help="Path (or symlink) to nuScenes root containing samples/LIDAR_TOP/.")
    p.add_argument("--bin", type=Path, default=None,
                   help="Specific .pcd.bin to use; otherwise picks the first sorted file.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print("=" * 68)
    print("LiDAR VAE ENCODER — REAL nuScenes SAMPLE SMOKE TEST")
    print("=" * 68)
    print(f"  device         : {args.device}")
    print(f"  nuscenes_root  : {args.nuscenes_root}")

    bin_path = args.bin if args.bin is not None else find_one_lidar_bin(args.nuscenes_root)
    print(f"  LiDAR sample   : {bin_path.name}")

    # 1. Raw point cloud
    t0 = time.perf_counter()
    points = load_nuscenes_lidar_bin(str(bin_path))
    print(f"\n[1] Raw point cloud ({(time.perf_counter()-t0)*1000:.1f} ms)")
    print(f"  shape                       {points.shape}  (cols = x, y, z, intensity, ring)")
    r = np.sqrt((points[:, :3] ** 2).sum(axis=1))
    print(f"  range  (m)  : min={r.min():.2f}  max={r.max():.2f}  mean={r.mean():.2f}")
    print(f"  intensity   : min={points[:,3].min():.1f}  max={points[:,3].max():.1f}")
    print(f"  ring_index  : unique = {sorted(set(points[:,4].astype(int).tolist()))[:5]}..."
          f"{sorted(set(points[:,4].astype(int).tolist()))[-3:]}  "
          f"({len(set(points[:,4].astype(int).tolist()))} beams)")
    print(f"  fraction beyond 100 m clamp : {(r > 100).mean()*100:.2f} %")

    # 2. Range-image projection
    t0 = time.perf_counter()
    img = point_cloud_to_range_image(points)
    print(f"\n[2] Range image ({(time.perf_counter()-t0)*1000:.1f} ms)")
    print(f"  shape                       {img.shape}  [3, H={H_DEFAULT}, W={W_DEFAULT}]")
    validity = img[2]
    fill_rate = validity.mean() * 100
    print(f"  cell fill rate              {fill_rate:.1f}%  "
          f"({int(validity.sum())} of {validity.size} cells have a return)")
    summarize_array("range channel    [0]", img[0])
    summarize_array("intensity channel[1]", img[1])
    summarize_array("validity channel [2]", img[2])

    # 3. Encode
    device = torch.device(args.device)
    x = torch.from_numpy(img).unsqueeze(0).to(device)  # [1, 3, 32, 1024]

    vae = LiDARVAE(in_channels=3, latent_channels=8, base_channels=32).to(device).eval()
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"\n[3] Encode through (random-init) LiDARVAE  ({n_params/1e6:.2f} M params total)")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        mu, logvar = vae.encode(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  encode time                 {enc_ms:.1f} ms")
    summarize_array("input  x      [1,3,32,1024]", x)
    summarize_array("latent mu     [1,8, 8, 256]", mu)
    summarize_array("latent logvar [1,8, 8, 256]", logvar)
    print(f"  implied sigma stats         "
          f"min={(0.5*logvar).exp().min().item():.4f}  "
          f"max={(0.5*logvar).exp().max().item():.4f}  "
          f"mean={(0.5*logvar).exp().mean().item():.4f}")

    # 4. Decode
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        x_hat = vae.decode(mu)              # use mu (eval semantics, deterministic)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[4] Decode mu -> reconstructed range image")
    print(f"  decode time                 {dec_ms:.1f} ms")
    summarize_array("recon x_hat   [1,3,32,1024]", x_hat)
    print("  expected at init: every channel ≈ 0.5 (zero-init decoder head + sigmoid)")

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  peak CUDA memory            {mem_mb:.0f} MiB  (full forward, B=1)")

    print("\n" + "=" * 68)
    print("SMOKE TEST PASSED — full VAE runs end-to-end on a real nuScenes sample.")
    print("=" * 68)


if __name__ == "__main__":
    main()
