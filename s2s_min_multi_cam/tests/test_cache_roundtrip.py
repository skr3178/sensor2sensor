"""Round-trip test for the latent cache.

Picks one cached sample, reloads the live encoders (SD VAE + LiDAR VAE), re-runs
the encoding on the original nuScenes files, and compares against the cached
values. Cached and live should match to within fp32 / fp16-quantization noise.

Run:
    env/bin/python s2s_min/tests/test_cache_roundtrip.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from data.cached_latents import CachedLatentsDataset
from data.range_image import load_nuscenes_lidar_bin, point_cloud_to_range_image
from models.image_encoder import FrozenSDVAEEncoder
from models.lidar_vae import LiDARVAE
from models.raymap import build_raymap

NUSCENES_ROOT  = Path("nuscenes")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
IMG_VAE_DIR    = Path("s2s_min/checkpoints/sd15_vae")
CACHE_DIR      = Path("s2s_min/out/cached_latents")
IMG_H, IMG_W   = 256, 448
NATIVE_W, NATIVE_H = 1600, 900
SD_DOWNSAMPLE  = 8


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def make_T(translation, rotation):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_rotmat(rotation)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def scale_intrinsics(K_native):
    K = K_native.astype(np.float32).copy()
    K[0, 0] *= IMG_W / NATIVE_W; K[0, 2] *= IMG_W / NATIVE_W
    K[1, 1] *= IMG_H / NATIVE_H; K[1, 2] *= IMG_H / NATIVE_H
    return K


def load_rgb(jpg_path: Path) -> torch.Tensor:
    img = Image.open(jpg_path).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


def find_records_for_token(target_token: str):
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    cam, lid = None, None
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] != target_token:
            continue
        chan = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if chan == "CAM_FRONT": cam = sd
        elif chan == "LIDAR_TOP": lid = sd
    return cam, lid, cs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    # ---- pick a cached sample ----
    ds = CachedLatentsDataset(CACHE_DIR)
    print(f"{len(ds)} samples in cache. Picking first.")
    item = ds[0]
    sample_token = item["sample_token"]
    print(f"sample token: {sample_token}\n")

    # ---- live encoders ----
    print("loading live encoders...")
    img_enc = FrozenSDVAEEncoder(local_dir=IMG_VAE_DIR).to(device)
    ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs = {k: v for k, v in ckpt["config"].items()
                   if k in inspect.signature(LiDARVAE.__init__).parameters}
    lidar_vae = LiDARVAE(**arch_kwargs).to(device).eval()
    lidar_vae.load_state_dict(ckpt["state_dict"])
    lidar_vae.requires_grad_(False)

    # ---- live forward ----
    cam_rec, lid_rec, cs = find_records_for_token(sample_token)
    cs_cam = cs[cam_rec["calibrated_sensor_token"]]

    rgb = load_rgb(NUSCENES_ROOT / cam_rec["filename"]).unsqueeze(0).to(device)
    image_latent_live = img_enc(rgb)[0].cpu().numpy().astype(np.float32)

    K_scaled = scale_intrinsics(np.array(cs_cam["camera_intrinsic"], dtype=np.float32))
    T = make_T(cs_cam["translation"], cs_cam["rotation"])
    raymap_live = build_raymap(
        torch.from_numpy(K_scaled), torch.from_numpy(T),
        H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE,
    )[0].cpu().numpy().astype(np.float32)

    pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / lid_rec["filename"]))
    range_img = torch.from_numpy(point_cloud_to_range_image(pc)).unsqueeze(0).to(device)
    with torch.no_grad():
        mu_live, _ = lidar_vae.encode(range_img)
    mu_live_np = mu_live[0].cpu().numpy().astype(np.float32)

    # ---- compare ----
    cached_image_latent = item["image_latent"].cpu().numpy()
    cached_raymap       = item["raymap"].cpu().numpy()
    cached_mu           = item["mu"].cpu().numpy()

    print("\nround-trip max-abs-diff (cached vs live):")

    # image_latent: encoder runs in fp16 → small drift across runs is expected
    # (cuDNN algorithm choice non-determinism), so allow a loose threshold here.
    d_img = float(np.abs(cached_image_latent - image_latent_live).max())
    print(f"  image_latent : {d_img:.2e}  "
          f"(loose threshold: fp16 cuDNN noise, accept < 0.05)")
    assert d_img < 0.05, f"image_latent drift {d_img} too large"

    # raymap: pure-function on fp32, should be bit-exact.
    d_ray = float(np.abs(cached_raymap - raymap_live).max())
    print(f"  raymap       : {d_ray:.2e}  (strict: should be 0.0)")
    assert d_ray == 0.0, f"raymap should be bit-exact, got {d_ray}"

    # LiDAR VAE: fp32 forward, also bit-exact (cuDNN determinism for our small ops).
    d_mu = float(np.abs(cached_mu - mu_live_np).max())
    print(f"  mu (LiDAR z) : {d_mu:.2e}  (strict: should be ~0.0 to numerical noise)")
    assert d_mu < 1e-4, f"mu drift {d_mu} too large"

    # ---- summary ----
    print("\nshape contract:")
    print(f"  image_latent: {cached_image_latent.shape}   expected (4, 32, 56)")
    print(f"  raymap      : {cached_raymap.shape}         expected (6, 32, 56)")
    print(f"  mu          : {cached_mu.shape}             expected (8, 8, 256)")
    assert cached_image_latent.shape == (4, 32, 56)
    assert cached_raymap.shape == (6, 32, 56)
    assert cached_mu.shape == (8, 8, 256)

    print("\nOK — cache round-trip verified.")


if __name__ == "__main__":
    main()
