"""M2 — Pre-encode all subset latents to disk.

Walks the 10-scene subset once. For every paired (CAM_FRONT, LIDAR_TOP) keyframe:
  - Encode CAM_FRONT through the frozen SD 1.5 VAE → `image_latent [4, 32, 56]`
  - Build raymap on the 32×56 latent grid                → `raymap     [6, 32, 56]`
  - Encode LIDAR_TOP range image through the frozen LiDAR VAE → `mu     [8, 8, 256]`

Saves each sample as `{sample_token}.npz` under `s2s_min/out/cached_latents/`.
M3's training loop reads from this cache directly — neither encoder is loaded
during diffusion training, freeing VRAM and saving ~60 ms per step.

### Design choice: μ-only caching (not sampled latents)

We cache the LiDAR VAE posterior **mean** (μ) and use it directly as the
diffusion target in M3. This is the standard latent-diffusion approach (Stable
Diffusion does the same with the SD VAE: it caches the encoder mean, never the
reparameterized sample).

**Future possibility — sampled latents as regularization:**
Some VAE-diffusion variants cache both μ AND logvar, then sample
`z = μ + σ·ε` with fresh `ε` per step. This introduces additional variance from
the VAE's posterior, which can act as a mild regularizer on the diffusion model
(it sees a slightly different latent for the same sample each epoch). To enable
this, change `--save-logvar` to True and update the M3 training loop to do the
reparameterize at read-time. We deliberately skip this in v1 because:
  - SD-family precedent uses μ-only and it works.
  - Sampling at read-time adds 0.1 ms per step but slight loss-curve noise.
  - The extra logvar tensor is small (~64 KB per sample) but doubles cache size.

Run:
    env/bin/python s2s_min/train/cache_latents.py
Output:
    s2s_min/out/cached_latents/<sample_token>.npz   (~135 KB per sample)
    s2s_min/out/cached_latents/MANIFEST.json        (sample-token list + stats)
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
from PIL import Image

from data.range_image import load_nuscenes_lidar_bin, point_cloud_to_range_image
from models.image_encoder import FrozenSDVAEEncoder
from models.lidar_vae import LiDARVAE
from models.raymap import build_raymap

NUSCENES_ROOT  = Path("nuscenes")
SUBSET_TOKENS  = Path("s2s_min/out/subset_scene_tokens.txt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
IMG_VAE_DIR    = Path("s2s_min/checkpoints/sd15_vae")
OUT_DIR        = Path("s2s_min/out/cached_latents")
IMG_H, IMG_W   = 256, 448
NATIVE_W, NATIVE_H = 1600, 900
SD_DOWNSAMPLE  = 8


# ----------------------------- helpers ----------------------------------
def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def make_T(translation, rotation_quat_wxyz):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_rotmat(rotation_quat_wxyz)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def scale_intrinsics(K_native):
    K = K_native.astype(np.float32).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def load_rgb_minus1_to_1(jpg_path: Path) -> torch.Tensor:
    img = Image.open(jpg_path).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


def collect_paired_keyframes() -> list[tuple[dict, dict, dict]]:
    """Returns list of (cam_record, lidar_record, calibrated_sensor_dict) tuples
    for every sample in the subset that has both CAM_FRONT and LIDAR_TOP keyframes."""
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample = json.loads((meta / "sample.json").read_text())
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}

    subset = set(SUBSET_TOKENS.read_text().split())
    samples_in = {s["token"]: s for s in sample if s["scene_token"] in subset}

    cam_records: dict[str, dict] = {}
    lid_records: dict[str, dict] = {}
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] not in samples_in:
            continue
        chan = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if chan == "CAM_FRONT": cam_records[sd["sample_token"]] = sd
        elif chan == "LIDAR_TOP": lid_records[sd["sample_token"]] = sd

    # Only samples that have both modalities.
    paired = []
    for tok in cam_records:
        if tok in lid_records:
            paired.append((cam_records[tok], lid_records[tok], cs))
    return paired


# ----------------------------- main --------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-logvar", action="store_true",
                        help="Also cache LiDAR VAE logvar (enables future sampled-latent variant; "
                             "doubles cache size for negligible value in v1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only cache the first N samples (for debugging).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode samples whose .npz already exists. Default: skip.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- discover samples ----
    pairs = collect_paired_keyframes()
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"paired samples to cache: {len(pairs)}")

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
    mu_stats = []
    image_stats = []
    n_skipped = 0
    n_failed = 0
    n_written = 0
    t0 = time.time()
    for i, (cam_rec, lid_rec, cs) in enumerate(pairs):
        sample_token = cam_rec["sample_token"]
        out_path = OUT_DIR / f"{sample_token}.npz"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        try:
            # Image side.
            rgb = load_rgb_minus1_to_1(NUSCENES_ROOT / cam_rec["filename"]).unsqueeze(0).to(device)
            image_latent = img_enc(rgb)[0].cpu().numpy().astype(np.float32)   # [4, 32, 56]

            # Raymap.
            cs_cam = cs[cam_rec["calibrated_sensor_token"]]
            K_scaled = scale_intrinsics(np.array(cs_cam["camera_intrinsic"], dtype=np.float32))
            T_cam2ego = make_T(cs_cam["translation"], cs_cam["rotation"])
            raymap = build_raymap(
                torch.from_numpy(K_scaled), torch.from_numpy(T_cam2ego),
                H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE,
            )[0].cpu().numpy().astype(np.float32)                              # [6, 32, 56]

            # LiDAR side.
            pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / lid_rec["filename"]))
            range_img = torch.from_numpy(point_cloud_to_range_image(pc)).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, logvar = lidar_vae.encode(range_img)
            mu_np     = mu[0].cpu().numpy().astype(np.float32)                 # [8, 8, 256]
            logvar_np = logvar[0].cpu().numpy().astype(np.float32)             # [8, 8, 256]

            # Save.
            payload: dict[str, np.ndarray | str] = {
                "image_latent": image_latent,
                "raymap":       raymap,
                "mu":           mu_np,
                "sample_token": np.array(sample_token),    # for round-trip identification
            }
            if args.save_logvar:
                payload["logvar"] = logvar_np

            np.savez_compressed(out_path, **payload)
            n_written += 1
            mu_stats.append((mu_np.mean(), mu_np.std()))
            image_stats.append((image_latent.mean(), image_latent.std()))

        except Exception as e:
            print(f"  [FAIL] {sample_token}: {e}")
            n_failed += 1
            continue

        if (i + 1) % 50 == 0 or i + 1 == len(pairs):
            elapsed = time.time() - t0
            rate = (n_written + n_skipped) / max(elapsed, 1e-6)
            print(f"  {i+1:>4}/{len(pairs)}  written={n_written} skipped={n_skipped} failed={n_failed}  "
                  f"({rate:.1f} samples/s)")

    # ---- manifest ----
    elapsed = time.time() - t0
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.npz"))
    manifest = {
        "n_paired_samples_in_subset": len(pairs),
        "n_written_this_run": n_written,
        "n_skipped_existing": n_skipped,
        "n_failed": n_failed,
        "save_logvar": args.save_logvar,
        "wall_time_seconds": round(elapsed, 1),
        "total_cache_bytes": total_bytes,
        "total_cache_mb": round(total_bytes / 1e6, 2),
        "tensor_shapes": {
            "image_latent": [4, 32, 56],
            "raymap":       [6, 32, 56],
            "mu":           [8, 8, 256],
            "logvar":       [8, 8, 256] if args.save_logvar else None,
        },
        "stats": {
            "mu_mean":   float(np.mean([s[0] for s in mu_stats])) if mu_stats else None,
            "mu_std":    float(np.mean([s[1] for s in mu_stats])) if mu_stats else None,
            "image_mean": float(np.mean([s[0] for s in image_stats])) if image_stats else None,
            "image_std":  float(np.mean([s[1] for s in image_stats])) if image_stats else None,
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
