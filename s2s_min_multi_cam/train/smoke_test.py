"""M0 — end-to-end smoke test.

Loads ONE paired (CAM_FRONT, LIDAR_TOP) nuScenes keyframe, runs it through the
full pipeline: SD VAE image encoder + raymap + pre-pool KV → LiDAR VAE encoder
→ DDPM noise injection → LiDARUNet forward → MSE on v-target → backward
→ one Adam step. Reports every intermediate tensor shape, the loss value, and
peak VRAM.

Pass criterion (per min_pipeline_plan.md §M0):
  - finite, non-NaN loss
  - no shape errors
  - peak VRAM < 6 GB on the 11.6 GB RTX 3060 (expected: 2–3 GB)

Run:
    env/bin/python s2s_min/train/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Local modules.
from data.range_image import load_nuscenes_lidar_bin, point_cloud_to_range_image
from models.diffusion import DiffusionWrapper
from models.image_encoder import FrozenSDVAEEncoder
from models.lidar_vae import LiDARVAE
from models.raymap import build_raymap
from models.unet import LiDARUNet, count_params

# ----------------------------- config -----------------------------------
NUSCENES_ROOT = Path("nuscenes")  # symlink at project root
SUBSET_TOKENS = Path("s2s_min/out/subset_scene_tokens.txt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
IMG_VAE_DIR = Path("s2s_min/checkpoints/sd15_vae")

IMG_H, IMG_W = 256, 448            # SD VAE input
NATIVE_W, NATIVE_H = 1600, 900     # nuScenes CAM_FRONT native resolution
SD_DOWNSAMPLE = 8                  # SD VAE spatial /8 → latent 32×56
KV_POOL_H, KV_POOL_W = 8, 64       # adaptive pool target for KV context


# ----------------------------- data utils -------------------------------
def quat_wxyz_to_rotmat(q):
    """nuScenes quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def find_paired_keyframe():
    """Walk metadata to find ONE sample with both CAM_FRONT and LIDAR_TOP keyframes."""
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample = json.loads((meta / "sample.json").read_text())
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}

    subset = set(SUBSET_TOKENS.read_text().split())
    samples_in_subset = [s for s in sample if s["scene_token"] in subset]
    samples_by_token = {s["token"]: s for s in samples_in_subset}

    cam_records: dict[str, dict] = {}
    lidar_records: dict[str, dict] = {}
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] not in samples_by_token:
            continue
        channel = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if channel == "CAM_FRONT":
            cam_records[sd["sample_token"]] = sd
        elif channel == "LIDAR_TOP":
            lidar_records[sd["sample_token"]] = sd

    # Use the first sample that has both modalities.
    for tok in cam_records:
        if tok in lidar_records:
            return cam_records[tok], lidar_records[tok], cs, sensor
    raise RuntimeError("no sample in subset has both CAM_FRONT and LIDAR_TOP keyframes")


def load_rgb_minus1_to_1(jpg_path: Path) -> torch.Tensor:
    """[3, 256, 448] float in [-1, 1]."""
    img = Image.open(jpg_path).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


def scale_intrinsics(K_native: np.ndarray) -> np.ndarray:
    """Scale 1600×900 intrinsics to 448×256 (the SD-VAE input resolution).

    Returns a `[3, 3]` float32 K matrix at the resized input resolution.
    The raymap builder further scales this by 1/SD_DOWNSAMPLE internally to get
    intrinsics on the 32×56 latent grid.
    """
    K = K_native.astype(np.float32).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K


# ----------------------------- main --------------------------------------
def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ===== 1. Find one paired sample. =====
    cam_rec, lidar_rec, cs, sensor = find_paired_keyframe()
    sample_token = cam_rec["sample_token"]
    print(f"\nsample token: {sample_token}")
    print(f"  CAM_FRONT : {cam_rec['filename']}")
    print(f"  LIDAR_TOP : {lidar_rec['filename']}")

    cs_cam = cs[cam_rec["calibrated_sensor_token"]]
    K_native = np.array(cs_cam["camera_intrinsic"], dtype=np.float32)
    K_scaled = scale_intrinsics(K_native)
    T_cam2ego = np.eye(4, dtype=np.float32)
    T_cam2ego[:3, :3] = quat_wxyz_to_rotmat(cs_cam["rotation"])
    T_cam2ego[:3, 3] = np.array(cs_cam["translation"], dtype=np.float32)
    print(f"  K (resized 448×256, fx≈{K_scaled[0,0]:.1f}, fy≈{K_scaled[1,1]:.1f})")
    print(f"  T_cam2ego translation: {T_cam2ego[:3, 3]}")

    # ===== 2. Load tensors. =====
    rgb = load_rgb_minus1_to_1(NUSCENES_ROOT / cam_rec["filename"]).unsqueeze(0).to(device)
    pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / lidar_rec["filename"]))
    range_img_np = point_cloud_to_range_image(pc)
    range_img = torch.from_numpy(range_img_np).unsqueeze(0).to(device)
    K_t = torch.from_numpy(K_scaled).unsqueeze(0).to(device)
    T_t = torch.from_numpy(T_cam2ego).unsqueeze(0).to(device)
    print(f"\nloaded tensors:")
    print(f"  rgb         : {tuple(rgb.shape)}  dtype={rgb.dtype}  range=[{rgb.min():+.3f}, {rgb.max():+.3f}]")
    print(f"  range_img   : {tuple(range_img.shape)}  range=[{range_img.min():.3f}, {range_img.max():.3f}]")
    print(f"  K_scaled    : {tuple(K_t.shape)}")
    print(f"  T_cam2ego   : {tuple(T_t.shape)}")

    # ===== 3. Build pretrained components (frozen). =====
    print("\nbuilding pretrained components...")
    image_encoder = FrozenSDVAEEncoder(local_dir=IMG_VAE_DIR).to(device)

    lidar_ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs_keys = set(inspect.signature(LiDARVAE.__init__).parameters)
    arch_kwargs = {k: v for k, v in lidar_ckpt["config"].items() if k in arch_kwargs_keys}
    lidar_vae = LiDARVAE(**arch_kwargs).to(device).eval()
    lidar_vae.load_state_dict(lidar_ckpt["state_dict"])
    lidar_vae.requires_grad_(False)
    print(f"  FrozenSDVAEEncoder: {sum(p.numel() for p in image_encoder.parameters())/1e6:.2f} M (frozen)")
    print(f"  LiDARVAE          : {sum(p.numel() for p in lidar_vae.parameters())/1e6:.2f} M (frozen, step={lidar_ckpt.get('step', '?')})")

    # ===== 4. Build trainable components. =====
    unet = LiDARUNet().to(device)
    diffusion = DiffusionWrapper()
    print(f"  LiDARUNet         : {count_params(unet)/1e6:.2f} M (trainable)")

    # ===== 5. Forward pass: encode both modalities, build KV, noise, predict. =====
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()

    # Image side: SD VAE → image latent → cat with raymap → pool → kv_context.
    image_latent = image_encoder(rgb)
    print(f"\nencoded:")
    print(f"  image_latent (post scaling_factor): {tuple(image_latent.shape)}  "
          f"mean={image_latent.mean():+.3f}  std={image_latent.std():.3f}")

    raymap = build_raymap(K_t, T_t, H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE)
    print(f"  raymap                              : {tuple(raymap.shape)}  "
          f"|origin|={raymap[:, :3].norm(dim=1).mean():.2f}m  "
          f"|dir|≈{raymap[:, 3:].norm(dim=1).mean():.3f} (should be ≈1)")

    kv_full = torch.cat([image_latent, raymap], dim=1)
    kv_context = F.adaptive_avg_pool2d(kv_full, (KV_POOL_H, KV_POOL_W))
    print(f"  kv_full                             : {tuple(kv_full.shape)}")
    print(f"  kv_context (pre-pooled)             : {tuple(kv_context.shape)}")

    # LiDAR side: LiDAR VAE → latent → noise.
    with torch.no_grad():
        mu, logvar = lidar_vae.encode(range_img)
    z_lidar = mu  # eval-mode: take the mean
    print(f"  z_lidar (LiDAR VAE μ)               : {tuple(z_lidar.shape)}  "
          f"mean={z_lidar.mean():+.3f}  std={z_lidar.std():.3f}")

    noise = torch.randn_like(z_lidar)
    t = diffusion.sample_timesteps(batch_size=1, device=device)
    z_noisy = diffusion.add_noise(z_lidar, noise, t)
    v_target = diffusion.get_target(z_lidar, noise, t)
    print(f"  t                                   : {t.tolist()}")
    print(f"  z_noisy                             : {tuple(z_noisy.shape)}")
    print(f"  v_target (v-prediction)             : {tuple(v_target.shape)}")

    # U-Net forward.
    v_pred = unet(z_noisy, t, kv_context)
    print(f"  v_pred                              : {tuple(v_pred.shape)}")

    # Loss + backward + one optimizer step.
    loss = F.mse_loss(v_pred, v_target)
    print(f"\nloss            : {loss.item():.6f}  (finite: {torch.isfinite(loss).item()})")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    # ===== 6. Report. =====
    print(f"\nstep time       : {dt*1000:.1f} ms")
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM       : {peak_gb*1000:.0f} MB  ({peak_gb:.2f} GB / 6.00 GB budget)")
        assert peak_gb < 6.0, f"VRAM {peak_gb:.2f} GB exceeds 6 GB M0 budget"
    assert torch.isfinite(loss).item(), "loss is not finite — M0 failed"

    print("\n" + "=" * 70)
    print("M0 SMOKE TEST PASSED.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
