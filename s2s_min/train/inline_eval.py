"""In-training-loop evaluation helpers.

Two metrics, both cheaper than MSE for stopping criterion (per `lidar-unet.md §15.x`):

  1. One-step recovery cos(ẑ₀, μ) at fixed timesteps  (~5 sec, every ~250 steps)
     Catches conditioning regressions invisible to v-MSE; analogous to §11.3 diagnostic.

  2. Held-out CD-3D-raw on N=16 fixed tokens          (~30 sec, every ~500 steps)
     The truly aligned signal; what we'd be judged on at deploy. Direct
     replacement for MSE-based best-detection.

Both compute on the LIVE U-Net weights (not EMA) for speed — the trend
across steps is the signal we care about; absolute number may differ by a
few % from a final EMA-weighted eval but the optimum step location matches.

Reuses canonical helpers (no duplication):
  - encode_dinov3 from scripts/dinov3_heldout_demo.py (live DINOv3 timm encode)
  - scale_intrinsics / make_T from train/cache_latents.py
  - build_raymap from models/raymap.py
  - chamfer_distance from eval/chamfer.py
  - range_image_to_point_cloud from data/range_image.py
  - raw_lidar_for_sample from scripts/run_m4_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure scripts/ is importable for encode_dinov3 + raw_lidar_for_sample.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from data.range_image import range_image_to_point_cloud
from eval.chamfer import chamfer_distance
from models.raymap import build_raymap
from train.cache_latents import SD_DOWNSAMPLE, make_T, scale_intrinsics

NUSCENES = Path("nuscenes")
KV_POOL_H, KV_POOL_W = 8, 64


def _index_cam_lid(training_cache_dir: Path):
    """Return (lid_by_sample, cam_front_by_sample) dicts. Loads nuScenes JSON once."""
    import json
    meta = NUSCENES / "v1.0-trainval"
    sd = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sn = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    lid_by_sample: dict = {}
    cam_by_sample: dict = {}
    for r in sd:
        if not r["is_key_frame"]:
            continue
        ch = sn[cs[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if ch == "LIDAR_TOP":
            lid_by_sample[r["sample_token"]] = (r, cs[r["calibrated_sensor_token"]])
        elif ch == "CAM_FRONT":
            cam_by_sample[r["sample_token"]] = (r, cs[r["calibrated_sensor_token"]])
    return lid_by_sample, cam_by_sample


def build_heldout_eval_set(
    n_samples: int,
    seed: int,
    training_cache_dir: Path,
    device: str,
    sdvae_cache_for_mu: Path,
    proj,
    vae=None,
):
    """Pre-load N held-out samples ONCE (live DINOv3 encode + raymap + raw GT + μ).

    Returns a list of dicts with keys: token, kv [1,10,8,64], raw_pc [N,3], mu [1,8,8,256] (or None).

    Held-out = tokens NOT in `training_cache_dir`. μ for the cos-sim diagnostic is
    either loaded from `sdvae_cache_for_mu` if available, or encoded LIVE via the
    provided `vae` (LiDARVAE.encode) from the raw .pcd.bin. If neither is possible,
    mu is set to None and the cos-sim diag will skip those samples.

    `proj` is the DINOv3Proj module; passed in so we use the SAME projection as training.
    """
    from dinov3_heldout_demo import encode_dinov3
    from run_m4_demo import raw_lidar_for_sample
    from data.range_image import point_cloud_to_range_image

    lid_by, cam_by = _index_cam_lid(training_cache_dir)
    training_tokens = {p.stem for p in training_cache_dir.glob("*.npz")}
    sdvae_tokens = {p.stem for p in sdvae_cache_for_mu.glob("*.npz")}

    # Held-out = has cam+lid metadata AND NOT in training cache.
    candidates = sorted((set(lid_by) & set(cam_by)) - training_tokens)
    is_in_dist = len(candidates) < n_samples
    if is_in_dist:
        print(f"  ⚠ Inline-eval set is IN-DISTRIBUTION: "
              f"only {len(candidates)} truly-held-out tokens vs {n_samples} requested.")
        print(f"    (Training cache covers the entire trainval cam+lid pool. "
              f"CD trend still useful as a progress signal but does NOT catch overfit. "
              f"For true held-out eval, run scripts/dinov3_heldout_demo.py post-hoc "
              f"against a smaller training cache.)")
        candidates = sorted(set(lid_by) & set(cam_by))

    rng = np.random.RandomState(seed)
    picks = sorted(rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False))

    eval_set = []
    for tok in picks:
        cam_rec, cam_cs = cam_by[tok]
        lid_rec, _      = lid_by[tok]

        # Live image encode
        img = Image.open(NUSCENES / cam_rec["filename"])
        feat = encode_dinov3(img, device)  # [1, 384, 14, 24]

        # Live raymap from CAM_FRONT calibration
        K_scaled = scale_intrinsics(np.array(cam_cs["camera_intrinsic"], dtype=np.float32))
        T_cam2ego = make_T(cam_cs["translation"], cam_cs["rotation"])
        raymap = build_raymap(
            torch.from_numpy(K_scaled), torch.from_numpy(T_cam2ego),
            H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE,
        ).to(device)  # [1, 6, 32, 56]

        # Build kv_context (DINOv3 path)
        p4 = proj(feat)
        p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
        kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

        # μ: prefer cached, fall back to live encode if VAE provided
        if tok in sdvae_tokens:
            mu = torch.from_numpy(np.load(sdvae_cache_for_mu / f"{tok}.npz")["mu"]).unsqueeze(0).to(device)
        elif vae is not None:
            from data.range_image import load_nuscenes_lidar_bin
            pc5 = load_nuscenes_lidar_bin(str(NUSCENES / lid_rec["filename"]))
            rng_img = torch.from_numpy(point_cloud_to_range_image(pc5)).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = vae.encode(rng_img)
        else:
            mu = None

        # Raw .pcd.bin for CD evaluation
        raw_pc = raw_lidar_for_sample(tok)[:, :3]

        eval_set.append({"token": tok, "kv": kv, "mu": mu, "raw_pc": raw_pc})
    return eval_set


@torch.no_grad()
def one_step_cos_diag(unet, diffusion, eval_sample, device, timesteps=(0, 500, 999)):
    """For one held-out sample, compute cos(ẑ₀, μ) at fixed timesteps via analytic inversion.

    For each t, add Gaussian noise (fixed seed for reproducibility), run U-Net once,
    invert: ẑ₀ = √α·z_t − √(1−α)·v_pred. Compare to μ.

    Returns dict {t: cos_value}, or empty dict if no μ available.
    """
    mu = eval_sample.get("mu")
    if mu is None:
        return {}
    kv = eval_sample["kv"]
    alphas_cumprod = diffusion.train_scheduler.alphas_cumprod.to(device)
    out = {}
    for t_int in timesteps:
        torch.manual_seed(42 + t_int)
        t = torch.tensor([t_int], device=device, dtype=torch.long)
        if t_int == 0:
            noise = torch.zeros_like(mu); z_noisy = mu.clone()
        else:
            noise = torch.randn_like(mu)
            z_noisy = diffusion.add_noise(mu, noise, t)
        v_pred = unet(z_noisy, t, kv)
        a = alphas_cumprod[t_int].sqrt()
        s = (1.0 - alphas_cumprod[t_int]).sqrt()
        z0_hat = a * z_noisy - s * v_pred
        c = F.cosine_similarity(z0_hat.flatten(1), mu.flatten(1), dim=-1).mean().item()
        out[t_int] = c
    return out


@torch.no_grad()
def heldout_cd_eval(unet, vae, diffusion, eval_set, cfg_scale, device, seed: int = 42):
    """Run DDIM-25 + decode + Chamfer for each held-out sample.

    Returns (mean_cd_3d, mean_cd_bev). Uses fixed seed per-sample for reproducibility.
    """
    cds_3d = []
    cds_bev = []
    for i, sample in enumerate(eval_set):
        torch.manual_seed(seed + i)
        z = diffusion.ddim_sample_cfg(
            unet=unet, shape=(1, 8, 8, 256), kv_context=sample["kv"],
            device=torch.device(device), cfg_scale=cfg_scale,
        )
        rng = vae.decode(z)[0].cpu().numpy().clip(0, 1)
        pc_pred = range_image_to_point_cloud(rng)
        cds_3d.append(chamfer_distance(pc_pred, sample["raw_pc"])["cd"])
        cds_bev.append(chamfer_distance(pc_pred, sample["raw_pc"], use_xy_only=True)["cd"])
    return float(np.mean(cds_3d)), float(np.mean(cds_bev))
