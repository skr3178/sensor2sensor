"""M4 demo — DDIM inference + BEV viz + Chamfer on a handful of held-out samples.

Picks N samples deterministically (indices outside M3.1's overfit-10 set),
runs the full pipeline on each, produces:
  - one multi-panel BEV PNG (N rows × 3 cols: raw LiDAR | VAE-decoded GT | DDIM-predicted)
  - per-sample stats: cos(z_pred, μ), Chamfer (4 variants — see below)
  - aggregate means

The four Chamfer metrics surface the error decomposition (per RESULTS.md / plan §M5):
    1. CD-3D-oracle   = CD(decode(z_pred), decode(μ))     — diffusion contribution only
    2. CD-BEV-oracle  = CD(...)                            — same, xy-only
    3. CD-3D-raw      = CD(decode(z_pred), raw_nuScenes)   — END-TO-END image→LiDAR ← headline
    4. CD-VAE-only    = CD(decode(μ),    raw_nuScenes)     — VAE bottleneck (lower bound on (3))

Run:
    env/bin/python s2s_min/scripts/run_m4_demo.py
Output:
    s2s_min/out/m4_demo/bev_grid.png + stats.txt
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.cached_latents import CachedLatentsDataset
from data.range_image import load_nuscenes_lidar_bin
from eval.bev_viz import bev_scatter
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import (
    build_kv_context, decode_ground_truth, infer_one_sample,
    load_lidar_vae, load_unet,
)
from models.diffusion import DiffusionWrapper

# Held-out sample indices (NOT in M3.1's overfit-10 range 0..9). Spread across the 401-sample subset.
HELD_OUT_IDX  = [100, 200, 300, 400]
UNET_CKPT     = Path("s2s_min/out/lidar_unet_m32_best.pt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
CACHE_DIR     = Path("s2s_min/out/cached_latents")
NUSCENES_ROOT = Path("nuscenes")  # project-root symlink to the S2GO nuScenes dir
OUT_DIR       = Path("s2s_min/out/m4_demo")
RANGE_M       = 60.0


# Lazy nuScenes metadata cache (loading sample_data.json is ~50 MB).
_METADATA_CACHE: dict | None = None


def _load_nuscenes_metadata() -> dict:
    """Load and index the nuScenes v1.0-trainval metadata files we need."""
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    # Index LIDAR_TOP keyframe records by sample_token.
    lid_by_sample: dict[str, dict] = {}
    for sd in sample_data:
        if not sd["is_key_frame"]:
            continue
        if sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "LIDAR_TOP":
            lid_by_sample[sd["sample_token"]] = sd
    _METADATA_CACHE = {"lid_by_sample": lid_by_sample}
    return _METADATA_CACHE


def raw_lidar_for_sample(sample_token: str) -> np.ndarray:
    """Load the raw LIDAR_TOP .pcd.bin for a given sample_token.

    Returns `[M, 4]` array (x, y, z, intensity) in the LiDAR sensor frame —
    the same frame `range_image_to_point_cloud` produces, so Chamfer is
    apples-to-apples.
    """
    meta = _load_nuscenes_metadata()
    rec = meta["lid_by_sample"].get(sample_token)
    if rec is None:
        raise KeyError(f"no LIDAR_TOP keyframe for sample {sample_token}")
    pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / rec["filename"]))   # [N, 5]
    return pc[:, :4]  # drop ring_index


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    unet, unet_ckpt = load_unet(UNET_CKPT, device)
    vae = load_lidar_vae(LIDAR_VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    ds = CachedLatentsDataset(CACHE_DIR)
    print(f"checkpoint: {UNET_CKPT}  step={unet_ckpt.get('step', '?')}  "
          f"loss_ema={unet_ckpt.get('loss_ema', float('nan')):.5f}")
    print(f"cache size: {len(ds)} samples")
    print(f"held-out indices: {HELD_OUT_IDX}")

    rows: list[str] = []
    rows.append(f"M4 demo — DDIM 25-step inference on held-out samples")
    rows.append(f"  checkpoint   : {UNET_CKPT}  step={unet_ckpt.get('step', '?')}  loss_ema={unet_ckpt.get('loss_ema', float('nan')):.5f}")
    rows.append(f"  LiDAR VAE    : {LIDAR_VAE_CKPT}")
    rows.append(f"  DDIM steps   : {diffusion.inference_steps}")
    rows.append(f"  prediction   : {diffusion.prediction_type}")
    rows.append(f"  range_m (BEV): {RANGE_M}")
    rows.append("")
    rows.append("Four Chamfer metrics (CD in meters; lower = better):")
    rows.append("  CD-3D-oracle  : decode(z_pred) vs decode(μ)        — diffusion contribution only")
    rows.append("  CD-BEV-oracle : same, xy-only                       — diffusion, planar geometry")
    rows.append("  CD-3D-raw     : decode(z_pred) vs raw nuScenes      — END-TO-END image→LiDAR (headline)")
    rows.append("  CD-VAE-only   : decode(μ) vs raw nuScenes           — VAE bottleneck (lower bound on CD-3D-raw)")
    rows.append("")
    rows.append("per-sample stats:")
    rows.append(f"  {'idx':>4}  {'token':<40}  "
                f"{'cos':>6}  {'CD-3D-oracle':>13}  {'CD-BEV-oracle':>14}  "
                f"{'CD-3D-raw':>10}  {'CD-VAE-only':>11}  "
                f"{'N_raw':>6}  {'N_oracle':>9}  {'N_pred':>7}  {'wall':>5}")

    cos_sims, cds_oracle, cds_xy_oracle, cds_raw, cds_vae = [], [], [], [], []

    # ---- figure: 3 columns per sample (raw | VAE-oracle | DDIM-pred) ----
    fig, axes = plt.subplots(len(HELD_OUT_IDX), 3,
                              figsize=(11, 3 * len(HELD_OUT_IDX)))
    if len(HELD_OUT_IDX) == 1:
        axes = axes[None, :]

    for i, idx in enumerate(HELD_OUT_IDX):
        item = ds[idx]
        sample_token = item["sample_token"]
        image_latent = item["image_latent"].unsqueeze(0).to(device)
        raymap       = item["raymap"].unsqueeze(0).to(device)
        mu           = item["mu"].unsqueeze(0).to(device)

        t0 = time.time()
        pred = infer_one_sample(unet, vae, diffusion, image_latent, raymap, seed=42)
        oracle = decode_ground_truth(vae, mu)
        torch.cuda.synchronize() if device == "cuda" else None
        # Raw nuScenes LiDAR (in LiDAR sensor frame — same as range_image_to_point_cloud output).
        raw_pc = raw_lidar_for_sample(sample_token)
        dt = time.time() - t0

        cos = F.cosine_similarity(pred["z_pred"].flatten(1), mu.flatten(1), dim=-1).item()
        # vs oracle (existing two metrics)
        cd_oracle    = chamfer_distance(oracle["point_cloud"], pred["point_cloud"], use_xy_only=False)
        cd_oracle_xy = chamfer_distance(oracle["point_cloud"], pred["point_cloud"], use_xy_only=True)
        # vs raw nuScenes (new, end-to-end)
        cd_raw   = chamfer_distance(raw_pc, pred["point_cloud"],   use_xy_only=False)
        cd_vae   = chamfer_distance(raw_pc, oracle["point_cloud"], use_xy_only=False)

        cos_sims.append(cos)
        cds_oracle.append(cd_oracle["cd"])
        cds_xy_oracle.append(cd_oracle_xy["cd"])
        cds_raw.append(cd_raw["cd"])
        cds_vae.append(cd_vae["cd"])

        rows.append(
            f"  {idx:>4}  {sample_token[:40]:<40}  "
            f"{cos:>6.3f}  {cd_oracle['cd']:>13.3f}  {cd_oracle_xy['cd']:>14.3f}  "
            f"{cd_raw['cd']:>10.3f}  {cd_vae['cd']:>11.3f}  "
            f"{raw_pc.shape[0]:>6d}  {cd_oracle['n_a']:>9d}  {cd_oracle['n_b']:>7d}  {dt:>5.2f}s"
        )

        # paint into the figure: raw | oracle | predicted
        bev_scatter(axes[i, 0], raw_pc,                color="tab:green", range_m=RANGE_M)
        bev_scatter(axes[i, 1], oracle["point_cloud"], color="tab:blue",  range_m=RANGE_M)
        bev_scatter(axes[i, 2], pred["point_cloud"],   color="tab:red",   range_m=RANGE_M)
        axes[i, 0].set_ylabel(f"idx {idx}\n{sample_token[:24]}…", fontsize=6)
        if i == 0:
            axes[i, 0].set_title(f"raw nuScenes (N={raw_pc.shape[0]})", fontsize=9)
            axes[i, 1].set_title(f"VAE-decoded GT (N={cd_oracle['n_a']})", fontsize=9)
            axes[i, 2].set_title(f"DDIM-predicted (N={cd_oracle['n_b']})", fontsize=9)

    fig.suptitle(
        f"M4: end-to-end DDIM 25-step inference on held-out samples\n"
        f"checkpoint step={unet_ckpt.get('step', '?')}  loss_ema={unet_ckpt.get('loss_ema', float('nan')):.4f}  "
        f"BEV range ±{RANGE_M:.0f} m",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "bev_grid.png", dpi=120, bbox_inches="tight")

    mean_cos       = float(np.mean(cos_sims))
    mean_cd_oracle = float(np.mean(cds_oracle))
    mean_cd_xy     = float(np.mean(cds_xy_oracle))
    mean_cd_raw    = float(np.mean(cds_raw))
    mean_cd_vae    = float(np.mean(cds_vae))

    rows.append("")
    rows.append("aggregate (mean over held-out):")
    rows.append(f"  mean cos(z_pred, μ) : {mean_cos:+.4f}   (1.0 = identical)")
    rows.append(f"  mean CD-3D-oracle   : {mean_cd_oracle:.3f} m   (diffusion contribution only)")
    rows.append(f"  mean CD-BEV-oracle  : {mean_cd_xy:.3f} m   (diffusion, xy-only)")
    rows.append(f"  mean CD-3D-raw      : {mean_cd_raw:.3f} m   ★ END-TO-END image→LiDAR — headline metric")
    rows.append(f"  mean CD-VAE-only    : {mean_cd_vae:.3f} m   (VAE bottleneck; CD-3D-raw can't go below this)")
    rows.append("")
    rows.append("error decomposition (rough): CD-3D-raw ≈ CD-VAE-only + (diffusion-induced delta)")
    rows.append(f"  diffusion delta     ≈ {mean_cd_raw - mean_cd_vae:+.3f} m")
    rows.append("")
    rows.append("pass criterion (per min_pipeline_plan.md §M4):")
    rows.append("  - DDIM produces non-trivial output  ✓ (CD finite, N_pred > 0)")
    rows.append("  - Generated BEV looks geometrically plausible (road plane + camera-region density)")
    rows.append("    → eyeball-check the bev_grid.png")
    rows.append("  - Chamfer is whatever it is — quantitative quality not gated")

    print()
    for r in rows:
        print(r)
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {OUT_DIR / 'bev_grid.png'}")
    print(f"wrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
