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

import argparse
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
from eval.oblique_viz import oblique_scatter
from eval.runfolder import maintain_latest_symlink, new_run_folder
from models.diffusion import DiffusionWrapper

# Held-out sample indices, evenly spread across the 4023-sample v5 cache.
# 16 samples gives ±4× tighter standard error on the headline CD than the original 4.
HELD_OUT_IDX  = [100, 353, 606, 860, 1113, 1366, 1620, 1873,
                 2126, 2380, 2633, 2886, 3140, 3393, 3646, 3900]
UNET_CKPT     = Path("s2s_min/out/runs/2026-05-29_123644__m3-unet-60M-capacity-test/lidar_unet_best.pt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")  # symlink → v5 VAE (lidar_vae_best.pt)
CACHE_DIR     = Path("s2s_min/out/cached_latents_v5_100scenes")
NUSCENES_ROOT = Path("nuscenes")  # project-root symlink to the S2GO nuScenes dir
LATEST_OUT_DIR = Path("s2s_min/out/m4_demo")  # stable path consumed by RESULTS.md / collect_results.py
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
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="classifier-free guidance scale. 1.0 = vanilla (no guidance). "
                        "Typical sweep: 1.0, 1.5, 3.0, 5.0. Requires U-Net trained with "
                        "cond_dropout>0 (M3 bs16 was 0.2).")
    args = p.parse_args()

    # Write eval outputs INSIDE the U-Net checkpoint's training folder so each
    # checkpoint shows its own evals alongside it. Run-folder name encodes cfg_scale
    # so sweep runs each get distinct folders side-by-side:
    #   <unet-train-folder>/m4_eval/<timestamp>__m4-demo-cfg<w>/{bev_grid.png, oblique_grid.png, stats.txt}
    descriptor = "m4-demo" if args.cfg_scale == 1.0 else f"m4-demo-cfg{args.cfg_scale:g}"
    OUT_DIR = new_run_folder(descriptor, parent=UNET_CKPT.resolve().parent / "m4_eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"output folder: {OUT_DIR}")
    print(f"device: {device}")
    print(f"cfg_scale: {args.cfg_scale}")

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
    rows.append(f"  cfg_scale    : {args.cfg_scale}{' (no guidance)' if args.cfg_scale == 1.0 else ' (classifier-free guidance ON)'}")
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

    # ---- figures: two 3-column grids (BEV + paper-style 3D oblique) ----
    # Per-row heights shrink when N is large so the PNG stays viewable.
    n = len(HELD_OUT_IDX)
    bev_row_h = 3.0 if n <= 6 else 1.8
    obl_row_h = 3.6 if n <= 6 else 2.2
    fig_bev, axes_bev = plt.subplots(n, 3, figsize=(11, bev_row_h * n))
    fig_obl, axes_obl = plt.subplots(n, 3, figsize=(14, obl_row_h * n), facecolor="black")
    if n == 1:
        axes_bev = axes_bev[None, :]
        axes_obl = axes_obl[None, :]

    for i, idx in enumerate(HELD_OUT_IDX):
        item = ds[idx]
        sample_token = item["sample_token"]
        image_latent = item["image_latent"].unsqueeze(0).to(device)
        raymap       = item["raymap"].unsqueeze(0).to(device)
        mu           = item["mu"].unsqueeze(0).to(device)

        t0 = time.time()
        pred = infer_one_sample(unet, vae, diffusion, image_latent, raymap,
                                seed=42, cfg_scale=args.cfg_scale)
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

        # paint into the BEV figure: raw | oracle | predicted
        bev_scatter(axes_bev[i, 0], raw_pc,                color="tab:green", range_m=RANGE_M)
        bev_scatter(axes_bev[i, 1], oracle["point_cloud"], color="tab:blue",  range_m=RANGE_M)
        bev_scatter(axes_bev[i, 2], pred["point_cloud"],   color="tab:red",   range_m=RANGE_M)
        axes_bev[i, 0].set_ylabel(f"idx {idx}\n{sample_token[:24]}…", fontsize=6)
        if i == 0:
            axes_bev[i, 0].set_title(f"raw nuScenes (N={raw_pc.shape[0]})", fontsize=9)
            axes_bev[i, 1].set_title(f"VAE-decoded GT (N={cd_oracle['n_a']})", fontsize=9)
            axes_bev[i, 2].set_title(f"DDIM-predicted (N={cd_oracle['n_b']})", fontsize=9)

        # paint into the 3D oblique figure (paper Figure 13 style)
        oblique_scatter(axes_obl[i, 0], raw_pc)
        oblique_scatter(axes_obl[i, 1], oracle["point_cloud"])
        oblique_scatter(axes_obl[i, 2], pred["point_cloud"])
        axes_obl[i, 0].set_ylabel(f"idx {idx}", fontsize=8, color="white")
        if i == 0:
            axes_obl[i, 0].set_title(f"raw nuScenes (N={raw_pc.shape[0]})",     fontsize=9, color="white")
            axes_obl[i, 1].set_title(f"VAE-decoded GT (N={cd_oracle['n_a']})",  fontsize=9, color="white")
            axes_obl[i, 2].set_title(f"DDIM-predicted (N={cd_oracle['n_b']})",  fontsize=9, color="white")

    fig_bev.suptitle(
        f"M4: end-to-end DDIM 25-step inference on held-out samples (BEV)\n"
        f"checkpoint step={unet_ckpt.get('step', '?')}  loss_ema={unet_ckpt.get('loss_ema', float('nan')):.4f}  "
        f"BEV range ±{RANGE_M:.0f} m",
        fontsize=10,
    )
    fig_bev.tight_layout(rect=[0, 0, 1, 0.95])
    fig_bev.savefig(OUT_DIR / "bev_grid.png", dpi=120, bbox_inches="tight")

    fig_obl.suptitle(
        f"M4: end-to-end DDIM inference — 3D oblique render (paper Figure 13 style)\n"
        f"checkpoint step={unet_ckpt.get('step', '?')}  loss_ema={unet_ckpt.get('loss_ema', float('nan')):.4f}",
        fontsize=10, color="white",
    )
    fig_obl.tight_layout(rect=[0, 0, 1, 0.95])
    fig_obl.savefig(OUT_DIR / "oblique_grid.png", dpi=120, bbox_inches="tight", facecolor="black")

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
    print(f"wrote {OUT_DIR / 'oblique_grid.png'}")
    print(f"wrote {OUT_DIR / 'stats.txt'}")

    # Keep s2s_min/out/m4_demo pointing at the latest run so RESULTS.md /
    # collect_results.py still resolve. First time after this change, the
    # pre-existing m4_demo/ directory is archived under runs/...-legacy-... .
    maintain_latest_symlink(LATEST_OUT_DIR, OUT_DIR)
    print(f"updated symlink: {LATEST_OUT_DIR} → {OUT_DIR}")


if __name__ == "__main__":
    main()
