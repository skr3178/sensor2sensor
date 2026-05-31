"""DINOv3 U-Net inference vs raw nuScenes LIDAR_TOP ground truth.

Takes the best DINOv3 checkpoint (from `s2s_min/out/.last_dinov3_run`), runs
DDIM-25 inference on a few cached samples, decodes via the LiDAR VAE, and
compares the predicted point cloud against the **raw .pcd.bin** ground truth
(NOT the oracle VAE-decode of μ — that comparison is `compare_encoders.py`).

Produces:
  - bev_grid.png  : N-row × 3-col panel (RAW GT | DINOv3 PRED | overlay)
  - stats.txt     : Chamfer (3D + BEV) vs raw GT per sample

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/scripts/dinov3_vs_raw_gt.py
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/scripts/dinov3_vs_raw_gt.py --cfg_scale 2.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.range_image import range_image_to_point_cloud
from eval.bev_viz import bev_scatter
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj

# Reuse run_m4_demo's nuScenes raw-LiDAR loader so we hit the same .pcd.bin /
# same sensor frame conventions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_m4_demo import raw_lidar_for_sample  # noqa: E402

# Defaults.
SDVAE_CACHE = Path("s2s_min/out/cached_latents_v5_100scenes")
D3_CACHE    = Path("s2s_min/out/cached_dinov3_v5_100scenes")
VAE_CKPT    = Path("s2s_min/out/lidar_vae_best.pt")
RANGE_M     = 60.0
SEED        = 42

# Sample tokens to evaluate. Pick 4 distinct scenes (token-sorted, evenly
# spaced) so they're as visually diverse as possible.
N_SAMPLES   = 4


def build_kv_dinov3(proj: DINOv3Proj, feat: torch.Tensor, raymap: torch.Tensor) -> torch.Tensor:
    """[B,384,14,24] → 1×1 conv to 4ch → bilinear up to raymap grid → cat raymap → pool."""
    p4 = proj(feat)
    p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))


@torch.no_grad()
def gen_pc(unet, vae, diffusion, kv_context, seed: int, cfg_scale: float, device):
    torch.manual_seed(seed)
    z = diffusion.ddim_sample_cfg(
        unet=unet, shape=(1, 8, 8, 256), kv_context=kv_context,
        device=torch.device(device), cfg_scale=cfg_scale,
    )
    rng = vae.decode(z)[0].cpu().numpy().clip(0, 1)            # [3, 32, 1024]
    return range_image_to_point_cloud(rng), rng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dinov3_ckpt", type=Path, default=None,
                   help="DINOv3 U-Net best.pt. Defaults to whatever .last_dinov3_run points at.")
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="classifier-free guidance scale (1.0 = vanilla, 2.5 = SD-VAE optimum)")
    p.add_argument("--n", type=int, default=N_SAMPLES, help="how many samples")
    p.add_argument("--out_dir", type=Path,
                   default=Path("s2s_min/out/dinov3_vs_raw_gt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Resolve DINOv3 ckpt from .last_dinov3_run if not given.
    if args.dinov3_ckpt is None:
        last = Path("s2s_min/out/.last_dinov3_run").read_text().strip()
        args.dinov3_ckpt = Path(last) / "lidar_unet_best.pt"
    assert args.dinov3_ckpt.exists(), f"missing: {args.dinov3_ckpt}"

    print("=" * 70)
    print("DINOv3 decode vs raw nuScenes ground truth")
    print("=" * 70)
    print(f"  device       : {device}")
    print(f"  DINOv3 ckpt  : {args.dinov3_ckpt}")
    print(f"  cfg_scale    : {args.cfg_scale}")
    print(f"  n samples    : {args.n}")
    print(f"  range_m (BEV): {RANGE_M}")

    # ---- load model + VAE + projection head ----
    unet, ckpt = load_unet(args.dinov3_ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)
    print(f"  DINOv3 step / loss_ema : {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}")

    # ---- pick sample tokens that exist in BOTH caches ----
    sdvae_toks = {p.stem for p in SDVAE_CACHE.glob("*.npz")}
    d3_toks    = {p.stem for p in D3_CACHE.glob("*.npz")}
    common = sorted(sdvae_toks & d3_toks)
    assert len(common) >= args.n
    # Evenly space across token list.
    step = len(common) // args.n
    picks = [common[i * step] for i in range(args.n)]
    print(f"  picked {len(picks)} tokens, step={step} across {len(common)} common samples")

    # ---- main loop ----
    rows = []
    rows.append(f"DINOv3 decode vs raw nuScenes LIDAR_TOP — {args.n} samples")
    rows.append(f"  ckpt        : {args.dinov3_ckpt}  step={ckpt.get('step')}  loss_ema={ckpt.get('loss_ema'):.4f}")
    rows.append(f"  cfg_scale   : {args.cfg_scale}")
    rows.append(f"  DDIM steps  : {diffusion.inference_steps}")
    rows.append("")
    rows.append("Three Chamfer metrics (m; lower = better):")
    rows.append("  CD-3D-raw    : decode(z_pred) vs raw nuScenes      — END-TO-END image→LiDAR")
    rows.append("  CD-BEV-raw   : same, xy-only                       — planar geometry")
    rows.append("  CD-VAE-only  : decode(μ)      vs raw nuScenes      — VAE-bottleneck (lower bound on CD-3D-raw)")
    rows.append("")
    rows.append(f"  {'idx':>3}  {'token':<36}  "
                f"{'CD-3D-raw':>10}  {'CD-BEV-raw':>11}  {'CD-VAE-only':>12}  "
                f"{'N_raw':>6}  {'N_pred':>7}  {'wall':>5}")

    fig, axes = plt.subplots(args.n, 3, figsize=(13, 4.0 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    cd3, cd_bev, cd_vae = [], [], []

    for i, tok in enumerate(picks):
        L = np.load(SDVAE_CACHE / f"{tok}.npz")
        D = np.load(D3_CACHE / f"{tok}.npz")
        raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
        mu     = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
        feat   = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)

        # ---- predicted (DINOv3 → DDIM → VAE decode → unproject) ----
        t0 = time.time()
        kv = build_kv_dinov3(proj, feat, raymap)
        pc_pred, _ = gen_pc(unet, vae, diffusion, kv, seed=SEED + i,
                            cfg_scale=args.cfg_scale, device=device)

        # ---- raw GT (.pcd.bin) ----
        pc_raw = raw_lidar_for_sample(tok)[:, :3]

        # ---- VAE-only oracle (for the lower-bound metric) ----
        with torch.no_grad():
            rng_oracle = vae.decode(mu)[0].cpu().numpy().clip(0, 1)
        pc_oracle = range_image_to_point_cloud(rng_oracle)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        # ---- Chamfer ----
        d_3d  = chamfer_distance(pc_pred,   pc_raw)["cd"]
        d_bev = chamfer_distance(pc_pred,   pc_raw, use_xy_only=True)["cd"]
        d_vae = chamfer_distance(pc_oracle, pc_raw)["cd"]
        cd3.append(d_3d)
        cd_bev.append(d_bev)
        cd_vae.append(d_vae)

        rows.append(f"  {i:>3}  {tok:<36}  "
                    f"{d_3d:>10.3f}  {d_bev:>11.3f}  {d_vae:>12.3f}  "
                    f"{len(pc_raw):>6d}  {len(pc_pred):>7d}  {dt:>5.2f}s")
        print(f"  [{i+1}/{args.n}] token={tok[:8]}…  CD-3D-raw={d_3d:.3f}  "
              f"CD-BEV={d_bev:.3f}  CD-VAE-only={d_vae:.3f}  ({dt:.2f}s)")

        # ---- BEV: GT | PRED | overlay ----
        bev_scatter(axes[i, 0], pc_raw,  color="tab:blue", range_m=RANGE_M)
        axes[i, 0].set_title(f"sample {i}: raw nuScenes GT ({len(pc_raw)} pts)")
        bev_scatter(axes[i, 1], pc_pred, color="tab:red",  range_m=RANGE_M)
        axes[i, 1].set_title(f"DINOv3 DDIM-25 pred ({len(pc_pred)} pts)  CD-3D={d_3d:.2f} m")

        ax = axes[i, 2]
        bev_scatter(ax, pc_raw,  color="tab:blue", range_m=RANGE_M, alpha=0.35, point_size=0.4)
        bev_scatter(ax, pc_pred, color="tab:red",  range_m=RANGE_M, alpha=0.55, point_size=0.4)
        ax.set_title("overlay: GT (blue) | pred (red)")

    rows.append("")
    rows.append("Summary (means across samples):")
    rows.append(f"  CD-3D-raw    : {np.mean(cd3):.3f} m  (median {np.median(cd3):.3f})")
    rows.append(f"  CD-BEV-raw   : {np.mean(cd_bev):.3f} m  (median {np.median(cd_bev):.3f})")
    rows.append(f"  CD-VAE-only  : {np.mean(cd_vae):.3f} m  (median {np.median(cd_vae):.3f})")
    rows.append("")
    rows.append(f"Interpretation:")
    rows.append(f"  CD-VAE-only is the floor the VAE itself can hit (image→LiDAR perfect oracle).")
    rows.append(f"  CD-3D-raw minus CD-VAE-only is the diffusion model's added error.")
    rows.append(f"  Diffusion gap = {np.mean(cd3) - np.mean(cd_vae):+.3f} m")

    fig.suptitle(
        f"DINOv3 decode vs raw nuScenes LIDAR_TOP — step {ckpt.get('step')}, "
        f"loss_ema {ckpt.get('loss_ema'):.4f}, cfg={args.cfg_scale}",
        fontsize=12,
    )
    fig.tight_layout()
    bev_path = args.out_dir / f"bev_grid_cfg{args.cfg_scale:g}.png"
    fig.savefig(bev_path, dpi=130)
    plt.close(fig)

    stats_path = args.out_dir / f"stats_cfg{args.cfg_scale:g}.txt"
    stats_path.write_text("\n".join(rows))
    print()
    print("\n".join(rows[-10:]))
    print()
    print(f"  saved BEV   : {bev_path}")
    print(f"  saved stats : {stats_path}")


if __name__ == "__main__":
    main()
