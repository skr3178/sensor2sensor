"""Oblique-grid + image-overlay visualization on DINOv3 ckpt, K1 + CFG=3.5.

Produces two figures per sample set:

  1. oblique_grid.png — N rows × 3 cols:
       [raw nuScenes GT | VAE-decoded oracle decode(μ) | DINOv3 DDIM pred]
     in paper Fig-13 oblique chase-cam style (height-colored).

  2. image_overlay.png — N rows × 2 cols:
       [CAM_FRONT image + projected GT LiDAR | CAM_FRONT image + projected pred LiDAR]
     points colored by range (turbo), z-clipped, projected via the nuScenes
     extrinsics: LiDAR sensor frame → ego → CAM_FRONT frame → image plane.

Uses the current diffusion.py defaults (clip_sample=False ← K1 fix). User
supplies --cfg_scale (default 3.5, the Phase-1 optimum).

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/scripts/dinov3_oblique_and_image_overlay.py \
        --n 4 --cfg_scale 3.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.range_image import load_nuscenes_lidar_bin, range_image_to_point_cloud
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from eval.oblique_viz import oblique_scatter
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj
from run_m4_demo import raw_lidar_for_sample

SDVAE_CACHE = Path("s2s_min/out/cached_latents_v5_100scenes")
D3_CACHE    = Path("s2s_min/out/cached_dinov3_v5_100scenes")
VAE_CKPT    = Path("s2s_min/out/lidar_vae_best.pt")
NUSCENES    = Path("nuscenes")


# ───────────────────────────── nuScenes metadata cache ─────────────────────────
_META = None
def load_meta():
    global _META
    if _META is not None:
        return _META
    sd = json.loads((NUSCENES / "v1.0-trainval" / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((NUSCENES / "v1.0-trainval" / "calibrated_sensor.json").read_text())}
    sn = {s["token"]: s for s in json.loads((NUSCENES / "v1.0-trainval" / "sensor.json").read_text())}
    lid_by_sample, cam_by_sample = {}, {}
    for r in sd:
        if not r["is_key_frame"]:
            continue
        ch = sn[cs[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if ch == "LIDAR_TOP":
            lid_by_sample[r["sample_token"]] = (r, cs[r["calibrated_sensor_token"]])
        elif ch == "CAM_FRONT":
            cam_by_sample[r["sample_token"]] = (r, cs[r["calibrated_sensor_token"]])
    _META = (lid_by_sample, cam_by_sample)
    return _META


# ───────────────────────────── geometry helpers ─────────────────────────────
def quat_to_rot(q):
    """nuScenes uses [w, x, y, z] quaternions."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),       2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)],
    ])

def build_T(translation, rotation_quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rotation_quat)
    T[:3, 3]  = translation
    return T

def project_lidar_to_cam(points_lid, cam_cs, lid_cs):
    """[N, 3] in LiDAR frame → [M, 4] = (u, v, depth, original_idx) in front of camera."""
    T_lid2ego = build_T(lid_cs["translation"], lid_cs["rotation"])
    T_cam2ego = build_T(cam_cs["translation"], cam_cs["rotation"])
    T_ego2cam = np.linalg.inv(T_cam2ego)
    T_lid2cam = T_ego2cam @ T_lid2ego

    pts_h = np.concatenate([points_lid, np.ones((len(points_lid), 1))], axis=1)  # [N, 4]
    pts_cam = (T_lid2cam @ pts_h.T).T[:, :3]                                    # [N, 3]

    # Behind camera → drop
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    orig_idx = np.where(in_front)[0]

    K = np.array(cam_cs["camera_intrinsic"])
    uvw = (K @ pts_cam.T).T                  # [M, 3]
    uv  = uvw[:, :2] / uvw[:, 2:3]           # [M, 2]
    depth = pts_cam[:, 2]                    # [M] in meters

    return np.column_stack([uv, depth, orig_idx])


# ───────────────────────────── inference helper ─────────────────────────────
@torch.no_grad()
def gen_pc(unet, vae, diffusion, kv_context, seed, cfg_scale, device):
    torch.manual_seed(seed)
    z = diffusion.ddim_sample_cfg(
        unet=unet, shape=(1, 8, 8, 256), kv_context=kv_context,
        device=torch.device(device), cfg_scale=cfg_scale,
    )
    rng = vae.decode(z)[0].cpu().numpy().clip(0, 1)
    return range_image_to_point_cloud(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3_ckpt", type=Path, default=None,
                    help="Defaults to whatever .last_dinov3_run points to.")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cfg_scale", type=float, default=3.5,
                    help="Phase 1 optimum is 3.5 (see lidar-unet.md §12.3).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("s2s_min/out/dinov3_oblique_image_overlay"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    if args.dinov3_ckpt is None:
        args.dinov3_ckpt = Path(open("s2s_min/out/.last_dinov3_run").read().strip()) / "lidar_unet_best.pt"
    assert args.dinov3_ckpt.exists()

    print("=" * 70)
    print("DINOv3 oblique + image overlay (K1 fix + CFG sweep optimum)")
    print("=" * 70)
    print(f"  ckpt        : {args.dinov3_ckpt}")
    print(f"  cfg_scale   : {args.cfg_scale}")
    print(f"  n samples   : {args.n}")
    print(f"  device      : {device}")

    unet, ckpt = load_unet(args.dinov3_ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)
    print(f"  ckpt step / loss_ema  : {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}")
    print(f"  K1 clip_sample        : {diffusion.inference_scheduler.config.clip_sample}")

    lid_by_sample, cam_by_sample = load_meta()

    # Pick n tokens that are in BOTH caches AND have CAM_FRONT+LIDAR_TOP metadata.
    common = sorted(
        ({p.stem for p in SDVAE_CACHE.glob("*.npz")} & {p.stem for p in D3_CACHE.glob("*.npz")})
        & set(lid_by_sample.keys()) & set(cam_by_sample.keys())
    )
    step = len(common) // args.n
    picks = [common[i * step] for i in range(args.n)]
    print(f"  picked {len(picks)} tokens evenly across {len(common)} common samples")

    # ─────────────────────── figures ───────────────────────
    # Oblique grid: N rows × 3 cols (raw GT | oracle | pred) on black bg
    obl_h = 3.6 if args.n <= 4 else 2.2
    fig_obl, ax_obl = plt.subplots(args.n, 3, figsize=(14, obl_h * args.n), facecolor="black")
    if args.n == 1:
        ax_obl = ax_obl[None, :]

    # Image overlay: N rows × 2 cols (img + GT LiDAR | img + pred LiDAR)
    fig_img, ax_img = plt.subplots(args.n, 2, figsize=(16, 4 * args.n))
    if args.n == 1:
        ax_img = ax_img[None, :]

    cd_list = []

    for i, tok in enumerate(picks):
        L = np.load(SDVAE_CACHE / f"{tok}.npz")
        D = np.load(D3_CACHE / f"{tok}.npz")
        raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
        mu     = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
        feat   = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)

        # KV context for DINOv3
        p4 = proj(feat)
        p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
        kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

        # ─── Generate predicted point cloud ───
        t0 = time.time()
        pc_pred = gen_pc(unet, vae, diffusion, kv, args.seed + i, args.cfg_scale, device)
        # VAE oracle (decode of cached μ)
        with torch.no_grad():
            rng_oracle = vae.decode(mu)[0].cpu().numpy().clip(0, 1)
        pc_oracle = range_image_to_point_cloud(rng_oracle)
        # Raw GT
        pc_raw = raw_lidar_for_sample(tok)[:, :3]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        cd_raw = chamfer_distance(pc_pred, pc_raw)["cd"]
        cd_oracle = chamfer_distance(pc_pred, pc_oracle)["cd"]
        cd_vae = chamfer_distance(pc_oracle, pc_raw)["cd"]
        cd_list.append((cd_raw, cd_oracle, cd_vae))
        print(f"  [{i+1}/{args.n}] tok={tok[:8]}…  "
              f"CD-3D-raw={cd_raw:.3f}  CD-3D-oracle={cd_oracle:.3f}  CD-VAE-only={cd_vae:.3f}  "
              f"({dt:.2f}s)")

        # ─── Oblique row ───
        oblique_scatter(ax_obl[i, 0], pc_raw)
        oblique_scatter(ax_obl[i, 1], pc_oracle)
        oblique_scatter(ax_obl[i, 2], pc_pred)
        ax_obl[i, 0].set_ylabel(f"sample {i}\n{tok[:8]}…", color="white", fontsize=8)
        if i == 0:
            ax_obl[i, 0].set_title(f"Raw nuScenes GT (N={len(pc_raw)})", color="white", fontsize=10)
            ax_obl[i, 1].set_title(f"VAE oracle decode(μ) (N={len(pc_oracle)})", color="white", fontsize=10)
            ax_obl[i, 2].set_title(f"DINOv3 DDIM pred, cfg={args.cfg_scale} (N={len(pc_pred)})", color="white", fontsize=10)

        # ─── Image overlay row ───
        cam_rec, cam_cs = cam_by_sample[tok]
        lid_rec, lid_cs = lid_by_sample[tok]
        img = np.array(Image.open(NUSCENES / cam_rec["filename"]))     # [H, W, 3], uint8
        H, W = img.shape[:2]

        for col, (pc, label, color_label) in enumerate([
            (pc_raw, f"raw GT (N={len(pc_raw)})",  "GT"),
            (pc_pred, f"DINOv3 pred cfg={args.cfg_scale} (CD-3D-raw={cd_raw:.2f}m)", "pred"),
        ]):
            ax = ax_img[i, col]
            ax.imshow(img)
            uvd = project_lidar_to_cam(pc[:, :3], cam_cs, lid_cs)  # only x,y,z; some may be [N, 4+]
            # Filter to image bounds + reasonable range
            mask = (uvd[:, 0] >= 0) & (uvd[:, 0] < W) & (uvd[:, 1] >= 0) & (uvd[:, 1] < H) & (uvd[:, 2] < 80)
            uvd = uvd[mask]
            sc = ax.scatter(uvd[:, 0], uvd[:, 1], c=uvd[:, 2],
                            s=2.0, alpha=0.65, cmap="turbo", vmin=2.0, vmax=60.0,
                            edgecolors="none")
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"sample {i} — {label}", fontsize=9)
            if col == 1:
                plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="depth (m)")

    cd_arr = np.array(cd_list)
    print()
    print(f"  Means across {args.n} samples:")
    print(f"    CD-3D-raw     : {cd_arr[:, 0].mean():.3f} m")
    print(f"    CD-3D-oracle  : {cd_arr[:, 1].mean():.3f} m")
    print(f"    CD-VAE-only   : {cd_arr[:, 2].mean():.3f} m")
    print(f"    Diffusion gap : {cd_arr[:, 0].mean() - cd_arr[:, 2].mean():+.3f} m")

    fig_obl.suptitle(
        f"Oblique grid — DINOv3 ckpt step {ckpt.get('step')}, K1 + CFG={args.cfg_scale}\n"
        f"mean CD-3D-raw {cd_arr[:, 0].mean():.3f} m, mean CD-VAE-only {cd_arr[:, 2].mean():.3f} m",
        color="white", fontsize=11,
    )
    fig_obl.tight_layout()
    obl_path = args.out_dir / f"oblique_grid_cfg{args.cfg_scale:g}.png"
    fig_obl.savefig(obl_path, dpi=120, facecolor="black")
    plt.close(fig_obl)

    fig_img.suptitle(
        f"Image + projected LiDAR — DINOv3 ckpt, K1 + CFG={args.cfg_scale}, "
        f"mean CD-3D-raw {cd_arr[:, 0].mean():.3f} m",
        fontsize=11,
    )
    fig_img.tight_layout()
    img_path = args.out_dir / f"image_overlay_cfg{args.cfg_scale:g}.png"
    fig_img.savefig(img_path, dpi=120)
    plt.close(fig_img)

    # stats
    stats_path = args.out_dir / f"stats_cfg{args.cfg_scale:g}.txt"
    stats_path.write_text(
        f"DINOv3 ckpt: {args.dinov3_ckpt}\n"
        f"step / loss_ema: {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}\n"
        f"cfg_scale: {args.cfg_scale}\n"
        f"K1 clip_sample: {diffusion.inference_scheduler.config.clip_sample}\n\n"
        f"Per-sample:\n" +
        "\n".join(f"  {t[:8]}…  CD-3D-raw={r:.4f}  CD-3D-oracle={o:.4f}  CD-VAE-only={v:.4f}"
                  for t, (r, o, v) in zip(picks, cd_list)) +
        f"\n\nMeans:\n"
        f"  CD-3D-raw     : {cd_arr[:, 0].mean():.4f} m\n"
        f"  CD-3D-oracle  : {cd_arr[:, 1].mean():.4f} m\n"
        f"  CD-VAE-only   : {cd_arr[:, 2].mean():.4f} m\n"
        f"  Diffusion gap : {cd_arr[:, 0].mean() - cd_arr[:, 2].mean():+.4f} m\n"
    )

    print()
    print(f"  saved oblique grid : {obl_path}")
    print(f"  saved image overlay: {img_path}")
    print(f"  saved stats        : {stats_path}")


if __name__ == "__main__":
    main()
