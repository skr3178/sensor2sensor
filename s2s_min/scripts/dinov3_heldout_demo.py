"""Run the DINOv3 ckpt on TRULY HELD-OUT nuScenes images.

The earlier demo scripts (dinov3_vs_raw_gt.py, dinov3_oblique_and_image_overlay.py)
default to tokens from the DINOv3 training cache (4023 samples in v5_100scenes).
This script picks tokens OUTSIDE that cache so we can see actual generalization
performance with K1 + CFG=3.5 applied.

Reuses canonical helpers (don't reinvent):
  - DINOv3 timm encoding from train/cache_dinov3.py (vit_small_patch16_dinov3.lvd1689m,
    224×384 input, 14×24 patch grid, prefix tokens stripped)
  - Raymap math from train/cache_latents.py (scale_intrinsics, make_T, build_raymap)
  - LiDAR VAE + U-Net loaders from eval/decode_to_pointcloud.py
  - BEV + image-overlay viz from existing scripts

Output: BEV grid + image-overlay grid + CD-3D-raw table; held-out CDs compared
to training-set baseline (1.730 m from §12.3) to surface the generalization gap.
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Canonical helpers (reused, not reinvented)
from data.range_image import range_image_to_point_cloud
from eval.bev_viz import bev_scatter
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj
from models.raymap import build_raymap
from run_m4_demo import raw_lidar_for_sample

# Canonical DINOv3 encoding + intrinsics scaling (matches the cache)
from train.cache_dinov3 import DINOV3_MODEL, HP, WP, GH, GW
from train.cache_latents import scale_intrinsics, make_T, SD_DOWNSAMPLE

# Reused geometry from the oblique-image script
from dinov3_oblique_and_image_overlay import load_meta, project_lidar_to_cam

NUSCENES = Path("nuscenes")
D3_CACHE = Path("s2s_min/out/cached_dinov3_v5_100scenes")  # training-set sentinel
VAE_CKPT = Path("s2s_min/out/lidar_vae_best.pt")
RANGE_M  = 60.0


# Cache the DINOv3 model + normalization on first use (avoid per-sample reload).
_TIMM = {"model": None, "mean": None, "std": None, "npfx": None}


def load_dinov3_model(device: str):
    if _TIMM["model"] is None:
        import timm
        _TIMM["model"] = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0).to(device).eval()
        _TIMM["mean"]  = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _TIMM["std"]   = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        _TIMM["npfx"]  = getattr(_TIMM["model"], "num_prefix_tokens", 1)
    return _TIMM["model"], _TIMM["mean"], _TIMM["std"], _TIMM["npfx"]


@torch.no_grad()
def encode_dinov3(img_pil: Image.Image, device: str) -> torch.Tensor:
    """Reproduces train/cache_dinov3.py per-sample encoding for a single image.
    Returns: [1, 384, 14, 24] (= GH, GW)."""
    model, mean, std, npfx = load_dinov3_model(device)
    im = img_pil.convert("RGB").resize((WP, HP), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    x = (arr - mean) / std
    tok = model.forward_features(x)                              # [1, npfx + GH*GW, 384]
    patch = tok[:, npfx:, :].transpose(1, 2).reshape(1, tok.shape[-1], GH, GW)
    return patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3_ckpt", type=Path, default=None,
                    help="Defaults to .last_dinov3_run")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cfg_scale", type=float, default=3.5,
                    help="Phase 1 optimum, see lidar-unet.md §12.3")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/dinov3_heldout_demo"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    if args.dinov3_ckpt is None:
        args.dinov3_ckpt = Path(open("s2s_min/out/.last_dinov3_run").read().strip()) / "lidar_unet_best.pt"

    print("=" * 70)
    print("DINOv3 ckpt — HELD-OUT samples (not in 4k-sample training cache)")
    print("=" * 70)
    print(f"  ckpt        : {args.dinov3_ckpt}")
    print(f"  cfg_scale   : {args.cfg_scale}")
    print(f"  n samples   : {args.n}")

    # ─── Load models ───
    unet, ckpt = load_unet(args.dinov3_ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)
    print(f"  K1 clip_sample : {diffusion.inference_scheduler.config.clip_sample}")

    # ─── Pick truly held-out tokens ───
    training_tokens = {p.stem for p in D3_CACHE.glob("*.npz")}
    lid_by_sample, cam_by_sample = load_meta()
    candidates = sorted((set(lid_by_sample) & set(cam_by_sample)) - training_tokens)
    print(f"  training tokens (in cache)            : {len(training_tokens)}")
    print(f"  HELD-OUT pool (cam+lid, NOT in cache) : {len(candidates)}")

    rng = np.random.RandomState(args.seed)
    picks = sorted(rng.choice(candidates, size=args.n, replace=False))

    # ─── Generate predictions ───
    fig_bev, ax_bev = plt.subplots(args.n, 3, figsize=(13, 4.0 * args.n))
    fig_img, ax_img = plt.subplots(args.n, 2, figsize=(16, 4.0 * args.n))
    if args.n == 1:
        ax_bev = ax_bev[None, :]; ax_img = ax_img[None, :]

    cd_list = []
    for i, tok in enumerate(picks):
        cam_rec, cam_cs = cam_by_sample[tok]
        lid_rec, lid_cs = lid_by_sample[tok]

        # Live DINOv3 encoding (matches cached_dinov3 build)
        img_pil = Image.open(NUSCENES / cam_rec["filename"])
        t0 = time.time()
        feat = encode_dinov3(img_pil, device)                                # [1, 384, 14, 24]

        # Live raymap (matches cache_latents)
        K_scaled = scale_intrinsics(np.array(cam_cs["camera_intrinsic"], dtype=np.float32))
        T_cam2ego = make_T(cam_cs["translation"], cam_cs["rotation"])
        raymap = build_raymap(
            torch.from_numpy(K_scaled), torch.from_numpy(T_cam2ego),
            H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE,
        ).to(device)                                                          # [1, 6, 32, 56]

        # KV context (same as dinov3_vs_raw_gt.py)
        p4 = proj(feat)
        p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
        kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

        # DDIM with K1 + CFG=3.5
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            z_pred = diffusion.ddim_sample_cfg(
                unet=unet, shape=(1, 8, 8, 256), kv_context=kv,
                device=torch.device(device), cfg_scale=args.cfg_scale,
            )
            rng_pred = vae.decode(z_pred)[0].cpu().numpy().clip(0, 1)
        pc_pred = range_image_to_point_cloud(rng_pred)
        pc_raw  = raw_lidar_for_sample(tok)[:, :3]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        cd_raw = chamfer_distance(pc_pred, pc_raw)["cd"]
        cd_bev = chamfer_distance(pc_pred, pc_raw, use_xy_only=True)["cd"]
        cd_list.append((cd_raw, cd_bev))
        print(f"  [{i+1}/{args.n}] tok={tok[:8]}…  CD-3D-raw={cd_raw:.3f}  CD-BEV={cd_bev:.3f}  "
              f"({dt:.2f}s)")

        # BEV row
        bev_scatter(ax_bev[i, 0], pc_raw,  color="tab:blue", range_m=RANGE_M)
        ax_bev[i, 0].set_title(f"sample {i}: raw HELD-OUT GT (N={len(pc_raw)})")
        bev_scatter(ax_bev[i, 1], pc_pred, color="tab:red",  range_m=RANGE_M)
        ax_bev[i, 1].set_title(f"DINOv3 pred K1+cfg={args.cfg_scale} (N={len(pc_pred)})  CD={cd_raw:.2f}m")
        bev_scatter(ax_bev[i, 2], pc_raw,  color="tab:blue", range_m=RANGE_M, alpha=0.35, point_size=0.4)
        bev_scatter(ax_bev[i, 2], pc_pred, color="tab:red",  range_m=RANGE_M, alpha=0.55, point_size=0.4)
        ax_bev[i, 2].set_title("overlay: GT (blue) | pred (red)")

        # Image overlay row
        H_im, W_im = img_pil.height, img_pil.width
        for col, (pc, label) in enumerate([
            (pc_raw,  f"raw GT N={len(pc_raw)}"),
            (pc_pred, f"pred N={len(pc_pred)} CD={cd_raw:.2f}m"),
        ]):
            ax = ax_img[i, col]
            ax.imshow(img_pil)
            uvd = project_lidar_to_cam(pc[:, :3], cam_cs, lid_cs)
            mask = (uvd[:, 0] >= 0) & (uvd[:, 0] < W_im) & (uvd[:, 1] >= 0) & (uvd[:, 1] < H_im) & (uvd[:, 2] < 80)
            uvd = uvd[mask]
            sc = ax.scatter(uvd[:, 0], uvd[:, 1], c=uvd[:, 2],
                            s=2.0, alpha=0.65, cmap="turbo", vmin=2.0, vmax=60.0,
                            edgecolors="none")
            ax.set_xlim(0, W_im); ax.set_ylim(H_im, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"sample {i} (HELD-OUT) — {label}", fontsize=9)
            if col == 1:
                plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="depth (m)")

    cd_arr = np.array(cd_list)
    print()
    print(f"  HELD-OUT mean: CD-3D-raw = {cd_arr[:,0].mean():.3f} m,  CD-BEV = {cd_arr[:,1].mean():.3f} m")
    print(f"  Train-set ref (§12.3, 16 samples, same K1+cfg=3.5): CD-3D-raw = 1.730 m")
    delta = cd_arr[:, 0].mean() - 1.730
    print(f"  Held-out − train Δ = {delta:+.3f} m  ({delta/1.730*100:+.1f}%)")

    fig_bev.suptitle(
        f"HELD-OUT DINOv3 (step {ckpt.get('step')}, K1 + cfg={args.cfg_scale})\n"
        f"mean CD-3D-raw {cd_arr[:,0].mean():.3f} m   "
        f"(training-set ref: 1.730 m, Δ {delta:+.3f} m)",
        fontsize=11,
    )
    fig_bev.tight_layout()
    fig_bev.savefig(args.out_dir / f"bev_heldout_cfg{args.cfg_scale:g}.png", dpi=120)
    plt.close(fig_bev)

    fig_img.suptitle(
        f"HELD-OUT image + projected LiDAR (cfg={args.cfg_scale}, mean CD={cd_arr[:,0].mean():.3f} m)",
        fontsize=11,
    )
    fig_img.tight_layout()
    fig_img.savefig(args.out_dir / f"image_heldout_cfg{args.cfg_scale:g}.png", dpi=120)
    plt.close(fig_img)

    (args.out_dir / f"stats_cfg{args.cfg_scale:g}.txt").write_text(
        f"HELD-OUT eval — DINOv3 ckpt = {args.dinov3_ckpt}\n"
        f"cfg_scale: {args.cfg_scale}\n"
        f"K1 clip_sample: {diffusion.inference_scheduler.config.clip_sample}\n\n"
        f"Picked {len(picks)} tokens from {len(candidates)} HELD-OUT candidates "
        f"(NOT in training cache of {len(training_tokens)})\n\n" +
        "\n".join(f"  {t}  CD-3D-raw={cd_arr[i,0]:.4f}  CD-BEV={cd_arr[i,1]:.4f}"
                  for i, t in enumerate(picks)) +
        f"\n\nMean CD-3D-raw (held-out): {cd_arr[:,0].mean():.4f} m\n"
        f"Mean CD-BEV   (held-out): {cd_arr[:,1].mean():.4f} m\n"
        f"Train-set ref (§12.3):    1.7300 m\n"
        f"Held-out − train:         {delta:+.4f} m  ({delta/1.730*100:+.1f}%)\n"
    )
    print(f"\n  saved bev   : {args.out_dir / f'bev_heldout_cfg{args.cfg_scale:g}.png'}")
    print(f"  saved image : {args.out_dir / f'image_heldout_cfg{args.cfg_scale:g}.png'}")
    print(f"  saved stats : {args.out_dir / f'stats_cfg{args.cfg_scale:g}.txt'}")


if __name__ == "__main__":
    main()
