"""Phase 0 — Inference sanity check on the A-v2 overfit-1 checkpoint.

If the model has truly memorized its one training sample, running DDIM-25
inference on that exact sample's conditioning should produce a LiDAR latent
nearly identical to the cached `mu`, and a decoded point cloud nearly
identical to the VAE-decode of `mu` (the oracle target).

This is the strongest possible test that:
  1. The model CAN solve image→LiDAR end-to-end
  2. The DINOv3 conditioning pathway carries enough signal
  3. The KV pool / cross-attention / decoder pipeline all function correctly

Metrics computed:
  - cos(z_pred, μ)          : latent-space similarity (memorization quality)
  - CD-3D-oracle  = CD(decode(z_pred), decode(μ))     : diffusion-only error
  - CD-3D-raw     = CD(decode(z_pred), raw nuScenes)  : end-to-end error
  - CD-VAE-only   = CD(decode(μ), raw nuScenes)       : VAE bottleneck floor

Visualization: 1×3 grid showing raw GT | VAE oracle | A-v2 prediction.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.range_image import range_image_to_point_cloud
from eval.bev_viz import bev_scatter
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj
from run_m4_demo import raw_lidar_for_sample

SDVAE_CACHE = Path("s2s_min/out/cached_latents_v5_100scenes")
D3_CACHE    = Path("s2s_min/out/cached_dinov3_v5_100scenes")
VAE_CKPT    = Path("s2s_min/out/lidar_vae_best.pt")
RANGE_M     = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="DINOv3 U-Net checkpoint (e.g. A-v2's best.pt)")
    ap.add_argument("--token", type=str, default=None,
                    help="sample token to test. Default: first alphabetical (== overfit-1 sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg_scale", type=float, default=1.0,
                    help="CFG scale. 1.0 = vanilla. Note: A-v2 was trained with cond_dropout=0.0 "
                         "so CFG won't behave normally — keep at 1.0.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("s2s_min/out/phase0_a2_sanity"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Resolve token (default = first cached sample, the one A-v2 memorized).
    if args.token is None:
        args.token = sorted(p.stem for p in D3_CACHE.glob("*.npz"))[0]
    assert (SDVAE_CACHE / f"{args.token}.npz").exists()
    assert (D3_CACHE / f"{args.token}.npz").exists()

    print("=" * 70)
    print("Phase 0 — A-v2 inference sanity check")
    print("=" * 70)
    print(f"  ckpt        : {args.ckpt}")
    print(f"  token       : {args.token}")
    print(f"  cfg_scale   : {args.cfg_scale}")
    print(f"  device      : {device}")

    unet, ckpt = load_unet(args.ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)
    print(f"  ckpt step / loss_ema : {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}")
    print(f"  U-Net params         : {sum(p.numel() for p in unet.parameters())/1e6:.2f} M")

    # ---- load the sample ----
    L = np.load(SDVAE_CACHE / f"{args.token}.npz")
    D = np.load(D3_CACHE / f"{args.token}.npz")
    raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
    mu     = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
    feat   = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)
    print(f"  mu shape    : {mu.shape}, mean={mu.mean().item():.3f}, std={mu.std().item():.3f}")
    print(f"  raymap shape: {raymap.shape}")
    print(f"  dinov3 feat : {feat.shape}")

    # ---- build KV context (DINOv3 → 4ch → upsample → cat raymap → pool) ----
    p4 = proj(feat)
    p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
    kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))
    print(f"  kv context  : {kv.shape}")

    # ---- DDIM inference ----
    t0 = time.time()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z_pred = diffusion.ddim_sample_cfg(
            unet=unet, shape=mu.shape, kv_context=kv,
            device=torch.device(device), cfg_scale=args.cfg_scale,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    t_ddim = time.time() - t0

    # ---- cos sim in latent space ----
    cos = F.cosine_similarity(z_pred.flatten(1), mu.flatten(1), dim=-1).item()
    diff = (z_pred - mu).pow(2).mean().item()

    # ---- decode both latents ----
    with torch.no_grad():
        rng_pred = vae.decode(z_pred)[0].cpu().numpy().clip(0, 1)   # [3, 32, 1024]
        rng_oracle = vae.decode(mu)[0].cpu().numpy().clip(0, 1)
    pc_pred   = range_image_to_point_cloud(rng_pred)
    pc_oracle = range_image_to_point_cloud(rng_oracle)
    pc_raw    = raw_lidar_for_sample(args.token)[:, :3]

    # ---- Chamfer distances ----
    cd_oracle = chamfer_distance(pc_pred,   pc_oracle)["cd"]
    cd_raw    = chamfer_distance(pc_pred,   pc_raw)["cd"]
    cd_vae    = chamfer_distance(pc_oracle, pc_raw)["cd"]
    cd_bev_oracle = chamfer_distance(pc_pred, pc_oracle, use_xy_only=True)["cd"]
    cd_bev_raw    = chamfer_distance(pc_pred, pc_raw,    use_xy_only=True)["cd"]

    # ---- print summary ----
    print()
    print("=" * 70)
    print("Results — A-v2 inference on its memorized sample")
    print("=" * 70)
    print(f"  Wall-clock (DDIM-25)        : {t_ddim:.2f}s")
    print()
    print(f"  cos(z_pred, μ)              : {cos:+.4f}     (1.000 = perfect memorization)")
    print(f"  ‖z_pred − μ‖²/n (MSE)       : {diff:.4f}")
    print()
    print(f"  CD-3D-oracle  vs decode(μ)  : {cd_oracle:.3f} m   (diffusion-only error)")
    print(f"  CD-BEV-oracle vs decode(μ)  : {cd_bev_oracle:.3f} m   (xy-only)")
    print(f"  CD-3D-raw     vs raw .pcd   : {cd_raw:.3f} m   (END-TO-END image→LiDAR)")
    print(f"  CD-BEV-raw    vs raw .pcd   : {cd_bev_raw:.3f} m   (xy-only)")
    print(f"  CD-VAE-only   decode(μ)→raw : {cd_vae:.3f} m   (VAE bottleneck floor)")
    print()
    print(f"  N_raw    : {len(pc_raw):>6d}")
    print(f"  N_oracle : {len(pc_oracle):>6d}")
    print(f"  N_pred   : {len(pc_pred):>6d}")
    print()
    print("Interpretation:")
    if cos > 0.9 and cd_oracle < 0.3:
        print("  ✓ STRONG MEMORIZATION: cos>0.9 and CD-3D-oracle <0.3 m.")
        print("  → Architecture is fully capable end-to-end. Any plateau on full data")
        print("    is a data/encoder/scale issue, NOT an architectural bug.")
    elif cos > 0.7:
        print("  ⚠ PARTIAL memorization: cos>0.7 but not yet 0.9+. More training would")
        print("    likely close the gap. The pathway works.")
    else:
        print("  ✗ WEAK memorization: cos≤0.7 despite training loss looking low.")
        print("  → Suggests a sampling-time vs training-time mismatch. Investigate")
        print("    DDIM schedule, CFG handling, or noise scale.")

    # ---- BEV plot: 3 panels (raw | oracle | pred) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    bev_scatter(axes[0], pc_raw,    color="tab:blue", range_m=RANGE_M)
    axes[0].set_title(f"raw nuScenes LIDAR_TOP ({len(pc_raw)} pts)")
    bev_scatter(axes[1], pc_oracle, color="tab:green", range_m=RANGE_M)
    axes[1].set_title(f"VAE oracle: decode(μ) ({len(pc_oracle)} pts)")
    bev_scatter(axes[2], pc_pred,   color="tab:red", range_m=RANGE_M)
    axes[2].set_title(f"A-v2 prediction: decode(DDIM(z|cond)) ({len(pc_pred)} pts)")
    fig.suptitle(
        f"Phase 0 — A-v2 memorized-sample inference   "
        f"token={args.token[:8]}…   step={ckpt.get('step')}   loss_ema={ckpt.get('loss_ema'):.4f}\n"
        f"cos(z_pred,μ)={cos:.3f}   CD-3D-oracle={cd_oracle:.2f}m   CD-3D-raw={cd_raw:.2f}m   CD-VAE-only={cd_vae:.2f}m",
        fontsize=11,
    )
    fig.tight_layout()
    bev_path = args.out_dir / "bev_3col.png"
    fig.savefig(bev_path, dpi=130)
    plt.close(fig)

    # ---- range-image plot: 3 rows (raw range channel | oracle | pred) ----
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5))
    titles = ["raw GT (range channel from .pcd, projected)", "VAE oracle decode(μ)", "A-v2 prediction"]
    # For raw, we need to project the raw point cloud back to a range image for visualization.
    # Easiest: synthesize a range image from pc_raw via the inverse mapping. Use the existing
    # range_image module — but it expects [N,5] with ring_index. We don't have ring_index here.
    # Skip raw range image — just show oracle and pred.
    from data.range_image import point_cloud_to_range_image
    raw5 = raw_lidar_for_sample(args.token)  # [N, 4] but raw_lidar_for_sample drops ring
    # Re-read with ring index for proper range image.
    from data.range_image import load_nuscenes_lidar_bin
    rec_pc5 = load_nuscenes_lidar_bin(
        str(Path("nuscenes") / Path(
            __import__('json').loads((Path("nuscenes/v1.0-trainval/sample_data.json")).read_text())
            [0]['filename'] if False else 'placeholder'
        ))
    ) if False else None
    # Simpler: just show oracle vs pred range images. Raw range image needs ring info we lose.
    for i, (rng, ttl) in enumerate([(rng_oracle, "VAE oracle decode(μ)"),
                                     (rng_pred,   "A-v2 prediction")]):
        ax = axes[i + 1]
        ax.imshow(rng[0], cmap="turbo", aspect="auto", vmin=0, vmax=0.5)
        ax.set_title(ttl)
        ax.set_xlabel("azimuth col (0=back, 512=forward, 1023=back)")
        ax.set_ylabel("elevation row")
    axes[0].axis("off")
    axes[0].text(0.5, 0.5,
                 "(raw GT range image omitted — requires beam ring index)",
                 ha="center", va="center", fontsize=10, transform=axes[0].transAxes)
    fig.suptitle(
        f"Phase 0 — A-v2 range images   token={args.token[:8]}…   "
        f"cos={cos:.3f}  CD-3D-oracle={cd_oracle:.2f}m",
        fontsize=11,
    )
    fig.tight_layout()
    rng_path = args.out_dir / "range_images.png"
    fig.savefig(rng_path, dpi=130)
    plt.close(fig)

    # ---- save stats ----
    stats_path = args.out_dir / "stats.txt"
    stats_path.write_text(f"""Phase 0 — A-v2 inference sanity check
{'=' * 70}
ckpt   : {args.ckpt}
token  : {args.token}
step   : {ckpt.get('step')}
loss_ema (training) : {ckpt.get('loss_ema'):.4f}

cos(z_pred, μ)        : {cos:+.4f}
‖z_pred − μ‖² / n      : {diff:.4f}

CD-3D-oracle (m)      : {cd_oracle:.4f}      (decode(z_pred) vs decode(μ))
CD-BEV-oracle (m)     : {cd_bev_oracle:.4f}
CD-3D-raw (m)         : {cd_raw:.4f}      (decode(z_pred) vs raw .pcd.bin)
CD-BEV-raw (m)        : {cd_bev_raw:.4f}
CD-VAE-only (m)       : {cd_vae:.4f}      (decode(μ) vs raw — VAE floor)

N_raw / N_oracle / N_pred : {len(pc_raw)} / {len(pc_oracle)} / {len(pc_pred)}
DDIM-25 wall-clock (s) : {t_ddim:.2f}
""")
    print(f"\n  saved BEV plot   : {bev_path}")
    print(f"  saved range plot : {rng_path}")
    print(f"  saved stats      : {stats_path}")


if __name__ == "__main__":
    main()
