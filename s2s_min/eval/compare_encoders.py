"""Head-to-head decode eval: DINOv3 (Option B) vs SD-VAE baseline U-Net.

For a fixed set of samples, each model generates a LiDAR point cloud (DDIM-25 -> LiDAR-VAE
decode -> unproject) and we measure CD-3D against the SAME target = VAE-decode of the cached
`mu` (the oracle both models are trained to reproduce). Noise seed is matched per-sample across
the two models, so the ONLY difference is the image conditioning.

This is the measurement the loss curve cannot make: does the better encoder produce LiDAR that
matches the conditioned scene better?

Caveat: samples are in-distribution (both models trained on the 100-scene set) — this is a
relative encoder comparison, not held-out generalization.

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/eval/compare_encoders.py --n 40
Outputs: s2s_min/out/encoder_eval/  (cd_results.json, cd_compare.png, qualitative_range.png, summary.md)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.range_image import range_image_to_point_cloud
from models.diffusion import DiffusionWrapper
from models.lidar_vae import LiDARVAE
from models.dinov3_proj import DINOv3Proj
from eval.decode_to_pointcloud import load_unet, load_lidar_vae, KV_POOL_H, KV_POOL_W
from eval.chamfer import chamfer_distance

DINOV3_CKPT_DEFAULT = None  # resolved from .last_dinov3_run
SDVAE_CKPT = Path("s2s_min/out/runs/2026-05-28_161242__m3-unet-v5cache-50ep-bs16/lidar_unet_best.pt")
VAE_CKPT = Path("s2s_min/out/lidar_vae_best.pt")
LAT_CACHE = Path("s2s_min/out/cached_latents_v5_100scenes")
D3_CACHE = Path("s2s_min/out/cached_dinov3_v5_100scenes")


def kv_sdvae(image_latent, raymap):
    return F.adaptive_avg_pool2d(torch.cat([image_latent, raymap], 1), (KV_POOL_H, KV_POOL_W))


def kv_dinov3(proj, feat, raymap):
    p4 = proj(feat)                                                        # [B,4,14,24]
    p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(torch.cat([p4, raymap], 1), (KV_POOL_H, KV_POOL_W))


@torch.no_grad()
def gen_pc(unet, vae, diffusion, kv, seed, cfg_scale, device):
    torch.manual_seed(seed)
    z = diffusion.ddim_sample_cfg(unet=unet, shape=(1, 8, 8, 256), kv_context=kv,
                                  device=torch.device(device), cfg_scale=cfg_scale)
    rng = vae.decode(z)[0].cpu().numpy().clip(0, 1)                        # [3,32,1024]
    return range_image_to_point_cloud(rng), rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3_ckpt", type=Path, default=None)
    ap.add_argument("--sdvae_ckpt", type=Path, default=SDVAE_CKPT)
    ap.add_argument("--n", type=int, default=40, help="number of samples to eval")
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/encoder_eval"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    d3_ckpt = args.dinov3_ckpt
    if d3_ckpt is None:
        run = Path(Path("s2s_min/out/.last_dinov3_run").read_text().strip())
        d3_ckpt = run / "lidar_unet_best.pt"
    print(f"device={device}\nDINOv3 ckpt: {d3_ckpt}\nSD-VAE ckpt: {args.sdvae_ckpt}")

    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()

    unet_d3, ck_d3 = load_unet(d3_ckpt, device)
    proj = DINOv3Proj(ck_d3["dinov3_proj"]["mean"].squeeze().tolist(),
                      ck_d3["dinov3_proj"]["std"].squeeze().tolist()).to(device).eval()
    proj.load_state_dict(ck_d3["dinov3_proj"]); proj.requires_grad_(False)
    unet_sd, ck_sd = load_unet(args.sdvae_ckpt, device)
    print(f"  DINOv3 step={ck_d3.get('step')} loss_ema={ck_d3.get('loss_ema')}")
    print(f"  SD-VAE step={ck_sd.get('step')} loss_ema={ck_sd.get('loss_ema')}")

    # shared sample set: tokens present in BOTH caches
    lat = {p.stem for p in LAT_CACHE.glob("*.npz")}
    d3 = {p.stem for p in D3_CACHE.glob("*.npz")}
    toks = sorted(lat & d3)
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(toks), size=min(args.n, len(toks)), replace=False)
    pick_tokens = [toks[i] for i in pick]
    print(f"evaluating {len(pick_tokens)} samples (cfg_scale={args.cfg_scale})")

    rows = []
    qual = []  # (rng_gt, rng_d3, rng_sd) for first few
    for j, tk in enumerate(pick_tokens):
        L = np.load(LAT_CACHE / f"{tk}.npz")
        D = np.load(D3_CACHE / f"{tk}.npz")
        raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
        mu = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
        img = torch.from_numpy(L["image_latent"]).unsqueeze(0).to(device)
        feat = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            rng_gt = vae.decode(mu)[0].cpu().numpy().clip(0, 1)
        pc_gt = range_image_to_point_cloud(rng_gt)

        seed = args.seed * 100000 + j   # matched noise across both models, varies per sample
        pc_d3, r_d3 = gen_pc(unet_d3, vae, diffusion, kv_dinov3(proj, feat, raymap), seed, args.cfg_scale, device)
        pc_sd, r_sd = gen_pc(unet_sd, vae, diffusion, kv_sdvae(img, raymap), seed, args.cfg_scale, device)

        cd_d3 = chamfer_distance(pc_d3, pc_gt)["cd"]
        cd_sd = chamfer_distance(pc_sd, pc_gt)["cd"]
        cd_d3_bev = chamfer_distance(pc_d3, pc_gt, use_xy_only=True)["cd"]
        cd_sd_bev = chamfer_distance(pc_sd, pc_gt, use_xy_only=True)["cd"]
        rows.append(dict(token=tk, cd_dinov3=cd_d3, cd_sdvae=cd_sd,
                         cd_dinov3_bev=cd_d3_bev, cd_sdvae_bev=cd_sd_bev,
                         n_gt=len(pc_gt), n_d3=len(pc_d3), n_sd=len(pc_sd)))
        if j < 4:
            qual.append((rng_gt[0], r_d3[0], r_sd[0]))   # range channel
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(pick_tokens)}  cd_d3={cd_d3:.3f} cd_sd={cd_sd:.3f}")

    d3v = np.array([r["cd_dinov3"] for r in rows])
    sdv = np.array([r["cd_sdvae"] for r in rows])
    d3b = np.array([r["cd_dinov3_bev"] for r in rows])
    sdb = np.array([r["cd_sdvae_bev"] for r in rows])
    win = float((d3v < sdv).mean())

    summary = dict(
        n=len(rows), cfg_scale=args.cfg_scale,
        cd3d=dict(dinov3_mean=float(d3v.mean()), sdvae_mean=float(sdv.mean()),
                  dinov3_median=float(np.median(d3v)), sdvae_median=float(np.median(sdv)),
                  improvement_pct=float(100 * (sdv.mean() - d3v.mean()) / sdv.mean())),
        cd_bev=dict(dinov3_mean=float(d3b.mean()), sdvae_mean=float(sdb.mean())),
        dinov3_win_rate=win,
        dinov3_ckpt=str(d3_ckpt), sdvae_ckpt=str(args.sdvae_ckpt),
    )
    (args.out_dir / "cd_results.json").write_text(json.dumps(dict(summary=summary, per_sample=rows), indent=2))

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(sdv, d3v, s=18, alpha=0.6, color="#6a1b9a")
    lim = [0, max(sdv.max(), d3v.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="parity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SD-VAE  CD-3D (m)"); ax.set_ylabel("DINOv3  CD-3D (m)")
    ax.set_title(f"Paired CD-3D (below diagonal = DINOv3 better)\nwin rate {win*100:.0f}%")
    ax.legend()
    ax = axes[1]
    ax.boxplot([sdv, d3v], labels=["SD-VAE", "DINOv3"], showmeans=True)
    ax.set_ylabel("CD-3D (m)")
    ax.set_title(f"CD-3D: SD-VAE {sdv.mean():.3f}  vs  DINOv3 {d3v.mean():.3f}  "
                 f"({summary['cd3d']['improvement_pct']:+.1f}%)")
    fig.suptitle(f"DINOv3 vs SD-VAE — generated-LiDAR CD-3D, {len(rows)} samples, cfg={args.cfg_scale}", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "cd_compare.png", dpi=130); plt.close(fig)

    if qual:
        fig, axes = plt.subplots(len(qual), 3, figsize=(15, 2.0 * len(qual)))
        if len(qual) == 1: axes = axes[None, :]
        for r, (g, d, s) in enumerate(qual):
            for c, (im, ttl) in enumerate([(g, "GT (VAE decode of μ)"), (d, "DINOv3"), (s, "SD-VAE")]):
                axes[r, c].imshow(im, cmap="turbo", aspect="auto", vmin=0, vmax=0.5)
                axes[r, c].set_title(ttl if r == 0 else ""); axes[r, c].axis("off")
        fig.suptitle("Generated range images (range channel) — GT vs DINOv3 vs SD-VAE", fontsize=12)
        fig.tight_layout(); fig.savefig(args.out_dir / "qualitative_range.png", dpi=130); plt.close(fig)

    md = [f"# Encoder decode eval — DINOv3 vs SD-VAE\n",
          f"- {len(rows)} samples, cfg_scale={args.cfg_scale}, matched noise seeds, target = VAE-decode(μ)",
          f"- DINOv3 ckpt: `{d3_ckpt}` (step {ck_d3.get('step')})",
          f"- SD-VAE ckpt: `{args.sdvae_ckpt}` (step {ck_sd.get('step')})\n",
          "| metric | SD-VAE | DINOv3 | Δ |", "|---|---|---|---|",
          f"| CD-3D mean (m) | {sdv.mean():.3f} | {d3v.mean():.3f} | {summary['cd3d']['improvement_pct']:+.1f}% |",
          f"| CD-3D median (m) | {np.median(sdv):.3f} | {np.median(d3v):.3f} | |",
          f"| CD-BEV mean (m) | {sdb.mean():.3f} | {d3b.mean():.3f} | |",
          f"| DINOv3 win rate | | {win*100:.0f}% | |",
          "\n_In-distribution relative comparison; lower CD = generated LiDAR closer to the conditioned scene._"]
    (args.out_dir / "summary.md").write_text("\n".join(md))

    print("\n" + "=" * 60)
    print(f"CD-3D mean:  SD-VAE {sdv.mean():.3f}   DINOv3 {d3v.mean():.3f}   "
          f"({summary['cd3d']['improvement_pct']:+.1f}%)")
    print(f"DINOv3 win rate: {win*100:.0f}%   (CD-BEV: SD {sdb.mean():.3f} / D3 {d3b.mean():.3f})")
    print(f"wrote results + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
