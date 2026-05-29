"""M4-style oblique 3D point-cloud grid: GT oracle vs DINOv3 vs SD-VAE, per sample.

Rows = samples, columns = [GT (VAE decode of μ) | DINOv3 pred | SD-VAE pred], rendered in the
paper Fig-13 oblique chase-cam style (oblique_viz.oblique_scatter), colored by height.

Reuses the generation helpers from compare_encoders.py (same KV builders, same matched-noise
seeds), so the clouds correspond to the CD-3D eval.

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/eval/oblique_grid_compare.py --k 6
Outputs: s2s_min/out/encoder_eval/oblique_grid.png
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.range_image import range_image_to_point_cloud
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj
from eval.decode_to_pointcloud import load_unet, load_lidar_vae
from eval.chamfer import chamfer_distance
from eval.oblique_viz import oblique_scatter
from eval.compare_encoders import (kv_sdvae, kv_dinov3, gen_pc, SDVAE_CKPT, VAE_CKPT,
                                    LAT_CACHE, D3_CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3_ckpt", type=Path, default=None)
    ap.add_argument("--sdvae_ckpt", type=Path, default=SDVAE_CKPT)
    ap.add_argument("--k", type=int, default=6, help="number of sample rows")
    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--out", type=Path, default=Path("s2s_min/out/encoder_eval/oblique_grid.png"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    d3_ckpt = args.dinov3_ckpt or (Path(Path("s2s_min/out/.last_dinov3_run").read_text().strip()) / "lidar_unet_best.pt")
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    unet_d3, ck_d3 = load_unet(d3_ckpt, device)
    proj = DINOv3Proj(ck_d3["dinov3_proj"]["mean"].squeeze().tolist(),
                      ck_d3["dinov3_proj"]["std"].squeeze().tolist()).to(device).eval()
    proj.load_state_dict(ck_d3["dinov3_proj"]); proj.requires_grad_(False)
    unet_sd, _ = load_unet(args.sdvae_ckpt, device)
    print(f"DINOv3 step={ck_d3.get('step')}  | SD-VAE {args.sdvae_ckpt.name}")

    # same sample selection as compare_encoders (seed, intersection of caches)
    toks = sorted({p.stem for p in LAT_CACHE.glob("*.npz")} & {p.stem for p in D3_CACHE.glob("*.npz")})
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(toks), size=min(args.k, len(toks)), replace=False)
    pick_tokens = [toks[i] for i in pick]
    print(f"rendering {len(pick_tokens)} samples")

    rows = []
    for j, tk in enumerate(pick_tokens):
        L = np.load(LAT_CACHE / f"{tk}.npz"); D = np.load(D3_CACHE / f"{tk}.npz")
        raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
        mu = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
        img = torch.from_numpy(L["image_latent"]).unsqueeze(0).to(device)
        feat = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            rng_gt = vae.decode(mu)[0].cpu().numpy().clip(0, 1)
        pc_gt = range_image_to_point_cloud(rng_gt)
        seed = args.seed * 100000 + j
        pc_d3, _ = gen_pc(unet_d3, vae, diffusion, kv_dinov3(proj, feat, raymap), seed, args.cfg_scale, device)
        pc_sd, _ = gen_pc(unet_sd, vae, diffusion, kv_sdvae(img, raymap), seed, args.cfg_scale, device)
        cd_d3 = chamfer_distance(pc_d3, pc_gt)["cd"]
        cd_sd = chamfer_distance(pc_sd, pc_gt)["cd"]
        rows.append((pc_gt, pc_d3, pc_sd, cd_d3, cd_sd))
        print(f"  {j+1}/{len(pick_tokens)}  cd_d3={cd_d3:.2f} cd_sd={cd_sd:.2f}")

    K = len(rows)
    fig, axes = plt.subplots(K, 3, figsize=(13.5, 3.6 * K), facecolor="black")
    if K == 1: axes = axes[None, :]
    col_titles = ["GT  (VAE decode of μ)", "DINOv3", "SD-VAE"]
    for r, (pc_gt, pc_d3, pc_sd, cd_d3, cd_sd) in enumerate(rows):
        cells = [(pc_gt, col_titles[0]),
                 (pc_d3, f"{col_titles[1]}   CD={cd_d3:.2f}m"),
                 (pc_sd, f"{col_titles[2]}   CD={cd_sd:.2f}m")]
        for c, (pc, ttl) in enumerate(cells):
            oblique_scatter(axes[r, c], pc, color_by="z", cmap=args.cmap, point_size=0.6, alpha=0.75)
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=10, color="white")
            else:
                # keep CD annotation on every row for the pred columns
                if c > 0:
                    axes[r, c].set_title(ttl.split("  ")[-1], fontsize=9, color="#bbbbbb")
    fig.suptitle(f"M4 oblique grid — GT vs DINOv3 vs SD-VAE  (cfg={args.cfg_scale}, height-colored)",
                 fontsize=12, color="white")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120, bbox_inches="tight", facecolor="black")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
