"""Gate test: do depth-aware encoders clear the raymap baseline?

Standalone follow-up to `sdvae_depth_probe.py`. Same camera-plane depth target and
same per-pixel 1x1 probe, but compares conditioning encoders head-to-head:

    sdvae      : frozen SD 1.5 VAE latent (4ch)           <- current encoder (the loser)
    raymap     : ray origin+dir (6ch)                     <- geometry-only baseline to beat (r~0.86)
    dinov2     : DINOv2-small patch features (384ch)       <- ENCODER-SWAP signal (option 2)
    da_depth   : Depth-Anything-V2 depth output (1ch)      <- C5 HYBRID signal (option 1) [if available]
    *+ray      : encoder concatenated with raymap          <- the real conditioning shape

The decisive question is NOT "encoder-alone > raymap" (raymap already gets ~0.86 from
the ground-plane prior). It is: does (encoder + raymap) add depth RESIDUAL over raymap
alone? SD-VAE+ray was 0.845 < ray-alone 0.863 (it HURTS). A useful encoder must push
(enc+ray) clearly above ray-alone.

NO cache rebuild, NO U-Net training. Reuses the depth-target + probe machinery from
sdvae_depth_probe.py. ~30 min on GPU.

Run:
    env/bin/python s2s_min/diagnostics/encoder_probe.py --limit 500
Outputs: s2s_min/out/encoder_probe/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# reuse everything from the SD-VAE probe
from diagnostics.sdvae_depth_probe import (
    assemble, PixelProbe, train_probe, metrics, H_LAT, W_LAT,
)

DINO_ID = "facebook/dinov2-small"
DA_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ------------------------- encoder feature extraction --------------------
@torch.no_grad()
def extract_dinov2(rgb_paths, device, batch=16):
    """DINOv2-small patch features resampled to the 32x56 latent grid -> [N,384,32,56].
    DINOv2 is the Depth-Anything-V2 backbone, so this is the encoder-swap signal."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(DINO_ID).to(device).eval()
    # 224x392 = (16x28) patches at patch_size 14; aspect 0.571 == 256/448
    Hp, Wp = 224, 392
    gh, gw = Hp // 14, Wp // 14
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(rgb_paths), batch):
        imgs = []
        for p in rgb_paths[s:s + batch]:
            im = Image.open(p).convert("RGB").resize((Wp, Hp), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
        x = torch.stack(imgs).to(device)
        x = (x - mean) / std
        out = model(pixel_values=x).last_hidden_state          # [B, 1+gh*gw, 384]
        patch = out[:, 1:, :].transpose(1, 2).reshape(-1, out.shape[-1], gh, gw)  # [B,384,gh,gw]
        f = torch.nn.functional.interpolate(patch, size=(H_LAT, W_LAT),
                                            mode="bilinear", align_corners=False)
        feats.append(f.cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(feats).float()                            # [N,384,32,56]


@torch.no_grad()
def extract_da_depth(rgb_paths, device, batch=16):
    """Depth-Anything-V2 depth output resampled to 32x56 -> [N,1,32,56], or None if unavailable.
    This is exactly the channel C5 would concatenate."""
    try:
        from transformers import AutoModelForDepthEstimation
    except Exception as e:
        print(f"  [da] transformers missing: {e}"); return None
    try:
        model = AutoModelForDepthEstimation.from_pretrained(DA_ID).to(device).eval()
    except Exception as e:
        print(f"  [da] weights unavailable ({repr(e)[:120]}); skipping Depth-Anything."); return None
    Hp, Wp = 252, 448            # multiples of 14, aspect 0.5625 ~ 256/448
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(rgb_paths), batch):
        imgs = []
        for p in rgb_paths[s:s + batch]:
            im = Image.open(p).convert("RGB").resize((Wp, Hp), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
        x = ((torch.stack(imgs).to(device)) - mean) / std
        d = model(pixel_values=x).predicted_depth               # [B,Hp,Wp] (inverse-relative)
        d = torch.nn.functional.interpolate(d[:, None], size=(H_LAT, W_LAT),
                                            mode="bilinear", align_corners=False)
        feats.append(d.cpu())
    del model; torch.cuda.empty_cache()
    print(f"  [da] Depth-Anything-V2 depth extracted: {DA_ID}")
    return torch.cat(feats).float()                             # [N,1,32,56]


def standardize(feat_np, tr_idx):
    m = feat_np[tr_idx].mean((0, 2, 3), keepdims=True)
    s = feat_np[tr_idx].std((0, 2, 3), keepdims=True) + 1e-6
    return (feat_np - m) / s


# ------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/encoder_probe"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    imgs, rays, depths, valids, rgb_paths, tokens = assemble(args.cache, args.limit)
    N = imgs.shape[0]; total_valid = int(valids.sum())
    print(f"valid cells: {total_valid:,}")

    print("extracting DINOv2-small features ...")
    t = time.time(); dino = extract_dinov2(rgb_paths, device).numpy(); print(f"  done ({time.time()-t:.0f}s) {dino.shape}")
    print("extracting Depth-Anything-V2 depth ...")
    da = extract_da_depth(rgb_paths, device)
    da = None if da is None else da.numpy()

    # split
    rng = np.random.RandomState(0); perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac)); va_idx, tr_idx = perm[:n_val], perm[n_val:]

    # standardize features
    sd_n = standardize(imgs, tr_idx)
    dn_n = standardize(dino, tr_idx)
    da_n = None if da is None else standardize(da, tr_idx)
    ray = rays  # leave metric

    # log-depth target
    log_depth = np.zeros_like(depths); log_depth[valids] = np.log(np.clip(depths[valids], 0.5, 100.0))

    to = lambda a: torch.from_numpy(a.astype(np.float32))
    y_cpu, m_cpu = to(log_depth), to(valids.astype(np.float32))
    tr = torch.from_numpy(tr_idx); va = torch.from_numpy(va_idx)

    # assemble conditions (CPU; moved to GPU one at a time to fit 12GB)
    conds = {
        "sdvae":      to(sd_n),
        "raymap":     to(ray),
        "sdvae+ray":  torch.cat([to(sd_n), to(ray)], 1),
        "dinov2":     to(dn_n),
        "dinov2+ray": torch.cat([to(dn_n), to(ray)], 1),
    }
    if da_n is not None:
        conds["da_depth"] = to(da_n)
        conds["da_depth+ray"] = torch.cat([to(da_n), to(ray)], 1)

    results, val_curves, pvs = {}, {}, {}
    yv_g, mv_g = y_cpu[va].to(device), m_cpu[va].to(device)
    yt_g, mt_g = y_cpu[tr].to(device), m_cpu[tr].to(device)
    for name, X in conds.items():
        Xt = X[tr].to(device); Xv = X[va].to(device)
        print(f"probe {name} ({X.shape[1]}ch)")
        _, met, curve, pv = train_probe(Xt, yt_g, mt_g, Xv, yv_g, mv_g,
                                        in_ch=X.shape[1], device=device, epochs=args.epochs)
        results[name] = met; val_curves[name] = curve; pvs[name] = pv
        print(f"   {met}")
        del Xt, Xv; torch.cuda.empty_cache()

    # mean floor
    mean_log = (yt_g * mt_g).sum() / mt_g.sum()
    pv_mean = torch.full_like(yv_g, mean_log.item())
    results["mean"] = metrics(pv_mean, yv_g, mv_g)

    # ---------------------------- plots ----------------------------------
    order = [c for c in ["sdvae", "sdvae+ray", "raymap", "dinov2", "dinov2+ray",
                         "da_depth", "da_depth+ray", "mean"] if c in results]
    cmap = {"sdvae": "#d62728", "sdvae+ray": "#ff9896", "raymap": "#ff7f0e",
            "dinov2": "#2ca02c", "dinov2+ray": "#98df8a", "da_depth": "#1f77b4",
            "da_depth+ray": "#aec7e8", "mean": "#7f7f7f"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, key, title in [(axes[0], "pearson", "Pearson r log-depth ↑"),
                           (axes[1], "absrel", "AbsRel ↓"),
                           (axes[2], "delta1", "δ<1.25 ↑")]:
        vals = [results[c][key] for c in order]
        bars = ax.bar(order, vals, color=[cmap[c] for c in order])
        ax.set_title(title); ax.tick_params(axis="x", rotation=35)
        if key == "pearson":
            ax.axhline(results["raymap"][key], ls="--", c="k", lw=1, label="raymap baseline")
            ax.legend()
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Encoder gate — depth probe, {N} samples, {total_valid:,} valid cells", fontsize=13)
    fig.tight_layout(); fig.savefig(args.out_dir / "encoder_metrics.png", dpi=130); plt.close(fig)

    # qualitative: dense depth maps for 4 val samples
    pick = list(range(min(4, n_val)))
    show = [c for c in ["sdvae", "dinov2", "da_depth"] if c in pvs]
    ncol = 2 + len(show)
    fig, axes = plt.subplots(len(pick), ncol, figsize=(3.3*ncol, 3.0*len(pick)))
    if len(pick) == 1: axes = axes[None, :]
    vmax = float(np.percentile(depths[valids], 95))
    for r, k in enumerate(pick):
        gi = va_idx[k]
        rgb = np.asarray(Image.open(rgb_paths[gi]).convert("RGB").resize((W_LAT*6, H_LAT*6)))
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("CAM_FRONT" if r == 0 else "")
        gt = np.ma.masked_where(~valids[gi, 0].astype(bool), depths[gi, 0])
        axes[r, 1].imshow(gt, cmap="turbo", vmin=0, vmax=vmax); axes[r, 1].set_title("GT depth" if r == 0 else "")
        for c, name in enumerate(show):
            dmap = np.exp(pvs[name][k, 0].numpy().clip(np.log(0.5), np.log(120)))
            axes[r, 2+c].imshow(dmap, cmap="turbo", vmin=0, vmax=vmax)
            axes[r, 2+c].set_title(f"probe: {name}" if r == 0 else "")
        for c in range(ncol): axes[r, c].axis("off")
    fig.suptitle("Dense depth predicted by each encoder's probe (GT is sparse LiDAR)")
    fig.tight_layout(); fig.savefig(args.out_dir / "encoder_qualitative.png", dpi=130); plt.close(fig)

    # decision card
    r_ray = results["raymap"]["pearson"]
    def lift(c): return results[c]["pearson"] - r_ray if c in results else float("nan")
    best_swap = "dinov2+ray"; best_hybrid = "da_depth+ray" if "da_depth+ray" in results else None
    fig = plt.figure(figsize=(11.5, 6.2)); ax = fig.add_subplot(111); ax.axis("off")
    lines = [("Encoder gate — does a depth-aware encoder beat the raymap prior?", 15, "black"),
             ("", 6, "black"),
             (f"raymap-alone baseline (r):           {r_ray:.3f}", 12, "black"),
             (f"sdvae + ray  lift over raymap:        {lift('sdvae+ray'):+.3f}   (current encoder)", 12, cmap['sdvae']),
             (f"DINOv2 + ray lift over raymap:        {lift('dinov2+ray'):+.3f}   (encoder swap)", 12, cmap['dinov2'])]
    if best_hybrid:
        lines.append((f"DepthAnything + ray lift over raymap: {lift('da_depth+ray'):+.3f}   (C5 hybrid)", 12, cmap['da_depth']))
    sd_hurts = lift("sdvae+ray") < 0.0
    swap_ok = lift("dinov2+ray") > 0.03
    hyb_ok = best_hybrid is not None and lift("da_depth+ray") > 0.03
    if swap_ok or hyb_ok:
        verdict = "GREEN LIGHT — a depth-aware encoder adds real depth residual."
        action = ("=> Worth the cache-rebuild + retrain. Recommend: "
                  + ("C5 (DepthAnything depth channel) " if hyb_ok else "")
                  + ("/ DINOv2 swap" if swap_ok else "") + ", plus B1 pos-enc.")
        vcol = "#2ca02c"
    else:
        verdict = "NO LIFT — even depth-aware features don't beat the geometry prior here."
        action = "=> Re-examine the target/probe before spending retrain budget; encoder swap not justified yet."
        vcol = "#d62728"
    lines += [("", 6, "black"), (verdict, 14, vcol), (action, 12, vcol)]
    if sd_hurts:
        lines += [("", 6, "black"),
                  ("(confirms SD-VAE actively HURTS: sdvae+ray < raymap-alone)", 10, "#555555")]
    y = 0.96
    for txt, sz, col in lines:
        ax.text(0.02, y, txt, fontsize=sz, color=col, family="monospace", transform=ax.transAxes, va="top")
        y -= 0.052 + sz*0.0015
    fig.savefig(args.out_dir / "decision_summary.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # json + md
    out = dict(n_samples=N, n_valid_cells=total_valid, has_depth_anything=da_n is not None,
               results=results, raymap_baseline=r_ray,
               lifts={c: lift(c) for c in ["sdvae+ray", "dinov2+ray"] + (["da_depth+ray"] if best_hybrid else [])},
               verdict=verdict)
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2))
    md = ["# Encoder gate probe\n", f"- samples {N}, valid cells {total_valid:,}, "
          f"Depth-Anything available: {da_n is not None}\n",
          "| condition | Pearson↑ | AbsRel↓ | δ<1.25↑ | R²↑ | lift vs raymap |",
          "|---|---|---|---|---|---|"]
    for c in order:
        r = results[c]; lv = f"{lift(c):+.3f}" if c not in ("raymap", "mean") else "—"
        md.append(f"| {c} | {r['pearson']:.3f} | {r['absrel']:.3f} | {r['delta1']:.3f} | {r['r2']:.3f} | {lv} |")
    md += [f"\n## {verdict}\n", action]
    (args.out_dir / "summary.md").write_text("\n".join(md))

    print("\n" + "=" * 70 + f"\n{verdict}\n{action}\nwrote plots+results to {args.out_dir}/")


if __name__ == "__main__":
    main()
