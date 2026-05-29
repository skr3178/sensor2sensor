"""Depth probe with the image encoder SWAPPED: DINOv2-small replaces the SD 1.5 VAE.

Mirror of `sdvae_depth_probe.py` but DINOv2 features stand in the "img" slot — so the
output folder is 1:1 comparable to `s2s_min/out/depth_probe/` (the SD-VAE run). Same
camera-plane depth target, same per-pixel 1x1 probe, same conditions and plot layout:
    img      : DINOv2-small features (384ch)   <- the swapped-in encoder
    ray      : raymap (6ch)
    img+ray  : both
    img_shuf : DINOv2 paired with another sample's depth (null control)
    mean     : constant floor

Writes to a SEPARATE dir (default s2s_min/out/depth_probe_dinov2/) — does NOT touch the
existing SD-VAE plots.

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/diagnostics/depth_probe_dinov2.py --limit 800
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

from diagnostics.sdvae_depth_probe import assemble, train_probe, metrics, H_LAT, W_LAT
from diagnostics.encoder_probe import extract_dinov2, standardize

ENC = "DINOv2"   # label used throughout the plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/depth_probe_dinov2"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    imgs_sd, rays, depths, valids, rgb_paths, tokens = assemble(args.cache, args.limit)
    N = imgs_sd.shape[0]; total_valid = int(valids.sum())
    print(f"valid cells: {total_valid:,}")

    print("extracting DINOv2-small features (replacing SD-VAE) ...")
    t = time.time(); dino = extract_dinov2(rgb_paths, device).numpy()
    print(f"  done ({time.time()-t:.0f}s) {dino.shape}")

    rng = np.random.RandomState(0); perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac)); va_idx, tr_idx = perm[:n_val], perm[n_val:]

    img = standardize(dino, tr_idx)          # DINOv2 is now "img"
    ray = rays
    log_depth = np.zeros_like(depths); log_depth[valids] = np.log(np.clip(depths[valids], 0.5, 100.0))

    to = lambda a: torch.from_numpy(a.astype(np.float32))
    y_cpu, m_cpu = to(log_depth), to(valids.astype(np.float32))
    tr = torch.from_numpy(tr_idx); va = torch.from_numpy(va_idx)
    yt, mt = y_cpu[tr].to(device), m_cpu[tr].to(device)
    yv, mv = y_cpu[va].to(device), m_cpu[va].to(device)

    shuf = rng.permutation(N)
    conds = {
        "img":      to(img),
        "ray":      to(ray),
        "img+ray":  torch.cat([to(img), to(ray)], 1),
        "img_shuf": to(img[shuf]),
    }
    results, val_curves, pvs = {}, {}, {}
    for name, X in conds.items():
        print(f"probe {name} ({X.shape[1]}ch)")
        _, met, curve, pv = train_probe(X[tr].to(device), yt, mt, X[va].to(device), yv, mv,
                                        in_ch=X.shape[1], device=device, epochs=args.epochs)
        results[name] = met; val_curves[name] = curve; pvs[name] = pv
        print(f"   {met}"); torch.cuda.empty_cache()

    mean_log = (yt * mt).sum() / mt.sum()
    pv_mean = torch.full_like(yv, mean_log.item())
    results["mean"] = metrics(pv_mean, yv, mv)
    val_curves["mean"] = [float(((pv_mean - yv) ** 2 * mv).sum() / mv.sum())] * args.epochs

    # ----- plots (same layout as depth_probe/) -----
    order = ["img", "img+ray", "ray", "img_shuf", "mean"]
    colors = {"img": "#2ca02c", "img+ray": "#1f77b4", "ray": "#ff7f0e",
              "img_shuf": "#9467bd", "mean": "#7f7f7f"}
    labels = {"img": f"{ENC} img", "img+ray": f"{ENC}+ray", "ray": "raymap only",
              "img_shuf": f"{ENC} (shuffled)", "mean": "mean floor"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, title in [(axes[0], "absrel", "AbsRel  (lower=better)"),
                           (axes[1], "rmse", "RMSE m  (lower=better)"),
                           (axes[2], "pearson", "Pearson r log-depth  (higher=better)"),
                           (axes[3], "delta1", "δ<1.25  (higher=better)")]:
        vals = [results[c][key] for c in order]
        bars = ax.bar([labels[c] for c in order], vals, color=[colors[c] for c in order])
        ax.set_title(title, fontsize=11); ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{ENC} depth probe (encoder SWAPPED in for SD-VAE) — {N} samples, "
                 f"{total_valid:,} valid cells", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "metrics_bar.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in order:
        ax.plot(val_curves[c], label=labels[c], color=colors[c])
    ax.set_xlabel("epoch"); ax.set_ylabel("val masked MSE (log-depth)")
    ax.set_title(f"{ENC} probe validation loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "training_curves.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, c in zip(axes, ["img", "ray"]):
        mm = mv.reshape(-1).bool().cpu()
        gt = torch.exp(yv.reshape(-1).cpu()[mm]).numpy()
        pr = torch.exp(pvs[c].reshape(-1)[mm].clamp(np.log(0.5), np.log(120))).numpy()
        ax.scatter(gt, pr, s=2, alpha=0.15, color=colors[c])
        lim = [0, np.percentile(gt, 99)]; ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("GT depth (m)"); ax.set_ylabel("predicted depth (m)")
        ax.set_title(f"{labels[c]}   r={results[c]['pearson']:.3f}  AbsRel={results[c]['absrel']:.3f}")
    fig.suptitle("Predicted vs ground-truth depth (held-out cells)")
    fig.tight_layout(); fig.savefig(args.out_dir / "scatter_pred_vs_gt.png", dpi=130); plt.close(fig)

    # qualitative — SAME 4-column layout as depth_probe/qualitative.png
    pick = list(range(min(4, n_val)))
    fig, axes = plt.subplots(len(pick), 4, figsize=(14, 3.0*len(pick)))
    if len(pick) == 1: axes = axes[None, :]
    vmax = float(np.percentile(depths[valids], 95))
    for r, k in enumerate(pick):
        gi = va_idx[k]
        rgb = np.asarray(Image.open(rgb_paths[gi]).convert("RGB").resize((W_LAT*8, H_LAT*8)))
        axes[r, 0].imshow(rgb); axes[r, 0].set_title("CAM_FRONT" if r == 0 else "")
        gt = np.ma.masked_where(~valids[gi, 0].astype(bool), depths[gi, 0])
        axes[r, 1].imshow(gt, cmap="turbo", vmin=0, vmax=vmax); axes[r, 1].set_title("GT depth (LiDAR, sparse)" if r == 0 else "")
        di = np.exp(pvs["img"][k, 0].numpy().clip(np.log(0.5), np.log(120)))
        dr = np.exp(pvs["ray"][k, 0].numpy().clip(np.log(0.5), np.log(120)))
        axes[r, 2].imshow(di, cmap="turbo", vmin=0, vmax=vmax); axes[r, 2].set_title(f"probe: {ENC} img" if r == 0 else "")
        axes[r, 3].imshow(dr, cmap="turbo", vmin=0, vmax=vmax); axes[r, 3].set_title("probe: raymap only" if r == 0 else "")
        for c in range(4): axes[r, c].axis("off")
    fig.suptitle(f"Qualitative depth probing — {ENC} encoder (dense pred; GT is sparse LiDAR)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "qualitative.png", dpi=130); plt.close(fig)

    # decision card
    r_img, r_ray = results["img"]["pearson"], results["ray"]["pearson"]
    r_shuf = results["img_shuf"]["pearson"]
    carries = (r_img - r_ray > 0.05) and (r_img - r_shuf > 0.10)
    fig = plt.figure(figsize=(11, 6)); ax = fig.add_subplot(111); ax.axis("off")
    verdict = (f"{ENC} FEATURES CARRY DEPTH" if carries else f"{ENC} FEATURES ARE DEPTH-IMPOVERISHED")
    vcol = "#2ca02c" if carries else "#d62728"
    action = ("=> Encoder swap is justified: DINOv2 supplies the depth signal SD-VAE lacked.\n"
              "   Proceed to cache-rebuild + U-Net retrain on DINOv2 conditioning (+ B1 pos-enc)."
              if carries else "=> Unexpected: re-check extraction before committing.")
    lines = [(f"{ENC} depth probe — verdict (encoder swapped in for SD-VAE)", 16, "black"),
             (verdict, 16, vcol), ("", 6, "black"),
             (f"Pearson r (log-depth):   img={r_img:.3f}   ray={r_ray:.3f}   img_shuf={r_shuf:.3f}", 12, "black"),
             (f"AbsRel:                  img={results['img']['absrel']:.3f}   ray={results['ray']['absrel']:.3f}"
              f"   floor={results['mean']['absrel']:.3f}", 12, "black"),
             (f"img lift over position prior (r):  {r_img-r_ray:+.3f}   (need > +0.05)", 12, "black"),
             (f"img lift over null/shuffle (r):    {r_img-r_shuf:+.3f}   (need > +0.10)", 12, "black"),
             ("", 6, "black"), (action, 13, vcol), ("", 6, "black"),
             ("Compare side-by-side with s2s_min/out/depth_probe/ (the SD-VAE run).", 10, "#555555")]
    y = 0.95
    for txt, sz, col in lines:
        ax.text(0.02, y, txt, fontsize=sz, color=col, family="monospace", transform=ax.transAxes, va="top")
        y -= 0.045 + sz*0.0016 + txt.count("\n")*0.05
    fig.savefig(args.out_dir / "decision_summary.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    out = dict(encoder="dinov2-small", n_samples=N, n_valid_cells=total_valid,
               results=results, verdict=verdict)
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2))
    md = [f"# {ENC} depth probe (encoder swapped in for SD-VAE)\n",
          f"- samples {N}, valid cells {total_valid:,}\n",
          "| condition | AbsRel↓ | RMSE↓ | δ<1.25↑ | Pearson↑ | R²↑ |", "|---|---|---|---|---|---|"]
    for c in order:
        r = results[c]
        md.append(f"| {labels[c]} | {r['absrel']:.3f} | {r['rmse']:.2f} | {r['delta1']:.3f} | {r['pearson']:.3f} | {r['r2']:.3f} |")
    md += [f"\n## Verdict: **{verdict}**\n", action,
           "\nCompare against `s2s_min/out/depth_probe/` (SD-VAE)."]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print("\n" + "=" * 70 + f"\n{verdict}\nwrote results + 5 plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
