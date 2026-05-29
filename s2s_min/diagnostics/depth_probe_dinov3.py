"""Depth probe with DINOv3-small (timm) as the image encoder — head-to-head with DINOv2/SD-VAE.

DINOv3 is a strict dense-feature upgrade over DINOv2 at the same size. timm 1.0.26 ships the
architecture (`vit_small_patch16_dinov3.lvd1689m`); weights are gated on HF and must be cached
once (HF token + license acceptance + network). After that this runs offline.

Mirrors depth_probe_dinov2.py exactly (same target, per-pixel probe, conditions, plot layout) so
its output folder is 1:1 comparable to depth_probe/ (SD-VAE) and depth_probe_dinov2/ (DINOv2).

Run (once weights are cached):
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/diagnostics/depth_probe_dinov3.py --limit 800
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from diagnostics.sdvae_depth_probe import assemble, train_probe, metrics, H_LAT, W_LAT
from diagnostics.encoder_probe import standardize

DINOV3_MODEL = "vit_small_patch16_dinov3.lvd1689m"
ENC = "DINOv3"


@torch.no_grad()
def extract_dinov3(rgb_paths, device, batch=16):
    """DINOv3-small patch features -> [N, embed_dim, 32, 56]. patch16, strips prefix tokens."""
    import timm
    model = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0).to(device).eval()
    Hp, Wp = 224, 384                      # /16 -> 14x24 patch grid, aspect ~0.583
    gh, gw = Hp // 16, Wp // 16
    npfx = getattr(model, "num_prefix_tokens", 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(rgb_paths), batch):
        ims = [torch.from_numpy(np.asarray(
            Image.open(p).convert("RGB").resize((Wp, Hp), Image.BICUBIC), np.float32) / 255.0
        ).permute(2, 0, 1) for p in rgb_paths[s:s + batch]]
        x = (torch.stack(ims).to(device) - mean) / std
        tok = model.forward_features(x)                       # [B, npfx+gh*gw, D]
        patch = tok[:, npfx:, :].transpose(1, 2).reshape(-1, tok.shape[-1], gh, gw)
        f = torch.nn.functional.interpolate(patch, size=(H_LAT, W_LAT), mode="bilinear", align_corners=False)
        feats.append(f.cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(feats).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/depth_probe_dinov3"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _, rays, depths, valids, rgb_paths, tokens = assemble(args.cache, args.limit)
    N = len(rgb_paths); total_valid = int(valids.sum())
    print(f"valid cells: {total_valid:,}")
    print(f"extracting {ENC} ({DINOV3_MODEL}) ..."); t = time.time()
    feat = extract_dinov3(rgb_paths, device).numpy(); print(f"  done ({time.time()-t:.0f}s) {feat.shape}")

    rng = np.random.RandomState(0); perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac)); va_idx, tr_idx = perm[:n_val], perm[n_val:]
    img = standardize(feat, tr_idx); ray = rays
    log_depth = np.zeros_like(depths); log_depth[valids] = np.log(np.clip(depths[valids], 0.5, 100))
    to = lambda a: torch.from_numpy(a.astype(np.float32))
    y, m = to(log_depth), to(valids.astype(np.float32))
    tr, va = torch.from_numpy(tr_idx), torch.from_numpy(va_idx)
    yt, mt, yv, mv = y[tr].to(device), m[tr].to(device), y[va].to(device), m[va].to(device)
    shuf = rng.permutation(N)
    conds = {"img": to(img), "ray": to(ray), "img+ray": torch.cat([to(img), to(ray)], 1),
             "img_shuf": to(img[shuf])}
    results, val_curves, pvs = {}, {}, {}
    for name, X in conds.items():
        print(f"probe {name} ({X.shape[1]}ch)")
        _, met, curve, pv = train_probe(X[tr].to(device), yt, mt, X[va].to(device), yv, mv,
                                        in_ch=X.shape[1], device=device, epochs=args.epochs)
        results[name] = met; val_curves[name] = curve; pvs[name] = pv; print(f"   {met}")
        torch.cuda.empty_cache()
    ml = (yt * mt).sum() / mt.sum()
    results["mean"] = metrics(torch.full_like(yv, ml.item()), yv, mv)
    val_curves["mean"] = [float(((torch.full_like(yv, ml.item()) - yv)**2 * mv).sum()/mv.sum())]*args.epochs

    order = ["img", "img+ray", "ray", "img_shuf", "mean"]
    colors = {"img": "#9467bd", "img+ray": "#1f77b4", "ray": "#ff7f0e", "img_shuf": "#c5b0d5", "mean": "#7f7f7f"}
    labels = {"img": f"{ENC} img", "img+ray": f"{ENC}+ray", "ray": "raymap only",
              "img_shuf": f"{ENC} (shuffled)", "mean": "mean floor"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, ttl in [(axes[0], "absrel", "AbsRel ↓"), (axes[1], "rmse", "RMSE m ↓"),
                         (axes[2], "pearson", "Pearson r ↑"), (axes[3], "delta1", "δ<1.25 ↑")]:
        vals = [results[c][key] for c in order]
        b = ax.bar([labels[c] for c in order], vals, color=[colors[c] for c in order])
        ax.set_title(ttl); ax.tick_params(axis="x", rotation=30)
        for bb, v in zip(b, vals): ax.text(bb.get_x()+bb.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{ENC} depth probe — {N} samples, {total_valid:,} cells", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir/"metrics_bar.png", dpi=130); plt.close(fig)

    pick = list(range(min(4, n_val)))
    fig, axes = plt.subplots(len(pick), 4, figsize=(14, 3.0*len(pick)))
    if len(pick) == 1: axes = axes[None, :]
    vmax = float(np.percentile(depths[valids], 95))
    for r, k in enumerate(pick):
        gi = va_idx[k]
        rgb = np.asarray(Image.open(rgb_paths[gi]).convert("RGB").resize((W_LAT*8, H_LAT*8)))
        axes[r,0].imshow(rgb); axes[r,0].set_title("CAM_FRONT" if r==0 else "")
        gt = np.ma.masked_where(~valids[gi,0].astype(bool), depths[gi,0])
        axes[r,1].imshow(gt, cmap="turbo", vmin=0, vmax=vmax); axes[r,1].set_title("GT depth" if r==0 else "")
        di = np.exp(pvs["img"][k,0].numpy().clip(np.log(0.5), np.log(120)))
        dr = np.exp(pvs["ray"][k,0].numpy().clip(np.log(0.5), np.log(120)))
        axes[r,2].imshow(di, cmap="turbo", vmin=0, vmax=vmax); axes[r,2].set_title(f"probe: {ENC}" if r==0 else "")
        axes[r,3].imshow(dr, cmap="turbo", vmin=0, vmax=vmax); axes[r,3].set_title("probe: raymap" if r==0 else "")
        for c in range(4): axes[r,c].axis("off")
    fig.suptitle(f"Qualitative depth probing — {ENC}", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir/"qualitative.png", dpi=130); plt.close(fig)

    out = dict(encoder=DINOV3_MODEL, n_samples=N, results=results)
    (args.out_dir/"results.json").write_text(json.dumps(out, indent=2, default=float))
    md = [f"# {ENC} depth probe ({DINOV3_MODEL})\n", f"- {N} samples, {total_valid:,} cells\n",
          "| condition | AbsRel | RMSE | δ<1.25 | Pearson | R² |", "|---|---|---|---|---|---|"]
    for c in order:
        r = results[c]; md.append(f"| {labels[c]} | {r['absrel']:.3f} | {r['rmse']:.2f} | {r['delta1']:.3f} | {r['pearson']:.3f} | {r['r2']:.3f} |")
    md.append("\nCompare against depth_probe/ (SD-VAE) and depth_probe_dinov2/ (DINOv2).")
    (args.out_dir/"summary.md").write_text("\n".join(md))
    print(f"\nwrote results + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
