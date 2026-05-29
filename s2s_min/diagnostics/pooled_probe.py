"""Probes 1-3: does the DINOv2 depth signal survive POOLING, RAYMAP-baseline, and CHANNEL reduction?

Follow-up to encoder_probe.py addressing three honest caveats:

  Probe 1 (pooling): the U-Net sees conditioning pooled to (8,64), not the 32x56 grid the
    earlier probe used. Pool features (and the depth target) to (8,64) and re-probe. If depth
    survives, the win is real for the U-Net's actual input.

  Probe 2 (raymap baseline): the right comparison is (DINOv2+raymap) vs (SD-VAE+raymap), since
    raymap already carries a strong geometry prior. Quantify DINOv2's marginal contribution.

  Probe 3 (channel reduction): cross-attn can't take 384ch as-is. Project DINOv2 384 -> N for
    N in {4,8,16,32,64} via (a) a LEARNED linear projection (upper bound — matches a trainable
    projection in the pipeline) and (b) PCA (unsupervised floor). Also test the concrete combo
    dinov2->4 + sdvae(4) + raymap(6) = 14ch.

All at the pooled (8,64) resolution for 1 & 2; Probe 3 at full 32x56 to isolate the channel effect.
NO training of the diffusion model. Reuses depth target + DINOv2 extraction.

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/diagnostics/pooled_probe.py --limit 600
Outputs: s2s_min/out/pooled_probe/
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.sdvae_depth_probe import assemble, PixelProbe, metrics, H_LAT, W_LAT
from diagnostics.encoder_probe import extract_dinov2

POOL = (8, 64)


def pool_feat(t):                      # [N,C,32,56] -> [N,C,8,64]
    return F.adaptive_avg_pool2d(t, POOL)


def pool_depth(depths, valids):        # masked-average depth + mask onto (8,64)
    dt = torch.from_numpy(depths.astype(np.float32))
    vt = torch.from_numpy(valids.astype(np.float32))
    num = F.adaptive_avg_pool2d(dt * vt, POOL)
    den = F.adaptive_avg_pool2d(vt, POOL)
    valid = den > 1e-6
    depth = torch.where(valid, num / den.clamp(min=1e-6), torch.zeros_like(num))
    return depth.numpy(), valid.numpy()


def standardize(feat, tr):
    m = feat[tr].mean((0, 2, 3), keepdims=True)
    s = feat[tr].std((0, 2, 3), keepdims=True) + 1e-6
    return (feat - m) / s


def train(model, Xtr, ytr, mtr, Xva, yva, mva, device, epochs=70, lr=2e-3, bs=64, seed=0):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]; curve = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            pred = model(Xtr[idx]); m = mtr[idx]
            loss = (((pred - ytr[idx]) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva)
            curve.append(float((((pv - yva) ** 2) * mva).sum() / mva.sum().clamp(min=1)))
    model.eval()
    with torch.no_grad():
        pv = model(Xva)
    return metrics(pv, yva, mva), curve, pv.cpu()


class ProjProbe(nn.Module):            # learned 1x1 projection in_ch->proj, then per-pixel MLP
    def __init__(self, in_ch, proj):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj, 1, bias=False)
        self.head = PixelProbe(proj)
    def forward(self, x): return self.head(self.proj(x))


def pca_project(feat, tr, ncomp, device):
    """Unsupervised PCA on train cells -> project all features to ncomp channels."""
    X = torch.from_numpy(feat).to(device)               # [N,C,h,w]
    N, C, h, w = X.shape
    flat = X[torch.from_numpy(tr).to(device)].permute(0, 2, 3, 1).reshape(-1, C)
    mean = flat.mean(0, keepdim=True)
    fc = flat - mean
    cov = (fc.T @ fc) / fc.shape[0]
    evals, evecs = torch.linalg.eigh(cov)               # ascending
    comps = evecs[:, -ncomp:]                            # [C,ncomp]
    proj = ((X.permute(0, 2, 3, 1).reshape(-1, C) - mean) @ comps)
    return proj.reshape(N, h, w, ncomp).permute(0, 3, 1, 2).contiguous().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/pooled_probe"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sd, ray, depths, valids, rgb_paths, tokens = assemble(args.cache, args.limit)
    N = sd.shape[0]
    print("extracting DINOv2 ..."); t = time.time()
    dino = extract_dinov2(rgb_paths, device).numpy(); print(f"  done ({time.time()-t:.0f}s) {dino.shape}")

    rng = np.random.RandomState(0); perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac)); va, tr = perm[:n_val], perm[n_val:]
    trt, vat = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device)

    # ---- full-res log-depth (for Probe 3) ----
    ld_full = np.zeros_like(depths); ld_full[valids] = np.log(np.clip(depths[valids], 0.5, 100))
    yF = torch.from_numpy(ld_full.astype(np.float32)).to(device)
    mF = torch.from_numpy(valids.astype(np.float32)).to(device)

    # ---- pooled depth target (for Probes 1&2) ----
    dpool, vpool = pool_depth(depths, valids)
    ldp = np.zeros_like(dpool); ldp[vpool] = np.log(np.clip(dpool[vpool], 0.5, 100))
    yP = torch.from_numpy(ldp.astype(np.float32)).to(device)
    mP = torch.from_numpy(vpool.astype(np.float32)).to(device)
    print(f"pooled valid cells: {int(vpool.sum()):,}  ({vpool.sum()/N:.0f}/sample)")

    to = lambda a: torch.from_numpy(a.astype(np.float32))
    results = {}

    # ===================== Probes 1 & 2 : pooled conditions =====================
    sd_p = standardize(pool_feat(to(sd)).numpy(), tr)
    dn_p = standardize(pool_feat(to(dino)).numpy(), tr)
    ry_p = pool_feat(to(ray)).numpy()
    pooled = {
        "sdvae_p":       sd_p,
        "raymap_p":      ry_p,
        "sdvae+ray_p":   np.concatenate([sd_p, ry_p], 1),     # current actual baseline
        "dinov2_p":      dn_p,
        "dinov2+ray_p":  np.concatenate([dn_p, ry_p], 1),     # proposed
    }
    print("\n== Probes 1&2 (pooled to 8x64) ==")
    for name, feat in pooled.items():
        X = to(feat).to(device)
        met, _, _ = train(PixelProbe(feat.shape[1]), X[trt], yP[trt], mP[trt],
                          X[vat], yP[vat], mP[vat], device, args.epochs)
        results[name] = met; print(f"  {name:14s} ({feat.shape[1]:3d}ch) {met}")
        del X; torch.cuda.empty_cache()
    # mean floor (pooled)
    ml = (yP[trt] * mP[trt]).sum() / mP[trt].sum()
    results["mean_p"] = metrics(torch.full_like(yP[vat], ml.item()), yP[vat], mP[vat])

    # ===================== Probe 3 : channel reduction (full-res 32x56) =====================
    dino_std = standardize(dino, tr)
    Xfull = to(dino_std).to(device)            # [N,384,32,56]
    Ns = [4, 8, 16, 32, 64]
    learned, pca = {}, {}
    print("\n== Probe 3a: LEARNED projection 384->N (full 32x56) ==")
    for n_ch in Ns:
        met, _, _ = train(ProjProbe(384, n_ch), Xfull[trt], yF[trt], mF[trt],
                          Xfull[vat], yF[vat], mF[vat], device, args.epochs)
        learned[n_ch] = met["pearson"]; print(f"  N={n_ch:3d}  pearson={met['pearson']:.3f}  absrel={met['absrel']:.3f}")
    # full 384 reference
    met384, _, _ = train(PixelProbe(384), Xfull[trt], yF[trt], mF[trt], Xfull[vat], yF[vat], mF[vat], device, args.epochs)
    learned[384] = met384["pearson"]; print(f"  N=384  pearson={met384['pearson']:.3f} (full)")

    print("\n== Probe 3b: PCA projection 384->N (full 32x56) ==")
    for n_ch in Ns:
        proj = pca_project(dino_std, tr, n_ch, device)
        Xp = to(proj).to(device)
        met, _, _ = train(PixelProbe(n_ch), Xp[trt], yF[trt], mF[trt], Xp[vat], yF[vat], mF[vat], device, args.epochs)
        pca[n_ch] = met["pearson"]; print(f"  N={n_ch:3d}  pearson={met['pearson']:.3f}")
        del Xp; torch.cuda.empty_cache()

    # concrete combo: dinov2->4 (learned) + sdvae(4) + raymap(6) = 14ch, pooled
    print("\n== combo: [dinov2->4] + sdvae(4) + raymap(6) = 14ch (pooled) ==")
    class ComboProbe(nn.Module):
        def __init__(self):
            super().__init__(); self.proj = nn.Conv2d(384, 4, 1, bias=False); self.head = PixelProbe(14)
        def forward(self, d, rest): return self.head(torch.cat([self.proj(d), rest], 1))
    dn_p_t = to(dn_p).to(device); rest_t = to(np.concatenate([sd_p, ry_p], 1)).to(device)
    combo = ComboProbe().to(device); opt = torch.optim.Adam(combo.parameters(), 2e-3)
    for ep in range(args.epochs):
        p = torch.randperm(trt.numel(), device=device)
        for s in range(0, trt.numel(), 64):
            idx = trt[p[s:s+64]]; pred = combo(dn_p_t[idx], rest_t[idx]); m = mP[idx]
            loss = (((pred - yP[idx])**2)*m).sum()/m.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
    combo.eval()
    with torch.no_grad(): pvc = combo(dn_p_t[vat], rest_t[vat])
    results["combo14_p"] = metrics(pvc, yP[vat], mP[vat]); print(f"  combo14 {results['combo14_p']}")

    # ============================== plots ==============================
    # P1/P2 bars
    order = ["sdvae_p", "sdvae+ray_p", "raymap_p", "dinov2_p", "dinov2+ray_p", "combo14_p", "mean_p"]
    colors = ["#d62728", "#ff9896", "#ff7f0e", "#2ca02c", "#98df8a", "#1f77b4", "#7f7f7f"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, key, ttl in [(axes[0], "pearson", "Pearson r (pooled 8×64) ↑"), (axes[1], "absrel", "AbsRel (pooled) ↓")]:
        vals = [results[c][key] for c in order]
        b = ax.bar(order, vals, color=colors); ax.tick_params(axis="x", rotation=30); ax.set_title(ttl)
        if key == "pearson":
            ax.axhline(results["raymap_p"][key], ls="--", c="k", lw=1, label="raymap (pooled)")
            ax.axhline(0.953, ls=":", c="g", lw=1, label="dinov2 @32×56 (unpooled)"); ax.legend(fontsize=8)
        for bar, v in zip(b, vals): ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Probes 1&2 — depth survival through pooling to (8,64). {N} samples.", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "pooled_metrics.png", dpi=130); plt.close(fig)

    # P3 curve
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = Ns + [384]
    ax.plot(xs, [learned[n] for n in xs], "o-", color="#2ca02c", label="learned projection")
    ax.plot(Ns, [pca[n] for n in Ns], "s--", color="#1f77b4", label="PCA projection")
    ax.axhline(results["sdvae+ray_p"]["pearson"], ls="--", c="#d62728", lw=1, label="SD-VAE+ray (pooled)")
    ax.axhline(0.85, ls=":", c="k", lw=1, label="0.85 target")
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in xs])
    ax.set_xlabel("projected channels N"); ax.set_ylabel("Pearson r (log-depth, 32×56)")
    ax.set_title("Probe 3 — DINOv2 depth retention vs channel count"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "dim_reduction.png", dpi=130); plt.close(fig)

    # decision card
    p_pool_loss = 0.953 - results["dinov2+ray_p"]["pearson"]
    marg = results["dinov2+ray_p"]["pearson"] - results["sdvae+ray_p"]["pearson"]
    min_ch = next((n for n in Ns if learned[n] >= 0.85), None)
    fig = plt.figure(figsize=(11.5, 6.4)); ax = fig.add_subplot(111); ax.axis("off")
    L = [("Pooled / raymap / channel probes — verdict", 15, "black"), ("", 5, "black"),
         (f"P1 pooling: dinov2+ray  32×56=0.953 -> pooled(8,64)={results['dinov2+ray_p']['pearson']:.3f}"
          f"   (loss {p_pool_loss:+.3f})", 12, "black"),
         (f"P2 baseline: sdvae+ray(pooled)={results['sdvae+ray_p']['pearson']:.3f}  "
          f"dinov2+ray(pooled)={results['dinov2+ray_p']['pearson']:.3f}  margin={marg:+.3f}", 12, "black"),
         (f"   dinov2-alone(pooled)={results['dinov2_p']['pearson']:.3f}   "
          f"raymap-alone(pooled)={results['raymap_p']['pearson']:.3f}", 12, "black"),
         (f"P3 learned proj: N=4->{learned[4]:.3f}  8->{learned[8]:.3f}  16->{learned[16]:.3f}  "
          f"32->{learned[32]:.3f}  64->{learned[64]:.3f}  full->{learned[384]:.3f}", 11, "black"),
         (f"   PCA proj:        N=4->{pca[4]:.3f}  8->{pca[8]:.3f}  16->{pca[16]:.3f}  "
          f"32->{pca[32]:.3f}  64->{pca[64]:.3f}", 11, "black"),
         (f"   combo [dinov2->4 + sdvae4 + ray6 = 14ch] pooled = {results['combo14_p']['pearson']:.3f}", 11, "#1f77b4"),
         ("", 5, "black"),
         (f"min learned channels for r>=0.85: {min_ch if min_ch else '>64'}", 13,
          "#2ca02c" if min_ch and min_ch <= 32 else "#d62728"),
         (f"depth survives pooling: {'YES' if results['dinov2+ray_p']['pearson']>0.85 else 'DEGRADED'}", 13,
          "#2ca02c" if results['dinov2+ray_p']['pearson'] > 0.85 else "#d62728"),
         (f"dinov2 beats sdvae through the real (pooled) pipeline: {'YES' if marg>0.03 else 'marginal'}", 13,
          "#2ca02c" if marg > 0.03 else "#d62728")]
    y = 0.96
    for txt, sz, col in L:
        ax.text(0.02, y, txt, fontsize=sz, color=col, family="monospace", transform=ax.transAxes, va="top")
        y -= 0.058 + sz*0.0012
    fig.savefig(args.out_dir / "decision_summary.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    out = dict(n_samples=N, pooled=results, learned_proj=learned, pca_proj=pca,
               pooling_loss=p_pool_loss, dinov2_margin_over_sdvae=marg, min_channels_0p85=min_ch)
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2, default=float))
    md = ["# Pooled / raymap / channel-reduction probes\n", f"- {N} samples, pooled to (8,64)\n",
          "## Probes 1&2 (pooled 8×64)", "| condition | Pearson | AbsRel | δ<1.25 | R² |", "|---|---|---|---|---|"]
    for c in order:
        r = results[c]; md.append(f"| {c} | {r['pearson']:.3f} | {r['absrel']:.3f} | {r['delta1']:.3f} | {r['r2']:.3f} |")
    md += ["\n## Probe 3 (channels → Pearson @32×56)", "| N | learned | PCA |", "|---|---|---|"]
    for n in Ns: md.append(f"| {n} | {learned[n]:.3f} | {pca[n]:.3f} |")
    md.append(f"| 384 | {learned[384]:.3f} | — |")
    md += [f"\n- pooling loss (dinov2+ray): {p_pool_loss:+.3f}",
           f"- dinov2 margin over sdvae (pooled): {marg:+.3f}",
           f"- min learned channels for r≥0.85: {min_ch}"]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"\nwrote results + plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
