"""Phase 0 diagnostics — direct architectural sanity checks.

Tests, all on the A-v2 checkpoint and its single memorized sample:

  1. **t=0 identity test**: With z_noisy=μ and noise=0, v_target should be 0
     and v_pred should be ≈ 0. Then z_pred = μ via the v-prediction inversion.
     Cosine similarity should be ~1.0. A low cos here means the U-Net can't
     even pass through identity at t=0 → fundamental architecture bug.

  1b. Sweep timesteps t ∈ {0, 10, 100, 500, 900}: one-step prediction from
      z_t = add_noise(μ, ε, t) → v_pred. Compare v_pred to v_target and recover
      ẑ_0. Cos(ẑ_0, μ) tells us per-timestep prediction quality.

  2. **Latent stats**: per-channel mean / std of cached μ vs DDIM-25 inference
     z_pred. Big divergence → normalization bug.

  4. **Histograms** of |z_noisy|, |v_pred|, |z_pred| at varied t. Out-of-range
     values (e.g. v_pred std > 5) indicate the U-Net is broken.

(Diagnostic 3 — MLP regression baseline — is in a separate script if needed.)

Run:
  HF_HUB_OFFLINE=1 env/bin/python s2s_min/scripts/phase0_diagnostics.py \
      --ckpt s2s_min/out/runs/<A-v2-folder>/lidar_unet_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj

SDVAE_CACHE = Path("s2s_min/out/cached_latents_v5_100scenes")
D3_CACHE    = Path("s2s_min/out/cached_dinov3_v5_100scenes")
VAE_CKPT    = Path("s2s_min/out/lidar_vae_best.pt")


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/phase0_diagnostics"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    if args.token is None:
        # Match the overfit-1 sample picker: alphabetical intersection of caches.
        lat = {p.stem for p in SDVAE_CACHE.glob("*.npz")}
        d3  = {p.stem for p in D3_CACHE.glob("*.npz")}
        args.token = sorted(lat & d3)[0]

    print("=" * 70)
    print("Phase 0 Diagnostics — A-v2 architecture sanity tests")
    print("=" * 70)
    print(f"  ckpt   : {args.ckpt}")
    print(f"  token  : {args.token}")
    print(f"  device : {device}")

    unet, ckpt = load_unet(args.ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)

    # ---- load sample ----
    L = np.load(SDVAE_CACHE / f"{args.token}.npz")
    D = np.load(D3_CACHE / f"{args.token}.npz")
    raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
    mu     = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
    feat   = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)

    # ---- build KV context ----
    p4 = proj(feat)
    p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
    kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

    # ─────────────────────────────────────────────────────────────────
    # Diagnostic 1: t=0 identity, plus sweep over timesteps
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Diagnostic 1: per-timestep one-step prediction quality")
    print("─" * 70)
    print(f"  At each t, feed z_t = add_noise(μ, ε, t) and ask the U-Net for v_pred.")
    print(f"  Recover ẑ_0 = α·z_t - σ·v_pred. Compare ẑ_0 to μ via cos sim and MSE.")
    print()
    print(f"  {'t':>5}  {'α(t)':>7}  {'σ(t)':>7}  "
          f"{'||v_target||':>13}  {'||v_pred||':>11}  "
          f"{'v-MSE':>9}  {'cos(ẑ₀,μ)':>11}  {'||ẑ₀−μ||/||μ||':>15}")

    sched = diffusion.train_scheduler
    alphas_cumprod = sched.alphas_cumprod.to(device)
    timesteps_to_test = [0, 10, 50, 100, 250, 500, 750, 900, 999]

    sweep_results = []  # list of dicts for plot
    for t_int in timesteps_to_test:
        t = torch.tensor([t_int], device=device, dtype=torch.long)
        # Make noise deterministic per-timestep so results are reproducible.
        torch.manual_seed(42 + t_int)

        # Special handling for t=0: scheduler does z_t = sqrt(1)·z + sqrt(0)·ε = z.
        if t_int == 0:
            noise = torch.zeros_like(mu)  # no noise at t=0 — pure identity test
            z_noisy = mu.clone()
        else:
            noise = torch.randn_like(mu)
            z_noisy = diffusion.add_noise(mu, noise, t)

        v_target = diffusion.get_target(mu, noise, t)
        with torch.no_grad():
            v_pred = unet(z_noisy, t, kv)

        # Inversion: from z_t and v_pred, recover ẑ_0 = α·z_t - σ·v_pred  (v_pred path)
        alpha_t = alphas_cumprod[t_int].sqrt()       # scalar
        sigma_t = (1.0 - alphas_cumprod[t_int]).sqrt()
        z0_hat = alpha_t * z_noisy - sigma_t * v_pred

        v_mse = (v_pred - v_target).pow(2).mean().item()
        cos_z0 = cos_sim(z0_hat, mu)
        rel_err = ((z0_hat - mu).pow(2).mean().sqrt() / mu.pow(2).mean().sqrt()).item()

        print(f"  {t_int:>5d}  {alpha_t.item():>7.4f}  {sigma_t.item():>7.4f}  "
              f"{v_target.norm().item():>13.3f}  {v_pred.norm().item():>11.3f}  "
              f"{v_mse:>9.4f}  {cos_z0:>+11.4f}  {rel_err:>15.4f}")

        sweep_results.append(dict(t=t_int, v_mse=v_mse, cos_z0=cos_z0,
                                  rel_err=rel_err, v_pred=v_pred, z_noisy=z_noisy,
                                  z0_hat=z0_hat, v_target=v_target))

    # Headline for diagnostic 1.
    t0_row = sweep_results[0]
    print()
    print(f"  TEST 1 (t=0 identity, no noise):  "
          f"v_target=0 expected, ||v_pred|| = {t0_row['v_pred'].norm().item():.3f}, "
          f"cos(ẑ₀,μ) = {t0_row['cos_z0']:+.4f}")
    if t0_row['cos_z0'] > 0.99:
        print(f"  → PASS: U-Net correctly passes through identity at t=0.")
    elif t0_row['cos_z0'] > 0.9:
        print(f"  → WEAK PASS: identity is approximate. Likely fine — small artifacts.")
    else:
        print(f"  → FAIL: U-Net does NOT pass through identity at t=0.")
        print(f"  → This is a fundamental architecture bug.")

    # ─────────────────────────────────────────────────────────────────
    # Diagnostic 2: latent stats μ vs DDIM-25 inference z_pred
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Diagnostic 2: latent stats — cached μ vs DDIM-25 inference z_pred")
    print("─" * 70)
    torch.manual_seed(42)
    with torch.no_grad():
        z_pred = diffusion.ddim_sample_cfg(
            unet=unet, shape=mu.shape, kv_context=kv,
            device=torch.device(device), cfg_scale=1.0,
        )

    print(f"  μ        : shape={tuple(mu.shape)}, mean={mu.mean():+.4f}, std={mu.std():.4f}, "
          f"min={mu.min():+.3f}, max={mu.max():+.3f}")
    print(f"  z_pred   : shape={tuple(z_pred.shape)}, mean={z_pred.mean():+.4f}, std={z_pred.std():.4f}, "
          f"min={z_pred.min():+.3f}, max={z_pred.max():+.3f}")
    print()
    print(f"  Per-channel stats (8 LiDAR-VAE latent channels):")
    print(f"  {'ch':>3}  {'μ mean':>10}  {'μ std':>10}  {'z_pred mean':>14}  {'z_pred std':>13}  {'std ratio':>11}")
    for c in range(mu.shape[1]):
        mu_m  = mu[:, c].mean().item()
        mu_s  = mu[:, c].std().item()
        zp_m  = z_pred[:, c].mean().item()
        zp_s  = z_pred[:, c].std().item()
        ratio = zp_s / max(mu_s, 1e-6)
        print(f"  {c:>3d}  {mu_m:>+10.4f}  {mu_s:>10.4f}  {zp_m:>+14.4f}  {zp_s:>13.4f}  {ratio:>11.3f}")

    overall_std_ratio = z_pred.std().item() / mu.std().item()
    print()
    print(f"  Overall std ratio (z_pred / μ): {overall_std_ratio:.3f}")
    if abs(overall_std_ratio - 1.0) < 0.15:
        print(f"  → PASS: z_pred std within 15% of μ std.")
    elif 0.5 < overall_std_ratio < 2.0:
        print(f"  → WEAK: z_pred std off by {abs(overall_std_ratio - 1.0)*100:.0f}% — borderline.")
    else:
        print(f"  → FAIL: z_pred std off by {abs(overall_std_ratio - 1.0)*100:.0f}% — normalization bug likely.")

    # ─────────────────────────────────────────────────────────────────
    # Diagnostic 4: histograms of z_noisy, v_pred, z_pred (recovered) at varied t
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Diagnostic 4: histograms of intermediate tensors at varied t")
    print("─" * 70)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    show_ts = [0, 100, 500, 900]
    for col, t_int in enumerate(show_ts):
        row = next(r for r in sweep_results if r["t"] == t_int)
        for ax_row, (key, label, color) in enumerate([
            ("z_noisy", "z_noisy", "tab:blue"),
            ("v_pred",  "v_pred",  "tab:red"),
            ("z0_hat",  "ẑ₀ (recovered)", "tab:green"),
        ]):
            ax = axes[ax_row, col]
            v = row[key].cpu().numpy().flatten()
            ax.hist(v, bins=80, color=color, alpha=0.7)
            ax.set_title(f"t={t_int}  {label}\nmean={v.mean():+.2f} std={v.std():.2f}",
                         fontsize=9)
            ax.set_xlim(-5, 5)
            ax.grid(True, alpha=0.3)
            # Overlay μ for the ẑ₀ row to show how close the recovery is.
            if key == "z0_hat":
                ax.hist(mu.cpu().numpy().flatten(), bins=80, color="black",
                        alpha=0.25, histtype="step", linewidth=1.5,
                        label="μ (target)")
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"Phase 0 Diagnostics — A-v2 ckpt step {ckpt.get('step')}, loss_ema {ckpt.get('loss_ema'):.4f}\n"
        f"token={args.token[:8]}…   z_pred(DDIM-25) std/μ-std ratio = {overall_std_ratio:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    hist_path = args.out_dir / "histograms.png"
    fig.savefig(hist_path, dpi=110)
    plt.close(fig)

    # Sweep plot: cos(ẑ₀, μ) and v-MSE vs t
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    ts = [r["t"] for r in sweep_results]
    cos_vals = [r["cos_z0"] for r in sweep_results]
    vmse_vals = [r["v_mse"] for r in sweep_results]
    ax1.plot(ts, cos_vals, "o-", color="tab:blue", label="cos(ẑ₀, μ)")
    ax1.set_xlabel("timestep t")
    ax1.set_ylabel("cos(ẑ₀, μ)", color="tab:blue")
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax2 = ax1.twinx()
    ax2.plot(ts, vmse_vals, "s-", color="tab:red", label="v-MSE", alpha=0.7)
    ax2.set_ylabel("v-MSE", color="tab:red")
    ax2.set_yscale("log")
    fig.suptitle(f"One-step prediction quality across timesteps — A-v2 ckpt\n"
                 f"loss_ema train = {ckpt.get('loss_ema'):.4f}", fontsize=10)
    fig.tight_layout()
    sweep_path = args.out_dir / "timestep_sweep.png"
    fig.savefig(sweep_path, dpi=110)
    plt.close(fig)

    # Save stats
    stats_path = args.out_dir / "stats.txt"
    with stats_path.open("w") as f:
        f.write(f"Phase 0 Diagnostics — A-v2 ckpt\n{'=' * 70}\n")
        f.write(f"ckpt   : {args.ckpt}\n")
        f.write(f"token  : {args.token}\n")
        f.write(f"step   : {ckpt.get('step')}\n")
        f.write(f"loss_ema train : {ckpt.get('loss_ema'):.4f}\n\n")
        f.write(f"--- Diagnostic 1: per-timestep one-step prediction ---\n")
        f.write(f"{'t':>5}  {'v_mse':>9}  {'cos(z0,μ)':>11}  {'rel_err':>10}\n")
        for r in sweep_results:
            f.write(f"{r['t']:>5}  {r['v_mse']:>9.4f}  {r['cos_z0']:>+11.4f}  {r['rel_err']:>10.4f}\n")
        f.write(f"\n--- Diagnostic 2: latent stats ratio ---\n")
        f.write(f"z_pred.std / μ.std = {overall_std_ratio:.4f}\n")

    print()
    print(f"  saved histograms : {hist_path}")
    print(f"  saved sweep plot : {sweep_path}")
    print(f"  saved stats      : {stats_path}")


if __name__ == "__main__":
    main()
