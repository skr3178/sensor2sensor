"""Phase 0 inference-only ablations on the A-v2 ckpt.

Tests two axes that don't require retraining:

  A. Step count: DDIM-25 (baseline), 50, 100, 250, 500, 999.
     Predicts: more steps → less compounding error → cos sim ↑.

  B. Stochasticity (DDIM eta): 0.0 (baseline = deterministic),
     0.1, 0.3, 0.5, 1.0 (= DDPM).
     Predicts: re-injected noise keeps the trajectory near training distribution.

Reports cos(z_pred, μ) and std(z_pred)/std(μ) for each config, plus a
1-line verdict.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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


@torch.no_grad()
def ddim_sample_with_eta(
    unet, diffusion: DiffusionWrapper, shape, kv_context, device, n_steps: int, eta: float,
    seed: int = 42,
):
    """DDIM with explicit n_steps and eta (stochasticity)."""
    diffusion.inference_scheduler.set_timesteps(n_steps, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(*shape, device=device, generator=g)
    for t in diffusion.inference_scheduler.timesteps:
        t_batch = t.expand(shape[0]).to(device)
        v_pred = unet(z, t_batch, kv_context)
        z = diffusion.inference_scheduler.step(
            v_pred, t, z, eta=eta, generator=g
        ).prev_sample
    return z


def cos_sim(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    if args.token is None:
        lat = {p.stem for p in SDVAE_CACHE.glob("*.npz")}
        d3  = {p.stem for p in D3_CACHE.glob("*.npz")}
        args.token = sorted(lat & d3)[0]

    print("=" * 70)
    print("Phase 0 Inference Ablation — DDIM step count × eta sweep")
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

    L = np.load(SDVAE_CACHE / f"{args.token}.npz")
    D = np.load(D3_CACHE / f"{args.token}.npz")
    raymap = torch.from_numpy(L["raymap"]).unsqueeze(0).to(device)
    mu     = torch.from_numpy(L["mu"]).unsqueeze(0).to(device)
    feat   = torch.from_numpy(D["feat"].astype(np.float32)).unsqueeze(0).to(device)

    p4 = proj(feat)
    p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
    kv = F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

    mu_std = mu.std().item()
    print(f"  μ std (target): {mu_std:.4f}\n")

    # ───── Axis A: step count, eta=0 (deterministic) ─────
    print("─" * 70)
    print("AXIS A: step count, eta=0 (deterministic DDIM)")
    print("─" * 70)
    print(f"  {'steps':>6}  {'wall(s)':>8}  {'cos(z,μ)':>10}  {'z_pred std':>12}  {'std ratio':>11}")
    for n in [25, 50, 100, 250, 500, 999]:
        t0 = time.time()
        z = ddim_sample_with_eta(unet, diffusion, mu.shape, kv, device, n, eta=0.0)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0
        c = cos_sim(z, mu)
        s = z.std().item()
        print(f"  {n:>6d}  {dt:>8.2f}  {c:>+10.4f}  {s:>12.4f}  {s/mu_std:>11.3f}")

    # ───── Axis B: stochasticity (eta) at DDIM-25 and DDIM-100 ─────
    print()
    print("─" * 70)
    print("AXIS B: eta sweep at fixed step counts")
    print("─" * 70)
    print(f"  {'n_steps':>7}  {'eta':>5}  {'wall(s)':>8}  {'cos(z,μ)':>10}  {'std ratio':>11}")
    for n in [25, 100]:
        for eta in [0.0, 0.1, 0.3, 0.5, 1.0]:
            t0 = time.time()
            z = ddim_sample_with_eta(unet, diffusion, mu.shape, kv, device, n, eta=eta)
            torch.cuda.synchronize() if device == "cuda" else None
            dt = time.time() - t0
            c = cos_sim(z, mu)
            s = z.std().item()
            print(f"  {n:>7d}  {eta:>5.2f}  {dt:>8.2f}  {c:>+10.4f}  {s/mu_std:>11.3f}")

    # ───── Axis C: noise dilation (start from N(0, σ²·I) with σ<1) ─────
    # Hypothesis: DDIM starts from N(0,I) but the model's trained distribution
    # at t=T is α_T·μ + σ_T·ε with α_T ≈ 0.027 and σ_T ≈ 0.9996. So the true
    # high-t distribution is dominated by the noise, and N(0,I) should be fine.
    # But maybe smaller initial noise helps the model stay closer to seen z values.
    print()
    print("─" * 70)
    print("AXIS C: initial noise scale (DDIM-25, eta=0)")
    print("─" * 70)
    print(f"  {'σ_init':>7}  {'cos(z,μ)':>10}  {'std ratio':>11}")
    for sigma_init in [0.5, 0.75, 1.0, 1.25, 1.5]:
        diffusion.inference_scheduler.set_timesteps(25, device=device)
        g = torch.Generator(device=device).manual_seed(42)
        z = torch.randn(*mu.shape, device=device, generator=g) * sigma_init
        for t in diffusion.inference_scheduler.timesteps:
            t_batch = t.expand(mu.shape[0]).to(device)
            v_pred = unet(z, t_batch, kv)
            z = diffusion.inference_scheduler.step(v_pred, t, z, eta=0.0).prev_sample
        c = cos_sim(z, mu)
        s = z.std().item()
        print(f"  {sigma_init:>7.2f}  {c:>+10.4f}  {s/mu_std:>11.3f}")

    print()
    print("=" * 70)
    print("Best config from above (eyeball): pick the highest cos(z,μ) row.")
    print("=" * 70)


if __name__ == "__main__":
    main()
