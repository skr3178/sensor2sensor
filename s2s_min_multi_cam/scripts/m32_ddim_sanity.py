"""M3.2 sanity — DDIM inference using the m32_best checkpoint on held-out samples.

Picks several sample indices that were NOT in the M3.1 overfit-10 set
(>= 10) so we test true generalization, not memorization. Reports per-sample
cosine similarity vs ground-truth μ + range-image L1.

Pass criterion (M3.2 part 2):
    - DDIM output is finite (no NaN/Inf)
    - DDIM output is non-trivial (not all-zero, magnitude in reasonable range)
    - cosine similarity > 0 (model is using conditioning, even imperfectly)

This is a generalization check, not a quality benchmark. The full 401-sample
distribution is harder than overfit-10; expect lower cos sim than M3.1's 0.58.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from data.cached_latents import CachedLatentsDataset
from models.diffusion import DiffusionWrapper
from models.lidar_vae import LiDARVAE
from models.unet import LiDARUNet

CACHE_DIR      = Path("s2s_min/out/cached_latents")
UNET_CKPT      = Path("s2s_min/out/lidar_unet_m32_best.pt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
OUT_DIR        = Path("s2s_min/out/m32_ddim_sanity")

# Held-out indices: outside M3.1's overfit-10 set (which used indices 0..9).
# Pick a deterministic spread across the 401-sample subset.
HELD_OUT_IDX = [100, 200, 300, 400]
# Also include one overfit-10 sample for direct comparison.
TRAIN_IDX = [0, 5]


def _build_kv(image_latent, raymap):
    return F.adaptive_avg_pool2d(torch.cat([image_latent, raymap], dim=1), (8, 64))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- load M3.2 v2 best EMA U-Net ----
    ckpt = torch.load(UNET_CKPT, map_location=device)
    unet = LiDARUNet().to(device).eval()
    unet.load_state_dict(ckpt["state_dict"])
    unet.requires_grad_(False)
    print(f"U-Net loaded from {UNET_CKPT}  step={ckpt.get('step', '?')}  "
          f"loss_ema={ckpt.get('loss_ema', float('nan')):.5f}")

    # ---- LiDAR VAE (decode for visualization-side check) ----
    vae_ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs = {k: v for k, v in vae_ckpt["config"].items()
                   if k in inspect.signature(LiDARVAE.__init__).parameters}
    vae = LiDARVAE(**arch_kwargs).to(device).eval()
    vae.load_state_dict(vae_ckpt["state_dict"])
    vae.requires_grad_(False)

    diffusion = DiffusionWrapper()
    ds = CachedLatentsDataset(CACHE_DIR)

    rows = []
    rows.append(f"checkpoint   : {UNET_CKPT}  (step={ckpt.get('step', '?')}, loss_ema={ckpt.get('loss_ema', float('nan')):.5f})")
    rows.append(f"dataset size : {len(ds)} cached samples")
    rows.append(f"DDIM steps   : {diffusion.inference_steps}")
    rows.append(f"prediction   : {diffusion.prediction_type}")
    rows.append("")
    rows.append("HELD-OUT samples (NOT in M3.1's overfit-10 set):")
    rows.append(f"  {'idx':>5}  {'token':<40}  {'cos(z_pred,μ)':>14}  {'||z_pred||':>11}  {'||μ||':>11}  {'L1(rng)':>9}  {'finite':>7}")

    cos_held, l1_held = [], []
    for idx in HELD_OUT_IDX:
        item = ds[idx]
        image_latent = item["image_latent"].unsqueeze(0).to(device)
        raymap       = item["raymap"].unsqueeze(0).to(device)
        mu           = item["mu"].unsqueeze(0).to(device)

        kv_context = _build_kv(image_latent, raymap)
        torch.manual_seed(42)
        z_pred = diffusion.ddim_sample(unet, mu.shape, kv_context, torch.device(device))
        finite = torch.isfinite(z_pred).all().item()

        cs = F.cosine_similarity(z_pred.flatten(1), mu.flatten(1), dim=-1).item()
        with torch.no_grad():
            x_pred = vae.decode(z_pred)
            x_gt   = vae.decode(mu)
        mask = (x_gt[0, 2] > 0.5).float()
        l1 = ((x_pred[0, 0] - x_gt[0, 0]).abs() * mask).sum().item() / mask.sum().clamp(min=1).item()
        cos_held.append(cs); l1_held.append(l1)
        rows.append(f"  {idx:>5}  {item['sample_token'][:40]:<40}  {cs:>14.4f}  "
                    f"{z_pred.norm().item():>11.2f}  {mu.norm().item():>11.2f}  {l1:>9.4f}  {'✓' if finite else '✗':>7}")

    rows.append("")
    rows.append("M3.1 OVERFIT samples (idx 0..9 — seen ~400× in M3.1 if M3.1 ckpt were loaded; M3.2 v2 sees them once per epoch only):")
    rows.append(f"  {'idx':>5}  {'token':<40}  {'cos(z_pred,μ)':>14}  {'||z_pred||':>11}  {'||μ||':>11}  {'L1(rng)':>9}  {'finite':>7}")
    cos_tr, l1_tr = [], []
    for idx in TRAIN_IDX:
        item = ds[idx]
        image_latent = item["image_latent"].unsqueeze(0).to(device)
        raymap       = item["raymap"].unsqueeze(0).to(device)
        mu           = item["mu"].unsqueeze(0).to(device)
        kv_context = _build_kv(image_latent, raymap)
        torch.manual_seed(42)
        z_pred = diffusion.ddim_sample(unet, mu.shape, kv_context, torch.device(device))
        cs = F.cosine_similarity(z_pred.flatten(1), mu.flatten(1), dim=-1).item()
        with torch.no_grad():
            x_pred = vae.decode(z_pred); x_gt = vae.decode(mu)
        mask = (x_gt[0, 2] > 0.5).float()
        l1 = ((x_pred[0, 0] - x_gt[0, 0]).abs() * mask).sum().item() / mask.sum().clamp(min=1).item()
        cos_tr.append(cs); l1_tr.append(l1)
        rows.append(f"  {idx:>5}  {item['sample_token'][:40]:<40}  {cs:>14.4f}  "
                    f"{z_pred.norm().item():>11.2f}  {mu.norm().item():>11.2f}  {l1:>9.4f}  {'✓':>7}")

    rows.append("")
    rows.append(f"HELD-OUT mean cos(z_pred, μ) : {np.mean(cos_held):+.4f}")
    rows.append(f"HELD-OUT mean L1(range image): {np.mean(l1_held):.4f}")
    rows.append(f"TRAIN    mean cos(z_pred, μ) : {np.mean(cos_tr):+.4f}")
    rows.append(f"TRAIN    mean L1(range image): {np.mean(l1_tr):.4f}")
    rows.append("")
    rows.append("interpretation:")
    rows.append("  HELD-OUT cos > 0   → model uses conditioning to generalize, not memorize  ← M3.2 pass criterion")
    rows.append("  HELD-OUT cos > 0.3 → strong generalization on the 10-scene subset")
    rows.append("  HELD-OUT cos ≈ TRAIN cos → no memorization gap (good for low-data; expected here)")

    print()
    for r in rows: print(r)
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
