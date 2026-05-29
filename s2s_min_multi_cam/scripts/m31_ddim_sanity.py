"""M3.1 sanity — DDIM inference on the 10 overfit samples.

Loads the best-EMA U-Net checkpoint, runs DDIM 25-step inference conditioned on
each cached sample's KV context, and compares the sampled latent to the
ground-truth μ. For an overfit-10 model, the sampled latent should be
*non-random* — i.e., close to the cached μ for the same sample.

Pass criterion (M3.1 part 2):
    - Sampled z is finite (no NaN/Inf)
    - Sampled z has reasonable magnitude (not all-zero, not exploded)
    - Cosine similarity between DDIM sample and ground-truth μ is well above
      what an untrained baseline produces

Also decodes both through the LiDAR VAE and reports the range-image L1 between
them, just to confirm the cycle is intact end-to-end.

Run:
    env/bin/python s2s_min/scripts/m31_ddim_sanity.py
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
UNET_CKPT      = Path("s2s_min/out/lidar_unet_best.pt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
OUT_DIR        = Path("s2s_min/out/m31_ddim_sanity")
N_OVERFIT      = 10


def _build_kv(image_latent, raymap):
    return F.adaptive_avg_pool2d(torch.cat([image_latent, raymap], dim=1), (8, 64))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- load U-Net (best EMA) ----
    ckpt = torch.load(UNET_CKPT, map_location=device)
    unet = LiDARUNet().to(device).eval()
    unet.load_state_dict(ckpt["state_dict"])
    unet.requires_grad_(False)
    print(f"U-Net loaded from {UNET_CKPT}  step={ckpt.get('step', '?')}  "
          f"loss_ema={ckpt.get('loss_ema', float('nan')):.5f}")

    # ---- load LiDAR VAE for decode-side eyeball ----
    vae_ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs = {k: v for k, v in vae_ckpt["config"].items()
                   if k in inspect.signature(LiDARVAE.__init__).parameters}
    vae = LiDARVAE(**arch_kwargs).to(device).eval()
    vae.load_state_dict(vae_ckpt["state_dict"])
    vae.requires_grad_(False)

    diffusion = DiffusionWrapper()

    # ---- iterate over the 10 overfit samples ----
    ds = CachedLatentsDataset(CACHE_DIR)
    n = min(N_OVERFIT, len(ds))
    print(f"\nrunning DDIM 25-step inference on {n} overfit samples...")

    rows = []
    rows.append(f"checkpoint   : {UNET_CKPT}  (step={ckpt.get('step', '?')}, loss_ema={ckpt.get('loss_ema', float('nan')):.5f})")
    rows.append(f"DDIM steps   : {diffusion.inference_steps}")
    rows.append(f"prediction   : {diffusion.prediction_type}")
    rows.append("")
    rows.append("per-sample stats (z_pred vs ground-truth μ; range_img L1 on valid pixels):")
    rows.append(f"  {'i':>3}  {'token':<40}  {'cos(z_pred,μ)':>14}  {'||z_pred||':>11}  {'||μ||':>11}  {'L1(rng_imgs)':>13}")

    cos_sims = []
    l1_ranges = []
    for i in range(n):
        item = ds[i]
        image_latent = item["image_latent"].unsqueeze(0).to(device)
        raymap       = item["raymap"].unsqueeze(0).to(device)
        mu           = item["mu"].unsqueeze(0).to(device)
        sample_token = item["sample_token"]

        kv_context = _build_kv(image_latent, raymap)

        # DDIM 25-step inference (no CFG for this sanity check; cond_dropout=0 implicit).
        torch.manual_seed(42)  # deterministic noise per sample for reproducibility
        z_pred = diffusion.ddim_sample(
            unet=unet,
            shape=mu.shape,
            kv_context=kv_context,
            device=torch.device(device),
        )

        # Sanity checks.
        assert torch.isfinite(z_pred).all(), f"sample {i}: z_pred has non-finite values"

        # Cosine similarity in flat space.
        cs = F.cosine_similarity(z_pred.flatten(1), mu.flatten(1), dim=-1).item()
        zn = z_pred.norm().item()
        mn = mu.norm().item()

        # Decode both and compare on valid pixels.
        with torch.no_grad():
            x_pred = vae.decode(z_pred)                 # [1, 3, 32, 1024]
            x_gt   = vae.decode(mu)
        mask = (x_gt[0, 2] > 0.5).float()
        denom = mask.sum().clamp(min=1).item()
        l1_range = ((x_pred[0, 0] - x_gt[0, 0]).abs() * mask).sum().item() / denom

        cos_sims.append(cs); l1_ranges.append(l1_range)
        rows.append(f"  {i:>3}  {sample_token[:40]:<40}  {cs:>14.4f}  {zn:>11.2f}  {mn:>11.2f}  {l1_range:>13.4f}")

    rows.append("")
    rows.append(f"MEAN cos(z_pred, μ): {np.mean(cos_sims):+.4f}   (1.0 = identical, 0.0 = random)")
    rows.append(f"MEAN L1(rng_imgs)  : {np.mean(l1_ranges):.4f}    (lower = better — range_clamp=100m means 0.01 ≈ 1m)")
    rows.append("")
    rows.append("interpretation:")
    rows.append("  cos > 0.5  → DDIM clearly conditioning on the input (M3.1 pass)")
    rows.append("  cos > 0.8  → strong overfit (expected target for 1000 steps × 10 samples)")
    rows.append("  cos ≈ 0    → conditioning ignored — pipeline bug")

    print()
    for r in rows:
        print(r)
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
