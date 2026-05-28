"""M3.0 smoke test — verify one optimizer step of train_diffusion works.

Builds the same training step as `train/train_diffusion.py` but inline (no CLI),
on ONE cached sample, and asserts:
  - loss is finite
  - 4 grad-accum micro-steps work (loss / 4 scaling correctly handled)
  - gradient norms are finite
  - one optimizer step lands without exception
  - peak VRAM stays under the M0/M3.0 budget (6 GB)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from data.cached_latents import CachedLatentsDataset
from models.diffusion import DiffusionWrapper
from models.unet import LiDARUNet, count_params

CACHE_DIR = Path("s2s_min/out/cached_latents")


def _build_kv_context(image_latent, raymap):
    return F.adaptive_avg_pool2d(torch.cat([image_latent, raymap], dim=1), (8, 64))


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1) Cached sample.
    ds = CachedLatentsDataset(CACHE_DIR)
    print(f"{len(ds)} samples in cache. Using sample 0.")
    item = ds[0]
    image_latent = item["image_latent"].unsqueeze(0).to(device)
    raymap       = item["raymap"].unsqueeze(0).to(device)
    mu           = item["mu"].unsqueeze(0).to(device)
    print(f"  image_latent: {tuple(image_latent.shape)}")
    print(f"  raymap      : {tuple(raymap.shape)}")
    print(f"  mu          : {tuple(mu.shape)}")

    # 2) Model + diffusion + optimizer.
    unet = LiDARUNet().to(device).train()
    diffusion = DiffusionWrapper()
    optim = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    print(f"  U-Net params : {count_params(unet)/1e6:.2f} M")
    print(f"  diffusion    : {diffusion.prediction_type} (T={diffusion.num_train_timesteps})")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 3) Four micro-batches, each calls loss / grad_accum.backward().
    GRAD_ACCUM = 4
    optim.zero_grad(set_to_none=True)
    mse_values = []
    print(f"\nrunning {GRAD_ACCUM} grad-accum micro-steps...")
    for micro_i in range(GRAD_ACCUM):
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            kv_context = _build_kv_context(image_latent, raymap)
            t = diffusion.sample_timesteps(mu.shape[0], device=device)
            noise = torch.randn_like(mu)
            z_noisy = diffusion.add_noise(mu, noise, t)
            v_target = diffusion.get_target(mu, noise, t)
            v_pred = unet(z_noisy, t, kv_context)
            loss = F.mse_loss(v_pred, v_target)
        scaler.scale(loss / GRAD_ACCUM).backward()
        mse_values.append(loss.item())
        print(f"  micro {micro_i+1}/{GRAD_ACCUM}: t={t.tolist()}  mse={loss.item():.5f}  "
              f"(finite={torch.isfinite(loss).item()})")
        assert torch.isfinite(loss).item(), f"loss non-finite at micro-step {micro_i}"

    # 4) Optimizer step with clip + scaler.
    scaler.unscale_(optim)
    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    print(f"\noptimizer step: grad_norm (post-unscale, pre-clip) = {grad_norm.item():.4f}")
    assert torch.isfinite(grad_norm).item(), f"non-finite grad_norm: {grad_norm}"

    # 5) Memory + summary.
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        print(f"peak VRAM    : {peak_mb:.0f} MiB  (M3.0 budget < 6000 MiB)")
        assert peak_mb < 6000, f"VRAM {peak_mb:.0f} MiB exceeds 6 GB M3.0 budget"

    print(f"mean mse over {GRAD_ACCUM} micro-steps: {sum(mse_values)/len(mse_values):.5f}")
    print("\nOK — M3.0 smoke test passed (one optimizer step, finite loss, grad clip, no OOM).")


if __name__ == "__main__":
    main()
