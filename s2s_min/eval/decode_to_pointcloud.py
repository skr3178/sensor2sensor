"""M4 inference orchestrator: cached latents → DDIM 25-step → LiDAR VAE decode → point cloud.

Two entry points:
  - `infer_one_sample()` — pure-function: takes the building blocks, returns z_pred,
    decoded range image, and the unprojected point cloud. Used by `run_m4_demo.py`.
  - `__main__`  — CLI: pick one cached sample by index, run inference, dump stats.

Composition reuses everything from M2/M3:
  - `CachedLatentsDataset`             (data/cached_latents.py)
  - `LiDARVAE.decode`                  (models/lidar_vae.py — frozen ckpt)
  - `LiDARUNet.forward`                (models/unet.py — frozen M3.2 v2 ckpt)
  - `DiffusionWrapper.ddim_sample`     (models/diffusion.py)
  - `range_image_to_point_cloud`       (data/range_image.py)
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from data.cached_latents import CachedLatentsDataset
from data.range_image import range_image_to_point_cloud
from models.diffusion import DiffusionWrapper
from models.lidar_vae import LiDARVAE
from models.unet import LiDARUNet

DEFAULT_UNET_CKPT  = Path("s2s_min/out/lidar_unet_m32_best.pt")
DEFAULT_VAE_CKPT   = Path("s2s_min/out/lidar_vae.pt")
DEFAULT_CACHE_DIR  = Path("s2s_min/out/cached_latents")
KV_POOL_H = 8
KV_POOL_W = 64


def build_kv_context(image_latent: torch.Tensor, raymap: torch.Tensor) -> torch.Tensor:
    """Same pre-pooled KV context used in training (see train_diffusion.py)."""
    return F.adaptive_avg_pool2d(torch.cat([image_latent, raymap], dim=1), (KV_POOL_H, KV_POOL_W))


def load_unet(ckpt_path: Path, device: str) -> tuple[LiDARUNet, dict]:
    """Load a U-Net checkpoint, honoring arch fields in the saved `config` dict.

    Back-compat: legacy checkpoints (pre-N-stage refactor) either omit
    `stem_channels`/`level_channels` or carry the hardcoded 3-stage defaults.
    Both cases work — defaults reproduce the legacy 3-stage build, and
    `LiDARUNet._load_from_state_dict` translates the old `enc_l*` / `dec_l*`
    state-dict keys to `encoders.N` / `decoders.N` automatically.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    arch_keys = set(inspect.signature(LiDARUNet.__init__).parameters)
    cfg = ckpt.get("config", {})
    arch_kwargs = {k: v for k, v in cfg.items() if k in arch_keys}
    if "level_channels" in arch_kwargs:
        arch_kwargs["level_channels"] = tuple(arch_kwargs["level_channels"])
    unet = LiDARUNet(**arch_kwargs).to(device).eval()
    unet.load_state_dict(ckpt["state_dict"])
    unet.requires_grad_(False)
    return unet, ckpt


def load_lidar_vae(ckpt_path: Path, device: str) -> LiDARVAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    arch_kwargs_keys = set(inspect.signature(LiDARVAE.__init__).parameters)
    arch_kwargs = {k: v for k, v in ckpt["config"].items() if k in arch_kwargs_keys}
    vae = LiDARVAE(**arch_kwargs).to(device).eval()
    vae.load_state_dict(ckpt["state_dict"])
    vae.requires_grad_(False)
    return vae


def infer_one_sample(
    unet: LiDARUNet,
    vae: LiDARVAE,
    diffusion: DiffusionWrapper,
    image_latent: torch.Tensor,         # [1, 4, 32, 56]
    raymap: torch.Tensor,                # [1, 6, 32, 56]
    seed: int | None = 42,
    cfg_scale: float = 1.0,
) -> dict:
    """Run DDIM inference + VAE decode + unprojection on one sample.

    Args:
        cfg_scale: classifier-free guidance scale. 1.0 = no guidance (the original
                   behavior — single conditional forward per DDIM step). Values
                   > 1.0 run a batched 2x U-Net forward per step and mix
                   unconditional + conditional predictions, sharpening the
                   image-conditioned output (see `DiffusionWrapper.ddim_sample_cfg`).

    Returns a dict with:
        z_pred        : [1, 8, 8, 256] sampled latent
        range_img     : [3, 32, 1024]  decoded range image in [0, 1]
        point_cloud   : [N, 4]         unprojected (x, y, z, intensity) on valid pixels
    """
    device = image_latent.device
    kv_context = build_kv_context(image_latent, raymap)

    if seed is not None:
        torch.manual_seed(seed)
    z_pred = diffusion.ddim_sample_cfg(
        unet=unet,
        shape=(1, 8, 8, 256),
        kv_context=kv_context,
        device=torch.device(device),
        cfg_scale=cfg_scale,
    )

    with torch.no_grad():
        range_img_t = vae.decode(z_pred)                       # [1, 3, 32, 1024]
    range_img = range_img_t[0].cpu().numpy().clip(0.0, 1.0)    # [3, 32, 1024]
    pc = range_image_to_point_cloud(range_img)                  # [N, 4]

    return {
        "z_pred": z_pred,
        "range_img": range_img,
        "point_cloud": pc,
    }


def decode_ground_truth(vae: LiDARVAE, mu: torch.Tensor) -> dict:
    """VAE-decode the cached μ (no diffusion). The 'oracle' the diffusion model
    would match perfectly if it had converged."""
    with torch.no_grad():
        range_img_t = vae.decode(mu)
    range_img = range_img_t[0].cpu().numpy().clip(0.0, 1.0)
    pc = range_image_to_point_cloud(range_img)
    return {"range_img": range_img, "point_cloud": pc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet_ckpt",   type=Path, default=DEFAULT_UNET_CKPT)
    p.add_argument("--vae_ckpt",    type=Path, default=DEFAULT_VAE_CKPT)
    p.add_argument("--cache_dir",   type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--idx",         type=int,  default=100,
                   help="cached-sample index to infer on (default 100 — held out from M3.1).")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--cfg_scale",   type=float, default=1.0,
                   help="classifier-free guidance scale (1.0 = no guidance). "
                        "Common values: 1.5, 3.0, 5.0. Requires the U-Net to have been "
                        "trained with cond_dropout > 0 (M3 bs16 used 0.2).")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    print(f"device: {device}")

    unet, unet_ckpt = load_unet(args.unet_ckpt, device)
    print(f"U-Net loaded  : {args.unet_ckpt}  step={unet_ckpt.get('step', '?')}  "
          f"loss_ema={unet_ckpt.get('loss_ema', float('nan')):.5f}")
    vae = load_lidar_vae(args.vae_ckpt, device)
    print(f"LiDAR VAE     : {args.vae_ckpt}")
    diffusion = DiffusionWrapper()
    print(f"DDIM steps    : {diffusion.inference_steps}")

    ds = CachedLatentsDataset(args.cache_dir)
    item = ds[args.idx]
    sample_token = item["sample_token"]
    print(f"\nsample [{args.idx}] token: {sample_token}")

    image_latent = item["image_latent"].unsqueeze(0).to(device)
    raymap       = item["raymap"].unsqueeze(0).to(device)
    mu           = item["mu"].unsqueeze(0).to(device)

    print(f"  image_latent: {tuple(image_latent.shape)}")
    print(f"  raymap      : {tuple(raymap.shape)}")
    print(f"  mu (GT)     : {tuple(mu.shape)}")

    pred = infer_one_sample(unet, vae, diffusion, image_latent, raymap,
                            seed=args.seed, cfg_scale=args.cfg_scale)
    gt   = decode_ground_truth(vae, mu)

    print(f"\npredicted:")
    print(f"  z_pred       : {tuple(pred['z_pred'].shape)}  "
          f"norm={pred['z_pred'].norm().item():.2f}  finite={torch.isfinite(pred['z_pred']).all().item()}")
    print(f"  range_img    : {pred['range_img'].shape}  range=[{pred['range_img'].min():.3f}, {pred['range_img'].max():.3f}]")
    print(f"  point_cloud  : {pred['point_cloud'].shape}  ({len(pred['point_cloud'])} points)")
    print(f"\nground truth (VAE decode of cached μ — the oracle):")
    print(f"  range_img    : {gt['range_img'].shape}  range=[{gt['range_img'].min():.3f}, {gt['range_img'].max():.3f}]")
    print(f"  point_cloud  : {gt['point_cloud'].shape}  ({len(gt['point_cloud'])} points)")


if __name__ == "__main__":
    main()
