"""Per-azimuth-column MSE diagnostic.

Disambiguates two hypotheses for the 0.32 mse_ema ceiling
(see s2s_min/docs/lidar-unet.md §10.5):

  H_FoV  — CAM_FRONT ~70° covers ~20% of LiDAR azimuth; the back ~80% has
           no image conditioning, so its MSE has a high irreducible floor.
           Predicts: front-facing columns << back-facing columns.

  H_pool — KV pool (PSNR 6.3 dB) destroys image content; even in the front
           FoV the model can't use the image effectively.
           Predicts: MSE is roughly uniform across azimuth.

We compute per-azimuth-column v-MSE on N held-out cached samples × T noise
timesteps, then average to a single [W=256] curve.

The azimuth convention (per data/range_image.py:70-72):
    azimuth = atan2(y, x)            ∈ [-π, π]
    col_range_img = ((azimuth + π) / (2π)) * 1024

So col 512 of the range image = forward (+x). After the LiDAR VAE's 4×
spatial compression, latent W=256, and **latent col 128 = forward**.
CAM_FRONT half-FoV ~35° → ±25 latent columns → front-band [103, 153].

Usage:
    env/bin/python -m s2s_min.scripts.azimuth_column_mse \
        --ckpt s2s_min/out/runs/2026-05-29_184618__m3-unet-60M-posenc-fix1-warmstart/lidar_unet_best.pt \
        --cache_dir s2s_min/out/cached_latents_v5_850scenes \
        --n_samples 32 \
        --out_dir s2s_min/out/azimuth_column_mse
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import time
from pathlib import Path


@contextlib.contextmanager
def contextlib_nullcontext():
    """Py3.8-compatible null context (contextlib.nullcontext was 3.7+ but
    safer to just have our own here for the conditional pos-enc toggle)."""
    yield

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Match the train_diffusion KV-pool convention.
KV_POOL_H, KV_POOL_W = 8, 64

# Latent W = 256. Front column = 128. CAM_FRONT half-FoV ≈ 35° → ±25 latent cols.
LATENT_W = 256
FRONT_CENTER = LATENT_W // 2   # 128
FRONT_HALF_WIDTH = int(35.0 / 180.0 * (LATENT_W // 2))   # 25
FRONT_LO, FRONT_HI = FRONT_CENTER - FRONT_HALF_WIDTH, FRONT_CENTER + FRONT_HALF_WIDTH


def build_kv_context(image_latent: torch.Tensor, raymap: torch.Tensor) -> torch.Tensor:
    """[B,4,32,56] + [B,6,32,56] → cat → pool to [B,10,8,64]."""
    kv_full = torch.cat([image_latent, raymap], dim=1)
    return F.adaptive_avg_pool2d(kv_full, (KV_POOL_H, KV_POOL_W))


def load_unet(ckpt_path: Path, device: torch.device):
    from s2s_min.models.unet import LiDARUNet
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch_keys = set(inspect.signature(LiDARUNet.__init__).parameters)
    cfg = ckpt.get("config", {})
    arch_kwargs = {k: v for k, v in cfg.items() if k in arch_keys}
    if "level_channels" in arch_kwargs:
        arch_kwargs["level_channels"] = tuple(arch_kwargs["level_channels"])
    unet = LiDARUNet(**arch_kwargs).to(device).eval()
    unet.load_state_dict(ckpt["state_dict"])
    unet.requires_grad_(False)
    return unet, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path,
                   default=Path("s2s_min/out/cached_latents_v5_850scenes"))
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--timesteps", type=int, nargs="+",
                   default=[100, 300, 500, 700, 900],
                   help="diffusion timesteps to average MSE over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=Path,
                   default=Path("s2s_min/out/azimuth_column_mse"))
    p.add_argument("--no_pos_enc", action="store_true",
                   help="disable Fix#1 cross-attention pos-enc during eval. "
                        "Use this when loading a pre-Fix#1 checkpoint (e.g. H1 source) "
                        "so the model behaves as it did during its own training.")
    p.add_argument("--per_elevation_row", action="store_true",
                   help="also compute MSE per (elevation_row, azimuth_col) heatmap "
                        "to localize the −90° spike to specific HDL-32E beams.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Per-azimuth-column MSE diagnostic")
    print("=" * 70)
    print(f"  ckpt        : {args.ckpt}")
    print(f"  cache_dir   : {args.cache_dir}")
    print(f"  n_samples   : {args.n_samples}")
    print(f"  timesteps   : {args.timesteps}")
    print(f"  device      : {device}")
    print(f"  front band  : latent cols [{FRONT_LO}, {FRONT_HI}] "
          f"(center={FRONT_CENTER}, half-width={FRONT_HALF_WIDTH})")

    # ---- load model and diffusion wrapper ----
    unet, ckpt = load_unet(args.ckpt, device)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  U-Net params: {n_params/1e6:.2f} M")
    print(f"  ckpt step / loss_ema : {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}")

    from s2s_min.models.diffusion import DiffusionWrapper
    diffusion = DiffusionWrapper()

    # ---- pick held-out sample indices ----
    cache_files = sorted([p for p in args.cache_dir.glob("*.npz")])
    assert len(cache_files) >= args.n_samples, \
        f"cache has only {len(cache_files)} samples, need {args.n_samples}"
    # Pick evenly across the tail of the cache (less likely to be heavily
    # over-fit during training shuffle).
    step = len(cache_files) // args.n_samples
    pick_indices = list(range(len(cache_files) - 1,
                              len(cache_files) - 1 - args.n_samples * step,
                              -step))[:args.n_samples]
    print(f"  picking {len(pick_indices)} samples, indices span "
          f"[{min(pick_indices)}, {max(pick_indices)}]")

    # ---- accumulate per-column MSE: [W]; optionally per-row × per-col [H, W] ----
    mse_per_col_sum = torch.zeros(LATENT_W, device=device, dtype=torch.float64)
    mse_per_hw_sum  = torch.zeros(8, LATENT_W, device=device, dtype=torch.float64)
    n_observations = 0   # samples * timesteps
    t_start = time.perf_counter()

    # Optional: disable Fix#1 pos-enc for fair eval of pre-Fix#1 checkpoints.
    from s2s_min.models.attention import disable_cross_attn_pos_enc
    pos_enc_ctx = disable_cross_attn_pos_enc() if args.no_pos_enc else contextlib_nullcontext()
    print(f"  cross-attn pos-enc: {'DISABLED (pre-Fix#1 eval)' if args.no_pos_enc else 'ENABLED'}")

    with pos_enc_ctx:
        for i, sample_idx in enumerate(pick_indices):
            npz = np.load(cache_files[sample_idx])
            image_latent = torch.from_numpy(npz["image_latent"]).unsqueeze(0).to(device)  # [1,4,32,56]
            raymap       = torch.from_numpy(npz["raymap"]).unsqueeze(0).to(device)        # [1,6,32,56]
            mu           = torch.from_numpy(npz["mu"]).unsqueeze(0).to(device)            # [1,8,8,256]

            kv_context = build_kv_context(image_latent, raymap)                           # [1,10,8,64]

            for t_int in args.timesteps:
                t = torch.tensor([t_int], device=device, dtype=torch.long)

                noise = torch.randn_like(mu)
                z_noisy  = diffusion.add_noise(mu, noise, t)
                v_target = diffusion.get_target(mu, noise, t)
                with torch.no_grad():
                    v_pred = unet(z_noisy, t, kv_context)

                sq_err = (v_pred - v_target).pow(2)                                       # [1,8,8,256]
                # Reduce over B, C, H — keep W axis.
                mse_per_col_sum += sq_err.mean(dim=(0, 1, 2)).double()                    # [256]
                # Also reduce over B, C — keep H, W for the heatmap.
                mse_per_hw_sum  += sq_err.mean(dim=(0, 1)).double()                       # [8, 256]
                n_observations += 1

            if (i + 1) % 4 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  [{i+1}/{len(pick_indices)} samples done, "
                      f"{n_observations} obs, {elapsed:.1f}s]")

    mse_per_col = (mse_per_col_sum / n_observations).cpu().numpy()                    # [256]
    mse_per_hw  = (mse_per_hw_sum  / n_observations).cpu().numpy()                    # [8, 256]

    # ---- analyze front vs back ----
    front_mask = np.zeros(LATENT_W, dtype=bool)
    front_mask[FRONT_LO:FRONT_HI] = True
    back_mask = ~front_mask

    mse_front_mean = mse_per_col[front_mask].mean()
    mse_back_mean  = mse_per_col[back_mask].mean()
    mse_overall    = mse_per_col.mean()
    ratio = mse_back_mean / mse_front_mean

    print()
    print("=" * 70)
    print("Per-azimuth-column MSE results")
    print("=" * 70)
    print(f"  observations     : {args.n_samples} samples × {len(args.timesteps)} timesteps "
          f"= {n_observations}")
    print(f"  overall mean MSE : {mse_overall:.4f}")
    print(f"  front-band MSE   : {mse_front_mean:.4f}  "
          f"(cols [{FRONT_LO},{FRONT_HI}], width={FRONT_HALF_WIDTH*2}, ~{FRONT_HALF_WIDTH*2/LATENT_W*100:.0f}% of azimuth)")
    print(f"  back-band MSE    : {mse_back_mean:.4f}  "
          f"(remaining ~{(LATENT_W-FRONT_HALF_WIDTH*2)/LATENT_W*100:.0f}% of azimuth)")
    print(f"  back / front     : {ratio:.3f}")
    print()
    print("Verdict:")
    if ratio > 1.15:
        print("  >> back/front ratio > 1.15 — STRONG signal that conditioning works")
        print("  >> in the front FoV and fails in the back. FoV asymmetry confirmed.")
        print("  >> NEXT MOVE: scope-B (6-camera surround input).")
    elif ratio > 1.05:
        print("  >> back/front ratio = 1.05-1.15 — MILD signal of FoV asymmetry, but")
        print("  >> not dominant. Both fixes (scope-B and Fix#2 KV pool) plausible.")
        print("  >> NEXT MOVE: cheap one first — Fix #2 (KV pool 16×128), then scope-B.")
    else:
        print("  >> back/front ratio < 1.05 — MSE is roughly uniform across azimuth.")
        print("  >> Model is NOT differentially using image even in the front FoV.")
        print("  >> Conditioning pathway is broken downstream of pos-enc.")
        print("  >> NEXT MOVE: Fix #2 (KV pool 16×128) or D1 (FiLM global cond).")

    # ---- save artifacts ----
    np.save(args.out_dir / "mse_per_col.npy", mse_per_col)
    np.save(args.out_dir / "mse_per_hw.npy",  mse_per_hw)
    stats = {
        "ckpt": str(args.ckpt),
        "ckpt_step": int(ckpt.get("step")),
        "ckpt_loss_ema": float(ckpt.get("loss_ema")),
        "n_samples": args.n_samples,
        "timesteps": args.timesteps,
        "n_observations": n_observations,
        "front_band_cols": [FRONT_LO, FRONT_HI],
        "mse_front_mean": float(mse_front_mean),
        "mse_back_mean": float(mse_back_mean),
        "mse_overall": float(mse_overall),
        "back_to_front_ratio": float(ratio),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top: per-column MSE curve.
    ax = axes[0]
    azimuth_deg = np.linspace(-180, 180, LATENT_W, endpoint=False)
    # Shift so col 0 is at azimuth -180; col 128 (front, +x) is at azimuth 0
    # Mapping: col=128 → azimuth=0, col=0 → azimuth=-180
    azimuth_deg = (np.arange(LATENT_W) - LATENT_W // 2) * (360.0 / LATENT_W)

    ax.plot(azimuth_deg, mse_per_col, lw=1.0, color="C0", alpha=0.7, label="MSE per column")
    # Smoothed (moving avg over 11 cols).
    kernel = np.ones(11) / 11.0
    mse_smooth = np.convolve(mse_per_col, kernel, mode="same")
    ax.plot(azimuth_deg, mse_smooth, lw=2.0, color="C3", label="smoothed (11-col MA)")
    # Mark front band.
    front_lo_deg = (FRONT_LO - LATENT_W // 2) * (360.0 / LATENT_W)
    front_hi_deg = (FRONT_HI - LATENT_W // 2) * (360.0 / LATENT_W)
    ax.axvspan(front_lo_deg, front_hi_deg, alpha=0.18, color="C2",
               label=f"CAM_FRONT FoV (~70°, cols {FRONT_LO}-{FRONT_HI})")
    # Mean lines.
    ax.axhline(mse_front_mean, color="C2", lw=1.5, ls="--",
               label=f"front mean = {mse_front_mean:.3f}")
    ax.axhline(mse_back_mean, color="C1", lw=1.5, ls="--",
               label=f"back mean = {mse_back_mean:.3f}")
    ax.set_xlabel("azimuth (degrees from forward; +x = 0°)")
    ax.set_ylabel("v-MSE")
    ax.set_title(
        f"Per-azimuth-column v-MSE   "
        f"ckpt step {ckpt.get('step')}, loss_ema {ckpt.get('loss_ema'):.4f}\n"
        f"back/front = {ratio:.3f}    ({n_observations} obs)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid(True, alpha=0.3)

    # Bottom: deviation from overall mean, colored.
    ax = axes[1]
    deviation = mse_per_col - mse_overall
    colors = ["C3" if d > 0 else "C2" for d in deviation]
    ax.bar(azimuth_deg, deviation, width=(360.0/LATENT_W),
           color=colors, alpha=0.6, edgecolor="none")
    ax.axvspan(front_lo_deg, front_hi_deg, alpha=0.18, color="C2")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("MSE − overall_mean")
    ax.set_title("deviation from mean (green = below avg, red = above)")
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = args.out_dir / "mse_per_col.png"
    fig.savefig(plot_path, dpi=110)
    print(f"\n  saved plot   : {plot_path}")
    print(f"  saved stats  : {args.out_dir / 'stats.json'}")
    print(f"  saved array  : {args.out_dir / 'mse_per_col.npy'}")

    if args.per_elevation_row:
        # H × W heatmap of MSE. Each row is an elevation band in the LiDAR
        # latent (8 latent rows from 4× compression of 32 HDL-32E beams →
        # roughly 4 beams per latent row).
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
        im = ax2.imshow(
            mse_per_hw, aspect="auto", origin="lower", cmap="viridis",
            extent=[-180.0, 180.0, 0, 8], vmin=0.15,
        )
        # mark front band
        ax2.axvspan(front_lo_deg, front_hi_deg, alpha=0.18, facecolor="none",
                    edgecolor="w", lw=1.5)
        ax2.set_xlabel("azimuth (deg from forward)")
        ax2.set_ylabel("LiDAR latent elevation row (0 = bottom beams, 7 = top beams)")
        ax2.set_title(
            f"v-MSE per (elevation row × azimuth col)   "
            f"ckpt step {ckpt.get('step')}, loss_ema {ckpt.get('loss_ema'):.4f}\n"
            f"pos_enc={'OFF' if args.no_pos_enc else 'ON'}    ({n_observations} obs)"
        )
        ax2.set_xticks([-180, -90, -45, 0, 45, 90, 180])
        plt.colorbar(im, ax=ax2, label="v-MSE")
        fig2.tight_layout()
        heatmap_path = args.out_dir / "mse_per_hw.png"
        fig2.savefig(heatmap_path, dpi=110)
        print(f"  saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
