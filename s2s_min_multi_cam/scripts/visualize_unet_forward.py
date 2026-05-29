"""Visualize the diffusion forward-corruption process + a fresh-U-Net forward.

Two viz purposes on ONE sample:
  1. Show what the U-Net's job will be: corrupted LiDAR latents at increasing
     timesteps (decoded back through the LiDAR VAE for visual inspection).
     This demonstrates the diffusion scheduler is wired correctly and visualizes
     the forward (q-sample) noising process.
  2. Sanity-check the U-Net forward at each timestep:
     - output shape matches input ✓
     - output magnitude is ~zero (zero-init head → fresh U-Net predicts v=0)
     - MSE loss between v_pred (≈ 0) and v_target is non-trivial and grows
       smoothly with t

For a fresh untrained U-Net, the *output values* are uninteresting (all near
zero). What we're validating is that the pipeline composes correctly.

Run:
    env/bin/python s2s_min/scripts/visualize_unet_forward.py
Output:
    s2s_min/out/unet_forward_samples/samples.png + stats.txt
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.range_image import (
    load_nuscenes_lidar_bin,
    point_cloud_to_range_image,
    range_image_to_point_cloud,
)
from models.diffusion import DiffusionWrapper
from models.lidar_vae import LiDARVAE
from models.unet import LiDARUNet

NUSCENES_ROOT = Path("nuscenes")
SUBSET_TOKENS = Path("s2s_min/out/subset_scene_tokens.txt")
LIDAR_VAE_CKPT = Path("s2s_min/out/lidar_vae.pt")
OUT_DIR        = Path("s2s_min/out/unet_forward_samples")

# Timesteps to visualize the forward-corruption process at.
T_STEPS = [0, 200, 500, 800, 999]


def find_lidar_keyframe():
    """First LIDAR_TOP keyframe in the subset."""
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sd = json.loads((meta / "sample_data.json").read_text())
    sample = json.loads((meta / "sample.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    subset = set(SUBSET_TOKENS.read_text().split())
    samples_in = {s["token"]: s for s in sample if s["scene_token"] in subset}
    for rec in sd:
        if not rec["is_key_frame"] or rec["sample_token"] not in samples_in:
            continue
        if sensor[cs[rec["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "LIDAR_TOP":
            return rec
    raise RuntimeError("no LIDAR_TOP keyframe in subset")


def _bev_scatter(pc: np.ndarray, ax, color: str, range_m: float = 60.0):
    x, y = pc[:, 0], pc[:, 1]
    ax.scatter(x, y, s=0.05, c=color, alpha=0.4, linewidths=0)
    ax.set_xlim(-range_m, range_m); ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- pick one sample + load + LiDAR VAE encode -----------------------
    rec = find_lidar_keyframe()
    print(f"sample: {rec['filename']}")
    pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / rec["filename"]))
    range_img_np = point_cloud_to_range_image(pc)
    x = torch.from_numpy(range_img_np).unsqueeze(0).to(device)
    print(f"input range_img: {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")

    # ---- LiDAR VAE (frozen) ----------------------------------------------
    ckpt = torch.load(LIDAR_VAE_CKPT, map_location=device)
    arch_kwargs = {k: v for k, v in ckpt["config"].items()
                   if k in inspect.signature(LiDARVAE.__init__).parameters}
    vae = LiDARVAE(**arch_kwargs).to(device).eval()
    vae.load_state_dict(ckpt["state_dict"])
    vae.requires_grad_(False)

    with torch.no_grad():
        mu, _ = vae.encode(x)                           # [1, 8, 8, 256]
        x_recon = vae.decode(mu).cpu().numpy()          # baseline: VAE round-trip
    print(f"latent mu: {tuple(mu.shape)}  mean={mu.mean():+.3f}  std={mu.std():.3f}")

    # ---- U-Net (fresh, untrained) + diffusion ----------------------------
    unet = LiDARUNet().to(device).eval()
    diffusion = DiffusionWrapper()
    # Fake KV context for the smoke check (random — fresh U-Net doesn't care).
    kv_context = torch.randn(1, 10, 8, 64, device=device)

    rows = []
    rows.append(f"sample: {rec['filename']}")
    rows.append(f"input range_img : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")
    rows.append(f"LiDAR latent mu : {tuple(mu.shape)}  mean={mu.mean().item():+.3f}  std={mu.std().item():.3f}")
    rows.append(f"U-Net params    : {sum(p.numel() for p in unet.parameters())/1e6:.2f} M (untrained, zero-init head)")
    rows.append("")
    rows.append("Forward-diffusion corruption + fresh U-Net forward at increasing t:")
    rows.append(f"  {'t':>4}  {'noisy_lat_std':>12}  {'recon_range_L1':>15}  "
                f"{'v_pred_max':>10}  {'v_target_std':>12}  {'mse(v_pred, v_target)':>20}")

    # For each timestep, decode the noised latent and run U-Net forward.
    decoded_rows = []  # list of (t, decoded_range_img_np, decoded_pc_np)
    for t_val in T_STEPS:
        t = torch.tensor([t_val], device=device)
        if t_val == 0:
            z_noisy = mu                                # clean latent (no noise)
            v_target = torch.zeros_like(mu)
        else:
            noise = torch.randn_like(mu)
            z_noisy = diffusion.add_noise(mu, noise, t)
            v_target = diffusion.get_target(mu, noise, t)

        with torch.no_grad():
            x_decoded = vae.decode(z_noisy).cpu().numpy()      # [1, 3, 32, 1024]
            v_pred = unet(z_noisy, t, kv_context)

        mse = torch.nn.functional.mse_loss(v_pred, v_target).item()
        l1_recon = (x_decoded - x.cpu().numpy()).__abs__().mean()
        rows.append(
            f"  {t_val:>4}  {z_noisy.std().item():>12.4f}  {l1_recon:>15.4f}  "
            f"{v_pred.abs().max().item():>10.2e}  {v_target.std().item():>12.4f}  "
            f"{mse:>20.4f}"
        )
        decoded_rows.append((t_val, x_decoded[0], range_image_to_point_cloud(x_decoded[0])))

    print("\n".join(rows))
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")

    # ---- figure: rows = timesteps, cols = [decoded range_img, BEV scatter] ----
    n_rows = len(T_STEPS) + 1  # + 1 row for the ground-truth reference at top
    fig = plt.figure(figsize=(14, 2.6 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.15)

    # Ground-truth reference row.
    ax_r = fig.add_subplot(gs[0, 0])
    ax_r.imshow(range_img_np[0], cmap="turbo", aspect="auto", vmin=0, vmax=1)
    ax_r.set_title("INPUT range_img (ground truth, channel 0 = range)", fontsize=9)
    ax_r.set_xticks([]); ax_r.set_yticks([])

    ax_b = fig.add_subplot(gs[0, 1])
    _bev_scatter(range_image_to_point_cloud(range_img_np), ax_b, "tab:blue")
    ax_b.set_title("BEV (input)", fontsize=8)

    # Per-timestep rows.
    for i, (t_val, decoded_img, decoded_pc) in enumerate(decoded_rows, start=1):
        ax_r = fig.add_subplot(gs[i, 0])
        ax_r.imshow(decoded_img[0].clip(0, 1), cmap="turbo", aspect="auto", vmin=0, vmax=1)
        ax_r.set_title(
            f"t = {t_val}  →  noised latent decoded back through VAE  (the U-Net's denoising target)",
            fontsize=9,
        )
        ax_r.set_xticks([]); ax_r.set_yticks([])

        ax_b = fig.add_subplot(gs[i, 1])
        if decoded_pc.shape[0] > 0:
            _bev_scatter(decoded_pc, ax_b, "tab:red")
        ax_b.set_title(f"BEV (t={t_val})", fontsize=8)

    fig.suptitle(
        "Diffusion forward-corruption process — what the LiDAR U-Net needs to learn to undo.\n"
        "(Fresh U-Net is zero-init, so v_pred ≈ 0 everywhere — see stats.txt for the per-t MSE values.)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "samples.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {OUT_DIR / 'samples.png'}")
    print(f"wrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
