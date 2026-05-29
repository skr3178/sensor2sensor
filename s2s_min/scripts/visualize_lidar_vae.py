"""Eyeball the LiDAR VAE on a handful of real nuScenes LIDAR_TOP keyframes.

For each sample, runs encode + decode and writes a PNG showing:

    [range strip — input above, decoded below]   [BEV scatter — input vs decoded overlaid]

Also prints per-channel reconstruction error (L1 on valid pixels for range/intensity,
BCE on validity), and the same in PSNR-like units where meaningful.

Run:
    env/bin/python s2s_min/scripts/visualize_lidar_vae.py
Output:
    s2s_min/out/lidar_vae_samples.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

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
from models.lidar_vae import LiDARVAE

NUSCENES_ROOT = Path("nuscenes")  # symlink at project root
SUBSET_TOKENS = Path("s2s_min/out/subset_scene_tokens.txt")
DEFAULT_CKPT  = Path("s2s_min/out/lidar_vae.pt")
DEFAULT_OUT_DIR = Path("s2s_min/out/lidar_vae_samples")
N_SAMPLES     = 4

# --- CLI overrides (eval_after.py uses these to redirect into a run-dir) ---
import argparse as _argparse
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
_p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
_p.add_argument("--n_samples", type=int, default=N_SAMPLES)
_args, _ = _p.parse_known_args()
CKPT      = _args.ckpt
OUT_DIR   = _args.out_dir
OUT_PNG   = OUT_DIR / "samples.png"
OUT_STATS = OUT_DIR / "stats.txt"
N_SAMPLES = _args.n_samples


def collect_lidar_paths() -> list[Path]:
    """One LIDAR_TOP keyframe per scene, drawn from the 10-scene subset, first N."""
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample = json.loads((meta / "sample.json").read_text())
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}

    subset = set(SUBSET_TOKENS.read_text().split())
    samples_in_subset = [s for s in sample if s["scene_token"] in subset]
    samples_by_token = {s["token"]: s for s in samples_in_subset}

    # First LIDAR_TOP keyframe per scene encountered.
    seen_scenes: set[str] = set()
    out: list[Path] = []
    for sd in sample_data:
        if not sd["is_key_frame"]:
            continue
        if sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"] != "LIDAR_TOP":
            continue
        if sd["sample_token"] not in samples_by_token:
            continue
        scene_t = samples_by_token[sd["sample_token"]]["scene_token"]
        if scene_t in seen_scenes:
            continue
        seen_scenes.add(scene_t)
        out.append(NUSCENES_ROOT / sd["filename"])
        if len(out) >= N_SAMPLES:
            break
    return out


def _bev_from_pc(pc: np.ndarray, color: str, ax, label: str, range_m: float = 60.0):
    """Top-down scatter of a (M, 4) point cloud."""
    x, y = pc[:, 0], pc[:, 1]
    ax.scatter(x, y, s=0.05, c=color, alpha=0.4, label=label, linewidths=0)
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint {CKPT} ...")
    ckpt = torch.load(CKPT, map_location="cuda")
    print(f"  step:   {ckpt.get('step', '<missing>')}")
    print(f"  config: {ckpt['config']}")
    # The checkpoint config includes training hyperparams (lam_*, lr, ema_decay,
    # etc.) that LiDARVAE.__init__ doesn't accept. Filter to model-architecture kwargs.
    import inspect
    arch_kwargs = set(inspect.signature(LiDARVAE.__init__).parameters)
    model_cfg = {k: v for k, v in ckpt["config"].items() if k in arch_kwargs}
    print(f"  arch kwargs (filtered): {model_cfg}")
    vae = LiDARVAE(**model_cfg).cuda().eval()
    vae.load_state_dict(ckpt["state_dict"])
    vae.requires_grad_(False)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"  params: {n_params/1e6:.2f} M")

    paths = collect_lidar_paths()
    print(f"\nusing {len(paths)} LIDAR_TOP samples:")
    for p in paths:
        print(f"  {p.name}")

    # Load every sample on CPU, then batch onto GPU for encode/decode.
    range_imgs_np = [point_cloud_to_range_image(load_nuscenes_lidar_bin(str(p))) for p in paths]
    range_imgs = torch.from_numpy(np.stack(range_imgs_np)).cuda()        # [N, 3, 32, 1024]
    print(f"\ninput range_imgs: {tuple(range_imgs.shape)}  dtype={range_imgs.dtype}")

    with torch.no_grad():
        mu, logvar = vae.encode(range_imgs)
        recon = vae.decode(mu)                                           # [N, 3, 32, 1024]
    print(f"latent mu       : {tuple(mu.shape)}  mean={mu.mean().item():+.3f}  std={mu.std().item():.3f}")
    print(f"latent logvar   : {tuple(logvar.shape)}  mean={logvar.mean().item():+.3f}  std={logvar.std().item():.3f}")
    print(f"recon            : {tuple(recon.shape)}  dtype={recon.dtype}")

    # ---- per-sample reconstruction quality on the three channels (print + save) ----
    rows: list[str] = []
    rows.append(f"checkpoint     : {CKPT}")
    rows.append(f"step           : {ckpt.get('step', '<missing>')}")
    rows.append(f"arch kwargs    : {model_cfg}")
    rows.append(f"params         : {n_params/1e6:.2f} M")
    rows.append(f"input shape    : {tuple(range_imgs.shape)}  (range/intensity/validity in [0, 1], fp32)")
    rows.append(f"latent mu      : {tuple(mu.shape)}  mean={mu.mean().item():+.3f}  std={mu.std().item():.3f}")
    rows.append(f"latent logvar  : {tuple(logvar.shape)}  mean={logvar.mean().item():+.3f}  std={logvar.std().item():.3f}")
    rows.append(f"recon shape    : {tuple(recon.shape)}")
    rows.append("")
    rows.append("per-sample reconstruction (lower = better; L1_range_m back-scaled to meters):")
    rows.append(f"  {'name':<60}  {'L1_range_m':>10}  {'L1_intens':>9}  {'BCE_valid':>9}  {'valid_acc':>9}")
    range_clamp_m = 100.0
    means = {"l1_range_m": 0.0, "l1_intensity": 0.0, "bce_valid": 0.0, "valid_acc": 0.0}
    for i, p in enumerate(paths):
        x, x_hat = range_imgs[i], recon[i]
        mask = x[2] > 0.5  # valid pixels
        denom = mask.sum().clamp(min=1).item()

        l1_range_m = ((x[0] - x_hat[0]).abs() * mask).sum().item() / denom * range_clamp_m
        l1_intensity = ((x[1] - x_hat[1]).abs() * mask).sum().item() / denom
        bce_valid = torch.nn.functional.binary_cross_entropy(x_hat[2], x[2]).item()
        valid_acc = ((x_hat[2] > 0.5) == (x[2] > 0.5)).float().mean().item()

        rows.append(f"  {p.name[:60]:<60}  {l1_range_m:10.3f}  {l1_intensity:9.4f}  "
                    f"{bce_valid:9.4f}  {valid_acc:9.3f}")
        means["l1_range_m"]   += l1_range_m   / N_SAMPLES
        means["l1_intensity"] += l1_intensity / N_SAMPLES
        means["bce_valid"]    += bce_valid    / N_SAMPLES
        means["valid_acc"]    += valid_acc    / N_SAMPLES
    rows.append(f"  {'MEAN':<60}  {means['l1_range_m']:10.3f}  {means['l1_intensity']:9.4f}  "
                f"{means['bce_valid']:9.4f}  {means['valid_acc']:9.3f}")

    print()
    for r in rows:
        print(r)
    OUT_STATS.write_text("\n".join(rows) + "\n")

    # ---- visualize: N_SAMPLES rows, 4 columns (range in / range out / valid in / BEV overlay) ----
    fig = plt.figure(figsize=(20, 3 * N_SAMPLES))
    gs = fig.add_gridspec(N_SAMPLES, 4, width_ratios=[3, 3, 3, 1.4], hspace=0.35, wspace=0.15)

    range_in_np  = range_imgs.cpu().numpy()
    range_out_np = recon.cpu().numpy().clip(0, 1)

    for i in range(N_SAMPLES):
        # column 0: input range (32 x 1024) coloured by depth
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(range_in_np[i, 0], cmap="turbo", aspect="auto", vmin=0, vmax=1)
        ax0.set_title("input range" if i == 0 else "", fontsize=9)
        ax0.set_ylabel(paths[i].name.split("__")[0], fontsize=6)
        ax0.set_xticks([]); ax0.set_yticks([])

        # column 1: decoded range
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(range_out_np[i, 0], cmap="turbo", aspect="auto", vmin=0, vmax=1)
        ax1.set_title("decoded range" if i == 0 else "", fontsize=9)
        ax1.set_xticks([]); ax1.set_yticks([])

        # column 2: input validity (binary mask) on top of decoded validity (continuous)
        ax2 = fig.add_subplot(gs[i, 2])
        # show recon validity in grey + true validity as red overlay where they differ
        rec_v = range_out_np[i, 2]
        true_v = range_in_np[i, 2]
        diff = np.abs(rec_v - true_v)
        ax2.imshow(rec_v, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax2.imshow(np.ma.masked_where(diff < 0.3, diff), cmap="Reds", alpha=0.7, aspect="auto", vmin=0, vmax=1)
        ax2.set_title("decoded validity (grey) + |diff| > 0.3 (red)" if i == 0 else "", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])

        # column 3: BEV scatter, input (blue) vs decoded (red)
        ax3 = fig.add_subplot(gs[i, 3])
        pc_in  = range_image_to_point_cloud(range_in_np[i])
        pc_out = range_image_to_point_cloud(range_out_np[i])
        _bev_from_pc(pc_in,  "tab:blue", ax3, "input",   range_m=60.0)
        _bev_from_pc(pc_out, "tab:red",  ax3, "decoded", range_m=60.0)
        if i == 0:
            ax3.set_title("BEV (blue=input, red=decoded)", fontsize=8)

    fig.suptitle(
        f"LiDAR VAE round-trip on real nuScenes LIDAR_TOP frames  "
        f"(checkpoint step={ckpt.get('step', '?')}, {n_params/1e6:.2f} M params)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\nwrote {OUT_PNG}")
    print(f"wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
