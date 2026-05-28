"""Eyeball the SD 1.5 VAE on a handful of real nuScenes CAM_FRONT frames.

Picks N samples deterministically from the metadata, runs encode + decode,
and writes a single PNG with columns:

    [original 256x448]  [each of 4 latent channels as grayscale]  [decoded recon]

Also prints per-sample latent stats. The decode is purely for visualization;
the production pipeline only uses .encode().

Run:
    env/bin/python s2s_min/scripts/visualize_image_vae.py
Output:
    s2s_min/out/image_vae_samples.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch  # ensure .torch submodule loads
import torch
from diffusers import AutoencoderKL
from PIL import Image

NUSCENES_ROOT = Path("/media/skr/storage/self_driving/S2GO/data/nuscenes")
CKPT          = Path("s2s_min/checkpoints/sd15_vae")
OUT_DIR       = Path("s2s_min/out/image_vae_samples")
OUT_PNG       = OUT_DIR / "samples.png"
OUT_STATS     = OUT_DIR / "stats.txt"
N_SAMPLES     = 4
IMG_H, IMG_W  = 256, 448

# Deterministic spread: pick every (total // N_SAMPLES)-th CAM_FRONT keyframe.
def collect_cam_front_paths() -> list[Path]:
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample_data = json.loads((meta / "sample_data.json").read_text())
    calibrated_sensor = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}

    kfs = [
        sd for sd in sample_data
        if sd["is_key_frame"]
        and sensor[calibrated_sensor[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT"
    ]
    # Spread the picks across the dataset for visual variety.
    step = max(1, len(kfs) // N_SAMPLES)
    chosen = kfs[::step][:N_SAMPLES]
    return [NUSCENES_ROOT / sd["filename"] for sd in chosen]


def load_rgb(path: Path) -> torch.Tensor:
    """Load + resize to (IMG_H, IMG_W). Return CHW float16 in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0           # HWC [0,1]
    arr = arr.transpose(2, 0, 1)                              # CHW
    arr = arr * 2.0 - 1.0                                     # [-1, 1]
    return torch.from_numpy(arr).to(torch.float16)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading VAE from {CKPT} ...")
    vae = AutoencoderKL.from_pretrained(
        CKPT, subfolder="vae", variant="fp16", torch_dtype=torch.float16,
    ).cuda().eval().requires_grad_(False)
    sf = vae.config.scaling_factor
    print(f"  scaling_factor = {sf}")

    paths = collect_cam_front_paths()
    print(f"using {len(paths)} CAM_FRONT samples:")
    for p in paths:
        print(f"  {p.name}")

    rgb_batch = torch.stack([load_rgb(p) for p in paths]).cuda()        # [N, 3, 256, 448]
    with torch.no_grad():
        z      = vae.encode(rgb_batch).latent_dist.mean * sf            # [N, 4, 32, 56]
        recon  = vae.decode((z / sf)).sample                            # [N, 3, 256, 448]

    # Per-sample latent stats — print AND save to stats.txt.
    header  = f"  {'name':<60}  {'mean':>8}  {'std':>6}  {'min':>7}  {'max':>7}"
    rows    = []
    rows.append(f"checkpoint     : {CKPT}")
    rows.append(f"scaling_factor : {sf}")
    rows.append(f"latent_channels: {vae.config.latent_channels}")
    rows.append(f"block_out_channels: {vae.config.block_out_channels}")
    rows.append(f"input shape    : [B={N_SAMPLES}, 3, {IMG_H}, {IMG_W}]  (RGB in [-1, 1], fp16)")
    rows.append(f"output shape   : {tuple(z.shape)}  (latent post scaling_factor)")
    rows.append("")
    rows.append("per-sample latent stats (post-scaling):")
    rows.append(header)
    for p, z_i in zip(paths, z):
        rows.append(f"  {p.name[:60]:<60}  {z_i.mean().item():+8.4f}  {z_i.std().item():6.3f}  "
                    f"{z_i.min().item():+7.3f}  {z_i.max().item():+7.3f}")
    print()
    for r in rows:
        print(r)
    OUT_STATS.write_text("\n".join(rows) + "\n")

    # Render: rows = samples, columns = [input, ch0, ch1, ch2, ch3, recon].
    cols = ["input RGB", "latent ch 0", "latent ch 1", "latent ch 2", "latent ch 3", "decoded recon"]
    fig, axes = plt.subplots(N_SAMPLES, len(cols),
                              figsize=(2.6 * len(cols), 2.6 * N_SAMPLES * IMG_H / IMG_W))
    for row in range(N_SAMPLES):
        # input
        img = ((rgb_batch[row].float().cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(paths[row].name.split("__")[0], fontsize=6)
        # 4 latent channels (each normalised independently for display)
        z_np = z[row].float().cpu().numpy()
        for c in range(4):
            ch = z_np[c]
            ch_n = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            axes[row, 1 + c].imshow(ch_n, cmap="viridis", aspect="auto")
        # decoded reconstruction
        rec = ((recon[row].float().cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        axes[row, 5].imshow(rec)
        for c in range(len(cols)):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=10)

    fig.suptitle(
        "SD 1.5 VAE on real nuScenes CAM_FRONT frames\n"
        f"input {IMG_H}x{IMG_W} RGB  →  latent [4, 32, 56]  →  decoded recon {IMG_H}x{IMG_W}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print()
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_STATS}")


if __name__ == "__main__":
    main()
