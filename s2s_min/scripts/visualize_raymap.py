"""Visualize the raymap on a few real nuScenes CAM_FRONT samples.

For each sample, plots:
    [input RGB]  [origin_x/y/z]  [dir_x]  [dir_y]  [dir_z]

Expected behaviour:
    - origin channels are constant per sample (camera position in ego frame).
    - dir_x varies LEFT-RIGHT across W (azimuth changes as you scan columns).
    - dir_y varies TOP-BOTTOM across H (elevation changes as you scan rows).
    - dir_z is mostly large positive (camera looks forward in ego frame).
    - per-pixel |direction| ≈ 1.0 everywhere.

Run:
    env/bin/python s2s_min/scripts/visualize_raymap.py
Output:
    s2s_min/out/raymap_samples/samples.png + stats.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.raymap import build_raymap

NUSCENES_ROOT = Path("/media/skr/storage/self_driving/S2GO/data/nuscenes")
OUT_DIR       = Path("s2s_min/out/raymap_samples")
N_SAMPLES     = 4
IMG_H, IMG_W  = 256, 448
NATIVE_W, NATIVE_H = 1600, 900
SD_DOWNSAMPLE = 8


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def scale_intrinsics(K_native):
    K = K_native.astype(np.float32).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def collect_cam_front_records(n: int):
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sd = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    kfs = [rec for rec in sd
           if rec["is_key_frame"]
           and sensor[cs[rec["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT"]
    step = max(1, len(kfs) // n)
    return [(kf, cs[kf["calibrated_sensor_token"]]) for kf in kfs[::step][:n]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = collect_cam_front_records(N_SAMPLES)
    print(f"using {len(records)} CAM_FRONT samples")

    rows = []
    Ks, Ts, rgbs, names = [], [], [], []
    for rec, cs_cam in records:
        K_native = np.array(cs_cam["camera_intrinsic"], dtype=np.float32)
        K_scaled = scale_intrinsics(K_native)
        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = quat_wxyz_to_rotmat(cs_cam["rotation"])
        T_cam2ego[:3, 3] = np.array(cs_cam["translation"], dtype=np.float32)
        Ks.append(K_scaled); Ts.append(T_cam2ego)
        img = Image.open(NUSCENES_ROOT / rec["filename"]).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
        rgbs.append(np.asarray(img, dtype=np.float32) / 255.0)
        names.append(rec["filename"].split("/")[-1])

    K_batch = torch.from_numpy(np.stack(Ks))
    T_batch = torch.from_numpy(np.stack(Ts))
    raymap = build_raymap(K_batch, T_batch, H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE)
    raymap_np = raymap.cpu().numpy()
    print(f"raymap: {tuple(raymap.shape)}  dtype={raymap.dtype}")

    # ---- stats ----
    rows.append(f"raymap shape: {tuple(raymap.shape)}")
    rows.append(f"latent grid : {raymap.shape[2]} × {raymap.shape[3]}  (SD-VAE 8× downsample of {IMG_H}×{IMG_W})")
    rows.append("")
    rows.append("per-sample sanity (origin should be constant; |dir| should be ≈ 1.0):")
    rows.append(f"  {'name':<60}  {'origin_xyz':<28}  {'|dir|_mean':>10}  {'|dir|_min':>10}  {'|dir|_max':>10}")
    for i, name in enumerate(names):
        ox, oy, oz = raymap_np[i, 0, 0, 0], raymap_np[i, 1, 0, 0], raymap_np[i, 2, 0, 0]
        d = raymap_np[i, 3:]  # [3, H, W]
        dnorm = np.sqrt((d ** 2).sum(axis=0))  # [H, W]
        rows.append(
            f"  {name[:60]:<60}  "
            f"({ox:+.2f},{oy:+.2f},{oz:+.2f})            "
            f"{dnorm.mean():>10.4f}  {dnorm.min():>10.4f}  {dnorm.max():>10.4f}"
        )

    rows.append("")
    rows.append("directional gradient check (per-sample averages over H, W):")
    rows.append(f"  {'name':<60}  {'dir_x':>8}  {'dir_y':>8}  {'dir_z':>8}  "
                f"{'dx(L→R)':>9}  {'dy(T→B)':>9}")
    for i, name in enumerate(names):
        dx, dy, dz = raymap_np[i, 3].mean(), raymap_np[i, 4].mean(), raymap_np[i, 5].mean()
        # gradient signs we expect:
        # dir_x leftmost column vs rightmost column — should differ (camera scans across azimuth)
        dx_lr = raymap_np[i, 3, :, -1].mean() - raymap_np[i, 3, :, 0].mean()
        dy_tb = raymap_np[i, 4, -1, :].mean() - raymap_np[i, 4, 0, :].mean()
        rows.append(f"  {name[:60]:<60}  {dx:>8.4f}  {dy:>8.4f}  {dz:>8.4f}  "
                    f"{dx_lr:>9.4f}  {dy_tb:>9.4f}")

    print("\n".join(rows))
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")

    # ---- figure: rows = samples, cols = [RGB, ox, oy, oz, dx, dy, dz] ----
    cols = ["input RGB", "origin_x", "origin_y", "origin_z", "dir_x", "dir_y", "dir_z"]
    fig, axes = plt.subplots(N_SAMPLES, len(cols),
                              figsize=(2.6 * len(cols), 2.0 * N_SAMPLES))
    if N_SAMPLES == 1:
        axes = axes[None, :]
    for r in range(N_SAMPLES):
        axes[r, 0].imshow(rgbs[r])
        axes[r, 0].set_ylabel(names[r].split("__")[0], fontsize=6)
        for c in range(6):
            ch = raymap_np[r, c]
            # symmetric color limits help read the gradient direction
            vmax = max(abs(ch.min()), abs(ch.max()), 1e-6)
            axes[r, c + 1].imshow(ch, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
        for c in range(len(cols)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=9)
    fig.suptitle(
        f"Raymap on the SD-VAE latent grid ({IMG_H}×{IMG_W} input → 32×56 latent, ÷{SD_DOWNSAMPLE})\n"
        f"Channels 0–2: ego-frame ray origin (constant per sample) — 3–5: unit direction",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "samples.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {OUT_DIR / 'samples.png'}")
    print(f"wrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
