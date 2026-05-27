"""Benchmark (B) — LiDAR re-projection round-trip for raymap correctness.

For a real paired nuScenes keyframe:
  1. Transform every LiDAR_TOP point into the ego frame.
  2. Project each point into the (scaled, 448×256) CAM_FRONT pixel grid using
     the calibrated camera intrinsics + extrinsics.
  3. Keep only points in front of the camera + inside the image bounds.
  4. Snap each (u, v) to its latent-grid cell (u // 8, v // 8) on the 32×56 grid.
  5. Look up our raymap at that cell → get `(origin_ego, dir_ego)`.
  6. Compute the angular error between `(point_ego − origin_ego)` and `dir_ego`.

Reports per-percentile angular error in degrees. Writes a side-by-side viz:
  - left: input image with projected LiDAR points coloured by angular error
  - right: histogram of per-point errors

The latent grid's 8× quantization sets a floor on the expected error: at
fx ≈ 354 px/rad (scaled), 1 pixel ≈ 0.16°, so a half-cell shift is ~0.6° and
a full-cell shift is ~1.3°. Mean angular error should land somewhere in that
neighbourhood for a correct raymap.

Run:
    env/bin/python s2s_min/scripts/benchmark_raymap.py
Output:
    s2s_min/out/raymap_benchmark/raymap_benchmark.png + stats.txt
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

from data.range_image import load_nuscenes_lidar_bin
from models.raymap import build_raymap

NUSCENES_ROOT  = Path("nuscenes")
SUBSET_TOKENS  = Path("s2s_min/out/subset_scene_tokens.txt")
OUT_DIR        = Path("s2s_min/out/raymap_benchmark")
IMG_H, IMG_W   = 256, 448
NATIVE_W, NATIVE_H = 1600, 900
SD_DOWNSAMPLE  = 8
H_LAT, W_LAT   = IMG_H // SD_DOWNSAMPLE, IMG_W // SD_DOWNSAMPLE   # 32, 56


# ----------------------------- helpers ----------------------------------
def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def make_T(translation, rotation_quat_wxyz):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_rotmat(rotation_quat_wxyz)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def scale_intrinsics(K_native: np.ndarray) -> np.ndarray:
    K = K_native.astype(np.float32).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def find_paired_keyframe():
    meta = NUSCENES_ROOT / "v1.0-trainval"
    sample = json.loads((meta / "sample.json").read_text())
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}
    subset = set(SUBSET_TOKENS.read_text().split())
    samples_in = {s["token"]: s for s in sample if s["scene_token"] in subset}
    cam, lid = {}, {}
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] not in samples_in:
            continue
        chan = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if chan == "CAM_FRONT": cam[sd["sample_token"]] = sd
        elif chan == "LIDAR_TOP": lid[sd["sample_token"]] = sd
    for tok in cam:
        if tok in lid:
            return cam[tok], lid[tok], cs
    raise RuntimeError("no paired CAM_FRONT + LIDAR_TOP found in subset")


# ----------------------------- main --------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cam_rec, lid_rec, cs = find_paired_keyframe()
    cs_cam = cs[cam_rec["calibrated_sensor_token"]]
    cs_lid = cs[lid_rec["calibrated_sensor_token"]]

    K_native = np.array(cs_cam["camera_intrinsic"], dtype=np.float32)
    K_scaled = scale_intrinsics(K_native)
    T_cam2ego = make_T(cs_cam["translation"], cs_cam["rotation"])
    T_lid2ego = make_T(cs_lid["translation"], cs_lid["rotation"])
    T_ego2cam = np.linalg.inv(T_cam2ego)

    print(f"sample: {cam_rec['sample_token']}")
    print(f"  CAM_FRONT  : {cam_rec['filename']}")
    print(f"  LIDAR_TOP  : {lid_rec['filename']}")
    print(f"  K scaled   : fx={K_scaled[0,0]:.2f} fy={K_scaled[1,1]:.2f} "
          f"cx={K_scaled[0,2]:.2f} cy={K_scaled[1,2]:.2f}")
    print(f"  T_cam2ego  : translation={T_cam2ego[:3,3].round(3).tolist()}")
    print(f"  T_lid2ego  : translation={T_lid2ego[:3,3].round(3).tolist()}")

    # ---- load LiDAR + transform to ego frame -----------------------------
    pc_lid = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / lid_rec["filename"]))[:, :3]   # [N, 3]
    print(f"\nloaded LiDAR: {pc_lid.shape[0]} points (LiDAR-frame)")
    pc_lid_h = np.concatenate([pc_lid, np.ones((pc_lid.shape[0], 1), dtype=np.float32)], axis=1)  # [N, 4]
    pc_ego = (T_lid2ego @ pc_lid_h.T).T[:, :3]                                          # [N, 3] ego
    pc_cam = (T_ego2cam @ np.concatenate([pc_ego, np.ones_like(pc_ego[:, :1])], axis=1).T).T[:, :3]   # [N, 3] cam

    # ---- project to pixel + filter in-frustum ----------------------------
    z = pc_cam[:, 2]
    in_front = z > 0.5                                                                  # >0.5m in front
    pc_cam_f = pc_cam[in_front]
    pc_ego_f = pc_ego[in_front]
    uv = (K_scaled @ pc_cam_f.T).T                                                      # [M, 3]
    uv = uv[:, :2] / uv[:, 2:3]
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    uv_valid = uv[in_bounds]
    pc_ego_valid = pc_ego_f[in_bounds]
    print(f"after frustum-cull: {pc_ego_valid.shape[0]} points (in-front & in-image)")

    # ---- build our raymap ------------------------------------------------
    raymap = build_raymap(
        torch.from_numpy(K_scaled),
        torch.from_numpy(T_cam2ego),
        H_latent=H_LAT, W_latent=W_LAT, downsample=SD_DOWNSAMPLE,
    )[0].cpu().numpy()                                                                  # [6, 32, 56]
    origin = raymap[:3]                                                                 # [3, H, W]
    direction = raymap[3:]                                                              # [3, H, W]

    # ---- look up raymap at each LiDAR pixel's latent cell ----------------
    u_lat = np.clip((uv_valid[:, 0] // SD_DOWNSAMPLE).astype(np.int64), 0, W_LAT - 1)
    v_lat = np.clip((uv_valid[:, 1] // SD_DOWNSAMPLE).astype(np.int64), 0, H_LAT - 1)
    o = origin[:, v_lat, u_lat].T                                                       # [M, 3]
    d = direction[:, v_lat, u_lat].T                                                    # [M, 3]
    d = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-8)                    # unit, just in case

    # ---- angular error between ray direction and (point - origin) --------
    rel = pc_ego_valid - o                                                              # [M, 3]
    rel_norm = np.linalg.norm(rel, axis=1, keepdims=True).clip(min=1e-6)
    rel_unit = rel / rel_norm
    cos_angle = (rel_unit * d).sum(axis=1).clip(-1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    print(f"\nangular errors over {len(angle_deg)} LiDAR points:")
    pcts = [50, 75, 90, 95, 99, 100]
    p_vals = np.percentile(angle_deg, pcts)
    rows = []
    rows.append(f"sample          : {cam_rec['sample_token']}")
    rows.append(f"N points        : {len(angle_deg)} (after in-frustum cull)")
    rows.append(f"latent grid     : {H_LAT}x{W_LAT}  (latent pixel ≈ "
                f"{np.degrees(SD_DOWNSAMPLE/K_scaled[0,0]):.2f}° wide, "
                f"{np.degrees(SD_DOWNSAMPLE/K_scaled[1,1]):.2f}° tall)")
    rows.append(f"")
    rows.append(f"angular error stats (degrees):")
    rows.append(f"  mean   : {angle_deg.mean():.3f}")
    rows.append(f"  median : {np.median(angle_deg):.3f}")
    for p, v in zip(pcts, p_vals):
        rows.append(f"  p{p:<5d}: {v:.3f}")
    rows.append(f"")
    rows.append(f"baseline (latent-cell quantization floor):")
    rows.append(f"  half-cell shift (~half a latent pixel): ≈ "
                f"{0.5 * np.degrees(SD_DOWNSAMPLE/K_scaled[0,0]):.2f}°")
    rows.append(f"  full-cell shift: ≈ {np.degrees(SD_DOWNSAMPLE/K_scaled[0,0]):.2f}°")
    rows.append(f"")
    rows.append(f"interpretation:")
    rows.append(f"  - mean << 1°  → raymap geometry is correct end-to-end")
    rows.append(f"  - mean ≈ 0.5°-2° → expected: errors are dominated by latent-grid quantization")
    rows.append(f"  - mean > 5°    → likely a frame-convention or scaling bug — investigate")
    print("\n".join(rows))
    (OUT_DIR / "stats.txt").write_text("\n".join(rows) + "\n")

    # ---- visualize: input image + projected LiDAR coloured by error -----
    img = Image.open(NUSCENES_ROOT / cam_rec["filename"]).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(np.asarray(img))
    sc = ax_img.scatter(
        uv_valid[:, 0], uv_valid[:, 1],
        c=angle_deg, cmap="turbo", s=4, alpha=0.8,
        vmin=0, vmax=max(2.0, p_vals[3]),  # cap at p95 or 2°, whichever larger
    )
    ax_img.set_xlim(0, IMG_W); ax_img.set_ylim(IMG_H, 0)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(
        f"LiDAR re-projection round-trip: {len(angle_deg)} points coloured by angular error vs. raymap "
        f"(mean = {angle_deg.mean():.2f}°)",
        fontsize=10,
    )
    cbar = plt.colorbar(sc, ax=ax_img, shrink=0.85, label="angular error (deg)")

    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(angle_deg, bins=60, color="tab:blue", edgecolor="white")
    ax_hist.axvline(angle_deg.mean(), color="tab:red", linestyle="--", linewidth=1, label=f"mean {angle_deg.mean():.2f}°")
    ax_hist.axvline(np.median(angle_deg), color="tab:green", linestyle="--", linewidth=1, label=f"median {np.median(angle_deg):.2f}°")
    ax_hist.set_xlabel("angular error (deg)")
    ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title("error distribution", fontsize=9)

    out_png = OUT_DIR / "raymap_benchmark.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    print(f"wrote {OUT_DIR / 'stats.txt'}")


if __name__ == "__main__":
    main()
