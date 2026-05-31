"""C5 viability probe — Spearman correlation between DA-v2 depth and LiDAR range.

For each of N samples:
  1. Load CAM_FRONT image + raw LIDAR_TOP .pcd.bin (with HDL-32E ring index).
  2. Run DA-v2 Metric Outdoor Small → predicted depth map at image resolution.
  3. Project LiDAR points (LiDAR sensor frame → ego → cam → image plane).
  4. For each projected point: pair (DA-v2 depth at that pixel, LiDAR range).
  5. Bin pairs by latent elevation row (ring // 4 → row 0..7).

Compute Spearman correlation per latent elevation row over the pooled
front-FoV pairs. Decision criteria (from user):
  - Top-beam |Spearman r| > 0.6  → C5 is high-confidence win. Proceed with
                                   cache rebuild + retrain.
  - 0.3 < |r| < 0.6              → C5 helps modestly. Worth doing.
  - |r| < 0.3                    → DA-v2 doesn't track LiDAR depth at long
                                   range. Skip to DPT or larger DA-v2.

DA-v2 output convention: inverse depth scaled by max_depth.
  depth_in_meters = max_depth / output (= 80 / output).
We use raw output and Spearman (rank-based) so the inverse relationship
just shows as a negative correlation — we report |r|.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.range_image import load_nuscenes_lidar_bin
from dinov3_oblique_and_image_overlay import build_T, load_meta

NUSCENES   = Path("nuscenes")
SDVAE      = Path("s2s_min/out/cached_latents_v5_100scenes")
D3         = Path("s2s_min/out/cached_dinov3_v5_100scenes")
DA_CKPT    = "s2s_min/checkpoints/depth_anything_v2_metric_outdoor_small"  # default; override via --da_ckpt

# HDL-32E → latent row mapping: 32 beams compressed to 8 rows (4 beams/row).
N_RINGS_RAW   = 32
N_LATENT_ROWS = 8


def project_lidar_to_image_with_ring(points_5, cam_cs, lid_cs, img_w, img_h):
    """
    points_5: [N, 5] = (x, y, z, intensity, ring) in LiDAR sensor frame.
    Returns: dict with keys u, v, range_lid, ring, az_deg (1-D arrays of length M).
             az_deg = atan2(y, x) in LIDAR sensor frame, degrees in [-180, +180].
             +x is forward → az_deg=0; CAM_FRONT FoV ≈ ±35°.
    """
    T_lid2ego = build_T(lid_cs["translation"], lid_cs["rotation"])
    T_cam2ego = build_T(cam_cs["translation"], cam_cs["rotation"])
    T_ego2cam = np.linalg.inv(T_cam2ego)
    T_lid2cam = T_ego2cam @ T_lid2ego

    xyz = points_5[:, :3]
    ring = points_5[:, 4].astype(np.int32)
    range_lid = np.linalg.norm(xyz, axis=1)   # LiDAR sensor-frame distance in m
    az_deg = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))  # LIDAR-frame azimuth

    pts_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    pts_cam = (T_lid2cam @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.5

    K = np.array(cam_cs["camera_intrinsic"])
    uvw = (K @ pts_cam[in_front].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

    # Apply both masks
    keep_first = np.where(in_front)[0]
    keep_idx = keep_first[in_bounds]
    return {
        "u":         uv[in_bounds, 0],
        "v":         uv[in_bounds, 1],
        "range_lid": range_lid[keep_idx],
        "ring":      ring[keep_idx],
        "az_deg":    az_deg[keep_idx],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_range", type=float, default=80.0,
                    help="ignore LiDAR points beyond this range (DA-v2 trained to 80 m)")
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/c5_depth_probe"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--da_ckpt", type=Path, default=Path(DA_CKPT),
                    help="Local path to DA-v2 model checkpoint (HF format).")
    args = ap.parse_args()
    da_ckpt_str = str(args.da_ckpt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("C5 viability probe — DA-v2 depth vs raw LIDAR range")
    print("=" * 70)
    print(f"  n_samples : {args.n}")
    print(f"  max_range : {args.max_range} m  (ignore LiDAR > this)")
    print(f"  device    : {args.device}")
    print(f"  DA ckpt   : {da_ckpt_str}")

    # ---- DA-v2 model ----
    proc = AutoImageProcessor.from_pretrained(da_ckpt_str)
    mdl  = AutoModelForDepthEstimation.from_pretrained(da_ckpt_str).to(args.device).eval()
    print(f"  DA-v2 params: {sum(p.numel() for p in mdl.parameters())/1e6:.2f} M")

    # ---- nuScenes metadata ----
    lid_by_sample, cam_by_sample = load_meta()

    # ---- Pick N samples that exist in both caches AND have cam+lidar metadata ----
    common = sorted(
        ({p.stem for p in SDVAE.glob("*.npz")} & {p.stem for p in D3.glob("*.npz")})
        & set(lid_by_sample.keys()) & set(cam_by_sample.keys())
    )
    rng = np.random.RandomState(args.seed)
    picks = sorted(rng.choice(common, size=min(args.n, len(common)), replace=False))
    print(f"  picked {len(picks)} samples (of {len(common)} candidates)")

    # ---- Per-cell accumulators (latent_row × azimuth_band) ----
    # nuScenes LIDAR_TOP is mounted rotated +90° vs vehicle forward, so
    # camera-forward maps to LiDAR-frame azimuth ≈ +90°. Empirically confirmed
    # by diagnostic: in-FoV points have median atan2(y,x) ≈ 92° (range 60-123°).
    # CAM_FRONT FoV ≈ ±35° around forward → LiDAR-frame az ∈ [55°, 125°].
    FORWARD_AZ = 90.0  # LiDAR-frame azimuth of vehicle-forward
    AZ_BANDS = [
        ("L2  -35°..-17°", FORWARD_AZ - 35.0, FORWARD_AZ - 17.5),
        ("L1  -17°..  0°", FORWARD_AZ - 17.5, FORWARD_AZ),
        ("R1    0°.. +17°", FORWARD_AZ,        FORWARD_AZ + 17.5),
        ("R2  +17°..+35°",  FORWARD_AZ + 17.5, FORWARD_AZ + 35.0),
    ]
    N_AZ = len(AZ_BANDS)
    # Per cell: store DA-v2 depth, v (image row), and LiDAR range.
    # v is the geometric/perspective baseline (lower-in-image → closer).
    pairs_by_cell = {(r, a): {"da": [], "v": [], "lid": []}
                     for r in range(N_LATENT_ROWS) for a in range(N_AZ)}
    # Control: count outside-FoV LiDAR points (no camera coverage → no DA-v2 signal)
    n_outside_fov = 0
    n_outside_fov_per_ring = np.zeros(N_RINGS_RAW, dtype=np.int64)
    n_used_total = 0
    t0 = time.time()

    for i, tok in enumerate(picks):
        # Image
        cam_rec, cam_cs = cam_by_sample[tok]
        lid_rec, lid_cs = lid_by_sample[tok]
        img = Image.open(NUSCENES / cam_rec["filename"])
        W, H = img.size

        # DA-v2 depth (raw output) at native processor resolution → bilinear back to (H, W)
        inputs = proc(images=img, return_tensors="pt").to(args.device)
        with torch.no_grad():
            depth_pred = mdl(**inputs).predicted_depth        # [1, h, w]
        depth_full = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()                              # [H, W]

        # LiDAR + projection
        pts5 = load_nuscenes_lidar_bin(str(NUSCENES / lid_rec["filename"]))   # [N, 5]
        proj = project_lidar_to_image_with_ring(pts5, cam_cs, lid_cs, W, H)

        # Filter to ≤ max_range
        in_range = proj["range_lid"] < args.max_range
        u  = proj["u"][in_range].astype(np.int32)
        v  = proj["v"][in_range].astype(np.int32)
        rl = proj["range_lid"][in_range]
        rg = proj["ring"][in_range]

        # Sample DA-v2 depth at projected pixel
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)
        da_depth = depth_full[v, u]                            # [M], DA-v2 inverse-depth value
        v_norm = v.astype(np.float64) / H                       # [M], image row baseline ∈ [0,1]
        az_deg = proj["az_deg"][in_range]                      # LiDAR-frame azimuth

        # Bin into (latent_row, azimuth_band) cells
        for ring in range(N_RINGS_RAW):
            row = ring // 4   # 0..7
            ring_mask = (rg == ring)
            if not ring_mask.any():
                continue
            for a_idx, (_, az_lo, az_hi) in enumerate(AZ_BANDS):
                cell_mask = ring_mask & (az_deg >= az_lo) & (az_deg < az_hi)
                if cell_mask.any():
                    pairs_by_cell[(row, a_idx)]["da"].extend(da_depth[cell_mask].tolist())
                    pairs_by_cell[(row, a_idx)]["v"].extend(v_norm[cell_mask].tolist())
                    pairs_by_cell[(row, a_idx)]["lid"].extend(rl[cell_mask].tolist())
        n_used_total += len(u)

        # Outside-FoV control: count LiDAR points NOT projected into the image
        all_in_range = pts5[np.linalg.norm(pts5[:, :3], axis=1) < args.max_range]
        n_outside_fov += len(all_in_range) - len(u)
        # Per-ring outside count
        for ring in range(N_RINGS_RAW):
            total_ring = (all_in_range[:, 4] == ring).sum()
            in_fov_ring = (rg == ring).sum()
            n_outside_fov_per_ring[ring] += int(total_ring - in_fov_ring)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(picks)}] {time.time()-t0:.1f}s elapsed, "
                  f"{n_used_total:,} total pairs collected")

    print(f"\n  Total in-FoV pairs : {n_used_total:,}")
    print(f"  Outside-FoV LiDAR points (control, no DA-v2 signal): {n_outside_fov:,} "
          f"({n_outside_fov/(n_outside_fov+n_used_total)*100:.1f}% of total)")

    # ---- Build per-cell matrices for DA-v2, v-baseline, random-shuffle ----
    spear_mat   = np.full((N_LATENT_ROWS, N_AZ), np.nan)  # |r(DA-v2, LiDAR)|
    spear_v_mat = np.full((N_LATENT_ROWS, N_AZ), np.nan)  # |r(v_coord, LiDAR)| baseline
    spear_random_mat = np.full((N_LATENT_ROWS, N_AZ), np.nan)  # null
    n_mat       = np.zeros((N_LATENT_ROWS, N_AZ), dtype=np.int64)
    med_mat     = np.full((N_LATENT_ROWS, N_AZ), np.nan)
    unique_mat  = np.zeros((N_LATENT_ROWS, N_AZ), dtype=np.int64)
    lift_mat    = np.full((N_LATENT_ROWS, N_AZ), np.nan)  # DA-v2 lift over v-baseline

    import warnings
    rng_shuffle = np.random.RandomState(args.seed + 1)
    for (r, a), cell in pairs_by_cell.items():
        da  = np.array(cell["da"])
        v_b = np.array(cell["v"])
        lid = np.array(cell["lid"])
        n = len(da)
        n_mat[r, a] = n
        if n < 100:
            continue
        med_mat[r, a] = float(np.median(lid))
        unique_mat[r, a] = len(np.unique(da))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # v-baseline (always defined, just image row)
            s_v, _ = spearmanr(v_b, lid)
            if not np.isnan(s_v):
                spear_v_mat[r, a] = abs(s_v)
            # Random shuffle null (should be ~0)
            shuffled = da.copy()
            rng_shuffle.shuffle(shuffled)
            s_rand, _ = spearmanr(shuffled, lid)
            if not np.isnan(s_rand):
                spear_random_mat[r, a] = abs(s_rand)
            # DA-v2 (NaN if saturated)
            if unique_mat[r, a] >= 5:
                s_r, _ = spearmanr(da, lid)
                if not np.isnan(s_r):
                    spear_mat[r, a] = abs(s_r)
                    lift_mat[r, a] = abs(s_r) - (abs(s_v) if not np.isnan(s_v) else 0.0)

    # ---- Per-row aggregate (across all 4 az bands) for the headline ----
    per_row_da     = np.full(N_LATENT_ROWS, np.nan)
    per_row_v      = np.full(N_LATENT_ROWS, np.nan)
    per_row_random = np.full(N_LATENT_ROWS, np.nan)
    per_row_n      = np.zeros(N_LATENT_ROWS, dtype=np.int64)
    per_row_med    = np.full(N_LATENT_ROWS, np.nan)
    for r in range(N_LATENT_ROWS):
        all_da, all_v, all_lid = [], [], []
        for a in range(N_AZ):
            all_da.extend(pairs_by_cell[(r, a)]["da"])
            all_v.extend(pairs_by_cell[(r, a)]["v"])
            all_lid.extend(pairs_by_cell[(r, a)]["lid"])
        all_da, all_v, all_lid = np.array(all_da), np.array(all_v), np.array(all_lid)
        per_row_n[r] = len(all_da)
        if len(all_da) > 100:
            per_row_med[r] = float(np.median(all_lid))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # DA-v2
                if len(np.unique(all_da)) > 5:
                    s_da, _ = spearmanr(all_da, all_lid)
                    if not np.isnan(s_da): per_row_da[r] = abs(s_da)
                # v-baseline
                s_v, _ = spearmanr(all_v, all_lid)
                if not np.isnan(s_v): per_row_v[r] = abs(s_v)
                # Random shuffle null
                shuffled = all_da.copy()
                rng_shuffle.shuffle(shuffled)
                s_rand, _ = spearmanr(shuffled, all_lid)
                if not np.isnan(s_rand): per_row_random[r] = abs(s_rand)
    per_row_abs = per_row_da  # kept name for backward-compat in plotting below

    # ---- Print per-row summary with baselines ----
    print()
    print("=" * 86)
    print(f"Per-row |Spearman r|: DA-v2 vs v-baseline (image-row perspective) vs random null")
    print("=" * 86)
    print(f"{'row':>4} {'n_pairs':>10} {'med_range':>11}  "
          f"{'DA-v2':>8} {'v_base':>8} {'random':>8}  "
          f"{'DA lift':>9}  {'verdict':<8}")
    for r in range(N_LATENT_ROWS):
        s_da, s_v, s_rd = per_row_da[r], per_row_v[r], per_row_random[r]
        n, mr = per_row_n[r], per_row_med[r]
        def fmt(x): return f"{x:>8.3f}" if not np.isnan(x) else f"{'N/A':>8}"
        if np.isnan(s_da):
            verdict = "sat/no" if n > 0 else "no_data"
            lift = "N/A".rjust(9)
        else:
            lift_val = s_da - (s_v if not np.isnan(s_v) else 0.0)
            lift = f"{lift_val:>+9.3f}"
            if s_da > 0.6:    verdict = "STRONG"
            elif s_da > 0.3:  verdict = "modest"
            else:             verdict = "weak"
        mr_str = f"{mr:>9.1f}m" if not np.isnan(mr) else f"{'N/A':>10}"
        print(f"{r:>4} {n:>10} {mr_str:>11}  "
              f"{fmt(s_da)} {fmt(s_v)} {fmt(s_rd)}  {lift}  {verdict:<8}")

    # ---- Per-cell heatmap table ----
    print()
    print("=" * 86)
    print(f"|Spearman r| per cell (DA-v2 vs LiDAR), rows × azimuth bands")
    print("=" * 86)
    header = f"{'row':>4}  " + " ".join(f"{lbl:>15}" for lbl, _, _ in AZ_BANDS) + f"  {'row_all':>8}"
    print(header)
    for r in range(N_LATENT_ROWS):
        row_str = f"{r:>4}  "
        for a in range(N_AZ):
            v = spear_mat[r, a]; n = n_mat[r, a]
            if np.isnan(v):
                if n == 0: cell = "      -        "
                elif unique_mat[r, a] < 5: cell = f" sat (n={n:>5})"
                else: cell = f" N/A (n={n:>5})"
            else:
                cell = f"{v:>5.3f} (n={n:>5})"
            row_str += f"{cell:>15} "
        row_all = per_row_abs[r]
        row_str += f"  {row_all:>8.3f}" if not np.isnan(row_all) else f"  {'N/A':>8}"
        print(row_str)

    # ---- Outside-FoV control ----
    print()
    print("=" * 70)
    print("Outside-FoV control (LiDAR points with no camera coverage):")
    print("=" * 70)
    print(f"  Total outside-FoV pairs : {n_outside_fov:,} "
          f"({n_outside_fov/(n_outside_fov+n_used_total)*100:.1f}% of all ≤80m points)")
    print(f"  Per-ring outside-FoV counts (= C5 cannot help these):")
    for ring in range(0, N_RINGS_RAW, 4):
        block = n_outside_fov_per_ring[ring:ring+4].sum()
        block_in = sum(n_mat[ring//4, :])
        total = block + block_in
        pct = 100 * block / total if total > 0 else 0
        print(f"    latent row {ring//4} (rings {ring}-{ring+3}): "
              f"{block:>8,} outside / {block_in:>6,} inside  ({pct:.0f}% outside)")

    # ---- Headline ----
    print()
    print("=" * 70)
    print("Decision summary")
    print("=" * 70)
    top_band_da = [per_row_da[r] for r in [5, 6, 7] if not np.isnan(per_row_da[r])]
    top_band_v  = [per_row_v[r]  for r in [5, 6, 7] if not np.isnan(per_row_v[r])]
    top_band_rd = [per_row_random[r] for r in [5, 6, 7] if not np.isnan(per_row_random[r])]
    if top_band_da:
        m_da = float(np.mean(top_band_da))
        m_v  = float(np.mean(top_band_v)) if top_band_v else 0.0
        m_rd = float(np.mean(top_band_rd)) if top_band_rd else 0.0
        print(f"  Top-beam (rows 5-7) mean |Spearman|:")
        print(f"    DA-v2     : {m_da:.4f}  ← the test signal")
        print(f"    v-baseline: {m_v:.4f}  (image-row perspective alone)")
        print(f"    random    : {m_rd:.4f}  (null floor — should be ~0)")
        print(f"    LIFT (DA-v2 − v-baseline): {m_da - m_v:+.4f}  ← what C5 adds OVER perspective")
        if m_da > 0.6:    print(f"  → C5 HIGH-CONFIDENCE WIN. Commit to cache rebuild + 4-hr retrain.")
        elif m_da > 0.3:  print(f"  → C5 MODEST WIN. Worth doing but expect smaller gain than ideal.")
        else:             print(f"  → C5 RISK: DA-v2 doesn't track LiDAR depth on top beams.")
    # Best cell across all (row, az):
    valid = ~np.isnan(spear_mat)
    if valid.any():
        rr, aa = np.unravel_index(np.argmax(np.where(valid, spear_mat, -1)), spear_mat.shape)
        print(f"  Best single cell: row {rr}, {AZ_BANDS[aa][0].strip()} → |r|={spear_mat[rr,aa]:.3f}")

    # ---- Plot: heatmap ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    plot_mat = np.where(np.isnan(spear_mat), 0.0, spear_mat)
    im = ax.imshow(plot_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for r in range(N_LATENT_ROWS):
        for a in range(N_AZ):
            v = spear_mat[r, a]; n = n_mat[r, a]
            if np.isnan(v):
                txt = "-" if n == 0 else ("sat" if unique_mat[r,a] < 5 else "N/A")
            else:
                txt = f"{v:.2f}"
            ax.text(a, r, f"{txt}\nn={n}", ha="center", va="center",
                    fontsize=8, color="black" if (np.isnan(v) or v > 0.4) else "white")
    ax.set_xticks(range(N_AZ))
    ax.set_xticklabels([lbl.strip() for lbl, _, _ in AZ_BANDS], rotation=20, ha="right")
    ax.set_yticks(range(N_LATENT_ROWS))
    ax.set_yticklabels([f"row {r}" for r in range(N_LATENT_ROWS)])
    ax.set_xlabel("azimuth band (LiDAR-frame, CAM_FRONT FoV)")
    ax.set_ylabel("latent elevation row")
    ax.set_title(
        f"C5 viability — |Spearman r| between DA-v2 depth and LiDAR range\n"
        f"per (elevation row × azimuth band) cell, {len(picks)} samples, max_range={args.max_range:g} m",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="|Spearman r|")
    fig.tight_layout()
    heat_path = args.out_dir / "spearman_heatmap.png"
    fig.savefig(heat_path, dpi=120)
    plt.close(fig)

    # ---- Plot: scatter per row (pooled over az for compactness) ----
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for r in range(N_LATENT_ROWS):
        ax = axes[r // 4, r % 4]
        all_da, all_lid = [], []
        for a in range(N_AZ):
            all_da.extend(pairs_by_cell[(r, a)]["da"])
            all_lid.extend(pairs_by_cell[(r, a)]["lid"])
        all_da, all_lid = np.array(all_da), np.array(all_lid)
        if len(all_da) > 0:
            idx = np.random.choice(len(all_da), size=min(5000, len(all_da)), replace=False)
            ax.scatter(all_da[idx], all_lid[idx], s=2, alpha=0.3, c="C0")
        s = per_row_abs[r]
        if np.isnan(s):
            ttl_color = "gray"
            ttl = f"row {r}  n={len(all_da):,}\n|Spearman|=N/A"
        else:
            ttl_color = "tab:green" if s > 0.6 else ("tab:orange" if s > 0.3 else "tab:red")
            ttl = f"row {r}  n={len(all_da):,}\n|Spearman|={s:.3f}"
        ax.set_xlabel("DA-v2 raw output (inv-depth)", fontsize=8)
        ax.set_ylabel("LiDAR range (m)", fontsize=8)
        ax.set_title(ttl, color=ttl_color, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"C5 probe — DA-v2 depth vs LiDAR range, pooled per latent row "
        f"(front-FoV, {len(picks)} samples)", fontsize=12,
    )
    fig.tight_layout()
    scatter_path = args.out_dir / "scatter_per_row.png"
    fig.savefig(scatter_path, dpi=110)
    plt.close(fig)

    # ---- Save stats ----
    stats_path = args.out_dir / "stats.txt"
    with stats_path.open("w") as f:
        f.write(f"C5 viability probe results\n{'=' * 70}\n")
        f.write(f"n_samples: {len(picks)}, max_range: {args.max_range} m, total pairs: {n_used_total}\n\n")
        f.write(f"AZ_BANDS = {AZ_BANDS}\n\n")
        f.write(f"Per-cell |Spearman r| (- = no pairs, sat = DA-v2 saturated):\n")
        f.write(f"{'row':>4}  " + " ".join(f"{lbl:>15}" for lbl, _, _ in AZ_BANDS) + f"  {'row_all':>8}\n")
        for r in range(N_LATENT_ROWS):
            row_str = f"{r:>4}  "
            for a in range(N_AZ):
                v, n = spear_mat[r, a], n_mat[r, a]
                if np.isnan(v):
                    if n == 0:
                        cell = "      -        "
                    elif unique_mat[r, a] < 5:
                        cell = f" sat (n={n:>5})"
                    else:
                        cell = f" N/A (n={n:>5})"
                else:
                    cell = f"{v:>5.3f} (n={n:>5})"
                row_str += f"{cell:>15} "
            row_all = per_row_abs[r]
            row_str += f"  {row_all:>8.3f}\n" if not np.isnan(row_all) else f"  {'N/A':>8}\n"
            f.write(row_str)
    print(f"\n  saved heatmap   : {heat_path}")
    print(f"  saved scatter   : {scatter_path}")
    print(f"  saved stats     : {stats_path}")


if __name__ == "__main__":
    main()
