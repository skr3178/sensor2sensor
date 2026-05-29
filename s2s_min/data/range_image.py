"""nuScenes LiDAR point cloud -> range image conversion.

Output format matches what the LiDAR VAE expects:
    [3, H=32, W=1024] in [0, 1]
    channels = (range_norm, intensity_norm, validity)

The 32-row layout uses the per-point `ring_index` recorded in the raw nuScenes
`.pcd.bin` file, which is the actual beam ID from the HDL-32E sensor (0..31).
This is preferable to binning elevation angles uniformly because the HDL-32E's
beams are not evenly spaced.
"""
from __future__ import annotations

import numpy as np

H_DEFAULT = 32
W_DEFAULT = 1024
RANGE_MAX_M = 100.0
INTENSITY_MAX = 255.0


def load_nuscenes_lidar_bin(path: str) -> np.ndarray:
    """Read a nuScenes `.pcd.bin` and return an `[N, 5]` array.

    Columns: (x, y, z, intensity, ring_index).
    """
    arr = np.fromfile(path, dtype=np.float32)
    assert arr.size % 5 == 0, f"Unexpected file size for {path}: {arr.size} floats"
    return arr.reshape(-1, 5)


def point_cloud_to_range_image(
    points: np.ndarray,
    H: int = H_DEFAULT,
    W: int = W_DEFAULT,
    range_max_m: float = RANGE_MAX_M,
    intensity_max: float = INTENSITY_MAX,
) -> np.ndarray:
    """Project a nuScenes LiDAR point cloud onto a 3-channel range image.

    Args:
        points: `[N, 5]` array `(x, y, z, intensity, ring_index)` in the sensor frame.
        H: number of elevation rows (32 for nuScenes HDL-32E).
        W: number of azimuth bins (default 1024 ≈ 0.35°/bin).
        range_max_m: clamp for range normalization (anything > this is dropped).
        intensity_max: nuScenes raw intensity is uint8 in [0, 255].

    Returns:
        `[3, H, W]` float32 image in [0, 1]:
            channel 0 — range / `range_max_m`
            channel 1 — intensity / `intensity_max`
            channel 2 — validity (1.0 where a return landed in this cell, 0.0 elsewhere)

        Cells without any return have all three channels set to 0.

    When multiple points fall in the same `(row, col)` cell, the **closest** one
    (smallest range) wins. This matches the standard range-image convention used
    by RangeLDM / X-Drive and avoids letting a far return shadow a near one.
    """
    assert points.ndim == 2 and points.shape[1] == 5, (
        f"Expected [N, 5] (x, y, z, intensity, ring_index), got {points.shape}"
    )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]
    ring = points[:, 4].astype(np.int32)

    r = np.sqrt(x * x + y * y + z * z)

    # Azimuth in [-pi, pi]; map to column [0, W). atan2 is continuous across 0/2pi.
    azimuth = np.arctan2(y, x)
    col = ((azimuth + np.pi) / (2.0 * np.pi) * W).astype(np.int32)
    col = np.mod(col, W)  # safety wrap, in case of float rounding at the seam

    # Filter: must have a valid beam ID and a finite, in-range distance.
    valid = (ring >= 0) & (ring < H) & (r > 0.0) & (r <= range_max_m)

    r = r[valid]
    intensity = intensity[valid]
    row = ring[valid]
    col = col[valid]

    range_norm = (r / range_max_m).astype(np.float32)
    intensity_norm = np.clip(intensity / intensity_max, 0.0, 1.0).astype(np.float32)

    img = np.zeros((3, H, W), dtype=np.float32)

    # Resolve collisions by writing the closest range last. NumPy vector assign
    # keeps the LAST write at a duplicate index, so sort by range descending.
    order = np.argsort(-r)
    rr = row[order]
    cc = col[order]
    img[0, rr, cc] = range_norm[order]
    img[1, rr, cc] = intensity_norm[order]
    img[2, rr, cc] = 1.0

    return img


def range_image_to_point_cloud(
    img: np.ndarray,
    H: int = H_DEFAULT,
    W: int = W_DEFAULT,
    range_max_m: float = RANGE_MAX_M,
    intensity_max: float = INTENSITY_MAX,
    elevations_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse projection: range image -> point cloud (lossy, used for sanity checks).

    Because the row index doesn't carry the actual beam elevation, we either need
    the HDL-32E beam-elevation table passed in, or we fall back to a uniform
    linear elevation in [-30.67°, +10.67°] (the HDL-32E FoV). The fallback is
    fine for an eyeball BEV plot but won't reproduce the original cloud exactly.

    Returns: `[M, 4]` array `(x, y, z, intensity)` for the valid cells only.
    """
    assert img.shape == (3, H, W), f"Expected [3, {H}, {W}], got {img.shape}"

    if elevations_deg is None:
        # HDL-32E: laser ID 0 is the BOTTOM beam (-30.67°), laser ID 31 the TOP (+10.67°).
        # `point_cloud_to_range_image` writes ring_index directly to `row`, so row 0 = laser 0 = bottom.
        # The previous direction (`linspace(10.67, -30.67)`) was reversed → unprojection garbled
        # elevation by up to ~41°, inflating round-trip Chamfer to ~6 m on a 32×1024 grid.
        elevations_deg = np.linspace(-30.67, 10.67, H, dtype=np.float32)

    mask = img[2] > 0.5
    rows, cols = np.nonzero(mask)
    r = img[0, rows, cols] * range_max_m
    intensity = img[1, rows, cols] * intensity_max

    elev = np.deg2rad(elevations_deg[rows])
    azim = (cols.astype(np.float32) / W) * 2.0 * np.pi - np.pi

    cos_e = np.cos(elev)
    x = r * cos_e * np.cos(azim)
    y = r * cos_e * np.sin(azim)
    z = r * np.sin(elev)

    return np.stack([x, y, z, intensity], axis=1)
