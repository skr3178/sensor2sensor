"""Waymo Open Dataset v2.0.1 LiDAR range-image decode -> 3-channel tensor.

Output format matches what the LiDAR VAE expects:
    [3, H=64, W=2048] in [0, 1]
    channels = (range_norm, intensity_norm, validity)

Waymo v2 stores the range image in parquet rows as a flat float `values` list
plus a `shape` triple `(H, W, C)`. For the TOP laser (laser_name=1) that
shape is `(64, 2650, 4)` with channels:
    0 = range_m            (0 means "no return")
    1 = intensity          (raw float, long-tailed; ~p99 < 1)
    2 = elongation         (dropped — paper figures show 3 channels for nuScenes
                            parity, and X-Drive's RangeLDM-style VAE doesn't use it)
    3 = is_in_no_label_zone (always -1 for v2 train/val without box-supervised
                             pretraining; not useful here)

Validity is synthesized as `range_m > 0`, matching how Waymo encodes "no return".
"""
from __future__ import annotations

import numpy as np

# Native Waymo TOP-laser range image (https://waymo.com/open/data/perception).
H_NATIVE = 64
W_NATIVE = 2650

# What we feed the VAE. 2048 is the closest multiple of 4 (the VAE downsamples
# /4 twice) below 2650; cropping centered preserves the forward arc and drops
# ~150 px from each side (~21° of the rear-facing azimuth).
H_DEFAULT = 64
W_DEFAULT = 2048

# Sensor + normalization knobs (the YAML can override these per dataset).
RANGE_MAX_M = 75.0          # Waymo TOP laser advertised max range; we clamp here.
INTENSITY_MAX = 1.5         # p99 ≈ 0.5; clipping at 1.5 retains 99.9 % of pixels
                            # while keeping the bright-retroreflector tail in.

# Waymo laser_name enum (https://github.com/waymo-research/waymo-open-dataset).
LASER_NAME_TOP = 1
LASER_NAME_FRONT = 2
LASER_NAME_SIDE_LEFT = 3
LASER_NAME_SIDE_RIGHT = 4
LASER_NAME_REAR = 5


def waymo_range_image_to_3ch(
    ri: np.ndarray,
    H_out: int = H_DEFAULT,
    W_out: int = W_DEFAULT,
    range_max_m: float = RANGE_MAX_M,
    intensity_max: float = INTENSITY_MAX,
) -> np.ndarray:
    """Convert a Waymo `(H, W, 4)` range image into a `(3, H_out, W_out)` tensor in [0, 1].

    Args:
        ri: float32 array `(H, W, 4)` straight out of the parquet row,
            channels `(range_m, intensity, elongation, no_label_zone)`.
        H_out, W_out: target spatial size. If smaller than the source, we
            center-crop; if equal, we pass through. Up-sampling is not
            supported here — feed the source at its native shape.
        range_max_m: clamp + divisor for the range channel.
        intensity_max: clamp + divisor for the intensity channel (long-tailed).

    Returns:
        float32 `(3, H_out, W_out)` in [0, 1]:
            channel 0 — range / range_max_m         (0 in no-return cells)
            channel 1 — clip(intensity, 0, im_max) / im_max
            channel 2 — validity in {0, 1}
    """
    assert ri.ndim == 3 and ri.shape[-1] == 4, (
        f"expected (H, W, 4) range image, got {ri.shape}"
    )
    H, W, _ = ri.shape

    rng = ri[..., 0].astype(np.float32)
    intensity = ri[..., 1].astype(np.float32)

    # "No return" cells have range == 0 in Waymo's encoding.
    validity = (rng > 0.0).astype(np.float32)

    rng_n = np.clip(rng / range_max_m, 0.0, 1.0)
    int_n = np.clip(intensity / intensity_max, 0.0, 1.0) * validity  # zero invalid cells
    rng_n = rng_n * validity

    img = np.stack([rng_n, int_n, validity], axis=0)  # (3, H, W)

    if H != H_out:
        assert H_out <= H, f"H_out={H_out} > source H={H} (up-sampling unsupported)"
        off = (H - H_out) // 2
        img = img[:, off : off + H_out, :]
    if W != W_out:
        assert W_out <= W, f"W_out={W_out} > source W={W} (up-sampling unsupported)"
        off = (W - W_out) // 2
        img = img[:, :, off : off + W_out]

    return np.ascontiguousarray(img)


def decode_lidar_row(
    values: np.ndarray | list,
    shape: list[int] | tuple[int, int, int],
) -> np.ndarray:
    """Reshape one parquet row's flat values list back into `(H, W, 4)`."""
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(tuple(shape))
