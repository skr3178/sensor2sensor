"""Waymo Open Dataset v2.0.1 LiDAR range-image decode -> 4-channel tensor.

Output format matches the paper's Waymo spec (range / intensity / elongation /
validity, 4 channels):

    [4, H=64, W=2048] in [0, 1]
    channels = (range_norm, intensity_norm, elongation_norm, validity)

Waymo v2 stores the range image in parquet rows as a flat float `values` list
plus a `shape` triple `(H, W, C)`. For the TOP laser (laser_name=1) that shape
is `(64, 2650, 4)` with native channels:

    0 = range_m              (0 means "no return")
    1 = intensity            (raw float, long-tailed; ~p99 < 1)
    2 = elongation           (waveform spread; ~p99 ≈ 1, max ≈ 1.5; discriminates
                              flat surfaces from foliage/vegetation)
    3 = is_in_no_label_zone  (always -1 for v2 train/val without box-supervised
                              pretraining; replaced here by a synthesized validity)

The 4th output channel is **validity** synthesized as `range_m > 0`, matching
how Waymo encodes "no return". Elongation is preserved as the 3rd channel — the
nuScenes HDL-32E doesn't measure it, but Waymo's Honeycomb LiDAR does, and the
paper's loss includes an L1_elongation term.
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
ELONGATION_MAX = 1.5        # p99 ≈ 1.04, max ≈ 1.5 from the empirical scan; the
                            # divisor matches intensity for consistent gradients.

# Waymo laser_name enum (https://github.com/waymo-research/waymo-open-dataset).
LASER_NAME_TOP = 1
LASER_NAME_FRONT = 2
LASER_NAME_SIDE_LEFT = 3
LASER_NAME_SIDE_RIGHT = 4
LASER_NAME_REAR = 5


def waymo_range_image_to_4ch(
    ri: np.ndarray,
    H_out: int = H_DEFAULT,
    W_out: int = W_DEFAULT,
    range_max_m: float = RANGE_MAX_M,
    intensity_max: float = INTENSITY_MAX,
    elongation_max: float = ELONGATION_MAX,
) -> np.ndarray:
    """Convert a Waymo `(H, W, 4)` range image into a `(4, H_out, W_out)` tensor in [0, 1].

    Args:
        ri: float32 array `(H, W, 4)` straight out of the parquet row,
            native channels `(range_m, intensity, elongation, no_label_zone)`.
        H_out, W_out: target spatial size. If smaller than the source, we
            center-crop; if equal, we pass through. Up-sampling is not
            supported here — feed the source at its native shape.
        range_max_m: clamp + divisor for the range channel.
        intensity_max: clamp + divisor for the intensity channel (long-tailed).
        elongation_max: clamp + divisor for the elongation channel.

    Returns:
        float32 `(4, H_out, W_out)` in [0, 1]:
            channel 0 — range / range_max_m              (0 in no-return cells)
            channel 1 — clip(intensity, 0, im_max) / im_max
            channel 2 — clip(elongation, 0, el_max) / el_max
            channel 3 — validity in {0, 1}
    """
    assert ri.ndim == 3 and ri.shape[-1] == 4, (
        f"expected (H, W, 4) range image, got {ri.shape}"
    )
    H, W, _ = ri.shape

    rng = ri[..., 0].astype(np.float32)
    intensity = ri[..., 1].astype(np.float32)
    elongation = ri[..., 2].astype(np.float32)

    # "No return" cells have range == 0 in Waymo's encoding.
    validity = (rng > 0.0).astype(np.float32)

    rng_n = np.clip(rng / range_max_m, 0.0, 1.0) * validity
    int_n = np.clip(intensity / intensity_max, 0.0, 1.0) * validity
    elong_n = np.clip(elongation / elongation_max, 0.0, 1.0) * validity

    img = np.stack([rng_n, int_n, elong_n, validity], axis=0)  # (4, H, W)

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
    values: "np.ndarray | list",
    shape: "list[int] | tuple[int, int, int]",
) -> np.ndarray:
    """Reshape one parquet row's flat values list back into `(H, W, 4)`."""
    arr = np.asarray(values, dtype=np.float32)
    return arr.reshape(tuple(shape))
