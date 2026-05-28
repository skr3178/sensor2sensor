"""LiDAR VAE training loss (M1).

Full loss (paper Eq. 1, full 4-channel form for Waymo; the 3-channel nuScenes
subset drops the elongation term):

    L_VAE =  lam_range            * L1_masked(x_range,       x_hat_range,       mask=x_validity)
          +  lam_intensity        * L1_masked(x_intensity,   x_hat_intensity,   mask=x_validity)
          +  lam_elongation       * L1_masked(x_elongation,  x_hat_elongation,  mask=x_validity)   # Waymo only
          +  lam_validity         * BCE(x_hat_validity, x_validity)
          +  lam_lpips_normals    * LPIPS(normals(x_range),     normals(x_hat_range))
          +  lam_lpips_intensity  * LPIPS(x_intensity,  x_hat_intensity)
          +  lam_lpips_validity   * LPIPS(x_validity,   x_hat_validity)
          +  lam_kl               * 0.5 * mean(mu^2 + sigma^2 - log sigma^2 - 1)

Channel order is inferred from `x.shape[1]`:
    3 channels (nuScenes, from `data/range_image.py`)
        0 = range, 1 = intensity, 2 = validity
    4 channels (Waymo, from `data/waymo_range_image.py`)
        0 = range, 1 = intensity, 2 = elongation, 3 = validity

`lam_elongation` is ignored for the 3-channel path (nuScenes HDL-32E does not
measure elongation).

Validity-masked L1: range and intensity errors are computed only at cells where
the ground-truth validity is 1. Invalid cells carry no LiDAR return so any value
there is meaningless; if we let the L1 see them the decoder would learn to
predict 0 at those cells to drive the unmasked loss down, smearing real returns.

LPIPS terms (range-derived normals, intensity, validity) match paper Eq. (5)(6):
perceptual similarity between predicted and ground-truth signal maps under a
frozen VGG-16 trunk. Inputs are mapped to [-1, 1] as the LPIPS package expects;
single-channel signals (intensity, validity) are replicated 3× to satisfy VGG's
3-channel input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normal-map estimation from a normalized range image
# ---------------------------------------------------------------------------
# HDL-32E nominal elevation FoV. Same default as data.range_image.
_HDL32E_TOP_DEG = 10.67
_HDL32E_BOT_DEG = -30.67
_RANGE_CLAMP_M = 100.0


def _default_elevations(H: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fallback per-row elevation table (radians). Replace with the real beam
    inclination table for exact normals; the linear fallback is good enough
    for a perceptual loss."""
    return torch.linspace(
        math.radians(_HDL32E_TOP_DEG), math.radians(_HDL32E_BOT_DEG),
        H, device=device, dtype=dtype,
    )


def range_image_to_normals(
    range_norm: torch.Tensor,
    range_clamp_m: float = _RANGE_CLAMP_M,
    elevations_rad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project a normalized range image to 3D and compute per-pixel unit normals.

    Args:
        range_norm: `[B, 1, H, W]` or `[B, H, W]` in [0, 1] (×range_clamp_m → m).
        range_clamp_m: same constant used at normalization time.
        elevations_rad: optional `[H]` tensor of beam-elevation angles in radians.
            If None, uses an equispaced HDL-32E proxy (10.67° → -30.67°).

    Returns:
        normals `[B, 3, H, W]`, unit length per pixel, values in [-1, 1].

    Notes:
        - Finite differences with **circular wrap on W** (azimuth periodic) and
          **replicate on H** (elevation non-periodic) — matches the same padding
          convention used by `CircularConv2d` elsewhere in the model.
        - Where the range gradient is degenerate (e.g. consecutive cells both 0)
          the normal magnitude underflows; we floor the denominator at 1e-6 so
          the unit vector falls back to the cross-product direction without NaN.
    """
    if range_norm.dim() == 4:
        # Expected [B, 1, H, W] — squeeze the channel dim.
        assert range_norm.shape[1] == 1, f"expected 1 channel, got {range_norm.shape[1]}"
        range_norm = range_norm.squeeze(1)
    B, H, W = range_norm.shape
    device, dtype = range_norm.device, range_norm.dtype

    range_m = range_norm * range_clamp_m                                # [B, H, W]

    if elevations_rad is None:
        elevations_rad = _default_elevations(H, device, dtype)
    elev = elevations_rad.view(1, H, 1)                                 # [1, H, 1]

    # Azimuth bin centers in [-π, π).
    azim = (torch.arange(W, device=device, dtype=dtype) / W) * (2 * math.pi) - math.pi
    azim = azim.view(1, 1, W)

    cos_e = torch.cos(elev)
    sin_e = torch.sin(elev)
    cos_a = torch.cos(azim)
    sin_a = torch.sin(azim)

    x = range_m * cos_e * cos_a                                          # [B, H, W]
    y = range_m * cos_e * sin_a
    z = range_m * sin_e.expand_as(range_m)
    p = torch.stack([x, y, z], dim=1)                                    # [B, 3, H, W]

    # ΔW with circular wrap on the W axis.
    p_w_right = torch.roll(p, shifts=-1, dims=-1)
    p_w_left  = torch.roll(p, shifts=+1, dims=-1)
    du = p_w_right - p_w_left                                            # [B, 3, H, W]

    # ΔH with edge replication on H (no wrap — elevation is non-periodic).
    p_padded = F.pad(p, (0, 0, 1, 1), mode="replicate")                  # [B, 3, H+2, W]
    p_h_down = p_padded[:, :, 2:, :]
    p_h_up   = p_padded[:, :, :-2, :]
    dv = p_h_down - p_h_up                                               # [B, 3, H, W]

    # Right-handed cross product n = du × dv.
    n = torch.cross(du, dv, dim=1)

    # Normalize to unit length, clamping the denominator to avoid NaN at edges.
    norm = n.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
    return n / norm                                                       # [B, 3, H, W] in [-1, 1]


# ---------------------------------------------------------------------------
# LPIPS helpers
# ---------------------------------------------------------------------------

def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    """LPIPS expects inputs in [-1, 1] with 3 channels.

    - Single-channel inputs (intensity, validity) are replicated 3×.
    - 3-channel inputs (normals) are passed through unchanged (already in [-1, 1]).
    - Other 1-channel sources in [0, 1] are mapped via `2x - 1`.
    """
    assert x.dim() == 4, f"expected [B, C, H, W], got {x.shape}"
    C = x.shape[1]
    if C == 1:
        # Map [0, 1] → [-1, 1] then replicate to 3 channels.
        x = x * 2.0 - 1.0
        return x.expand(-1, 3, -1, -1)
    if C == 3:
        # Normals are already in [-1, 1] by construction (unit vectors).
        return x
    raise ValueError(f"unsupported channel count for LPIPS input: {C}")


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------

def lidar_vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lam_range: float = 1.0,
    lam_intensity: float = 1.0,
    lam_elongation: float = 1.0,
    lam_validity: float = 1.0,
    lam_kl: float = 1e-6,
    lam_lpips_normals: float = 0.0,
    lam_lpips_intensity: float = 0.0,
    lam_lpips_validity: float = 0.0,
    lpips_module: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """4-8-term VAE loss; returns a dict with `total` and per-term scalars.

    The number of L1 reconstruction terms is inferred from the channel count
    of `x`: 3 → nuScenes (range/intensity/validity); 4 → Waymo
    (range/intensity/elongation/validity). LPIPS terms are skipped entirely
    (no VGG forward pass) if their λ is 0.

    Per-term entries are `.detach()`ed so callers can log them without holding
    the autograd graph.
    """
    assert x.shape == x_hat.shape, f"shape mismatch: {x.shape} vs {x_hat.shape}"
    C = x.shape[1]
    assert C in (3, 4), (
        f"expected 3 (nuScenes) or 4 (Waymo) channels, got {C}"
    )

    x_range, x_int = x[:, 0:1], x[:, 1:2]
    h_range, h_int = x_hat[:, 0:1], x_hat[:, 1:2]
    if C == 4:
        x_elong, x_valid = x[:, 2:3], x[:, 3:4]
        h_elong, h_valid = x_hat[:, 2:3], x_hat[:, 3:4]
    else:
        x_elong = h_elong = None
        x_valid = x[:, 2:3]
        h_valid = x_hat[:, 2:3]

    mask = x_valid
    denom = mask.sum().clamp(min=1.0)

    # ---- reconstruction terms (paper Eq. 3) ----
    loss_range = (mask * (x_range - h_range).abs()).sum() / denom
    loss_intensity = (mask * (x_int - h_int).abs()).sum() / denom
    if x_elong is not None:
        loss_elong = (mask * (x_elong - h_elong).abs()).sum() / denom
    else:
        loss_elong = None

    # ---- BCE on validity (paper Eq. 4) ----
    # Decoder head output already went through sigmoid. F.binary_cross_entropy
    # itself is autocast-unsafe (log(0) at saturated sigmoid). Force fp32 + disable
    # autocast around the BCE call.
    with torch.cuda.amp.autocast(enabled=False):
        loss_validity = F.binary_cross_entropy(h_valid.float(), x_valid.float())

    # ---- KL (paper Eq. 7) ----
    loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()

    out = {
        "total": None,                              # filled in below
        "L1_range": loss_range.detach(),
        "L1_intensity": loss_intensity.detach(),
        "BCE_validity": loss_validity.detach(),
        "KL": loss_kl.detach(),
    }

    total = (
        lam_range * loss_range
        + lam_intensity * loss_intensity
        + lam_validity * loss_validity
        + lam_kl * loss_kl
    )
    if loss_elong is not None:
        out["L1_elongation"] = loss_elong.detach()
        total = total + lam_elongation * loss_elong

    # ---- LPIPS terms (paper Eqs. 5, 6) — only computed when enabled ----
    use_lpips = (
        lam_lpips_normals + lam_lpips_intensity + lam_lpips_validity
    ) > 0.0
    if use_lpips:
        assert lpips_module is not None, "lpips_module required when any lam_lpips_* > 0"
        # LPIPS-VGG is fp32-only; run outside autocast for stability.
        with torch.cuda.amp.autocast(enabled=False):
            if lam_lpips_normals > 0:
                n_gt   = range_image_to_normals(x_range.float())
                n_hat  = range_image_to_normals(h_range.float())
                loss_lpips_normals = lpips_module(_to_lpips_input(n_gt),
                                                  _to_lpips_input(n_hat)).mean()
                total = total + lam_lpips_normals * loss_lpips_normals
                out["LPIPS_normals"] = loss_lpips_normals.detach()

            if lam_lpips_intensity > 0:
                loss_lpips_intensity = lpips_module(_to_lpips_input(x_int.float()),
                                                    _to_lpips_input(h_int.float())).mean()
                total = total + lam_lpips_intensity * loss_lpips_intensity
                out["LPIPS_intensity"] = loss_lpips_intensity.detach()

            if lam_lpips_validity > 0:
                loss_lpips_validity = lpips_module(_to_lpips_input(x_valid.float()),
                                                    _to_lpips_input(h_valid.float())).mean()
                total = total + lam_lpips_validity * loss_lpips_validity
                out["LPIPS_validity"] = loss_lpips_validity.detach()

    out["total"] = total
    return out
