"""LiDAR VAE training loss (M1).

4-term objective on 3-channel nuScenes range images:
    L_VAE = lam_range    * L1_masked(x_range,     x_hat_range,     mask=x_validity)
          + lam_intensity * L1_masked(x_intensity, x_hat_intensity, mask=x_validity)
          + lam_validity  * BCE(x_hat_validity, x_validity)
          + lam_kl        * 0.5 * mean(mu^2 + sigma^2 - log sigma^2 - 1)

Channel order is hard-coded to match `data/range_image.py`:
    0 = range (normalized to [0, 1])
    1 = intensity (normalized to [0, 1])
    2 = validity (binary {0, 1})

Validity-masked L1: range and intensity errors are computed only at cells where
the ground-truth validity is 1. Invalid cells carry no LiDAR return so any value
there is meaningless; if we let the L1 see them the decoder would learn to
predict 0 at those cells to drive the unmasked loss down, smearing real returns.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lidar_vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lam_range: float = 1.0,
    lam_intensity: float = 1.0,
    lam_validity: float = 1.0,
    lam_kl: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """4-term VAE loss; returns a dict with `total` and per-term scalars.

    Per-term entries are `.detach()`ed so callers can log them without holding
    the autograd graph.
    """
    assert x.shape == x_hat.shape, f"shape mismatch: {x.shape} vs {x_hat.shape}"
    assert x.shape[1] == 3, f"expected 3 channels (range/intensity/validity), got {x.shape[1]}"

    x_range, x_int, x_valid = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    h_range, h_int, h_valid = x_hat[:, 0:1], x_hat[:, 1:2], x_hat[:, 2:3]

    mask = x_valid
    denom = mask.sum().clamp(min=1.0)

    loss_range = (mask * (x_range - h_range).abs()).sum() / denom
    loss_intensity = (mask * (x_int - h_int).abs()).sum() / denom

    # Decoder head output already went through sigmoid. F.binary_cross_entropy
    # itself is flagged as autocast-unsafe (it would log(0) at saturated
    # sigmoid outputs). Force fp32 + disable autocast around the BCE call.
    # We can't switch to BCEWithLogits because the sigmoid lives inside the decoder.
    with torch.cuda.amp.autocast(enabled=False):
        loss_validity = F.binary_cross_entropy(h_valid.float(), x_valid.float())

    loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()

    total = (
        lam_range * loss_range
        + lam_intensity * loss_intensity
        + lam_validity * loss_validity
        + lam_kl * loss_kl
    )

    return {
        "total": total,
        "L1_range": loss_range.detach(),
        "L1_intensity": loss_intensity.detach(),
        "BCE_validity": loss_validity.detach(),
        "KL": loss_kl.detach(),
    }
