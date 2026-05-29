"""Build a per-pixel raymap on the SD-VAE latent grid.

Geometric conditioning input for the LiDAR U-Net: at every spatial location of
the image latent (`32 × 56` for our 256×448 input), store the camera ray's
origin and unit direction in the ego (vehicle) frame.

Concatenated channel-wise into the KV context, this gives the U-Net a
geometry-aware signal of "which way is the camera looking at this latent pixel"
on top of the appearance signal from the image latent itself.

Math (per architecture.md §2):
    for each (u, v) in the H_latent × W_latent grid:
        pixel_h    = K'^-1 @ [u, v, 1]^T          # ray dir in camera frame
        pixel_h   /= ||pixel_h||                   # unit
        dir_ego    = T_cam2ego[:3,:3] @ pixel_h    # rotate into ego frame
    raymap[:, 0:3] = T_cam2ego[:3, 3]              # ray origin (broadcast)
    raymap[:, 3:6] = dir_ego

Critical correctness rule: K' is the **scaled** intrinsics matrix
`K' = diag(1/s, 1/s, 1) @ K` where `s` is the SD-VAE downsample factor (8).
We compute on the latent grid directly — never bilinearly resample a raymap.

Adapted from `Reference_code/light-field-networks/geometry.py:get_ray_directions`,
with batched support and the scaled-intrinsics trick added.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def build_raymap(
    K: torch.Tensor,
    T_cam2ego: torch.Tensor,
    H_latent: int,
    W_latent: int,
    downsample: int = 8,
) -> torch.Tensor:
    """Build a [B, 6, H_latent, W_latent] raymap.

    Args:
        K:           [B, 3, 3] camera intrinsics at the **native image resolution**
                     (the image the SD VAE was fed before its 8× downsample).
                     If [3, 3] is passed, an explicit batch dim of 1 is added.
        T_cam2ego:   [B, 4, 4] camera-to-ego transformation. Camera frame
                     convention: right-down-forward (x-right, y-down, z-forward),
                     matching nuScenes and OpenCV.
        H_latent:    height of the latent grid (e.g. 32).
        W_latent:    width of the latent grid (e.g. 56).
        downsample:  spatial compression factor of the encoder feeding this raymap
                     (8 for SD 1.5 VAE).

    Returns:
        [B, 6, H_latent, W_latent] float32. Channels:
            0:3 — ego-frame ray origin xyz (broadcast across spatial dims)
            3:6 — ego-frame unit direction xyz
    """
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if T_cam2ego.dim() == 2:
        T_cam2ego = T_cam2ego.unsqueeze(0)
    assert K.shape[-2:] == (3, 3), f"K must end in (3,3), got {K.shape}"
    assert T_cam2ego.shape[-2:] == (4, 4), f"T must end in (4,4), got {T_cam2ego.shape}"
    assert K.shape[0] == T_cam2ego.shape[0], "K and T_cam2ego batch dims must match"

    B = K.shape[0]
    device = K.device

    # Scaled intrinsics: K' = diag(1/s, 1/s, 1) @ K — equivalent to dividing
    # fx, fy, cx, cy by `s` while keeping the last row as [0, 0, 1].
    K_scaled = K.clone()
    K_scaled[:, 0, :] = K_scaled[:, 0, :] / downsample  # row 0 (fx, 0, cx)
    K_scaled[:, 1, :] = K_scaled[:, 1, :] / downsample  # row 1 (0, fy, cy)
    # row 2 (0, 0, 1) untouched.

    K_inv = torch.linalg.inv(K_scaled)  # [B, 3, 3]

    # Pixel grid at the latent resolution. Use sub-pixel centers (+ 0.5) so each
    # ray points at the center of its latent cell, not the corner.
    u = torch.arange(W_latent, device=device, dtype=torch.float32) + 0.5  # [W]
    v = torch.arange(H_latent, device=device, dtype=torch.float32) + 0.5  # [H]
    vv, uu = torch.meshgrid(v, u, indexing="ij")                          # [H, W] each
    ones = torch.ones_like(uu)
    uv1 = torch.stack([uu, vv, ones], dim=-1)                             # [H, W, 3]

    # Ray direction in camera frame: K^-1 @ [u, v, 1]^T per pixel.
    # Reshape to [H*W, 3, 1] and batch-matmul with K_inv [B, 3, 3].
    uv1_flat = uv1.reshape(-1, 3).T                                       # [3, H*W]
    dirs_cam = K_inv @ uv1_flat                                           # [B, 3, H*W]
    dirs_cam = dirs_cam.reshape(B, 3, H_latent, W_latent)                 # [B, 3, H, W]

    # Rotate camera-frame directions into ego frame.
    R = T_cam2ego[:, :3, :3]                                              # [B, 3, 3]
    dirs_ego = R @ dirs_cam.reshape(B, 3, -1)                             # [B, 3, H*W]
    dirs_ego = dirs_ego.reshape(B, 3, H_latent, W_latent)

    # Normalize to unit length AFTER the rotation (rotation preserves length,
    # so doing it before/after is equivalent, but doing it last is conventional).
    dirs_ego = dirs_ego / dirs_ego.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Origin: shared across all pixels of a sample. Broadcast to [B, 3, H, W].
    origin = T_cam2ego[:, :3, 3].view(B, 3, 1, 1).expand(B, 3, H_latent, W_latent)

    raymap = torch.cat([origin, dirs_ego], dim=1)  # [B, 6, H, W]
    return raymap.contiguous()
