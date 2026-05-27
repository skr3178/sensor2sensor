"""Bidirectional Chamfer distance between two point clouds.

Pure-Python using `scipy.spatial.cKDTree` for k-NN — no CUDA extension build,
no extra deps. Suitable for the ~5k–30k-point clouds we generate in M4.

For comparison: LiDAR-Diffusion's `lidm/eval/metric_utils.py:compute_pairwise_cd`
uses a hand-built `chamfer3D.cu` CUDA op which is ~10× faster but requires
`python setup.py install` and matching CUDA versions. Overkill for our 4-sample
M4 demo.

CD definition (the "symmetric, mean-over-points" variant, matching LiDM / X-Drive):

    CD(A, B) = (1/|A|) Σ_{a ∈ A} min_{b ∈ B} ||a − b||  +
               (1/|B|) Σ_{b ∈ B} min_{a ∈ A} ||a − b||

Reported in meters (assuming input points are in the same metric frame).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(
    pc_a: np.ndarray,
    pc_b: np.ndarray,
    use_xy_only: bool = False,
) -> dict:
    """Bidirectional Chamfer distance.

    Args:
        pc_a: `[N, ≥3]` point cloud (uses first 3 dims, or first 2 if `use_xy_only`).
        pc_b: `[M, ≥3]` point cloud.
        use_xy_only: if True, compute BEV-plane Chamfer (ignores z). Useful when
                     the LiDAR VAE under-shoots elevation and you want to isolate
                     the planar geometry component.

    Returns:
        dict with keys:
            cd                 : float — bidirectional mean (meters)
            a_to_b_mean        : float — mean nearest-neighbour distance A→B
            b_to_a_mean        : float — mean nearest-neighbour distance B→A
            n_a, n_b           : int   — point counts
    """
    assert pc_a.ndim == 2 and pc_a.shape[1] >= 2, f"pc_a must be [N, ≥2], got {pc_a.shape}"
    assert pc_b.ndim == 2 and pc_b.shape[1] >= 2, f"pc_b must be [M, ≥2], got {pc_b.shape}"

    dims = 2 if use_xy_only else 3
    a = pc_a[:, :dims].astype(np.float64)
    b = pc_b[:, :dims].astype(np.float64)

    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            "cd": float("inf"),
            "a_to_b_mean": float("inf"),
            "b_to_a_mean": float("inf"),
            "n_a": int(a.shape[0]),
            "n_b": int(b.shape[0]),
        }

    tree_b = cKDTree(b)
    tree_a = cKDTree(a)

    # nearest-neighbour distance from each point in A to the closest in B (and vice versa).
    a_to_b, _ = tree_b.query(a, k=1)
    b_to_a, _ = tree_a.query(b, k=1)

    return {
        "cd": float(a_to_b.mean() + b_to_a.mean()),
        "a_to_b_mean": float(a_to_b.mean()),
        "b_to_a_mean": float(b_to_a.mean()),
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
    }
