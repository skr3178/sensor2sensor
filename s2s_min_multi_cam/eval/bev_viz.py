"""Top-down BEV (Bird's-Eye-View) scatter rendering.

Two functions:
  - `bev_scatter(ax, pc, ...)`              — draw one point cloud onto a matplotlib axis
  - `side_by_side_bev(pc_gt, pc_pred, ...)` — draw two side-by-side panels with a title

Pattern lifted from our own `scripts/visualize_lidar_vae.py:_bev_from_pc` (which
itself echoes the LiDAR-Diffusion sample.py viz conventions).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bev_scatter(
    ax,
    pc: np.ndarray,
    color: str = "tab:blue",
    label: str | None = None,
    range_m: float = 60.0,
    point_size: float = 0.10,
    alpha: float = 0.5,
) -> None:
    """Draw a single point cloud as a top-down scatter on the given axis.

    Args:
        ax:        matplotlib Axes.
        pc:        `[N, ≥2]` point cloud — only x,y columns are used.
        color:     matplotlib colour spec.
        label:     optional legend label.
        range_m:   axis half-extent in meters. Default 60 m (matches our raymap viz).
        point_size: matplotlib `s` parameter.
        alpha:     point alpha for blending overlap.
    """
    if pc.shape[0] == 0:
        return
    x, y = pc[:, 0], pc[:, 1]
    ax.scatter(x, y, s=point_size, c=color, alpha=alpha, label=label, linewidths=0)
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def side_by_side_bev(
    pc_gt: np.ndarray,
    pc_pred: np.ndarray,
    title: str = "",
    out_path: str | Path | None = None,
    range_m: float = 60.0,
) -> matplotlib.figure.Figure:
    """Render a 1×2 panel (ground truth | predicted) BEV.

    Args:
        pc_gt:    `[N, ≥2]` ground-truth point cloud.
        pc_pred:  `[M, ≥2]` model-predicted point cloud.
        title:    suptitle (e.g. sample token + per-sample metrics).
        out_path: if set, save the figure here. Returns the figure regardless.
        range_m:  axis half-extent.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    bev_scatter(axes[0], pc_gt,   color="tab:blue", range_m=range_m)
    bev_scatter(axes[1], pc_pred, color="tab:red",  range_m=range_m)
    axes[0].set_title(f"ground truth  ({len(pc_gt)} pts)", fontsize=9)
    axes[1].set_title(f"DDIM predicted ({len(pc_pred)} pts)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94] if title else None)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig
