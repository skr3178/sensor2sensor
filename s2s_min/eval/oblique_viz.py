"""3D oblique perspective renderer for LiDAR point clouds.

Matches the visualization style of the Sensor2Sensor paper's Figure 13 / 13a / 13b:
camera positioned behind and above the LiDAR sensor, looking forward; points
colored by height (z) with a perceptually-uniform colormap; black background.

Sibling to `bev_viz.py` (top-down) — same API shape so they're interchangeable.

Why not `mpl_toolkits.mplot3d`? Matplotlib's 3D backend is too slow on the
~32 k-point clouds M4 produces (~3–5 s per panel) and gives no fine-grained
control over depth-dependent point size. We project manually in numpy
(~10 ms for 32 k points) and scatter-plot the 2D result.

Frame convention: input point clouds are in the nuScenes LIDAR_TOP sensor
frame (+X forward, +Y left, +Z up), matching what
`s2s_min/data/range_image.py:range_image_to_point_cloud` produces. The default
camera (eye, target, up) renders that frame as a chase-cam view looking forward.
Override (eye, target, up) for other frames.

Two entry points:
  - `oblique_scatter(ax, pc, ...)`              — draw one cloud onto a matplotlib Axes
  - `side_by_side_oblique(pcs, titles, ...)`    — multi-panel layout matching the paper figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 world→camera matrix (OpenGL convention: camera looks down −Z in its own frame)."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)

    R = np.stack([right, new_up, -forward], axis=0)  # rows = camera basis in world
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M


def _project(
    pc_xyz: np.ndarray,
    view: np.ndarray,
    fov_deg: float,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project N points → 2D (NDC-like) coords + camera-frame depth.

    Returns:
        uv:    [N, 2] in roughly [-1, 1] × [-1, 1] (aspect-corrected on u).
        depth: [N]    positive = in front of camera.
    """
    homog = np.hstack([pc_xyz[:, :3], np.ones((pc_xyz.shape[0], 1))])  # [N, 4]
    cam = (view @ homog.T).T[:, :3]                                    # [N, 3]
    depth = -cam[:, 2]                                                 # OpenGL: forward = −Z
    f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
    safe = np.where(depth > 1e-6, depth, np.nan)
    u = (cam[:, 0] * f / safe) / aspect
    v = (cam[:, 1] * f / safe)
    return np.stack([u, v], axis=1), depth


def oblique_scatter(
    ax: "matplotlib.axes.Axes",
    pc: np.ndarray,
    *,
    color_by: str = "z",
    cmap: str = "viridis",
    eye: tuple[float, float, float] = (-15.0, 0.0, 6.0),
    target: tuple[float, float, float] = (25.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    fov_deg: float = 60.0,
    aspect: float = 1.5,
    point_size: float = 0.5,
    alpha: float = 0.7,
    z_clip: tuple[float, float] = (-3.0, 6.0),
    depth_range: tuple[float, float] = (1.0, 80.0),
) -> None:
    """Project a 3D point cloud onto a 2D oblique view and scatter-plot it.

    Args:
        ax:           matplotlib Axes (its facecolor is forced to black).
        pc:           `[N, ≥3]` point cloud (uses x, y, z; ignores intensity).
        color_by:     `"z"` colors by height; `"depth"` colors by camera-frame depth.
        cmap:         matplotlib colormap. `viridis` ≈ paper-style; try `plasma` / `turbo`.
        eye/target:   camera position and look-at point, in the point-cloud world frame.
                      Default: behind and above the LiDAR origin, looking forward along +X.
        up:           up vector. Default +Z (nuScenes LIDAR_TOP convention).
        fov_deg:      vertical field-of-view.
        aspect:       image-plane width / height.
        point_size:   base size; closer points are drawn larger (1/depth scaling).
        alpha:        per-point alpha.
        z_clip:       (min, max) for color when `color_by="z"` — stable colormap across
                      panels even if individual clouds have different z extents.
        depth_range:  cull points outside (near, far) meters in camera frame.
    """
    if pc.shape[0] == 0:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        return

    view = _look_at(np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float))
    uv, depth = _project(pc, view, fov_deg, aspect)

    keep = (
        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        & (depth >= depth_range[0]) & (depth <= depth_range[1])
        & (np.abs(uv[:, 0]) <= 1.0) & (np.abs(uv[:, 1]) <= 1.0)
    )
    uv = uv[keep]
    if uv.shape[0] == 0:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        return

    d = depth[keep]
    if color_by == "z":
        c = np.clip(pc[keep, 2], z_clip[0], z_clip[1])
    else:
        c = d

    # near points larger — gives the perspective "depth" feel of the paper figures
    sizes = point_size * np.clip(depth_range[1] / np.maximum(d, depth_range[0]), 0.3, 5.0)

    ax.scatter(uv[:, 0], uv[:, 1], s=sizes, c=c, cmap=cmap,
               alpha=alpha, linewidths=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([]); ax.set_yticks([])


def side_by_side_oblique(
    pcs: list[np.ndarray],
    titles: list[str],
    *,
    suptitle: str = "",
    out_path: str | Path | None = None,
    **scatter_kwargs,
) -> "matplotlib.figure.Figure":
    """Render N point clouds side-by-side in 3D oblique view (paper Figure 13 layout).

    Args:
        pcs:      list of `[N_i, ≥3]` point clouds.
        titles:   one short caption per panel (e.g. ["raw nuScenes", "VAE oracle", "DDIM pred"]).
        suptitle: optional figure-level title (drawn in white over the black background).
        out_path: if set, save figure here (parent dirs created).
        **scatter_kwargs: forwarded to `oblique_scatter` (eye, target, fov_deg, ...).
    """
    n = len(pcs)
    assert len(titles) == n, "len(titles) must match len(pcs)"
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), facecolor="black")
    if n == 1:
        axes = [axes]
    for ax, pc, title in zip(axes, pcs, titles):
        oblique_scatter(ax, pc, **scatter_kwargs)
        ax.set_title(title, fontsize=10, color="white")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color="white")
    fig.tight_layout(rect=[0, 0, 1, 0.94] if suptitle else None)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    return fig


if __name__ == "__main__":
    from eval.runfolder import new_run_folder

    # Smoke test: synthetic ground plane + vehicle box → confirms axis orientation
    # and that the default camera is pointed forward (+X).
    rng = np.random.default_rng(0)
    ground   = rng.uniform(low=[-30,  -20, -2.0], high=[60,  20, -1.7], size=(12000, 3))
    car_box  = rng.uniform(low=[ 15,  -1.0, -1.5], high=[18,  1.0,  0.5], size=( 1500, 3))
    building = rng.uniform(low=[ 30,    8,  -1.5], high=[55,  15,  4.0], size=( 3000, 3))
    pc = np.concatenate([ground, car_box, building], axis=0)
    out_dir = new_run_folder("oblique-viz-smoke")
    out = out_dir / "synthetic.png"
    side_by_side_oblique(
        [pc, pc, pc],
        ["raw", "oracle", "DDIM pred"],
        suptitle="oblique_viz smoke test (synthetic ground + car + building)",
        out_path=out,
    )
    print(f"wrote {out}")
