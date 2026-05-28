"""NuScenes keyframe loader returning LiDAR + all 6 surround cameras paired.

Single-pass metadata walk (no nuscenes-devkit) over `sample.json`,
`sample_data.json`, `calibrated_sensor.json`, `sensor.json`. For every sample
that has keyframes for both `LIDAR_TOP` and all 6 cameras in `CAMERA_ORDER`,
emit a dict combining:

  * LiDAR range image           [3, 32, 1024]   (range, intensity, validity)
  * 6 RGB tensors               [6, 3, 256, 448] in [-1, 1] (SD convention)
  * 6 scaled intrinsics         [6, 3, 3]       — K scaled to the 256x448 image
  * 6 camera-to-ego extrinsics  [6, 4, 4]
  * sample_token                str

Per-camera intrinsics differ (nuScenes surround cams have different focal
lengths and principal points). The K matrices returned here are pre-scaled
from native 1600x900 to the processed 256x448 image so they feed directly
into `models.raymap.build_raymap(downsample=8)`.

CAMERA_ORDER is fixed by this module and matches `configs/min.yaml -> nuscenes.cameras`.
The CrossViewFusion module is permutation-equivariant, but every downstream consumer
assumes this canonical ordering, so do not reshuffle it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.range_image import load_nuscenes_lidar_bin, point_cloud_to_range_image

# Fixed canonical ordering — used everywhere downstream.
CAMERA_ORDER: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
LIDAR_CHANNEL = "LIDAR_TOP"

# nuScenes native camera resolution. All surround cams share this.
NATIVE_W, NATIVE_H = 1600, 900
# Processed RGB resolution fed to the SD VAE.
IMG_W, IMG_H = 448, 256


def _quat_wxyz_to_rotmat(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _make_T(translation: list[float], rotation_quat_wxyz: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_wxyz_to_rotmat(rotation_quat_wxyz)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return T


def _scale_intrinsics(K_native: np.ndarray) -> np.ndarray:
    """Rescale K from native (1600x900) to processed (448x256). Same trick as
    `train/cache_latents.py:scale_intrinsics`."""
    K = K_native.astype(np.float32).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def _load_rgb_minus1_to_1(jpg_path: Path) -> torch.Tensor:
    img = Image.open(jpg_path).convert("RGB").resize((IMG_W, IMG_H), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


class NuScenesPairedKeyframes(Dataset):
    """Paired LiDAR + 6-camera keyframes from nuScenes v1.0-trainval.

    Args:
        nuscenes_root: path to the nuScenes root (contains samples/, v1.0-trainval/).
        scene_tokens:  optional list of scene tokens to restrict to. None → all scenes.
        cameras:       camera channel ordering. Defaults to CAMERA_ORDER. If you
                       override this you must update the CrossViewFusion view count.
    """

    def __init__(
        self,
        nuscenes_root: str | Path,
        scene_tokens: list[str] | None = None,
        cameras: tuple[str, ...] = CAMERA_ORDER,
    ):
        self.root = Path(nuscenes_root)
        self.cameras = tuple(cameras)

        meta_dir = self.root / "v1.0-trainval"
        sample = json.loads((meta_dir / "sample.json").read_text())
        sample_data = json.loads((meta_dir / "sample_data.json").read_text())
        cs_records = json.loads((meta_dir / "calibrated_sensor.json").read_text())
        sensor = json.loads((meta_dir / "sensor.json").read_text())

        sensor_by_token = {s["token"]: s for s in sensor}
        cs_by_token = {c["token"]: c for c in cs_records}
        channel_by_cs = {
            c_tok: sensor_by_token[c["sensor_token"]]["channel"]
            for c_tok, c in cs_by_token.items()
        }

        # Optionally filter samples by scene.
        if scene_tokens is not None:
            wanted_scenes = set(scene_tokens)
            samples_by_token = {
                s["token"]: s for s in sample if s["scene_token"] in wanted_scenes
            }
        else:
            samples_by_token = {s["token"]: s for s in sample}

        # Index sample_data records by (sample_token, channel) for keyframes only.
        # Each (sample_token, channel) pair has at most one keyframe.
        records: dict[tuple[str, str], dict] = {}
        for sd in sample_data:
            if not sd["is_key_frame"]:
                continue
            if sd["sample_token"] not in samples_by_token:
                continue
            chan = channel_by_cs[sd["calibrated_sensor_token"]]
            records[(sd["sample_token"], chan)] = sd

        # Keep only samples that have ALL needed channels.
        needed = (LIDAR_CHANNEL,) + self.cameras
        complete_tokens = [
            tok for tok in samples_by_token
            if all((tok, c) in records for c in needed)
        ]
        # Stable ordering by token for reproducibility.
        complete_tokens.sort()

        # Resolve per-sample records up front.
        self.entries: list[dict] = []
        for tok in complete_tokens:
            cam_recs = [records[(tok, c)] for c in self.cameras]
            lid_rec = records[(tok, LIDAR_CHANNEL)]
            self.entries.append({
                "sample_token": tok,
                "lid_filename": lid_rec["filename"],
                "cam_filenames": [r["filename"] for r in cam_recs],
                "cam_cs_tokens": [r["calibrated_sensor_token"] for r in cam_recs],
            })

        # Pre-extract per-camera intrinsics and extrinsics (sample-independent in
        # nuScenes — calibration only changes per scene-log, but we recompute per
        # sample anyway since the cost is negligible).
        self._cs_by_token = cs_by_token

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        # LiDAR.
        pc = load_nuscenes_lidar_bin(str(self.root / entry["lid_filename"]))
        range_img = torch.from_numpy(point_cloud_to_range_image(pc))   # [3, 32, 1024]

        # Cameras.
        cams = torch.stack(
            [_load_rgb_minus1_to_1(self.root / fn) for fn in entry["cam_filenames"]],
            dim=0,
        )  # [V, 3, IMG_H, IMG_W]

        # Per-camera scaled intrinsics + cam->ego extrinsics.
        K_list, T_list = [], []
        for cs_tok in entry["cam_cs_tokens"]:
            cs = self._cs_by_token[cs_tok]
            K_native = np.array(cs["camera_intrinsic"], dtype=np.float32)
            K_list.append(_scale_intrinsics(K_native))
            T_list.append(_make_T(cs["translation"], cs["rotation"]))
        cam_K = torch.from_numpy(np.stack(K_list, axis=0))             # [V, 3, 3]
        cam_T_cam2ego = torch.from_numpy(np.stack(T_list, axis=0))     # [V, 4, 4]

        return {
            "range_image":   range_img,
            "cams":          cams,
            "cam_K":         cam_K,
            "cam_T_cam2ego": cam_T_cam2ego,
            "sample_token":  entry["sample_token"],
        }


def load_subset_tokens(path: str | Path) -> list[str]:
    """Read a newline-delimited token list (output of scripts/select_subset.py)."""
    text = Path(path).read_text().strip()
    return [t for t in text.split("\n") if t]
