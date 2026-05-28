"""Minimal nuScenes LIDAR_TOP keyframe dataset (no nuscenes-devkit dep).

Reads the v1.0-trainval JSON metadata directly to enumerate LIDAR_TOP
keyframes for a chosen set of scenes, then returns each as a 3-channel
range image ready for the VAE.

For the M1 VAE we don't need paired cameras yet, so this loader is
LiDAR-only. The fuller `data/nuscenes_mini_paired.py` (M3) will add
CAM_FRONT + intrinsics/extrinsics on top of the same metadata walk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.range_image import load_nuscenes_lidar_bin, point_cloud_to_range_image

LIDAR_CHANNEL = "LIDAR_TOP"


class NuScenesLidarKeyframes(Dataset):
    """All LIDAR_TOP keyframes from a list of scene tokens.

    Args:
        nuscenes_root: path to the nuScenes root (contains samples/, sweeps/, v1.0-trainval/).
        scene_tokens:  optional list of scene tokens to restrict to. If None,
                       every keyframe in v1.0-trainval is returned.
        return_dict:   if True, returns `{"range_image": Tensor, "lidar_path": str}`;
                       if False (default), returns just the tensor.

    Each item is a `[3, 32, 1024]` float32 tensor in [0, 1].
    """

    def __init__(
        self,
        nuscenes_root: str | Path,
        scene_tokens: list[str] | None = None,
        return_dict: bool = False,
    ):
        self.root = Path(nuscenes_root)
        self.return_dict = return_dict

        meta_dir = self.root / "v1.0-trainval"
        sample = json.loads((meta_dir / "sample.json").read_text())
        sample_data = json.loads((meta_dir / "sample_data.json").read_text())
        calibrated_sensor = json.loads((meta_dir / "calibrated_sensor.json").read_text())
        sensor = json.loads((meta_dir / "sensor.json").read_text())

        sensor_channel_by_token = {s["token"]: s["channel"] for s in sensor}
        cs_to_channel = {
            c["token"]: sensor_channel_by_token[c["sensor_token"]]
            for c in calibrated_sensor
        }

        # All LIDAR_TOP keyframes, indexed by their parent sample token.
        lidar_by_sample = {
            sd["sample_token"]: sd
            for sd in sample_data
            if sd["is_key_frame"] and cs_to_channel[sd["calibrated_sensor_token"]] == LIDAR_CHANNEL
        }

        # Filter samples by scene if requested.
        if scene_tokens is not None:
            wanted = set(scene_tokens)
            samples = [s for s in sample if s["scene_token"] in wanted]
        else:
            samples = sample

        # Keep only samples that actually have a paired LIDAR keyframe.
        samples = [s for s in samples if s["token"] in lidar_by_sample]

        self.lidar_paths: list[str] = [
            str(self.root / lidar_by_sample[s["token"]]["filename"]) for s in samples
        ]

    def __len__(self) -> int:
        return len(self.lidar_paths)

    def __getitem__(self, idx: int):
        path = self.lidar_paths[idx]
        points = load_nuscenes_lidar_bin(path)
        img = point_cloud_to_range_image(points)
        tensor = torch.from_numpy(img)
        if self.return_dict:
            return {"range_image": tensor, "lidar_path": path}
        return tensor


def load_subset_tokens(path: str | Path) -> list[str]:
    """Read a newline-delimited token list (output of scripts/select_subset.py)."""
    text = Path(path).read_text().strip()
    return [t for t in text.split("\n") if t]
