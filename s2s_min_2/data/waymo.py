"""Minimal Waymo Open Dataset v2.0.1 TOP-LiDAR keyframe dataset.

Reads the `lidar` parquet component per segment, filters to the TOP laser
(`key.laser_name == 1`), decodes the per-row range image, and returns a
`[4, H, W]` tensor in [0, 1] ready for the VAE
(channels = range, intensity, elongation, validity).

Layout expected on disk (matches what `scripts/download_waymo_samples.sh` writes):

    {waymo_root}/{split}/lidar/{segment_id}.parquet

Each parquet holds 5 lasers × ~198 frames; we use only the 198 TOP rows per
file. With 20 training segments that's ~3,960 training keyframes — comparable
to the nuScenes 10-scene M1 subset (~401 keyframes) but ~10x larger.
"""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from data.waymo_range_image import (
    H_DEFAULT,
    W_DEFAULT,
    LASER_NAME_TOP,
    RANGE_MAX_M,
    INTENSITY_MAX,
    ELONGATION_MAX,
    decode_lidar_row,
    waymo_range_image_to_4ch,
)

# Parquet column names — long, so we alias them once.
_COL_LASER = "key.laser_name"
_COL_TS = "key.frame_timestamp_micros"
_COL_VALUES = "[LiDARComponent].range_image_return1.values"
_COL_SHAPE = "[LiDARComponent].range_image_return1.shape"


class WaymoLidarTopKeyframes(Dataset):
    """All TOP-LiDAR keyframes from a Waymo split (return-1 range image).

    Args:
        waymo_root: path to the Waymo v2 root (contains `training/`, `validation/`).
        split: which split to load (`"training"` or `"validation"`).
        H_out, W_out: target output shape. Defaults to (64, 2048).
        range_max_m, intensity_max, elongation_max: normalization constants
            forwarded to `waymo_range_image_to_4ch`.
        segments: optional list of segment IDs (basename without `.parquet`)
            to restrict to. If `None`, every parquet under
            `{waymo_root}/{split}/lidar/` is used.
        cache_size: how many parquet tables to keep in RAM (LRU). Each
            top-laser table is ~512 MiB after column-pruning + filter +
            combine, so the default 2 budgets ~1 GiB per DataLoader worker.
            Lower to 1 if you have many workers; raise if RAM is plentiful.
        return_dict: if True returns `{"range_image": Tensor, "segment": str,
            "timestamp": int}`; otherwise just the tensor.

    Each item is a `[4, H_out, W_out]` float32 tensor in [0, 1].
    """

    def __init__(
        self,
        waymo_root: str | Path,
        split: str = "training",
        H_out: int = H_DEFAULT,
        W_out: int = W_DEFAULT,
        range_max_m: float = RANGE_MAX_M,
        intensity_max: float = INTENSITY_MAX,
        elongation_max: float = ELONGATION_MAX,
        segments: List[str] | None = None,
        cache_size: int = 1,
        return_dict: bool = False,
    ):
        self.root = Path(waymo_root)
        self.split = split
        self.H_out = H_out
        self.W_out = W_out
        self.range_max_m = range_max_m
        self.intensity_max = intensity_max
        self.elongation_max = elongation_max
        self.return_dict = return_dict
        self._cache_size = cache_size

        lidar_dir = self.root / split / "lidar"
        if not lidar_dir.is_dir():
            raise FileNotFoundError(
                f"Waymo lidar dir not found at {lidar_dir}. "
                f"Run scripts/download_waymo_samples.sh first."
            )

        files = sorted(p for p in lidar_dir.iterdir() if p.suffix == ".parquet")
        if segments is not None:
            wanted = set(segments)
            files = [f for f in files if f.stem in wanted]
        if not files:
            raise FileNotFoundError(
                f"No matching .parquet files found under {lidar_dir} "
                f"(segments filter={segments})"
            )

        # Build the (file_idx, top_row_idx) flat index. We pre-read each
        # file's metadata only — actual range-image bytes load lazily.
        self.files: List[Path] = files
        self._index: List[Tuple[int, int]] = []
        for fi, f in enumerate(files):
            pf = pq.ParquetFile(f)
            # The parquet has 5 lasers × N frames rows; we want only the TOP
            # ones. Each file has the same number of frames per laser, so
            # n_top = num_rows / 5.
            n_top = pf.metadata.num_rows // 5
            self._index.extend((fi, ri) for ri in range(n_top))

        # OrderedDict serves as a tiny LRU cache: file_idx -> pyarrow.Table
        # already filtered to TOP and sorted by timestamp.
        self._cache: "OrderedDict[int, object]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _get_top_table(self, fi: int):
        cached = self._cache.get(fi)
        if cached is not None:
            self._cache.move_to_end(fi)
            return cached
        # Column-prune before reading so we never pay for return2 (the secondary
        # range image — twice the bytes of return1). Then top-filter and
        # combine_chunks to drop the original 990-row backing buffer.
        table = pq.read_table(
            self.files[fi],
            columns=[_COL_LASER, _COL_TS, _COL_VALUES, _COL_SHAPE],
        )
        mask = np.asarray(table[_COL_LASER]) == LASER_NAME_TOP
        table = table.filter(mask).combine_chunks()
        table = table.sort_by(_COL_TS)
        self._cache[fi] = table
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return table

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        fi, ri = self._index[idx]
        table = self._get_top_table(fi)
        # `.column(name)[i]` returns a pyarrow scalar; `.as_py()` materializes
        # the underlying python list (which we then go through numpy).
        shape = table.column(_COL_SHAPE)[ri].as_py()
        values = table.column(_COL_VALUES)[ri].as_py()
        ri_img = decode_lidar_row(values, shape)
        img = waymo_range_image_to_4ch(
            ri_img,
            H_out=self.H_out,
            W_out=self.W_out,
            range_max_m=self.range_max_m,
            intensity_max=self.intensity_max,
            elongation_max=self.elongation_max,
        )
        tensor = torch.from_numpy(img)
        if not self.return_dict:
            return tensor
        return {
            "range_image": tensor,
            "segment": self.files[fi].stem,
            "timestamp": int(table.column(_COL_TS)[ri].as_py()),
        }


class SegmentGroupedRandomSampler(Sampler[int]):
    """Yields dataset indices grouped by source parquet, with shuffling.

    Why this exists: `WaymoLidarTopKeyframes` caches one ~512 MiB pyarrow table
    per parquet, and `pq.read_table` + filter + combine_chunks transiently
    peaks at ~1.5 GiB during load. Vanilla `shuffle=True` jumps across all 20
    segments randomly, so each DataLoader worker constantly evicts and reloads
    tables — push that across two workers and the OS OOM-killer fires.

    This sampler instead:
      1. Shuffles segment order each epoch.
      2. Within each segment, shuffles its frame indices.
      3. Emits all of segment A's indices, then all of segment B's, etc.

    With `num_workers <= ~4` and `cache_size=1`, every worker keeps the
    segment it's currently chewing through pinned in RAM (one table = 512 MiB)
    and only loads a new one at segment boundaries (~20 batches at B=10).
    """

    def __init__(self, dataset: WaymoLidarTopKeyframes, seed: int = 0):
        self._index = dataset._index  # list of (file_idx, row_in_top_table)
        # Group dataset indices by file_idx once; segments are stable.
        self._by_file: dict[int, list[int]] = {}
        for di, (fi, _ri) in enumerate(self._index):
            self._by_file.setdefault(fi, []).append(di)
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + self._epoch)
        # Stable epoch counter so two calls in the same epoch give the same order;
        # advance it next time __iter__ is invoked.
        self._epoch += 1
        file_order = list(self._by_file.keys())
        rng.shuffle(file_order)
        for fi in file_order:
            ids = list(self._by_file[fi])
            rng.shuffle(ids)
            for di in ids:
                yield di

    def __len__(self) -> int:
        return len(self._index)
