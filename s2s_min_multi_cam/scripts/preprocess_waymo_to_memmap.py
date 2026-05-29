"""Decode every Waymo TOP-LiDAR keyframe into a packed numpy memmap.

Why this exists: reading 165 MiB parquet files, filtering to the TOP laser,
and `combine_chunks`-ing in pyarrow transiently peaks at ~3 GiB of RAM per
DataLoader worker. With two workers and any swap pressure that's enough to
trip the OS OOM-killer mid-epoch. Decoding once and storing as a flat,
memory-mappable float16 array reduces the per-worker steady-state to just
the OS page cache (basically free) and makes random access O(1).

Output layout (per split):
    {out_root}/{split}/range_images.npy      memmap, shape (N, 3, H, W), dtype float16
    {out_root}/{split}/manifest.json         per-index metadata (segment, ts)

Run from the repo root once after `scripts/download_waymo_samples.sh`:
    python s2s_min/scripts/preprocess_waymo_to_memmap.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S_DIR = REPO_ROOT / "s2s_min"
sys.path.insert(0, str(S2S_DIR))

import numpy as np
import pyarrow.parquet as pq

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

_COL_LASER = "key.laser_name"
_COL_TS = "key.frame_timestamp_micros"
_COL_VALUES = "[LiDARComponent].range_image_return1.values"
_COL_SHAPE = "[LiDARComponent].range_image_return1.shape"


def process_split(
    waymo_root: Path,
    out_root: Path,
    split: str,
    H_out: int,
    W_out: int,
    range_max_m: float,
    intensity_max: float,
    elongation_max: float,
) -> None:
    in_dir = waymo_root / split / "lidar"
    files = sorted(p for p in in_dir.iterdir() if p.suffix == ".parquet")
    if not files:
        print(f"  [skip] no parquets in {in_dir}")
        return

    # 1) Count frames up front so we can allocate the output memmap exactly.
    total_frames = 0
    per_file_n = []
    for f in files:
        n_top = pq.ParquetFile(f).metadata.num_rows // 5
        per_file_n.append(n_top)
        total_frames += n_top
    print(f"  segments: {len(files)}   top-laser frames: {total_frames}")

    out_split = out_root / split
    out_split.mkdir(parents=True, exist_ok=True)
    out_arr_path = out_split / "range_images.npy"

    # 2) Allocate the memmap and stream into it segment-by-segment.
    shape = (total_frames, 4, H_out, W_out)
    np.lib.format.open_memmap(
        str(out_arr_path), mode="w+", dtype=np.float16, shape=shape
    )
    # Reopen as a memmap for streaming writes (open_memmap returns a memmap
    # but releasing it lets us re-acquire below with a fresh handle).
    mm = np.load(out_arr_path, mmap_mode="r+")

    manifest = []
    cursor = 0
    t0 = time.perf_counter()
    for fi, (f, n_top) in enumerate(zip(files, per_file_n)):
        t_f = time.perf_counter()
        # Column-prune + filter + combine, just like the runtime loader.
        table = pq.read_table(
            f, columns=[_COL_LASER, _COL_TS, _COL_VALUES, _COL_SHAPE]
        )
        mask = np.asarray(table[_COL_LASER]) == LASER_NAME_TOP
        table = table.filter(mask).combine_chunks().sort_by(_COL_TS)
        assert table.num_rows == n_top, f"expected {n_top} top rows, got {table.num_rows}"

        values_col = table.column(_COL_VALUES)
        shape_col = table.column(_COL_SHAPE)
        ts_col = table.column(_COL_TS)

        for ri in range(n_top):
            ri_img = decode_lidar_row(values_col[ri].as_py(), shape_col[ri].as_py())
            img = waymo_range_image_to_4ch(
                ri_img,
                H_out=H_out,
                W_out=W_out,
                range_max_m=range_max_m,
                intensity_max=intensity_max,
                elongation_max=elongation_max,
            )
            mm[cursor] = img.astype(np.float16, copy=False)
            manifest.append({
                "segment": f.stem,
                "timestamp_us": int(ts_col[ri].as_py()),
            })
            cursor += 1

        # Drop the in-memory table aggressively before reading the next one;
        # otherwise pyarrow can hold the previous backing buffer until the
        # next allocation and we accumulate RAM across files.
        del table, values_col, shape_col, ts_col
        gc.collect()

        dt = time.perf_counter() - t_f
        dt_total = time.perf_counter() - t0
        print(f"  [{fi+1:2d}/{len(files)}] {f.stem} -> {n_top} frames "
              f"in {dt:5.1f}s   (split total {dt_total:5.1f}s, cursor={cursor})")

    assert cursor == total_frames, f"wrote {cursor}, expected {total_frames}"
    mm.flush()
    del mm  # close the memmap

    with open(out_split / "manifest.json", "w") as fh:
        json.dump({
            "n_frames": total_frames,
            "H": H_out,
            "W": W_out,
            "channels": 4,
            "range_max_m": range_max_m,
            "intensity_max": intensity_max,
            "elongation_max": elongation_max,
            "items": manifest,
        }, fh)

    size_mb = out_arr_path.stat().st_size / 2**20
    print(f"  -> wrote {out_arr_path}  ({size_mb:.0f} MiB on disk)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--waymo_root", type=Path, default=S2S_DIR / "data" / "waymo")
    p.add_argument("--out_root", type=Path, default=S2S_DIR / "data" / "waymo_packed")
    p.add_argument("--H_out", type=int, default=H_DEFAULT)
    p.add_argument("--W_out", type=int, default=W_DEFAULT)
    p.add_argument("--range_max_m", type=float, default=RANGE_MAX_M)
    p.add_argument("--intensity_max", type=float, default=INTENSITY_MAX)
    p.add_argument("--elongation_max", type=float, default=ELONGATION_MAX)
    p.add_argument("--splits", nargs="+", default=["training", "validation"])
    args = p.parse_args()

    print("=" * 70)
    print("Waymo -> packed numpy memmap preprocessor")
    print("=" * 70)
    print(f"  waymo_root  : {args.waymo_root}")
    print(f"  out_root    : {args.out_root}")
    print(f"  target shape: (N, 4, {args.H_out}, {args.W_out})  dtype float16")
    print(f"  range_max_m={args.range_max_m}  intensity_max={args.intensity_max}  "
          f"elongation_max={args.elongation_max}")
    for split in args.splits:
        print(f"\nProcessing split: {split}")
        process_split(
            args.waymo_root, args.out_root, split,
            args.H_out, args.W_out, args.range_max_m, args.intensity_max,
            args.elongation_max,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
