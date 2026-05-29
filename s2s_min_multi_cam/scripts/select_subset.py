"""Pick a reproducible subset of nuScenes-trainval scenes for M1/M3 training.

Per the plan ([min_pipeline_plan.md] subset definition), we take 10 scenes from
v1.0-trainval/scene.json sampled with a fixed RNG (`np.random.default_rng(seed)`)
rather than the first 10 (which would bias toward whatever ordering nuScenes
happened to use, often clustered by location).

Output: a newline-delimited list of scene tokens written to
    s2s_min/out/subset_scene_tokens.txt
Consumed by `data.nuscenes_mini.NuScenesLidarKeyframes`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuscenes_root", type=Path,
        default=Path(__file__).resolve().parents[2] / "nuscenes",
    )
    parser.add_argument("--n", type=int, default=10,
                        help="number of scenes to pick (default 10)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed (default 0) -- determines which scenes")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "out" / "subset_scene_tokens.txt")
    args = parser.parse_args()

    scene_path = args.nuscenes_root / "v1.0-trainval" / "scene.json"
    scenes = json.loads(scene_path.read_text())
    print(f"Total scenes in v1.0-trainval: {len(scenes)}")
    assert args.n <= len(scenes), f"asked for {args.n}, only {len(scenes)} available"

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(scenes), args.n, replace=False)
    # Sort by index so the output is stable across runs (independent of choice order).
    picked = [scenes[i] for i in sorted(indices)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(s["token"] for s in picked) + "\n")

    print(f"Picked {len(picked)} scenes (seed={args.seed}) -> {args.out}")
    for s in picked:
        print(f"  {s['token']}  {s['name']}  nbr_samples={s['nbr_samples']}")
    print(f"  total keyframes across subset: {sum(s['nbr_samples'] for s in picked)}")


if __name__ == "__main__":
    main()
