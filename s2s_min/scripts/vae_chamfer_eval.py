"""VAE-only round-trip Chamfer evaluation, with projection-floor baseline.

For each LIDAR_TOP keyframe in two pools (training scenes from the 10-scene
subset, and held-out scenes from outside it) we compute:

    1.  raw_pc                                            ← ground truth
    2.  pc_proj_only  = unproject(project(raw_pc))        ← projection floor
                        (NO VAE — bounds how well we can do given the lossy
                         spherical projection alone, with our linear elevation
                         fallback table)
    3.  pc_vae        = unproject(decode(encode(project(raw_pc))))
                        ← VAE round-trip
    4.  CD_floor      = chamfer(raw_pc_filtered, pc_proj_only)
    5.  CD_roundtrip  = chamfer(raw_pc_filtered, pc_vae)
    6.  VAE delta     = CD_roundtrip − CD_floor
                        ← what the VAE *adds* on top of the inherent projection
                          error. This is the apples-to-apples number to compare
                          against RangeLDM's ~0.01–0.02 m VAE-only published CD.

`raw_pc_filtered` keeps only points that fell inside the range-image envelope
(ring ∈ [0, 32), 0 < range ≤ 100 m) — i.e. the points the VAE *could* have
predicted. Points outside are unfair to score against.

Output:
    s2s_min/out/vae_chamfer_eval.json    (machine-readable per-sample + means)
Stdout: a human-readable table.

Run:
    env/bin/python s2s_min/scripts/vae_chamfer_eval.py
    env/bin/python s2s_min/scripts/vae_chamfer_eval.py --ckpt path/to/lidar_vae_best.pt
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S = REPO_ROOT / "s2s_min"
sys.path.insert(0, str(S2S))

import numpy as np
import torch

from data.range_image import (
    H_DEFAULT,
    W_DEFAULT,
    RANGE_MAX_M,
    load_nuscenes_lidar_bin,
    point_cloud_to_range_image,
    range_image_to_point_cloud,
)
from eval.chamfer import chamfer_distance
from models.lidar_vae import LiDARVAE


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _read_subset_tokens(subset_file: Path) -> set[str]:
    return {t for t in subset_file.read_text().split() if t}


def _enumerate_keyframes(nuscenes_root: Path) -> list[dict]:
    """All LIDAR_TOP keyframes with `scene_token` annotated; small enough to load whole."""
    meta = nuscenes_root / "v1.0-trainval"
    sample = json.loads((meta / "sample.json").read_text())
    sample_data = json.loads((meta / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((meta / "sensor.json").read_text())}

    sample_by_token = {s["token"]: s for s in sample}
    out = []
    for sd in sample_data:
        if not sd["is_key_frame"]:
            continue
        chan = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if chan != "LIDAR_TOP":
            continue
        s = sample_by_token.get(sd["sample_token"])
        if s is None:
            continue
        out.append({
            "path": str(nuscenes_root / sd["filename"]),
            "scene_token": s["scene_token"],
            "sample_token": sd["sample_token"],
            "filename": Path(sd["filename"]).name,
        })
    return out


def _pick_samples(
    keyframes: list[dict],
    subset_tokens: set[str],
    n_train: int,
    n_held_out: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Pick `n_train` keyframes from inside the subset and `n_held_out` from outside.

    For both pools we take **one keyframe per scene** to maximize visual diversity.
    """
    rng = np.random.default_rng(seed)

    by_scene_in:  dict[str, list[dict]] = {}
    by_scene_out: dict[str, list[dict]] = {}
    for k in keyframes:
        bucket = by_scene_in if k["scene_token"] in subset_tokens else by_scene_out
        bucket.setdefault(k["scene_token"], []).append(k)

    def pick(by_scene, n):
        if n == 0:
            return []
        scenes = sorted(by_scene.keys())
        rng.shuffle(scenes)
        picked = []
        for sc in scenes:
            picked.append(by_scene[sc][0])     # first keyframe of the scene (stable)
            if len(picked) >= n:
                break
        return picked

    return pick(by_scene_in, n_train), pick(by_scene_out, n_held_out)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _filter_pc(points: np.ndarray, range_max_m: float) -> np.ndarray:
    """Keep only points that fall inside the range-image envelope.

    Matches the filtering inside `point_cloud_to_range_image`: ring_index in
    `[0, H_DEFAULT)`, range in `(0, range_max_m]`. Returns `[M, 4]` (x, y, z, intensity).
    """
    r = np.sqrt((points[:, :3] ** 2).sum(axis=1))
    ring = points[:, 4].astype(np.int32)
    valid = (ring >= 0) & (ring < H_DEFAULT) & (r > 0.0) & (r <= range_max_m)
    return points[valid][:, :4].astype(np.float32)


def _eval_one(
    vae: LiDARVAE,
    bin_path: str,
    device: torch.device,
) -> dict:
    """Returns per-sample CD-floor, CD-roundtrip, and bookkeeping."""
    points = load_nuscenes_lidar_bin(bin_path)                # [N, 5]
    raw_filtered = _filter_pc(points, RANGE_MAX_M)            # [M, 4]

    # --- Projection only (no VAE) -----------------------------------------
    img = point_cloud_to_range_image(points)                  # [3, H, W]
    pc_proj_only = range_image_to_point_cloud(img)            # [P, 4]

    # --- VAE round-trip ---------------------------------------------------
    x = torch.from_numpy(img).unsqueeze(0).to(device)         # [1, 3, H, W]
    with torch.no_grad():
        mu, _ = vae.encode(x)
        recon = vae.decode(mu)[0].detach().cpu().numpy()      # [3, H, W]
    # Decoder validity may be a probability; threshold at 0.5 for unprojection.
    recon[2] = (recon[2] > 0.5).astype(np.float32)
    pc_vae = range_image_to_point_cloud(recon)                # [Q, 4]

    cd_floor      = chamfer_distance(raw_filtered, pc_proj_only)
    cd_roundtrip  = chamfer_distance(raw_filtered, pc_vae)

    return {
        "file":            Path(bin_path).name,
        "n_raw":           int(raw_filtered.shape[0]),
        "n_proj_only":     int(pc_proj_only.shape[0]),
        "n_vae":           int(pc_vae.shape[0]),
        "cd_floor":        cd_floor["cd"],
        "cd_roundtrip":    cd_roundtrip["cd"],
        "vae_delta":       cd_roundtrip["cd"] - cd_floor["cd"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=S2S / "out" / "lidar_vae_best.pt",
                   help="Path to the trained VAE checkpoint. Falls back to lidar_vae.pt if best is missing.")
    p.add_argument("--nuscenes_root", type=Path, default=REPO_ROOT / "nuscenes")
    p.add_argument("--subset_file", type=Path, default=S2S / "out" / "subset_scene_tokens.txt")
    p.add_argument("--n_train", type=int, default=4)
    p.add_argument("--n_held_out", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=Path, default=S2S / "out" / "vae_chamfer_eval.json")
    args = p.parse_args()

    if not args.ckpt.exists():
        fallback = args.ckpt.with_name("lidar_vae.pt")
        if fallback.exists():
            print(f"warning: {args.ckpt} missing; falling back to {fallback}")
            args.ckpt = fallback
        else:
            print(f"error: no checkpoint found at {args.ckpt} or fallback")
            sys.exit(1)

    print("=" * 78)
    print("VAE-only round-trip Chamfer eval")
    print("=" * 78)
    print(f"  checkpoint    : {args.ckpt}")
    print(f"  nuscenes_root : {args.nuscenes_root}")
    print(f"  device        : {args.device}")

    # ---- Load model ------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location=args.device)
    arch_kwargs = set(inspect.signature(LiDARVAE.__init__).parameters)
    model_cfg = {k: v for k, v in ckpt["config"].items() if k in arch_kwargs}
    vae = LiDARVAE(**model_cfg).to(args.device).eval()
    vae.load_state_dict(ckpt["state_dict"])
    vae.requires_grad_(False)
    print(f"  step          : {ckpt.get('step', '?')}")
    print(f"  l1_range_ema  : {ckpt.get('l1_range_ema', '<not saved>')}")
    print(f"  arch kwargs   : {model_cfg}")
    print(f"  params        : {sum(p.numel() for p in vae.parameters())/1e6:.2f} M")

    # ---- Pick samples ----------------------------------------------------
    subset = _read_subset_tokens(args.subset_file)
    keyframes = _enumerate_keyframes(args.nuscenes_root)
    train_picks, held_picks = _pick_samples(
        keyframes, subset, args.n_train, args.n_held_out, seed=args.seed
    )
    print(f"  total LIDAR_TOP keyframes : {len(keyframes)}")
    print(f"  scenes in subset (train)  : {len(subset)}")
    print(f"  scenes outside (held-out) : {len(set(k['scene_token'] for k in keyframes)) - len(subset)}")
    print(f"  picked: {len(train_picks)} training + {len(held_picks)} held-out")

    # ---- Evaluate --------------------------------------------------------
    device = torch.device(args.device)
    rows_train = []
    rows_held  = []
    t0 = time.perf_counter()
    print(f"\n  {'split':<10} {'n_raw':>7} {'n_proj':>7} {'n_vae':>7}  "
          f"{'CD_floor':>10} {'CD_roundtrip':>13} {'VAE_delta':>10}  file")
    for tag, picks, rows in [("train", train_picks, rows_train),
                              ("held-out", held_picks, rows_held)]:
        for k in picks:
            r = _eval_one(vae, k["path"], device)
            rows.append(r)
            print(f"  {tag:<10} {r['n_raw']:>7d} {r['n_proj_only']:>7d} {r['n_vae']:>7d}  "
                  f"{r['cd_floor']:>10.3f} {r['cd_roundtrip']:>13.3f} {r['vae_delta']:>10.3f}  "
                  f"{r['file']}")
    print(f"  (evaluated in {time.perf_counter() - t0:.1f}s)")

    # ---- Means + report --------------------------------------------------
    def _mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    summary = {
        "checkpoint": str(args.ckpt),
        "step": ckpt.get("step"),
        "l1_range_ema": ckpt.get("l1_range_ema"),
        "config": model_cfg,
        "train": {
            "rows": rows_train,
            "mean_cd_floor":     _mean(rows_train, "cd_floor"),
            "mean_cd_roundtrip": _mean(rows_train, "cd_roundtrip"),
            "mean_vae_delta":    _mean(rows_train, "vae_delta"),
        },
        "held_out": {
            "rows": rows_held,
            "mean_cd_floor":     _mean(rows_held, "cd_floor"),
            "mean_cd_roundtrip": _mean(rows_held, "cd_roundtrip"),
            "mean_vae_delta":    _mean(rows_held, "vae_delta"),
        },
    }

    print()
    print("=" * 78)
    print(f"  {'group':<10} {'CD_floor (m)':>14} {'CD_roundtrip (m)':>18} {'VAE_delta (m)':>15}")
    for grp in ("train", "held_out"):
        d = summary[grp]
        print(f"  {grp:<10} {d['mean_cd_floor']:>14.3f} "
              f"{d['mean_cd_roundtrip']:>18.3f} {d['mean_vae_delta']:>15.3f}")
    print("=" * 78)
    print()
    print("Interpretation:")
    print("  - CD_floor      = lossy projection alone (raw → range image → raw). Independent of the VAE.")
    print("  - CD_roundtrip  = the same path, but through the VAE encode + decode in between.")
    print("  - VAE_delta     = how much *extra* error the VAE introduces. ← compare against RangeLDM ~0.01–0.02 m.")
    print("  - held_out      = scenes the model has never seen during training (true generalization signal).")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    # Replace non-JSON floats (e.g. inf from missing best-ckpt) with strings before dumping.
    def _scrub(o):
        if isinstance(o, dict):
            return {k: _scrub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_scrub(v) for v in o]
        if isinstance(o, float) and (o != o or o == float("inf") or o == float("-inf")):
            return str(o)
        return o
    args.out_json.write_text(json.dumps(_scrub(summary), indent=2) + "\n")
    print(f"\nwrote: {args.out_json}")


if __name__ == "__main__":
    main()
