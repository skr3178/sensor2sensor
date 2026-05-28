"""M5 — Collect per-milestone numbers from s2s_min/out/ into one summary table.

Reads (read-only) every `stats.txt`, MANIFEST, and training `.log` file the
minimum pipeline produced, extracts the headline numbers, and prints a single
scannable table.

Use this as a 30-second smoke check before opening `RESULTS.md`:
  - if every milestone shows a numeric value, all stages ran and persisted output
  - if any cell shows "MISSING" or "PARSE-FAIL", inspect that milestone

Exit code 0 if all expected files exist and parse, 1 otherwise. No threshold
checking (numbers are floats; brittle to lock in).

Run:
    env/bin/python s2s_min/scripts/collect_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT_DIR = Path("s2s_min/out")

# Paths to every stats file / log the pipeline produces, indexed by milestone.
SOURCES = {
    "M2_cache":         OUT_DIR / "cached_latents" / "MANIFEST.json",
    "img_vae_verify":   OUT_DIR / "image_vae_samples" / "stats.txt",
    "lidar_vae_verify": OUT_DIR / "lidar_vae_samples" / "stats.txt",
    "raymap_bench":     OUT_DIR / "raymap_benchmark" / "stats.txt",
    "M3.1_ddim":        OUT_DIR / "m31_ddim_sanity" / "stats.txt",
    "M3.2_ddim":        OUT_DIR / "m32_ddim_sanity" / "stats.txt",
    "M3.1_log":         OUT_DIR / "train_diffusion_overfit10.log",
    "M3.2_log":         OUT_DIR / "train_diffusion_m32.log",
    "M4_demo":          OUT_DIR / "m4_demo" / "stats.txt",
}


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text()
    except Exception:
        return None


def _grep(text: str | None, pattern: str, group: int = 1) -> str:
    """Return the first regex match (or 'MISSING' if text is None / 'PARSE-FAIL' on no match)."""
    if text is None:
        return "MISSING"
    m = re.search(pattern, text)
    if m is None:
        return "PARSE-FAIL"
    return m.group(group).strip()


def _grep_last(text: str | None, pattern: str, group: int = 1) -> str:
    """Like `_grep` but returns the LAST match (for picking the final loss-curve line).

    Uses re.finditer so multi-group patterns work — re.findall returns tuples
    for those, which lose the `.strip()` interface.
    """
    if text is None:
        return "MISSING"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return "PARSE-FAIL"
    return matches[-1].group(group).strip()


def main() -> int:
    summaries: list[tuple[str, str]] = []
    missing = 0

    # ---- M2 cache (MANIFEST.json) ----
    mp = _read(SOURCES["M2_cache"])
    if mp is None:
        summaries += [("M2 cache", "MISSING — run train/cache_latents.py")]
        missing += 1
    else:
        m = json.loads(mp)
        summaries += [(
            "M2 cache",
            f"{m['n_paired_samples_in_subset']} samples, "
            f"{m['total_cache_mb']:.1f} MB, "
            f"wall {m['wall_time_seconds']:.1f}s, "
            f"μ_mean={m['stats']['mu_mean']:+.3f} μ_std={m['stats']['mu_std']:.3f}"
        )]

    # ---- Image VAE verify ----
    iv = _read(SOURCES["img_vae_verify"])
    iv_shape = _grep(iv, r"output shape\s*:\s*(\([^)]+\))")
    iv_sf    = _grep(iv, r"scaling_factor\s*:\s*([0-9.eE+-]+)")
    summaries += [("Image VAE verify", f"latent {iv_shape}  scaling_factor={iv_sf}")]
    if iv is None: missing += 1

    # ---- LiDAR VAE verify ----
    lv = _read(SOURCES["lidar_vae_verify"])
    lv_step   = _grep(lv, r"step\s+:\s+(\d+)")
    lv_params = _grep(lv, r"params\s+:\s+([0-9.]+ M)")
    lv_mean   = _grep_last(lv, r"MEAN\s+\S+\s+([0-9.]+)")
    lv_bce    = _grep_last(lv, r"\s([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$", group=3)
    summaries += [("LiDAR VAE verify",
                   f"step={lv_step}  params={lv_params}  mean L1_range_m={lv_mean}  BCE_valid={lv_bce}")]
    if lv is None: missing += 1

    # ---- Raymap benchmark ----
    rb = _read(SOURCES["raymap_bench"])
    rb_mean = _grep(rb, r"mean\s+:\s+([0-9.]+)")
    rb_max  = _grep(rb, r"p100\s*:\s*([0-9.]+)")
    rb_n    = _grep(rb, r"N points\s+:\s+(\d+)")
    summaries += [("Raymap benchmark", f"mean error {rb_mean}°  max {rb_max}°  N points {rb_n}")]
    if rb is None: missing += 1

    # ---- M3.1 overfit-10 ----
    m31_log  = _read(SOURCES["M3.1_log"])
    m31_ddim = _read(SOURCES["M3.1_ddim"])
    m31_steps = _grep(m31_log, r"Final:\s*step=(\d+)")
    m31_wall  = _grep(m31_log, r"total_time=([0-9.]+)s")
    m31_mse   = _grep(m31_log, r"final mse_ema:\s+([0-9.]+)")
    m31_cos   = _grep(m31_ddim, r"MEAN cos.+:\s+([+\-0-9.]+)")
    summaries += [("M3.1 overfit-10",
                   f"steps={m31_steps}  wall {m31_wall}s  final mse_ema={m31_mse}  DDIM cos {m31_cos}")]
    if m31_log is None or m31_ddim is None: missing += 1

    # ---- M3.2 v2 (full epoch with proper schedule) ----
    m32_log  = _read(SOURCES["M3.2_log"])
    m32_ddim = _read(SOURCES["M3.2_ddim"])
    m32_steps = _grep(m32_log, r"Final:\s*step=(\d+)")
    m32_wall  = _grep(m32_log, r"total_time=([0-9.]+)s")
    m32_mse   = _grep(m32_log, r"final mse_ema:\s+([0-9.]+)")
    m32_cos   = _grep(m32_ddim, r"HELD-OUT mean cos\(z_pred, μ\)\s*:\s+([+\-0-9.]+)")
    summaries += [("M3.2 v2 (5 epoch)",
                   f"steps={m32_steps}  wall {m32_wall}s  final mse_ema={m32_mse}  DDIM held-out cos {m32_cos}")]
    if m32_log is None or m32_ddim is None: missing += 1

    # ---- M4 demo (the headline) ----
    m4 = _read(SOURCES["M4_demo"])
    m4_cos       = _grep(m4, r"mean cos\(z_pred, μ\)\s*:\s+([+\-0-9.]+)")
    m4_cd_raw    = _grep(m4, r"mean CD-3D-raw\s+:\s+([0-9.]+)")
    m4_cd_vae    = _grep(m4, r"mean CD-VAE-only\s+:\s+([0-9.]+)")
    m4_cd_oracle = _grep(m4, r"mean CD-3D-oracle\s+:\s+([0-9.]+)")
    summaries += [("M4 demo (4 held-out)",
                   f"cos {m4_cos}  CD-3D-raw={m4_cd_raw} m  "
                   f"CD-VAE-only={m4_cd_vae} m  CD-3D-oracle={m4_cd_oracle} m")]
    if m4 is None: missing += 1

    # ---- print ----
    print("=" * 80)
    print("Sensor2Sensor minimum pipeline — collected results")
    print("=" * 80)
    for milestone, summary in summaries:
        marker = "✗" if ("MISSING" in summary or "PARSE-FAIL" in summary) else "✓"
        print(f"  {marker} {milestone:<22} {summary}")
    print("=" * 80)
    if missing:
        print(f"FAIL — {missing} source(s) missing. Run the missing stage(s) and re-run this script.")
        return 1
    print("OK — all expected output files exist and parse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
