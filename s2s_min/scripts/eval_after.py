"""Wait for a training run to finish, then dump evaluation artifacts into its run folder.

Pattern:
    1. Optionally wait for `--pid <N>` (or `--pid_file path`) to exit.
       Polls every 30 s with `kill(pid, 0)`.
    2. Run the three eval scripts, with output paths redirected into
       `{run_dir}/eval/`:
         a. plot_train_runs.py     → eval/loss_comparison.png + eval/loss_plots/*.png
         b. visualize_lidar_vae.py → eval/samples.png + eval/stats.txt
         c. vae_chamfer_eval.py    → eval/chamfer.json
    3. Write `{run_dir}/summary.md` — top-level human-readable run summary.

Doesn't touch any file outside the run dir (e.g. the global
`s2s_min/out/loss_comparison.png` stays whatever the last manual run wrote).

Run, detached, while v5 (or any future run) is still going:

    nohup setsid python -u s2s_min/scripts/eval_after.py \\
        --run_dir s2s_min/out/runs/2026-05-28_142002__v5-100scenes-bs16-lpips-nohup \\
        --pid_file s2s_min/out/v5.pid \\
        > /tmp/eval_after.log 2>&1 < /dev/null &
    disown

Or post-hoc, once a run is already finished:

    python s2s_min/scripts/eval_after.py --run_dir <path-to-run-dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S = REPO_ROOT / "s2s_min"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for_pid(pid: int, poll_seconds: int = 30) -> None:
    print(f"[eval_after] waiting for PID {pid} to exit (poll every {poll_seconds}s)...")
    t0 = time.time()
    while _pid_alive(pid):
        time.sleep(poll_seconds)
    print(f"[eval_after] PID {pid} exited after {time.time() - t0:.0f}s.")


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run a subprocess, capture combined stdout+stderr, print as it runs."""
    print(f"[eval_after] $ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    print(out)
    return proc.returncode, out


def _load_metadata(run_dir: Path) -> dict:
    p = run_dir / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_chamfer(eval_dir: Path) -> dict:
    p = eval_dir / "chamfer.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_viz_stats(eval_dir: Path) -> str:
    p = eval_dir / "stats.txt"
    if not p.exists():
        return ""
    return p.read_text()


# Historical runs that pre-date the run-folder layout. Hardcoded so the
# comparison table in summary.md doesn't lose them. Keep in sync with
# s2s_min/docs/lidar_vae.md §8.1.
HISTORICAL_RUNS = [
    {
        "tag": "v1", "scenes": 10, "keyframes": 401, "lambda_range": 1,
        "lpips": False, "lr_sched": "constant", "eff_batch": 8,
        "steps": 2513, "wallclock_min": 9.1, "vram": "446 MiB",
        "best_l1r_ema": "~0.015 (lost)", "final_l1r_ema": "0.188",
        "divergence_step": "~2 000", "usable": "✗",
        "vae_delta_heldout": "—",
    },
    {
        "tag": "v2", "scenes": 10, "keyframes": 401, "lambda_range": 50,
        "lpips": False, "lr_sched": "constant", "eff_batch": 8,
        "steps": 2513, "wallclock_min": 9.4, "vram": "446 MiB",
        "best_l1r_ema": "0.01147", "final_l1r_ema": "0.030",
        "divergence_step": "~950", "usable": "✗",
        "vae_delta_heldout": "—",
    },
    {
        "tag": "v3", "scenes": 10, "keyframes": 401, "lambda_range": 50,
        "lpips": False, "lr_sched": "cosine", "eff_batch": 8,
        "steps": 2513, "wallclock_min": 9.4, "vram": "446 MiB",
        "best_l1r_ema": "0.01116", "final_l1r_ema": "0.066",
        "divergence_step": "~1 050", "usable": "✗",
        "vae_delta_heldout": "—",
    },
    {
        "tag": "v4", "scenes": 10, "keyframes": 401, "lambda_range": 50,
        "lpips": True, "lr_sched": "cosine", "eff_batch": 8,
        "steps": 2513, "wallclock_min": 21.0, "vram": "1.2 GB",
        "best_l1r_ema": "0.00656", "final_l1r_ema": "0.00667",
        "divergence_step": "none", "usable": "✓",
        "vae_delta_heldout": "—",
    },
]


def _full_history_table(this_meta: dict, this_chamfer: dict) -> str:
    """Markdown table of v1..v5+, with the current run appended as the last column."""
    args = this_meta.get("args", {})
    n_scenes = "?"
    if "subset_file" in args and "_100scenes" in str(args["subset_file"]):
        n_scenes = 100
    elif "subset_file" in args:
        n_scenes = 10
    final_step = this_meta.get("final_step", "?")
    keyframes = "?"
    if isinstance(n_scenes, int):
        keyframes = "4 023" if n_scenes == 100 else "401"

    best = this_meta.get("best_l1_range_ema")
    final = this_meta.get("final_l1_range_ema")
    wallclock_sec = this_meta.get("wallclock_seconds", 0)
    eff_batch = args.get("batch_size", "?")
    if isinstance(eff_batch, int) and isinstance(args.get("grad_accum"), int):
        eff_batch = eff_batch * args["grad_accum"]
    vae_delta = "—"
    if this_chamfer.get("held_out", {}).get("mean_vae_delta") is not None:
        vae_delta = f"{this_chamfer['held_out']['mean_vae_delta']:.3f} m"

    current = {
        "tag": this_meta.get("description", "current"),
        "scenes": n_scenes,
        "keyframes": keyframes,
        "lambda_range": args.get("lam_range", "?"),
        "lpips": any(args.get(k, 0) > 0 for k in
                      ("lam_lpips_normals", "lam_lpips_intensity", "lam_lpips_validity")),
        "lr_sched": args.get("lr_schedule", "?"),
        "eff_batch": eff_batch,
        "steps": final_step,
        "wallclock_min": (wallclock_sec / 60.0) if isinstance(wallclock_sec, (int, float)) else "?",
        "vram": "?",   # not in metadata.json — would need log scraping
        "best_l1r_ema": f"{best:.5f}" if isinstance(best, float) else str(best),
        "final_l1r_ema": f"{final:.5f}" if isinstance(final, float) else str(final),
        "divergence_step": "none" if isinstance(best, float) and best < 0.01 else "?",
        "usable": "✓" if isinstance(final, float) and final < 0.05 else "?",
        "vae_delta_heldout": vae_delta,
    }

    all_runs = HISTORICAL_RUNS + [current]
    tags = [r["tag"] for r in all_runs]

    def row(label, key, fmt=str):
        return f"| {label} | " + " | ".join(fmt(r.get(key, "—")) for r in all_runs) + " |"

    def lpips_fmt(v): return "✓" if v else "✗"
    def wall_fmt(v):
        if isinstance(v, (int, float)): return f"{v:.1f} min"
        return str(v)

    header = "| | " + " | ".join(tags) + " |"
    sep    = "|---|" + "|".join(["---"] * len(all_runs)) + "|"

    lines = [
        header, sep,
        row("Scenes", "scenes"),
        row("Keyframes", "keyframes"),
        row("Effective batch", "eff_batch"),
        row("LPIPS terms", "lpips", lpips_fmt),
        row("LR schedule", "lr_sched"),
        row("Total optim steps", "steps"),
        row("Wall-clock", "wallclock_min", wall_fmt),
        row("Peak VRAM", "vram"),
        row("Best `l1_range_ema`", "best_l1r_ema"),
        row("Final `l1_range_ema`", "final_l1r_ema"),
        row("Divergence step", "divergence_step"),
        row("Final ckpt usable?", "usable"),
        row("VAE_delta on held-out", "vae_delta_heldout"),
    ]
    return "\n".join(lines)


def _gather_prior_runs(runs_root: Path, this_run: Path) -> list[dict]:
    """Return [{name, best_l1_range_ema, description, step}, ...] for sibling runs."""
    out = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or d == this_run:
            continue
        md_path = d / "metadata.json"
        if not md_path.exists():
            continue
        try:
            md = json.loads(md_path.read_text())
        except Exception:
            continue
        out.append({
            "name": d.name,
            "description": md.get("description", "?"),
            "best_l1_range_ema": md.get("best_l1_range_ema", "?"),
            "final_l1_range_ema": md.get("final_l1_range_ema", "?"),
            "final_step": md.get("final_step", "?"),
            "wallclock_seconds": md.get("wallclock_seconds", "?"),
        })
    return out


def _write_summary(run_dir: Path, eval_dir: Path) -> None:
    md = _load_metadata(run_dir)
    chamfer = _load_chamfer(eval_dir)
    viz_stats = _load_viz_stats(eval_dir)

    desc = md.get("description", "?")
    wall = md.get("wallclock_seconds", "?")
    step = md.get("final_step", "?")
    best_ema = md.get("best_l1_range_ema", "?")
    final_ema = md.get("final_l1_range_ema", "?")
    git = md.get("git_commit", "?")
    args_block = md.get("args", {})

    prior = _gather_prior_runs(run_dir.parent, run_dir)

    def _fmt_num(x, places=5):
        if isinstance(x, float):
            return f"{x:.{places}f}"
        return str(x)

    lines: list[str] = []
    lines += [
        f"# Run summary — `{desc}`",
        "",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        f"- **Run folder**: `{run_dir.name}`",
        f"- **Git commit**: `{git}`",
        f"- **Wall-clock**: {wall:.1f} s ({wall/60:.1f} min)" if isinstance(wall, (int, float))
            else f"- **Wall-clock**: {wall}",
        "",
        "## Headline metrics",
        "",
        f"- best `l1_range_ema`: **{_fmt_num(best_ema)}**"
        + (f" → ≈ {best_ema*100:.2f} m mean per-pixel range error" if isinstance(best_ema, float) else ""),
        f"- final `l1_range_ema`: **{_fmt_num(final_ema)}** (after {step} optimizer steps)",
    ]

    if chamfer:
        lines += [
            "",
            "## VAE-only Chamfer ([chamfer.json](eval/chamfer.json))",
            "",
            "| group     | CD_floor (m) | CD_roundtrip (m) | VAE_delta (m) |",
            "|-----------|-------------:|-----------------:|--------------:|",
        ]
        for grp in ("train", "held_out"):
            d = chamfer.get(grp, {})
            lines.append(
                f"| {grp:<9} | {d.get('mean_cd_floor', float('nan')):.3f} "
                f"| {d.get('mean_cd_roundtrip', float('nan')):.3f} "
                f"| {d.get('mean_vae_delta', float('nan')):.3f} |"
            )
        lines += [
            "",
            "_`VAE_delta` = `CD_roundtrip − CD_floor` — what the VAE adds on top of "
            "the projection floor. This is the apples-to-apples number to compare "
            "against RangeLDM's published ~0.01–0.02 m VAE-only Chamfer._",
        ]

    if viz_stats:
        # Extract just the MEAN row from visualize_lidar_vae's stats.txt
        mean_line = next((ln for ln in viz_stats.splitlines() if "MEAN" in ln), "")
        if mean_line:
            lines += [
                "",
                "## BEV reconstruction sanity ([eval/samples.png](eval/samples.png))",
                "",
                f"```",
                f"{viz_stats.splitlines()[-2] if 'name' in viz_stats else 'name                              L1_range_m  L1_intens  BCE_valid  valid_acc'}",
                f"{mean_line}",
                f"```",
                "",
                "_4 LIDAR_TOP keyframes pulled from the 10-scene subset (training samples)._",
            ]

    lines += [
        "",
        "## Recipe ([metadata.json](metadata.json) has the full args dict)",
        "",
    ]
    if args_block:
        keys_of_interest = [
            "epochs", "batch_size", "grad_accum", "lr", "lr_min",
            "lr_schedule", "lr_warmup_steps", "weight_decay",
            "lam_range", "lam_intensity", "lam_validity", "lam_kl",
            "lam_lpips_normals", "lam_lpips_intensity", "lam_lpips_validity",
            "ema_decay", "subset_file",
        ]
        for k in keys_of_interest:
            if k in args_block:
                lines.append(f"- `{k}` = `{args_block[k]}`")

    # --- Canonical 5-run comparison (matches lidar_vae.md §8.1) -------
    lines += [
        "",
        "## Full run history (v1 → this run)",
        "",
        _full_history_table(md, chamfer),
        "",
        "_v1-v4 numbers are hardcoded in `s2s_min/scripts/eval_after.py:HISTORICAL_RUNS` "
        "since those runs pre-date the run-folder layout. Keep in sync with "
        "[`docs/lidar_vae.md` §8.1](../../docs/lidar_vae.md)._",
    ]

    if prior:
        lines += [
            "",
            "## Sibling runs in this folder (run-folder layout only)",
            "",
            "| run | description | best l1_range_ema | final l1_range_ema | step | wall-clock (s) |",
            "|---|---|---|---|---|---|",
        ]
        rows = prior + [{
            "name": run_dir.name,
            "description": desc + " ← **this run**",
            "best_l1_range_ema": best_ema,
            "final_l1_range_ema": final_ema,
            "final_step": step,
            "wallclock_seconds": wall,
        }]
        for r in rows:
            lines.append(
                f"| `{r['name'][:32]}…` | {r['description']} | {_fmt_num(r['best_l1_range_ema'])} "
                f"| {_fmt_num(r['final_l1_range_ema'])} | {r['final_step']} | {r['wallclock_seconds']} |"
            )

    lines += [
        "",
        "## Eval artifacts (all in this folder)",
        "",
        "- [eval/loss_comparison.png](eval/loss_comparison.png) — 8-panel loss-term comparison vs prior runs",
        "- [eval/loss_plots/](eval/loss_plots/) — standalone PNG per loss term",
        "- [eval/samples.png](eval/samples.png) — BEV/range visualization on the best ckpt",
        "- [eval/stats.txt](eval/stats.txt) — per-sample reconstruction stats",
        "- [eval/chamfer.json](eval/chamfer.json) — VAE-only round-trip Chamfer + projection-floor baseline",
        "- [lidar_vae_best.pt](lidar_vae_best.pt) — what M2/M3 should load",
        "- [metadata.json](metadata.json) — full CLI args + final losses",
        "",
    ]

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"[eval_after] wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="Path to the run folder under s2s_min/out/runs/.")
    parser.add_argument("--pid", type=int, default=None,
                        help="Optional PID to wait on before running evals.")
    parser.add_argument("--pid_file", type=Path, default=None,
                        help="File containing a PID to wait on (alternative to --pid).")
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--skip_chamfer", action="store_true",
                        help="Skip the Chamfer eval (e.g. if it would take too long).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"[eval_after] run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(2)

    # ---- Wait for training to finish ----------------------------------
    pid = args.pid
    if pid is None and args.pid_file is not None and args.pid_file.exists():
        try:
            pid = int(args.pid_file.read_text().strip())
        except ValueError:
            pid = None
    if pid is not None:
        _wait_for_pid(pid, poll_seconds=args.poll_seconds)
    else:
        print("[eval_after] no PID to wait on; running evals immediately.")

    # ---- Prep eval folder --------------------------------------------
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    loss_plots_dir = eval_dir / "loss_plots"

    ckpt = run_dir / "lidar_vae_best.pt"
    if not ckpt.exists():
        # fall back to live ckpt if best.pt didn't get saved (e.g. very short run)
        ckpt = run_dir / "lidar_vae.pt"

    # ---- (a) Loss-curve snapshot -------------------------------------
    rc, _ = _run([
        sys.executable, "s2s_min/scripts/plot_train_runs.py",
        "--out",      eval_dir / "loss_comparison.png",
        "--out_solo", loss_plots_dir,
    ], cwd=REPO_ROOT)
    if rc != 0:
        print(f"[eval_after] WARN: plot_train_runs.py exited {rc}")

    # ---- (b) BEV / range visualization -------------------------------
    rc, _ = _run([
        sys.executable, "s2s_min/scripts/visualize_lidar_vae.py",
        "--ckpt", ckpt,
        "--out_dir", eval_dir,
    ], cwd=REPO_ROOT)
    if rc != 0:
        print(f"[eval_after] WARN: visualize_lidar_vae.py exited {rc}")

    # ---- (c) Chamfer eval --------------------------------------------
    if not args.skip_chamfer:
        rc, _ = _run([
            sys.executable, "s2s_min/scripts/vae_chamfer_eval.py",
            "--ckpt", ckpt,
            "--out_json", eval_dir / "chamfer.json",
        ], cwd=REPO_ROOT)
        if rc != 0:
            print(f"[eval_after] WARN: vae_chamfer_eval.py exited {rc}")

    # ---- (d) summary.md ----------------------------------------------
    _write_summary(run_dir, eval_dir)

    print(f"[eval_after] done. artifacts in {eval_dir}")


if __name__ == "__main__":
    main()
