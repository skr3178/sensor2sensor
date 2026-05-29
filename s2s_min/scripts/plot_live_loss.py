"""Live loss plot for an in-progress M3 training run.

Parses `<run_folder>/train.log`, extracts the per-step `mse` and smoothed `mse_ema`
values, and writes `<run_folder>/live_loss.png`. Designed to be invoked repeatedly —
each call overwrites the PNG. Pair with `watch -n 30` or a bash refresh loop while
training is running; open the PNG in an auto-refreshing image viewer to see progress.

Usage:
    python s2s_min/scripts/plot_live_loss.py \
        --run_dir s2s_min/out/runs/2026-05-28_161242__m3-unet-v5cache-50ep-bs16

Refresh loop (background):
    while kill -0 $(cat <run>/train.pid) 2>/dev/null; do
        python s2s_min/scripts/plot_live_loss.py --run_dir <run> >/dev/null
        sleep 30
    done
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_PAT = re.compile(
    r"\[step\s+(\d+)\s+epoch\s+(\d+)\s+t=\s*([\d.]+)s.*?"
    r"lr=([\d.eE+-]+)\].*?"
    r"mse_ema=([\d.]+)\s+mse=([\d.]+)"
)
EPOCH_END_PAT = re.compile(
    r"--\s+end of epoch (\d+)/\d+\s+--.*?mse_ema=([\d.]+)\s+best_so_far=([\d.]+)"
)


def parse_log(log_path: Path) -> dict:
    """Return per-step series and per-epoch best/end metrics."""
    txt = log_path.read_text()
    steps, mse, mse_ema, lrs, walls = [], [], [], [], []
    for m in STEP_PAT.finditer(txt):
        steps.append(int(m.group(1)))
        walls.append(float(m.group(3)))
        lrs.append(float(m.group(4)))
        mse_ema.append(float(m.group(5)))
        mse.append(float(m.group(6)))

    epoch_ends = []  # (epoch, mse_ema, best_so_far)
    for m in EPOCH_END_PAT.finditer(txt):
        epoch_ends.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    # Detect a "currently best" line (live ckpt) — train_diffusion prints
    # `  final mse_ema: 0.xxxxx` on graceful exit; not present mid-run.
    finished = "final mse_ema:" in txt

    return {
        "steps": steps, "mse": mse, "mse_ema": mse_ema, "lrs": lrs, "walls": walls,
        "epoch_ends": epoch_ends, "finished": finished,
    }


def plot(run_dir: Path, out_path: Path) -> None:
    data = parse_log(run_dir / "train.log")
    desc = (run_dir / "description.md").read_text().splitlines()[0].lstrip("# ").strip() \
           if (run_dir / "description.md").exists() else run_dir.name

    if not data["steps"]:
        # Empty log → just write a placeholder image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "no step lines parsed yet — training is starting up",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ---- loss panel ----
    ax_loss.plot(data["steps"], data["mse"], color="0.7", lw=0.8, alpha=0.6, label="mse (per step)")
    ax_loss.plot(data["steps"], data["mse_ema"], color="tab:blue", lw=1.6, label="mse_ema (smoothed)")

    # epoch boundaries (vertical lines)
    for ep, ema, best in data["epoch_ends"]:
        # find the step where this epoch ended
        # approximate by searching for the closest log step (epoch_end log line has no step)
        # — we instead use the last step of that epoch from our parsed series
        eps = [s for s, e in zip(data["steps"], [None]*len(data["steps"])) if False]  # placeholder
        # better: use raw txt re-parse to find step at end-of-epoch
    # simpler: shade alternating epochs using parsed epoch column
    # (skipped — keep visualization clean)

    # best-so-far per-epoch annotation
    if data["epoch_ends"]:
        last_ep, last_ema, last_best = data["epoch_ends"][-1]
        ax_loss.axhline(last_best, color="tab:green", lw=0.8, linestyle="--", alpha=0.7,
                        label=f"best so far ({last_best:.4f}, end ep {last_ep})")

    ax_loss.set_ylabel("loss")
    ax_loss.set_title(
        f"{desc}\n"
        f"step {data['steps'][-1]} (epoch {1 + data['epoch_ends'][-1][0] if data['epoch_ends'] else 0}) — "
        f"wall {data['walls'][-1]/60:.1f} min — "
        f"mse_ema={data['mse_ema'][-1]:.5f}"
        + ("  ← training in progress" if not data["finished"] else "  ← training DONE")
    )
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    # log-scale y if range is wide
    if max(data["mse_ema"]) / max(min(data["mse_ema"]), 1e-6) > 5.0:
        ax_loss.set_yscale("log")

    # ---- lr panel ----
    ax_lr.plot(data["steps"], data["lrs"], color="tab:orange", lw=1.2)
    ax_lr.set_xlabel("optimizer step")
    ax_lr.set_ylabel("lr")
    ax_lr.grid(True, alpha=0.3)
    ax_lr.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Path to the M3 run folder containing train.log.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <run_dir>/live_loss.png).")
    args = p.parse_args()
    out = args.out or (args.run_dir / "live_loss.png")
    plot(args.run_dir, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
