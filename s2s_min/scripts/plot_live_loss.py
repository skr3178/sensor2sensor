"""Live multi-metric training plot for an in-progress diffusion run.

Reads TWO data sources from `<run_folder>/`:

  1. `train.log`            — per-step `mse`/`mse_ema` (free), LR, epoch boundaries
  2. `<stem>_eval.jsonl`    — periodic held-out CD-3D + CD-BEV + cos-sim at t∈{0,500,999}
                              (written by `--inline_eval` per `train/inline_eval.py`)

Renders a 3-panel layout (per `lidar-unet.md §15.x` stopping-criterion fix):

    ┌─────────────────────────────────────────────────────┐
    │  Held-out CD-3D-raw (m)                             │ ← PRIMARY (the real metric)
    │  + CD-BEV-raw                                       │
    ├─────────────────────────────────────────────────────┤
    │  Cos-sim @ t=0, t=500, t=999                        │ ← conditioning-strength signal
    ├─────────────────────────────────────────────────────┤
    │  mse_ema (diagnostic only, NOT stopping signal)     │ ← demoted
    │  LR                                                 │
    └─────────────────────────────────────────────────────┘

If inline-eval JSONL doesn't exist (e.g. training run without `--inline_eval`),
falls back to legacy single-panel mse_ema + LR plot.

Usage:
    python s2s_min/scripts/plot_live_loss.py --run_dir <run_folder>

Refresh loop (background):
    while kill -0 $(cat <run>/train.pid) 2>/dev/null; do
        python s2s_min/scripts/plot_live_loss.py --run_dir <run> >/dev/null
        sleep 30
    done
"""
from __future__ import annotations

import argparse
import json
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


def parse_train_log(log_path: Path) -> dict:
    """Return per-step series + per-epoch end metrics + done flag."""
    txt = log_path.read_text() if log_path.exists() else ""
    steps, mse, mse_ema, lrs, walls = [], [], [], [], []
    for m in STEP_PAT.finditer(txt):
        steps.append(int(m.group(1)))
        walls.append(float(m.group(3)))
        lrs.append(float(m.group(4)))
        mse_ema.append(float(m.group(5)))
        mse.append(float(m.group(6)))

    epoch_ends = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
                  for m in EPOCH_END_PAT.finditer(txt)]
    finished = "final mse_ema:" in txt

    return {
        "steps": steps, "mse": mse, "mse_ema": mse_ema, "lrs": lrs, "walls": walls,
        "epoch_ends": epoch_ends, "finished": finished,
    }


def find_eval_jsonl(run_dir: Path) -> Path | None:
    """Look for `<stem>_eval.jsonl` in the run dir. Returns the first match or None."""
    candidates = sorted(run_dir.glob("*_eval.jsonl"))
    return candidates[0] if candidates else None


def parse_eval_jsonl(jsonl_path: Path) -> dict:
    """Read the per-event JSONL written by train/inline_eval.py. Returns separate
    arrays for cos-sim events and CD events (they fire at different cadences)."""
    cos_steps, cos_t0, cos_t500, cos_t999 = [], [], [], []
    cd_steps, cd_3d, cd_bev = [], [], []
    best_cd_held = float("inf")

    if not jsonl_path or not jsonl_path.exists():
        return {"cos_steps": cos_steps, "cos_t0": cos_t0, "cos_t500": cos_t500,
                "cos_t999": cos_t999, "cd_steps": cd_steps, "cd_3d": cd_3d,
                "cd_bev": cd_bev, "best_cd_held": best_cd_held}

    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = r.get("step")
        if step is None:
            continue
        if "cos_t0" in r:
            cos_steps.append(step)
            cos_t0.append(r.get("cos_t0"))
            cos_t500.append(r.get("cos_t500"))
            cos_t999.append(r.get("cos_t999"))
        if "cd_3d_mean" in r:
            cd_steps.append(step)
            cd_3d.append(r.get("cd_3d_mean"))
            cd_bev.append(r.get("cd_bev_mean"))
            best = r.get("best_cd_held")
            if best is not None and best < best_cd_held:
                best_cd_held = best

    return {"cos_steps": cos_steps, "cos_t0": cos_t0, "cos_t500": cos_t500,
            "cos_t999": cos_t999, "cd_steps": cd_steps, "cd_3d": cd_3d,
            "cd_bev": cd_bev, "best_cd_held": best_cd_held}


def plot(run_dir: Path, out_path: Path) -> None:
    train = parse_train_log(run_dir / "train.log")
    eval_path = find_eval_jsonl(run_dir)
    have_eval = eval_path is not None and eval_path.exists() and \
                len(parse_eval_jsonl(eval_path).get("cos_steps", [])) > 0
    desc = (run_dir / "description.md").read_text().splitlines()[0].lstrip("# ").strip() \
           if (run_dir / "description.md").exists() else run_dir.name

    if not train["steps"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "no step lines parsed yet — training is starting up",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    # Title
    progress_tag = "  ← training in progress" if not train["finished"] else "  ← training DONE"
    cur_step = train["steps"][-1]
    cur_wall = train["walls"][-1] / 60
    cur_ema = train["mse_ema"][-1]
    cur_ep = (train["epoch_ends"][-1][0] + 1) if train["epoch_ends"] else 0
    title = (f"{desc}\nstep {cur_step} (epoch {cur_ep}) — wall {cur_wall:.1f} min — "
             f"mse_ema={cur_ema:.5f}{progress_tag}")

    if not have_eval:
        # ── Legacy 2-panel layout (mse_ema + LR) for runs without --inline_eval ──
        fig, (ax_loss, ax_lr) = plt.subplots(
            2, 1, figsize=(11, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_loss.plot(train["steps"], train["mse"], color="0.7", lw=0.8, alpha=0.6, label="mse")
        ax_loss.plot(train["steps"], train["mse_ema"], color="tab:blue", lw=1.6, label="mse_ema")
        if train["epoch_ends"]:
            _, _, last_best = train["epoch_ends"][-1]
            ax_loss.axhline(last_best, color="tab:green", lw=0.8, ls="--", alpha=0.7,
                            label=f"best so far ({last_best:.4f})")
        ax_loss.set_ylabel("loss"); ax_loss.set_title(title); ax_loss.legend(loc="upper right", fontsize=9)
        ax_loss.grid(True, alpha=0.3)
        if max(train["mse_ema"]) / max(min(train["mse_ema"]), 1e-6) > 5.0:
            ax_loss.set_yscale("log")
        ax_lr.plot(train["steps"], train["lrs"], color="tab:orange", lw=1.2)
        ax_lr.set_xlabel("optimizer step"); ax_lr.set_ylabel("lr")
        ax_lr.set_yscale("log"); ax_lr.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    # ── 3-panel layout: CD (primary) + cos-sim + mse/LR (demoted) ──
    ev = parse_eval_jsonl(eval_path)
    fig, (ax_cd, ax_cos, ax_loss) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )

    # ── PRIMARY: held-out CD-3D-raw + CD-BEV ──
    if ev["cd_steps"]:
        ax_cd.plot(ev["cd_steps"], ev["cd_3d"], "o-", color="tab:red", lw=2.0, ms=5,
                    label=f"CD-3D-raw  (latest {ev['cd_3d'][-1]:.4f} m)")
        ax_cd.plot(ev["cd_steps"], ev["cd_bev"], "s-", color="tab:purple", lw=1.5, ms=4, alpha=0.7,
                    label=f"CD-BEV-raw (latest {ev['cd_bev'][-1]:.4f} m)")
        if ev["best_cd_held"] < float("inf"):
            ax_cd.axhline(ev["best_cd_held"], color="tab:green", lw=1.0, ls="--", alpha=0.7,
                          label=f"best CD-3D so far ({ev['best_cd_held']:.4f} m)")
        # Reference: 100scenes baseline (from lidar-unet.md §14 / §12.3)
        ax_cd.axhline(2.241, color="0.5", lw=0.8, ls=":", alpha=0.7,
                      label="100scenes baseline (held-out, n=16): 2.241 m")
    else:
        ax_cd.text(0.5, 0.5, "no CD eval data yet (--cd_eval_every cadence)",
                   ha="center", va="center", transform=ax_cd.transAxes, color="gray")
    ax_cd.set_ylabel("Chamfer distance (m)  ★ PRIMARY")
    ax_cd.set_title(title)
    ax_cd.legend(loc="upper right", fontsize=8)
    ax_cd.grid(True, alpha=0.3)

    # ── SECONDARY: cos-sim at fixed timesteps ──
    if ev["cos_steps"]:
        ax_cos.plot(ev["cos_steps"], ev["cos_t0"],   "o-", lw=1.2, ms=3, color="tab:blue",
                    label="cos @ t=0   (must stay ~1)")
        ax_cos.plot(ev["cos_steps"], ev["cos_t500"], "s-", lw=1.2, ms=3, color="tab:orange",
                    label="cos @ t=500 (mid-noise; rises as model learns)")
        ax_cos.plot(ev["cos_steps"], ev["cos_t999"], "^-", lw=1.2, ms=3, color="tab:green",
                    label="cos @ t=999 (high-noise; conditioning strength)")
        ax_cos.axhline(1.0,  color="black", lw=0.5, alpha=0.4)
        ax_cos.axhline(0.0,  color="black", lw=0.5, alpha=0.2)
    else:
        ax_cos.text(0.5, 0.5, "no cos-sim data yet (--cos_eval_every cadence)",
                    ha="center", va="center", transform=ax_cos.transAxes, color="gray")
    ax_cos.set_ylabel("cos(ẑ₀, μ)  ↑")
    ax_cos.set_ylim(-0.1, 1.05)
    ax_cos.legend(loc="lower right", fontsize=8)
    ax_cos.grid(True, alpha=0.3)

    # ── DEMOTED: mse_ema + LR (training-loss diagnostic only) ──
    ax_loss.plot(train["steps"], train["mse"], color="0.75", lw=0.6, alpha=0.4)
    ax_loss.plot(train["steps"], train["mse_ema"], color="tab:gray", lw=1.2,
                 label=f"mse_ema (diagnostic — see §15 for why not the stopping signal)")
    ax_loss.set_ylabel("training v-MSE  (diag)")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")
    # LR overlay on twin y
    ax_lr = ax_loss.twinx()
    ax_lr.plot(train["steps"], train["lrs"], color="tab:orange", lw=0.8, alpha=0.7)
    ax_lr.set_ylabel("lr", color="tab:orange")
    ax_lr.set_yscale("log")
    ax_lr.tick_params(axis="y", labelcolor="tab:orange")
    ax_loss.set_xlabel("optimizer step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Path to the run folder containing train.log (and optionally *_eval.jsonl).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <run_dir>/live_loss.png).")
    args = p.parse_args()
    out = args.out or (args.run_dir / "live_loss.png")
    plot(args.run_dir, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
