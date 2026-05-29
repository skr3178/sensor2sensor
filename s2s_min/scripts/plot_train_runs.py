"""Render a comparison plot of LiDAR VAE training runs from their .log files.

Output: s2s_min/out/loss_comparison.png

Panels (2 × 4 grid):
    total              (overall optimization signal)
    L1_range           (geometry — main quality metric)
    L1_intensity       (reflectance)
    BCE_validity       (return-mask canary — climbs first when training fails)
    KL                 (latent regularizer, λ_KL=1e-6)
    LPIPS_normals      (paper Eq. 6 — only present in v4+)
    LPIPS_intensity    (paper Eq. 6 — only present in v4+)
    LPIPS_validity     (paper Eq. 6 — only present in v4+)

v1/v2/v3 don't log the LPIPS terms — their lines are simply absent from those panels.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S = REPO_ROOT / "s2s_min"
LOGS = {
    "v1 (λ_range=1, constant LR, no EMA)":     S2S / "out" / "train_vae_epoch50.log",
    "v2 (λ_range=50, constant LR, EMA)":       S2S / "out" / "train_vae_epoch50_v2.log",
    "v3 (λ_range=50, cosine LR, EMA)":         S2S / "out" / "train_vae_epoch50_v3.log",
    "v4 (v3 + 3 LPIPS terms, 10 scenes)":      S2S / "out" / "train_vae_epoch50_v4_lpips.log",
    "v5 (v4 + 100 scenes, batch16)":           S2S / "out" / "train_vae_v5_nohup.log",
}
DEFAULT_OUT      = S2S / "out" / "loss_comparison.png"
DEFAULT_OUT_SOLO = S2S / "out" / "loss_plots"             # one PNG per loss term

# --- CLI overrides so eval_after.py can redirect into a per-run snapshot ---
import argparse as _argparse
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                help="Combined 8-panel PNG path.")
_p.add_argument("--out_solo", type=Path, default=DEFAULT_OUT_SOLO,
                help="Directory for the per-metric standalone PNGs.")
_args, _ = _p.parse_known_args()
OUT      = _args.out
OUT_SOLO = _args.out_solo

# Single regex to capture all numeric `key=value` pairs after the `[step …]` prefix.
STEP_RE = re.compile(r"\[step\s+(\d+)\s+epoch")
KV_RE   = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|inf)")

# What each panel shows. key matches the metric name as it appears in the log;
# scale=log for anything that spans orders of magnitude, linear for KL and total.
PANELS = [
    ("total",            "total loss",                                 "log"),
    ("L1_range",         "L1_range  (lower = better; main quality)",   "log"),
    ("L1_intensity",     "L1_intensity",                                "log"),
    ("BCE_validity",     "BCE_validity  (canary — first to spike)",     "log"),
    ("KL",               "KL  (regularizer; λ_KL=1e-6 caps effect)",    "linear"),
    ("LPIPS_normals",    "LPIPS_normals  (v4+ only)",                   "log"),
    ("LPIPS_intensity",  "LPIPS_intensity  (v4+ only)",                 "log"),
    ("LPIPS_validity",   "LPIPS_validity  (v4+ only)",                  "log"),
]

COLORS = {
    "v1 (λ_range=1, constant LR, no EMA)":     "#888888",   # gray
    "v2 (λ_range=50, constant LR, EMA)":       "#1f77b4",   # blue
    "v3 (λ_range=50, cosine LR, EMA)":         "#d62728",   # red
    "v4 (v3 + 3 LPIPS terms, 10 scenes)":      "#2ca02c",   # green
    "v5 (v4 + 100 scenes, batch16)":           "#9467bd",   # purple — newest committed
}


def parse_log(path: Path) -> dict[str, list[float]]:
    """Return {metric_name: list[float]} aligned with the 'step' list."""
    series: dict[str, list[float]] = {"step": []}
    for line in path.read_text().splitlines():
        sm = STEP_RE.search(line)
        if not sm:
            continue
        series["step"].append(int(sm.group(1)))
        kv = dict(KV_RE.findall(line))
        # Skip obvious non-metric keys (step, epoch, t, vram, lr).
        for k, v in kv.items():
            if k in ("step", "epoch", "t", "vram", "lr"):
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            series.setdefault(k, []).extend([float("nan")] * (len(series["step"]) - 1 - len(series.get(k, []))))
            series.setdefault(k, []).append(val)
    # Right-pad any short series with NaN so they align with `step`.
    n = len(series["step"])
    for k in list(series.keys()):
        if k == "step":
            continue
        if len(series[k]) < n:
            series[k].extend([float("nan")] * (n - len(series[k])))
    return series


def main():
    series_by_run: dict[str, dict[str, list[float]]] = {}
    for label, path in LOGS.items():
        if not path.exists():
            print(f"missing: {path}")
            continue
        s = parse_log(path)
        series_by_run[label] = s
        keys = sorted(k for k in s.keys() if k != "step")
        print(f"{label}: {len(s['step'])} log points; metrics = {keys}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)

    for ax, (metric, title, scale) in zip(axes.flat, PANELS):
        plotted = 0
        for label, s in series_by_run.items():
            if metric not in s:
                continue
            ax.plot(s["step"], s[metric], label=label, color=COLORS[label],
                    linewidth=1.5, alpha=0.9)
            plotted += 1
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer step")
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.3)
        if plotted == 0:
            ax.text(0.5, 0.5, "(metric not logged in any run)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#aaaaaa", fontsize=9)

    # Mark the divergence steps we identified for v1/v2/v3 (v4 has no divergence).
    div_steps = {
        "v1": (2000, "#888888"),
        "v2": ( 950, "#1f77b4"),
        "v3": (1050, "#d62728"),
    }
    for ax in axes.flat:
        for _, (step, color) in div_steps.items():
            ax.axvline(step, color=color, linestyle="--", linewidth=0.7, alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.99), fontsize=9, frameon=False)

    fig.suptitle("LiDAR VAE training — all loss terms across v1 / v2 / v3 / v4",
                 y=1.03, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote combined: {OUT}")

    # ----- Standalone per-panel PNGs --------------------------------------
    OUT_SOLO.mkdir(parents=True, exist_ok=True)
    for metric, title, scale in PANELS:
        fig1, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for label, s in series_by_run.items():
            if metric not in s:
                continue
            ax.plot(s["step"], s[metric], label=label, color=COLORS[label],
                    linewidth=1.8, alpha=0.9)
            plotted += 1
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(metric)
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.3)
        for _, (step, color) in div_steps.items():
            ax.axvline(step, color=color, linestyle="--", linewidth=0.7, alpha=0.4)
        if plotted == 0:
            ax.text(0.5, 0.5, "(metric not logged in any run)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#aaaaaa", fontsize=10)
        else:
            ax.legend(loc="best", fontsize=9, frameon=True)
        fig1.tight_layout()
        out_solo = OUT_SOLO / f"{metric}.png"
        fig1.savefig(out_solo, dpi=140, bbox_inches="tight")
        plt.close(fig1)
        print(f"wrote standalone: {out_solo}")


if __name__ == "__main__":
    main()
