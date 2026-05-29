"""Tiny helper: timestamped run folders for eval/diagnostic outputs.

Mirrors the convention training runs already use (e.g.
`s2s_min/out/runs/2026-05-28_142002__v5-100scenes-bs16-lpips-nohup/`).

Two functions:
  - `new_run_folder(descriptor)`            — create + return a fresh timestamped folder.
  - `maintain_latest_symlink(latest, run)`  — point a stable "latest" path at the new run.
      If `latest` exists as a regular directory (i.e. pre-cleanup state), its contents
      are archived into a sibling `*-legacy-*` run folder before the symlink replaces it.

Anything in `s2s_min/out/runs/` is intended to be self-contained and not shared with
other runs — so each invocation of an eval script writes to its own dated folder.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

RUNS_ROOT = Path("s2s_min/out/runs")


def new_run_folder(descriptor: str, parent: Path | None = None) -> Path:
    """Create and return `<parent>/YYYY-MM-DD_HHMMSS__<descriptor>/`.

    Default `parent` is `s2s_min/out/runs/`. Pass an explicit `parent` to nest the
    run folder under a specific checkpoint's training folder — for example,
    `parent=<unet-train-folder>/m4_eval` so the eval lives alongside the
    checkpoint that produced it.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if parent is None:
        parent = RUNS_ROOT
    folder = parent / f"{ts}__{descriptor}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def maintain_latest_symlink(latest: Path, target: Path) -> None:
    """Make `latest` resolve to `target`, archiving any pre-existing dir.

    - `latest` is a symlink → unlink, recreate.
    - `latest` is a regular dir → move its contents into a fresh legacy run folder,
      remove the empty dir, then create the symlink. Printed to stdout so the user
      can see the one-time migration happen.
    - `latest` doesn't exist → just create the symlink.

    The symlink is created as a path relative to `latest.parent` so it stays valid
    if the repo is moved.
    """
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists() and latest.is_dir():
        archive = new_run_folder(f"{latest.name}-legacy-pre-runs-cleanup")
        for item in latest.iterdir():
            shutil.move(str(item), str(archive / item.name))
        latest.rmdir()
        print(f"[runfolder] moved legacy {latest}/* → {archive}/")

    rel = target.resolve().relative_to(latest.resolve().parent if latest.exists()
                                       else latest.parent.resolve())
    latest.symlink_to(rel, target_is_directory=True)
