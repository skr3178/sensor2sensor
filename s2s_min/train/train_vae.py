"""M1: train the LiDAR VAE.

Two modes:

  Overfit gate (sanity check) — clamp to N samples, train for K steps:
      python -m s2s_min.train.train_vae --overfit 10 --steps 500

  Full epoch over the 10-scene subset (~401 keyframes):
      python -m s2s_min.train.train_vae --epochs 50

Run from the repo root. The dataset comes from `scripts/select_subset.py`
output; if that file doesn't exist yet, the whole trainval split is used
(NOT recommended for a 3060).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
S2S_DIR = REPO_ROOT / "s2s_min"
sys.path.insert(0, str(S2S_DIR))

import torch
from torch.utils.data import DataLoader, Subset

from data.nuscenes_mini import NuScenesLidarKeyframes, load_subset_tokens
from data.waymo import SegmentGroupedRandomSampler, WaymoLidarTopKeyframes
from models.lidar_vae import LiDARVAE
from train.losses import lidar_vae_loss


class WeightEMA:
    """Exponential moving average of model weights, shadow kept on CPU.

    Implements the same EMA scheme as the Sensor2Sensor paper (decay 0.999):
    shadow_t = decay * shadow_{t-1} + (1 - decay) * live_t.

    Keeping the shadow on CPU costs ~10 MB for our 2 M-param VAE (fp32) but
    avoids holding a second copy of weights in VRAM. Update is fast — one
    `.cpu()` copy per optimizer step.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone().to("cpu", dtype=torch.float32) \
                if v.is_floating_point() else v.detach().clone().cpu()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().to("cpu", dtype=torch.float32),
                    alpha=1.0 - self.decay,
                )
            else:
                self.shadow[k] = v.detach().clone().cpu()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


def _build_dataset(args):
    """Pick the right LiDAR backend based on --dataset and apply --overfit clamp."""
    if args.dataset == "nuscenes":
        tokens = (
            load_subset_tokens(args.subset_file)
            if args.subset_file.exists()
            else None
        )
        ds = NuScenesLidarKeyframes(args.nuscenes_root, scene_tokens=tokens)
    elif args.dataset == "waymo":
        ds = WaymoLidarTopKeyframes(
            args.waymo_root,
            split=args.waymo_split,
            H_out=args.waymo_h,
            W_out=args.waymo_w,
            range_max_m=args.waymo_range_max,
            intensity_max=args.waymo_intensity_max,
        )
    else:
        raise ValueError(f"unknown --dataset {args.dataset!r}")

    if args.overfit > 0:
        n = min(args.overfit, len(ds))
        ds = Subset(ds, list(range(n)))
    return ds


def _format_loss_dict(d: dict) -> str:
    return "  ".join(f"{k}={v.item():.5f}" for k, v in d.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["nuscenes", "waymo"], default="nuscenes",
                   help="LiDAR data backend. Defaults to nuscenes (existing M1 setup); "
                        "use 'waymo' to train on the Waymo Open Dataset v2.0.1 download.")
    p.add_argument("--nuscenes_root", type=Path, default=REPO_ROOT / "nuscenes")
    p.add_argument("--subset_file", type=Path,
                   default=S2S_DIR / "out" / "subset_scene_tokens.txt")
    p.add_argument("--waymo_root", type=Path, default=S2S_DIR / "data" / "waymo",
                   help="Waymo v2 root with training/ and validation/ subdirs of LiDAR parquets.")
    p.add_argument("--waymo_split", default="training", choices=["training", "validation"])
    p.add_argument("--waymo_h", type=int, default=64,
                   help="Target H for the Waymo TOP range image (native 64).")
    p.add_argument("--waymo_w", type=int, default=2048,
                   help="Target W (centered crop from native 2650; must be a multiple of 4).")
    p.add_argument("--waymo_range_max", type=float, default=75.0,
                   help="Clamp / divisor for the Waymo range channel (meters).")
    p.add_argument("--waymo_intensity_max", type=float, default=1.5,
                   help="Clamp / divisor for the Waymo intensity channel.")
    p.add_argument("--overfit", type=int, default=0,
                   help="if >0, clamp the dataset to N samples (overfit gate).")
    p.add_argument("--steps", type=int, default=0,
                   help="train for K optimizer steps (used with --overfit). "
                        "Mutually exclusive with --epochs.")
    p.add_argument("--epochs", type=int, default=0,
                   help="train for E full epochs over the dataset.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="micro-batches per optimizer step (effective batch = batch_size * grad_accum).")
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--lr_min", type=float, default=4e-6,
                   help="floor of the cosine schedule (default: lr/100).")
    p.add_argument("--lr_warmup_steps", type=int, default=200,
                   help="linear ramp from 0 to --lr over the first N optimizer steps.")
    p.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine",
                   help="cosine (default) decays lr -> lr_min after warmup; "
                        "constant disables the schedule.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lam_range", type=float, default=50.0,
                   help="L1_range weight. RangeLDM's nuScenes config uses 50; "
                        "range is the geometrically important channel.")
    p.add_argument("--lam_intensity", type=float, default=1.0)
    p.add_argument("--lam_validity", type=float, default=1.0)
    p.add_argument("--lam_kl", type=float, default=1e-6,
                   help="KL weight; X-Drive / RangeLDM default. Lower if posterior collapses.")
    # LPIPS — paper Eqs. (5)(6). Default 0 means "off"; set > 0 to enable.
    p.add_argument("--lam_lpips_normals", type=float, default=1.0,
                   help="LPIPS weight on normals derived from x_range via finite diffs. "
                        "Highest-leverage of the 3 LPIPS terms.")
    p.add_argument("--lam_lpips_intensity", type=float, default=1.0,
                   help="LPIPS weight on intensity (1-ch, replicated to 3 for VGG).")
    p.add_argument("--lam_lpips_validity", type=float, default=1.0,
                   help="LPIPS weight on validity (1-ch, replicated to 3 for VGG).")
    p.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone. 'vgg' (default) is the paper's choice.")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay for shadow weights (kept on CPU).")
    p.add_argument("--best_ema_alpha", type=float, default=0.99,
                   help="smoothing for the L1_range-EMA used to detect "
                        "new-best checkpoints.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=0,
                   help="if >0, save a checkpoint every N optimizer steps.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to live checkpoint .pt. If omitted, written under "
                        "out/runs/<timestamp>__<description>/lidar_vae.pt and the "
                        "top-level out/lidar_vae*.pt compat symlinks are updated. "
                        "Pass a path explicitly to skip the run-folder layout.")
    p.add_argument("--description", type=str, default="untitled",
                   help="Short tag appended to the timestamped run folder name. "
                        "Use spaces-free identifiers like 'v4-lpips-50ep'.")
    p.add_argument("--run_root", type=Path, default=S2S_DIR / "out" / "runs",
                   help="Root directory containing all timestamped run folders.")
    p.add_argument("--no_compat_symlinks", action="store_true",
                   help="Skip updating the back-compat symlinks at out/lidar_vae*.pt. "
                        "Only relevant when running with the default run-folder layout.")
    p.add_argument("--no_amp", action="store_true",
                   help="disable mixed precision (default: fp16 on CUDA).")
    args = p.parse_args()

    # ---- Run folder layout ----
    # When --checkpoint is not given, write into a timestamped subfolder of
    # --run_root and optionally update three compat symlinks at out/ root so
    # downstream scripts (visualize_*, eval/*, etc.) keep finding the latest
    # checkpoint at the historical paths.
    use_run_folder = args.checkpoint is None
    if use_run_folder:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Sanitize description to a single-token slug.
        slug = "".join(c if (c.isalnum() or c in "-_") else "-" for c in args.description.strip())
        run_dir = args.run_root / f"{stamp}__{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint = run_dir / "lidar_vae.pt"
        args._run_dir = run_dir
    else:
        args._run_dir = args.checkpoint.parent

    if args.steps == 0 and args.epochs == 0:
        p.error("specify either --steps (with --overfit) or --epochs")
    if args.steps > 0 and args.epochs > 0:
        p.error("--steps and --epochs are mutually exclusive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print("=" * 70)
    print("M1: LiDAR VAE TRAINING")
    print("=" * 70)
    print(f"  device                  : {device}")
    print(f"  mixed precision (fp16)  : {use_amp}")
    print(f"  batch_size x grad_accum : {args.batch_size} x {args.grad_accum} "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"  lr / weight_decay       : {args.lr} / {args.weight_decay}")
    print(f"  lr schedule             : {args.lr_schedule}  "
          f"(warmup={args.lr_warmup_steps} steps, lr_min={args.lr_min})")
    print(f"  loss weights            : range={args.lam_range}  intensity={args.lam_intensity}  "
          f"validity={args.lam_validity}  kl={args.lam_kl}")
    lpips_on = (args.lam_lpips_normals + args.lam_lpips_intensity + args.lam_lpips_validity) > 0
    print(f"  lpips weights           : normals={args.lam_lpips_normals}  "
          f"intensity={args.lam_lpips_intensity}  validity={args.lam_lpips_validity}  "
          f"({'enabled' if lpips_on else 'disabled'}, net={args.lpips_net})")
    print(f"  EMA decay               : {args.ema_decay}")
    if use_run_folder:
        print(f"  run dir                 : {args._run_dir}")
        print(f"  description             : {args.description!r}")
    print(f"  checkpoints out         :")
    print(f"    final (live)          : {args.checkpoint}")
    print(f"    final (EMA)           : {args.checkpoint.with_name('lidar_vae_ema.pt')}")
    print(f"    best (EMA, lowest L1_range-EMA): {args.checkpoint.with_name('lidar_vae_best.pt')}")
    if use_run_folder and not args.no_compat_symlinks:
        out_root = S2S_DIR / "out"
        print(f"  back-compat symlinks    : {out_root}/lidar_vae{{.pt,_ema.pt,_best.pt}} → {args._run_dir.name}/*")

    ds = _build_dataset(args)
    print(f"  dataset                 : {args.dataset} "
          f"({args.waymo_split if args.dataset == 'waymo' else 'trainval'})")
    print(f"  dataset size            : {len(ds)} keyframes "
          f"(overfit_n={args.overfit if args.overfit else 'off'})")
    # Waymo full-epoch: segment-grouped sampler keeps each worker on one
    # parquet at a time (matches the cache_size=1 budget of
    # WaymoLidarTopKeyframes). In --overfit mode the subset already lives in
    # one segment so plain shuffle is fine and the sampler doesn't apply.
    if args.dataset == "waymo" and not isinstance(ds, Subset):
        sampler = SegmentGroupedRandomSampler(ds, seed=args.seed)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        # Keep worker procs alive across epochs so the per-worker parquet cache
        # survives — otherwise we pay a ~1 s parquet read for every epoch boundary.
        persistent_workers=(args.num_workers > 0),
    )

    model = LiDARVAE(in_channels=3, latent_channels=8, base_channels=32).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  VAE params              : {n_params/1e6:.2f} M")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = WeightEMA(model, decay=args.ema_decay)

    # Frozen LPIPS evaluator (VGG-16 trunk by default). Only built when any
    # `lam_lpips_*` is non-zero — saves ~250 MB VRAM in the LPIPS-off case.
    lpips_module = None
    if lpips_on:
        import lpips  # local import: only required when LPIPS is enabled
        lpips_module = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
        for p_ in lpips_module.parameters():
            p_.requires_grad_(False)
        print(f"  LPIPS module            : net={args.lpips_net}, "
              f"{sum(p.numel() for p in lpips_module.parameters())/1e6:.2f} M params, frozen")

    # Compute total optimizer steps so the cosine schedule knows when to bottom out.
    if args.steps > 0:
        total_steps = args.steps
    else:
        opt_steps_per_epoch = max(1, len(loader) // args.grad_accum)
        total_steps = opt_steps_per_epoch * args.epochs
    decay_steps = max(1, total_steps - args.lr_warmup_steps)
    print(f"  total optimizer steps   : {total_steps}  "
          f"(decay window = {decay_steps} after warmup)")

    if args.lr_schedule == "cosine":
        # Linear warmup from 0 -> lr, then cosine decay from lr -> lr_min.
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1e-6, end_factor=1.0,
            total_iters=max(1, args.lr_warmup_steps),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=decay_steps, eta_min=args.lr_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, cosine],
            milestones=[args.lr_warmup_steps],
        )
    else:
        scheduler = None

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Derive EMA/best names from the live checkpoint so multiple datasets
    # (--checkpoint lidar_vae_waymo.pt vs lidar_vae_nuscenes.pt) don't clobber.
    ckpt_ema_path  = args.checkpoint.with_name(args.checkpoint.stem + "_ema.pt")
    ckpt_best_path = args.checkpoint.with_name(args.checkpoint.stem + "_best.pt")

    # EMA-smoothed L1_range, used to decide when to overwrite `lidar_vae_best.pt`.
    l1_range_ema: float | None = None
    best_l1_range_ema: float = float("inf")

    step = 0
    t_start = time.perf_counter()
    epoch = 0

    def _train_one_batch(x: torch.Tensor) -> dict:
        """One forward+backward; accumulates into the current optimizer step."""
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_hat, mu, logvar = model(x)
            losses = lidar_vae_loss(
                x, x_hat, mu, logvar,
                lam_range=args.lam_range,
                lam_intensity=args.lam_intensity,
                lam_validity=args.lam_validity,
                lam_kl=args.lam_kl,
                lam_lpips_normals=args.lam_lpips_normals,
                lam_lpips_intensity=args.lam_lpips_intensity,
                lam_lpips_validity=args.lam_lpips_validity,
                lpips_module=lpips_module,
            )
        loss = losses["total"] / args.grad_accum
        scaler.scale(loss).backward()
        return losses

    def _optimizer_step():
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        ema.update(model)                       # EMA shadow updates after each optim step

    def _log(step: int, losses: dict, extra: str = ""):
        dt = time.perf_counter() - t_start
        mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  [step {step:5d}  epoch {epoch:3d}  t={dt:6.1f}s  vram={mem:.0f}MiB  "
              f"lr={cur_lr:.2e}]  "
              f"l1r_ema={l1_range_ema:.5f}  "
              f"{_format_loss_dict(losses)}  {extra}")

    def _common_meta() -> dict:
        return {
            "step": step,
            "config": {
                "in_channels": 3, "latent_channels": 8, "base_channels": 32,
                "lam_range": args.lam_range, "lam_intensity": args.lam_intensity,
                "lam_validity": args.lam_validity, "lam_kl": args.lam_kl,
                "lam_lpips_normals": args.lam_lpips_normals,
                "lam_lpips_intensity": args.lam_lpips_intensity,
                "lam_lpips_validity": args.lam_lpips_validity,
                "lpips_net": args.lpips_net if lpips_on else None,
                "ema_decay": args.ema_decay,
                "lr": args.lr, "lr_min": args.lr_min,
                "lr_warmup_steps": args.lr_warmup_steps,
                "lr_schedule": args.lr_schedule,
            },
        }

    def _save_live():
        torch.save({"state_dict": model.state_dict(), **_common_meta()}, args.checkpoint)

    def _save_ema():
        torch.save({"state_dict": ema.state_dict(), **_common_meta()}, ckpt_ema_path)

    def _save_best():
        torch.save({
            "state_dict": ema.state_dict(),
            "l1_range_ema": best_l1_range_ema,
            **_common_meta(),
        }, ckpt_best_path)

    # ----- main loop -----
    optim.zero_grad(set_to_none=True)
    micro = 0
    last_losses: dict = {}

    WARMUP_STEPS_FOR_BEST = 50          # don't fire best-save during random-init noise

    def _after_optim_step() -> None:
        """Run after every optimizer step: update EMAs + maybe save best ckpt."""
        nonlocal l1_range_ema, best_l1_range_ema
        v = last_losses["L1_range"].item()
        l1_range_ema = v if l1_range_ema is None else (
            args.best_ema_alpha * l1_range_ema + (1.0 - args.best_ema_alpha) * v
        )
        if step >= WARMUP_STEPS_FOR_BEST and l1_range_ema < best_l1_range_ema:
            best_l1_range_ema = l1_range_ema
            _save_best()

    if args.steps > 0:
        iter_loader = iter(loader)
        while step < args.steps:
            try:
                x = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                epoch += 1
                x = next(iter_loader)
            last_losses = _train_one_batch(x)
            micro += 1
            if micro == args.grad_accum:
                _optimizer_step()
                step += 1
                micro = 0
                _after_optim_step()
                if step % args.log_every == 0 or step == 1:
                    _log(step, last_losses)
                if args.save_every > 0 and step % args.save_every == 0:
                    _save_live()
                    _save_ema()
        if step % args.log_every != 0:
            _log(step, last_losses)
    else:
        for epoch in range(args.epochs):
            for x in loader:
                last_losses = _train_one_batch(x)
                micro += 1
                if micro == args.grad_accum:
                    _optimizer_step()
                    step += 1
                    micro = 0
                    _after_optim_step()
                    if step % args.log_every == 0 or step == 1:
                        _log(step, last_losses)
                    if args.save_every > 0 and step % args.save_every == 0:
                        _save_live()
                        _save_ema()
            print(f"  -- end of epoch {epoch+1}/{args.epochs} --  "
                  f"l1_range_ema={l1_range_ema:.5f}  best_so_far={best_l1_range_ema:.5f}")
        # Flush any partial accumulation.
        if micro > 0:
            _optimizer_step()
            step += 1
            _after_optim_step()
    _save_live()
    _save_ema()

    print(f"\nFinal: step={step}, total_time={(time.perf_counter() - t_start):.1f}s")
    print(f"  live ckpt   : {args.checkpoint}")
    print(f"  EMA ckpt    : {ckpt_ema_path}")
    print(f"  best EMA    : {ckpt_best_path}  (l1_range_ema={best_l1_range_ema:.5f})")
    if last_losses:
        print(f"  final losses: {_format_loss_dict(last_losses)}")
        print(f"  final l1_range_ema: {l1_range_ema:.5f}")

    # ---- run-folder bookkeeping: metadata + description + compat symlinks ----
    if use_run_folder:
        _write_run_metadata(args, lpips_on, step, t_start, last_losses,
                            l1_range_ema, best_l1_range_ema)
        if not args.no_compat_symlinks:
            _update_compat_symlinks(args._run_dir, S2S_DIR / "out")
        print(f"\n  run folder           : {args._run_dir}")


def _write_run_metadata(args, lpips_on, final_step, t_start, last_losses,
                        final_l1r_ema, best_l1r_ema) -> None:
    """Drop a machine-readable metadata.json + a markdown description stub in the run folder."""
    import json
    import subprocess
    import time as _time
    from datetime import datetime

    run_dir: Path = args._run_dir

    # Best-effort git commit (don't fail the run if git isn't available).
    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    losses_snapshot = (
        {k: v.item() if hasattr(v, "item") else v for k, v in last_losses.items()}
        if last_losses else {}
    )

    meta = {
        "description": args.description,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "wallclock_seconds": round(_time.perf_counter() - t_start, 1),
        "git_commit": git_commit,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items() if not k.startswith("_")},
        "final_step": final_step,
        "final_l1_range_ema": final_l1r_ema,
        "best_l1_range_ema": best_l1r_ema,
        "final_losses": losses_snapshot,
        "lpips_enabled": lpips_on,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Human-readable summary stub — left for the user to expand with notes.
    desc_md = (
        f"# Training run: `{args.description}`\n\n"
        f"- **Started**: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- **Wall-clock**: {meta['wallclock_seconds']:.1f} s\n"
        f"- **Git commit**: `{git_commit or '<unavailable>'}`\n"
        f"- **Final step**: {final_step}\n"
        f"- **Best l1_range_ema**: {best_l1r_ema:.5f}\n"
        f"- **Final l1_range_ema**: {final_l1r_ema:.5f}\n\n"
        f"## Recipe\n\n"
        f"- λ_range / intensity / validity / kl = "
        f"{args.lam_range} / {args.lam_intensity} / {args.lam_validity} / {args.lam_kl}\n"
        f"- λ_lpips_normals / intensity / validity = "
        f"{args.lam_lpips_normals} / {args.lam_lpips_intensity} / {args.lam_lpips_validity} "
        f"({'on' if lpips_on else 'off'}, net={args.lpips_net})\n"
        f"- Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}, "
        f"schedule={args.lr_schedule} (warmup={args.lr_warmup_steps}, lr_min={args.lr_min})\n"
        f"- Batch × grad_accum: {args.batch_size} × {args.grad_accum}\n"
        f"- EMA decay: {args.ema_decay}\n"
        f"- Mixed precision: {not args.no_amp}\n\n"
        f"## Notes (write your observations here)\n\n"
        f"_TODO: what was different about this run? what worked / didn't?_\n"
    )
    (run_dir / "description.md").write_text(desc_md)


def _update_compat_symlinks(run_dir: Path, out_root: Path) -> None:
    """Point out/lidar_vae{,_ema,_best}.pt at the newest run's files.

    Uses relative symlinks so the tree is portable. Replaces existing files
    or symlinks at those paths; leaves real files alone if they're not
    symlinks AND `_can_replace_path` returns False (cautious by default).
    """
    import os
    for fname in ("lidar_vae.pt", "lidar_vae_ema.pt", "lidar_vae_best.pt"):
        src = run_dir / fname
        if not src.exists():
            continue                                # e.g. _best.pt may be absent if --steps < warmup
        link = out_root / fname
        if link.is_symlink() or not link.exists():
            try:
                link.unlink(missing_ok=True)
            except TypeError:                       # py3.7 fallback
                if link.exists() or link.is_symlink():
                    link.unlink()
            rel_target = os.path.relpath(src, link.parent)
            link.symlink_to(rel_target)
        else:
            # Real file at this path (legacy non-symlink). Don't clobber silently.
            print(f"  WARN: {link} is a real file, not a symlink — leaving alone. "
                  f"Move/delete it manually to enable compat-symlink updates.")

    # Always-current latest pointer to the run dir itself.
    latest = out_root / "latest_run"
    try:
        latest.unlink(missing_ok=True)
    except TypeError:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
    rel_run = os.path.relpath(run_dir, latest.parent)
    latest.symlink_to(rel_run, target_is_directory=True)


if __name__ == "__main__":
    main()
