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


def _build_dataset(nuscenes_root: Path, subset_file: Path, overfit_n: int):
    tokens = load_subset_tokens(subset_file) if subset_file.exists() else None
    ds = NuScenesLidarKeyframes(nuscenes_root, scene_tokens=tokens)
    if overfit_n > 0:
        n = min(overfit_n, len(ds))
        ds = Subset(ds, list(range(n)))
    return ds


def _format_loss_dict(d: dict) -> str:
    return "  ".join(f"{k}={v.item():.5f}" for k, v in d.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nuscenes_root", type=Path, default=REPO_ROOT / "nuscenes")
    p.add_argument("--subset_file", type=Path,
                   default=S2S_DIR / "out" / "subset_scene_tokens.txt")
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
    p.add_argument("--checkpoint", type=Path,
                   default=S2S_DIR / "out" / "lidar_vae.pt")
    p.add_argument("--no_amp", action="store_true",
                   help="disable mixed precision (default: fp16 on CUDA).")
    args = p.parse_args()

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
    print(f"  EMA decay               : {args.ema_decay}")
    print(f"  checkpoints out         :")
    print(f"    final (live)          : {args.checkpoint}")
    print(f"    final (EMA)           : {args.checkpoint.with_name('lidar_vae_ema.pt')}")
    print(f"    best (EMA, lowest L1_range-EMA): {args.checkpoint.with_name('lidar_vae_best.pt')}")

    ds = _build_dataset(args.nuscenes_root, args.subset_file, args.overfit)
    print(f"  dataset size            : {len(ds)} keyframes "
          f"(overfit_n={args.overfit if args.overfit else 'off'})")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
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
    ckpt_ema_path  = args.checkpoint.with_name("lidar_vae_ema.pt")
    ckpt_best_path = args.checkpoint.with_name("lidar_vae_best.pt")

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


if __name__ == "__main__":
    main()
