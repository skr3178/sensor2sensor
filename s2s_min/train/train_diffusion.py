"""M3: train the conditional LiDAR diffusion U-Net.

Reads pre-encoded latents from `s2s_min/out/cached_latents/` (produced by M2).
The two frozen encoders (SD VAE, LiDAR VAE) are NOT loaded — saves ~250 MB VRAM
and skips ~60 ms of encoder forward per step.

Three usage modes:

  M3.0 — one optimizer step on a single sample (smoke test):
      python -m s2s_min.train.train_diffusion --overfit 1 --steps 1

  M3.1 — overfit a fixed 10-sample subset for ~1000 steps:
      python -m s2s_min.train.train_diffusion --overfit 10 --steps 1000

  M3.2 — one full epoch on all 401 cached samples:
      python -m s2s_min.train.train_diffusion --epochs 1

Run from the repo root. Cache must exist at `--cache_dir` (default
`s2s_min/out/cached_latents/`) — run `train/cache_latents.py` first.

Structure mirrors `train/train_vae.py` exactly: same WeightEMA, same
gradient-accumulation loop, same cosine+warmup schedule, same three-checkpoint
scheme. The only diffusion-specific code is the inner training step
(`_train_one_batch`) which samples t, noises mu, and computes MSE between the
U-Net's v_pred and the scheduler's v_target.
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
import torch.nn.functional as F

# PyTorch's internal flash SDPA kernel has an `is_sm80` check that fails on
# RTX 3060 (sm_86, technically newer but trips the hardcoded check). Disable
# it so we fall back to mem-efficient SDPA, which works on sm_75+.
# Verified: 60M model forward+backward runs cleanly after this.
# Remove once we either upgrade PyTorch to 2.1+ or swap in the external
# flash-attn library directly.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
from torch.utils.data import DataLoader, Subset

from data.cached_latents import CachedLatentsDataset
from models.diffusion import DiffusionWrapper
from models.unet import LiDARUNet, count_params


KV_POOL_H = 8
KV_POOL_W = 64


class WeightEMA:
    """Exponential moving average of model weights, shadow kept on CPU.

    Same scheme as `train_vae.py:WeightEMA`. Decay 0.999 by default per the
    Sensor2Sensor paper. CPU shadow costs ~60 MB for our 14.81 M-param U-Net
    (fp32) — keeps VRAM headroom intact during M3 training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = (
                v.detach().clone().to("cpu", dtype=torch.float32)
                if v.is_floating_point()
                else v.detach().clone().cpu()
            )

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


def _collate(batch: list[dict]) -> dict:
    """Stack tensors; keep `sample_token` as a list of strings."""
    out: dict = {}
    for k in ("image_latent", "raymap", "mu"):
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    out["sample_token"] = [item["sample_token"] for item in batch]
    if "logvar" in batch[0]:
        out["logvar"] = torch.stack([item["logvar"] for item in batch], dim=0)
    return out


def _build_kv_context(image_latent: torch.Tensor, raymap: torch.Tensor) -> torch.Tensor:
    """[B, 4, 32, 56] + [B, 6, 32, 56] -> [B, 10, 32, 56] -> pool to [B, 10, 8, 64]."""
    kv_full = torch.cat([image_latent, raymap], dim=1)
    return F.adaptive_avg_pool2d(kv_full, (KV_POOL_H, KV_POOL_W))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=Path,
                   default=S2S_DIR / "out" / "cached_latents",
                   help="directory of pre-encoded .npz latents (output of cache_latents.py)")
    p.add_argument("--overfit", type=int, default=0,
                   help="if >0, clamp the dataset to N samples (overfit gate, M3.0/M3.1).")
    p.add_argument("--steps", type=int, default=0,
                   help="train for K optimizer steps. Mutually exclusive with --epochs.")
    p.add_argument("--epochs", type=int, default=0,
                   help="train for E full epochs over the dataset.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="micro-batches per optimizer step (effective batch = batch_size * grad_accum).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cond_dropout", type=float, default=0.2,
                   help="probability per sample of zeroing the kv_context (CFG hook, paper §B.2).")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="global-norm gradient clip; paper default 1.0.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--best_ema_alpha", type=float, default=0.99,
                   help="smoothing for the loss-EMA used to detect new-best checkpoints.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=0,
                   help="if >0, save a checkpoint every N optimizer steps.")
    p.add_argument("--checkpoint", type=Path,
                   default=S2S_DIR / "out" / "lidar_unet.pt")
    p.add_argument("--no_amp", action="store_true",
                   help="disable mixed precision (default: fp16 on CUDA).")
    # U-Net architecture knobs (forwarded to LiDARUNet). Defaults reproduce the
    # legacy 3-stage (96, 192, 384) ~14.81 M config bitwise (back-compat verified
    # via tests/test_unet_nstage_regression.py).
    p.add_argument("--stem_channels", type=int, default=96,
                   help="stem conv output channels. Wider stem = wider every level.")
    p.add_argument("--level_channels", type=int, nargs="+", default=[96, 192, 384],
                   metavar="C",
                   help="channels per U-Net level (last entry = bottleneck width). "
                        "3 entries → 3-stage; 4 entries → 4-stage paper-match. "
                        "Phase-1 target: 192 384 768 (~60M). Phase-4 target: "
                        "160 320 640 1024 (~125M).")
    args = p.parse_args()

    if args.steps == 0 and args.epochs == 0:
        p.error("specify either --steps (typically with --overfit) or --epochs")
    if args.steps > 0 and args.epochs > 0:
        p.error("--steps and --epochs are mutually exclusive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print("=" * 70)
    print("M3: LiDAR DIFFUSION U-NET TRAINING")
    print("=" * 70)
    print(f"  device                  : {device}")
    print(f"  mixed precision (fp16)  : {use_amp}")
    print(f"  batch_size x grad_accum : {args.batch_size} x {args.grad_accum} "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"  lr / weight_decay       : {args.lr} / {args.weight_decay}")
    print(f"  lr schedule             : {args.lr_schedule}  "
          f"(warmup={args.lr_warmup_steps} steps, lr_min={args.lr_min})")
    print(f"  cond dropout            : {args.cond_dropout}  (per-sample CFG hook)")
    print(f"  grad clip               : {args.grad_clip}")
    print(f"  EMA decay               : {args.ema_decay}")
    _stem_for_print = args.checkpoint.stem
    print(f"  checkpoints out         :")
    print(f"    final (live)          : {args.checkpoint}")
    print(f"    final (EMA)           : {args.checkpoint.with_name(f'{_stem_for_print}_ema.pt')}")
    print(f"    best (EMA, lowest loss-EMA): {args.checkpoint.with_name(f'{_stem_for_print}_best.pt')}")

    # ----- dataset -----
    ds: torch.utils.data.Dataset = CachedLatentsDataset(args.cache_dir)
    if args.overfit > 0:
        n = min(args.overfit, len(ds))
        ds = Subset(ds, list(range(n)))
    print(f"  dataset size            : {len(ds)} cached samples "
          f"(overfit_n={args.overfit if args.overfit else 'off'})")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=_collate,
    )

    # ----- model + diffusion -----
    unet = LiDARUNet(
        stem_channels=args.stem_channels,
        level_channels=tuple(args.level_channels),
    ).to(device)
    unet.train()
    n_params = count_params(unet)
    print(f"  U-Net arch              : stem={args.stem_channels}, "
          f"levels={tuple(args.level_channels)} ({len(args.level_channels)}-stage)")
    print(f"  U-Net params            : {n_params/1e6:.2f} M")
    diffusion = DiffusionWrapper()
    print(f"  diffusion               : {diffusion.prediction_type}, "
          f"T={diffusion.num_train_timesteps}, DDIM steps={diffusion.inference_steps}")

    # ----- optimizer + scheduler + EMA + AMP -----
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = WeightEMA(unet, decay=args.ema_decay)

    if args.steps > 0:
        total_steps = args.steps
    else:
        opt_steps_per_epoch = max(1, len(loader) // args.grad_accum)
        total_steps = opt_steps_per_epoch * args.epochs
    decay_steps = max(1, total_steps - args.lr_warmup_steps)
    print(f"  total optimizer steps   : {total_steps}  "
          f"(decay window = {decay_steps} after warmup)")

    if args.lr_schedule == "cosine":
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
    # Derive EMA/best paths from the live checkpoint's stem so distinct --checkpoint
    # values produce distinct EMA/best files (e.g. --checkpoint lidar_unet_m32.pt →
    # lidar_unet_m32_ema.pt / lidar_unet_m32_best.pt). Avoids cross-run overwrites.
    stem = args.checkpoint.stem
    ckpt_ema_path  = args.checkpoint.with_name(f"{stem}_ema.pt")
    ckpt_best_path = args.checkpoint.with_name(f"{stem}_best.pt")

    loss_ema: float | None = None
    best_loss_ema: float = float("inf")
    step = 0
    epoch = 0
    t_start = time.perf_counter()

    def _train_one_batch(batch: dict) -> dict:
        """One forward+backward; accumulates into the current optimizer step.

        Returns a dict with 'mse' (the loss component we report + smooth).
        """
        # All tensors come from cache as fp32 — promote to device.
        image_latent = batch["image_latent"].to(device, non_blocking=True)   # [B, 4, 32, 56]
        raymap       = batch["raymap"].to(device, non_blocking=True)         # [B, 6, 32, 56]
        mu           = batch["mu"].to(device, non_blocking=True)             # [B, 8, 8, 256]

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Build per-batch KV context.
            kv_context = _build_kv_context(image_latent, raymap)             # [B, 10, 8, 64]
            # Classifier-free-guidance hook: per-sample, zero kv_context with prob cond_dropout.
            if args.cond_dropout > 0:
                B = kv_context.shape[0]
                drop = (torch.rand(B, device=device) < args.cond_dropout)    # [B]
                kv_context = kv_context * (~drop).view(B, 1, 1, 1).to(kv_context.dtype)

            # Sample timesteps + noise; compute v_target via the diffusion wrapper.
            t = diffusion.sample_timesteps(mu.shape[0], device=device)
            noise = torch.randn_like(mu)
            z_noisy = diffusion.add_noise(mu, noise, t)
            v_target = diffusion.get_target(mu, noise, t)

            v_pred = unet(z_noisy, t, kv_context)
            loss = F.mse_loss(v_pred, v_target)

        scaler.scale(loss / args.grad_accum).backward()
        return {"mse": loss.detach()}

    def _optimizer_step():
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=args.grad_clip)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        ema.update(unet)

    def _log(step: int, losses: dict, extra: str = ""):
        dt = time.perf_counter() - t_start
        mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0
        cur_lr = optim.param_groups[0]["lr"]
        ema_str = f"{loss_ema:.5f}" if loss_ema is not None else "n/a"
        print(f"  [step {step:5d}  epoch {epoch:3d}  t={dt:6.1f}s  vram={mem:.0f}MiB  "
              f"lr={cur_lr:.2e}]  "
              f"mse_ema={ema_str}  "
              f"mse={losses['mse'].item():.5f}  {extra}")

    def _common_meta() -> dict:
        return {
            "step": step,
            "config": {
                # U-Net architecture (from CLI; defaults reproduce legacy 3-stage).
                "in_channels": 8, "out_channels": 8,
                "stem_channels": args.stem_channels,
                "level_channels": list(args.level_channels),
                "num_res_blocks": 2,
                "kv_channels": 10, "num_heads": 8,
                # Diffusion + training settings.
                "prediction_type": diffusion.prediction_type,
                "num_train_timesteps": diffusion.num_train_timesteps,
                "cond_dropout": args.cond_dropout,
                "ema_decay": args.ema_decay,
                "grad_clip": args.grad_clip,
                "lr": args.lr, "lr_min": args.lr_min,
                "lr_warmup_steps": args.lr_warmup_steps,
                "lr_schedule": args.lr_schedule,
                "weight_decay": args.weight_decay,
            },
        }

    def _save_live():
        torch.save({"state_dict": unet.state_dict(), **_common_meta()}, args.checkpoint)

    def _save_ema():
        torch.save({"state_dict": ema.state_dict(), **_common_meta()}, ckpt_ema_path)

    def _save_best():
        torch.save({
            "state_dict": ema.state_dict(),
            "loss_ema": best_loss_ema,
            **_common_meta(),
        }, ckpt_best_path)

    optim.zero_grad(set_to_none=True)
    micro = 0
    last_losses: dict = {}
    WARMUP_STEPS_FOR_BEST = 50

    def _after_optim_step() -> None:
        nonlocal loss_ema, best_loss_ema
        v = last_losses["mse"].item()
        loss_ema = v if loss_ema is None else (
            args.best_ema_alpha * loss_ema + (1.0 - args.best_ema_alpha) * v
        )
        if step >= WARMUP_STEPS_FOR_BEST and loss_ema < best_loss_ema:
            best_loss_ema = loss_ema
            _save_best()

    if args.steps > 0:
        iter_loader = iter(loader)
        while step < args.steps:
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                epoch += 1
                batch = next(iter_loader)
            last_losses = _train_one_batch(batch)
            micro += 1
            if micro == args.grad_accum:
                _optimizer_step()
                step += 1
                micro = 0
                _after_optim_step()
                if step % args.log_every == 0 or step == 1:
                    _log(step, last_losses)
                if args.save_every > 0 and step % args.save_every == 0:
                    _save_live(); _save_ema()
        if step % args.log_every != 0:
            _log(step, last_losses)
    else:
        for epoch in range(args.epochs):
            for batch in loader:
                last_losses = _train_one_batch(batch)
                micro += 1
                if micro == args.grad_accum:
                    _optimizer_step()
                    step += 1
                    micro = 0
                    _after_optim_step()
                    if step % args.log_every == 0 or step == 1:
                        _log(step, last_losses)
                    if args.save_every > 0 and step % args.save_every == 0:
                        _save_live(); _save_ema()
            ema_str = f"{loss_ema:.5f}" if loss_ema is not None else "n/a"
            best_str = f"{best_loss_ema:.5f}" if best_loss_ema < float("inf") else "n/a"
            print(f"  -- end of epoch {epoch+1}/{args.epochs} --  "
                  f"mse_ema={ema_str}  best_so_far={best_str}")
        if micro > 0:
            _optimizer_step()
            step += 1
            _after_optim_step()

    _save_live(); _save_ema()

    print(f"\nFinal: step={step}, total_time={(time.perf_counter() - t_start):.1f}s")
    print(f"  live ckpt   : {args.checkpoint}")
    print(f"  EMA ckpt    : {ckpt_ema_path}")
    best_str = f"{best_loss_ema:.5f}" if best_loss_ema < float("inf") else "(not yet saved — too few steps)"
    print(f"  best EMA    : {ckpt_best_path}  (loss_ema={best_str})")
    if last_losses:
        print(f"  final mse   : {last_losses['mse'].item():.5f}")
        ema_str = f"{loss_ema:.5f}" if loss_ema is not None else "n/a"
        print(f"  final mse_ema: {ema_str}")


if __name__ == "__main__":
    main()
