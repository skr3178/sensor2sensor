"""Diffusion wrapper — exposes a small surface over diffusers' schedulers.

Two responsibilities bundled into one class:
  - **Training noise injection** via `DDPMScheduler.add_noise()` + `get_velocity()`.
  - **Inference sampling** via `DDIMScheduler.step()` for 25-step DDIM.

We use v-prediction throughout (more stable than ε-prediction at small batch).

This is a thin wrapper — the heavy lifting stays inside diffusers. The wrapper
exists so M0's smoke_test, M3's training loop, and M4's inference all share
the same diffusion contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler


class DiffusionWrapper:
    """Bundles training (DDPM) and inference (DDIM) schedulers.

    Args:
        num_train_timesteps: total diffusion steps (default 1000, SD/ADM convention).
        beta_schedule:       diffusers beta schedule name ("scaled_linear" is SD's default).
        prediction_type:     "v_prediction" (recommended) or "epsilon".
        inference_steps:     DDIM step count for sampling (default 25, fast and stable).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "v_prediction",
        inference_steps: int = 25,
    ):
        common = dict(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.train_scheduler = DDPMScheduler(**common)
        # CRITICAL: clip_sample=False. The diffusers DDIMScheduler default clips
        # pred_original_sample to [-1, +1] on every step (meant for image diffusion
        # in pixel space). Our LiDAR latent has values up to ±5; the default clip
        # caps z_pred std at ~0.6 × μ std and destroys reconstruction quality.
        # Phase 0 root-cause analysis (s2s_min/docs/lidar-unet.md §11.9, 2026-05-31):
        # disabling this clip moved DDIM-25 cos(z_pred, μ) from 0.74 → 0.9955 on a
        # memorized sample. Training is unaffected (DDPMScheduler.add_noise() doesn't
        # clip; only step() does, which is inference-only).
        self.inference_scheduler = DDIMScheduler(**common, clip_sample=False)
        self.num_train_timesteps = num_train_timesteps
        self.inference_steps = inference_steps
        self.prediction_type = prediction_type

    # --- training helpers ------------------------------------------------

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Random integer timesteps in `[0, num_train_timesteps)`."""
        return torch.randint(
            low=0, high=self.num_train_timesteps,
            size=(batch_size,), device=device, dtype=torch.long,
        )

    def add_noise(self, z: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward (q-sample) noising: z_t = sqrt(αbar_t)·z + sqrt(1-αbar_t)·noise."""
        return self.train_scheduler.add_noise(z, noise, t)

    def get_target(self, z: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get the training target appropriate to the prediction_type.

        For v_prediction: v = α·noise - σ·z   (per Salimans & Ho 2022).
        For epsilon:     just `noise`.
        """
        if self.prediction_type == "v_prediction":
            return self.train_scheduler.get_velocity(z, noise, t)
        elif self.prediction_type == "epsilon":
            return noise
        else:
            raise ValueError(f"unknown prediction_type: {self.prediction_type}")

    # --- inference -------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        unet: nn.Module,
        shape: tuple[int, ...],
        kv_context: torch.Tensor,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDIM 25-step sampling loop.

        Args:
            unet:       the LiDARUNet (or any module with signature `(z, t, kv) -> v`).
            shape:      sample tensor shape, e.g. (B, 8, 8, 256).
            kv_context: [B, kv_channels, H_kv, W_kv] image+raymap conditioning.
            device:     where to allocate the noise.
            generator:  optional torch Generator for reproducible noise.

        Returns:
            Sampled latent of shape `shape`.
        """
        self.inference_scheduler.set_timesteps(self.inference_steps, device=device)
        z = torch.randn(*shape, device=device, generator=generator)

        for t in self.inference_scheduler.timesteps:
            t_batch = t.expand(shape[0]).to(device)
            model_out = unet(z, t_batch, kv_context)
            z = self.inference_scheduler.step(model_out, t, z).prev_sample

        return z

    @torch.no_grad()
    def ddim_sample_cfg(
        self,
        unet: nn.Module,
        shape: tuple[int, ...],
        kv_context: torch.Tensor,
        device: torch.device,
        cfg_scale: float = 3.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDIM sampling with classifier-free guidance.

        Identical to `ddim_sample` when `cfg_scale == 1.0`. Above 1.0, runs the
        U-Net on a 2× batch per step (concatenated unconditional + conditional)
        and mixes the two predictions via:

            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        The training prerequisite (the U-Net having seen `kv_context = 0` for
        some fraction of training steps) is satisfied by our `--cond_dropout 0.2`
        on the M3 bs16 run.

        Pattern: LiDAR-Diffusion (Reference_code/LiDAR-Diffusion/lidm/models/diffusion/ddim.py:175-179).
        Batched form is ~30 % faster than two sequential forward passes because
        the U-Net's small batch is doubled into one larger kernel launch.

        Args:
            unet, shape, kv_context, device, generator: as in `ddim_sample`.
            cfg_scale: guidance scale `w`. 1.0 = no guidance (= `ddim_sample`).
                       Typical range: 1.5 (subtle) … 7.5 (heavy, may over-saturate).

        Returns:
            Sampled latent of shape `shape`.
        """
        if cfg_scale == 1.0:
            return self.ddim_sample(unet, shape, kv_context, device, generator)

        self.inference_scheduler.set_timesteps(self.inference_steps, device=device)
        z = torch.randn(*shape, device=device, generator=generator)
        kv_uncond = torch.zeros_like(kv_context)

        for t in self.inference_scheduler.timesteps:
            t_batch = t.expand(shape[0]).to(device)
            # Batched CFG: [uncond half; cond half] concatenated on batch dim.
            z_in  = torch.cat([z, z], dim=0)                    # [2B, ...]
            t_in  = torch.cat([t_batch, t_batch], dim=0)
            kv_in = torch.cat([kv_uncond, kv_context], dim=0)
            pred_pair = unet(z_in, t_in, kv_in)
            pred_uncond, pred_cond = pred_pair.chunk(2, dim=0)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            z = self.inference_scheduler.step(pred, t, z).prev_sample

        return z
