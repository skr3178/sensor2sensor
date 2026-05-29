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
        self.inference_scheduler = DDIMScheduler(**common)
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
