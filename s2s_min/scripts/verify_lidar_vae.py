"""Smoke-load the trained LiDAR VAE checkpoint and verify shapes / round-trip.

Confirms:
  1. Checkpoint loads cleanly (state_dict keys + config match the module).
  2. Encoder produces mu, logvar of [B, 8, 8, 256] from a [B, 3, 32, 1024] input.
  3. Decoder produces [B, 3, 32, 1024] in [0, 1] from a [B, 8, 8, 256] latent.
  4. End-to-end forward returns the documented tuple.
  5. requires_grad_ is correctly off after freeze.

Run:  env/bin/python s2s_min/scripts/verify_lidar_vae.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from s2s_min.models.lidar_vae import LiDARVAE

CKPT = Path(__file__).resolve().parents[1] / "out" / "lidar_vae.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading: {CKPT}")
    ckpt = torch.load(CKPT, map_location=device)
    print(f"  step:   {ckpt.get('step', '<missing>')}")
    print(f"  config: {ckpt['config']}")

    vae = LiDARVAE(**ckpt["config"]).to(device).eval()
    missing, unexpected = vae.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    vae.requires_grad_(False)

    n_params = sum(p.numel() for p in vae.parameters())
    n_tensors = sum(1 for _ in vae.parameters())
    print(f"  params: {n_params/1e6:.2f} M across {n_tensors} tensors")
    print(f"  device: {device}")
    print(f"  requires_grad sum: {sum(p.requires_grad for p in vae.parameters())} (expect 0)")

    print("\n--- shape / round-trip ---")
    B = 2
    x = torch.rand(B, 3, 32, 1024, device=device)
    x[:, 2] = (x[:, 2] > 0.5).float()  # validity is 0/1 in practice
    with torch.no_grad():
        mu, logvar = vae.encode(x)
        print(f"encode: x {tuple(x.shape)} -> mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")
        assert mu.shape == (B, 8, 8, 256), mu.shape
        assert logvar.shape == (B, 8, 8, 256), logvar.shape

        z = mu  # eval mode -> reparameterize returns mu; sanity test it too
        x_hat = vae.decode(z)
        print(f"decode: z {tuple(z.shape)} -> x_hat {tuple(x_hat.shape)}")
        assert x_hat.shape == (B, 3, 32, 1024), x_hat.shape
        assert (x_hat >= 0).all() and (x_hat <= 1).all(), "x_hat out of [0, 1]"

        x_hat_fwd, mu_fwd, logvar_fwd = vae(x)
        assert torch.equal(x_hat, x_hat_fwd), "forward() != decode(encode())[μ] at eval"
        assert torch.equal(mu, mu_fwd)
        assert torch.equal(logvar, logvar_fwd)

    print("\n--- latent stats on random input ---")
    print(f"  mu     : mean={mu.mean().item():+.4f}  std={mu.std().item():.4f}  "
          f"min={mu.min().item():+.4f}  max={mu.max().item():+.4f}")
    print(f"  logvar : mean={logvar.mean().item():+.4f}  std={logvar.std().item():.4f}")
    print(f"  x_hat  : mean={x_hat.mean().item():.4f}  std={x_hat.std().item():.4f}")

    print("\nOK — checkpoint is ready for M2/M3/M4.")


if __name__ == "__main__":
    main()
