"""M1: LiDAR VAE shape + zero-init sanity check.

Verifies models.md sec 2.4 steps 3-5:
    encode: [B, 3, 32, 1024] -> mu, logvar each [B, 8, 8, 256]
    decode: [B, 8, 8, 256]   -> x_hat [B, 3, 32, 1024], all values in [0, 1]
    forward: full encode -> sample -> decode round trip
    reparameterize: deterministic in eval, stochastic in train

Also asserts the zero-init head property: at init the decoder outputs ~0.5
on every channel (because the head conv is zero-init and sigmoid(0) = 0.5).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.lidar_vae import LiDARVAE


def _assert_shape(name: str, actual: torch.Size, expected: tuple) -> None:
    expected = torch.Size(expected)
    if actual != expected:
        raise AssertionError(f"{name}: expected {tuple(expected)}, got {tuple(actual)}")
    print(f"  OK  {name:<40} -> {tuple(actual)}")


def test_encoder_shape():
    print("\n[1] Encoder: [B, 3, 32, 1024] -> mu, logvar [B, 8, 8, 256]")
    vae = LiDARVAE(in_channels=3, latent_channels=8, base_channels=32).eval()
    x = torch.randn(2, 3, 32, 1024)
    with torch.no_grad():
        mu, logvar = vae.encode(x)
    _assert_shape("mu",     mu.shape,     (2, 8, 8, 256))
    _assert_shape("logvar", logvar.shape, (2, 8, 8, 256))

    n_params = sum(p.numel() for p in vae.parameters())
    n_enc = sum(p.numel() for n, p in vae.named_parameters() if n.startswith("enc_"))
    n_dec = sum(p.numel() for n, p in vae.named_parameters() if n.startswith("dec_"))
    print(f"      params: encoder={n_enc/1e6:.2f} M  decoder={n_dec/1e6:.2f} M  "
          f"total={n_params/1e6:.2f} M")


def test_decoder_shape_and_zero_init():
    """Step 4 in models.md sec 2.4:
       - decode([B, 8, 8, 256]) -> x_hat [B, 3, 32, 1024]
       - at init, output is uniformly 0.5 (zero-init head + sigmoid)
    """
    print("\n[2] Decoder: [B, 8, 8, 256] -> x_hat [B, 3, 32, 1024]  +  zero-init -> 0.5")
    vae = LiDARVAE().eval()
    z = torch.randn(2, 8, 8, 256)
    with torch.no_grad():
        x_hat = vae.decode(z)
    _assert_shape("x_hat", x_hat.shape, (2, 3, 32, 1024))

    # Zero-init head means sigmoid(0) = 0.5 exactly, regardless of z.
    err = (x_hat - 0.5).abs().max().item()
    assert err < 1e-6, f"Zero-init head broken: max |x_hat - 0.5| = {err}"
    print(f"  OK  zero-init decoder output is 0.5 (max |x_hat - 0.5| = {err:.2e})")

    # Output must lie in [0, 1] (sigmoid).
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "decoder output escaped [0, 1]"
    print(f"  OK  output range = [{x_hat.min():.4f}, {x_hat.max():.4f}]  (subset of [0, 1])")


def test_forward_round_trip():
    """Full forward through reparameterize + decoder."""
    print("\n[3] forward round trip: input -> encode -> sample -> decode")
    vae = LiDARVAE().eval()
    x = torch.rand(2, 3, 32, 1024)                # already in [0, 1] like real data
    with torch.no_grad():
        x_hat, mu, logvar = vae(x)
    _assert_shape("x_hat",  x_hat.shape,  (2, 3, 32, 1024))
    _assert_shape("mu",     mu.shape,     (2, 8, 8, 256))
    _assert_shape("logvar", logvar.shape, (2, 8, 8, 256))
    print(f"  OK  x_hat range = [{x_hat.min():.4f}, {x_hat.max():.4f}]")


def test_reparameterize_train_vs_eval():
    """In train mode the sampler perturbs; in eval mode it returns mu exactly."""
    print("\n[4] reparameterize: train != eval")
    vae = LiDARVAE()
    mu = torch.zeros(1, 8, 8, 256)
    logvar = torch.zeros(1, 8, 8, 256)  # sigma = 1

    vae.eval()
    z_eval = vae.reparameterize(mu, logvar)
    assert torch.equal(z_eval, mu), "eval-mode reparameterize must return mu"
    print(f"  OK  eval z == mu (max diff = 0)")

    vae.train()
    torch.manual_seed(0)
    z_train_a = vae.reparameterize(mu, logvar)
    torch.manual_seed(1)
    z_train_b = vae.reparameterize(mu, logvar)
    diff = (z_train_a - z_train_b).abs().max().item()
    assert diff > 0.1, f"train-mode reparameterize should be stochastic, max diff {diff}"
    print(f"  OK  train z stochastic (max diff between two draws = {diff:.3f})")


def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("M1: LiDAR VAE ENCODER + DECODER SHAPE TEST (CPU)")
    print("=" * 60)

    test_encoder_shape()
    test_decoder_shape_and_zero_init()
    test_forward_round_trip()
    test_reparameterize_train_vs_eval()

    print("\n" + "=" * 60)
    print("M1 VAE SHAPE / ZERO-INIT TESTS PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
