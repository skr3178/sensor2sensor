"""M0 U-Net validation tests.

Runs on CPU. Validates:
  1. timestep_embedding shape + non-zero output.
  2. TimestepMLP shape.
  3. ResBlock with t_emb forward + backward.
  4. ResBlock without t_emb still works (LiDAR VAE backward compatibility).
  5. EncoderLevel new modes: t_emb + return_skip.
  6. EncoderLevel old mode (M-1 backward compatibility).
  7. Bottleneck shape.
  8. DecoderLevel shape with skip-concat.
  9. Full LiDARUNet: forward shape, backward gradient flow, all params get grads,
     param count in target range, zero-init head yields near-zero initial output.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.blocks import EncoderLevel, ResBlock
from models.timestep import TimestepMLP, timestep_embedding
from models.unet import Bottleneck, DecoderLevel, LiDARUNet, count_params


def _ok(name: str, actual, expected=None) -> None:
    if expected is not None:
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print(f"  OK  {name:<55} -> {actual}")


def test_timestep_embedding():
    print("\n[1] timestep_embedding(t=[1, 50, 999], dim=96)")
    t = torch.tensor([1, 50, 999])
    emb = timestep_embedding(t, dim=96)
    _ok("shape", tuple(emb.shape), (3, 96))
    assert emb.abs().max() > 0, "timestep_embedding produced all zeros"
    # Different timesteps should give different embeddings.
    assert (emb[0] - emb[2]).abs().max() > 0.1, "different t should give different emb"
    print(f"      stats: mean={emb.mean():+.3f} std={emb.std():.3f}")


def test_timestep_mlp():
    print("\n[2] TimestepMLP(96 -> 384)")
    mlp = TimestepMLP(in_dim=96, out_dim=384)
    out = mlp(torch.randn(2, 96))
    _ok("shape", tuple(out.shape), (2, 384))


def test_resblock_with_t_emb():
    print("\n[3] ResBlock(96, 192, t_emb_dim=384) with t_emb")
    block = ResBlock(in_ch=96, out_ch=192, t_emb_dim=384)
    x = torch.randn(2, 96, 8, 256, requires_grad=True)
    t_emb = torch.randn(2, 384)
    out = block(x, t_emb=t_emb)
    _ok("shape", tuple(out.shape), (2, 192, 8, 256))
    # Backward sanity: every param must at least receive a .grad tensor (graph reaches it).
    # Note: by design, zero-init conv2 BLOCKS gradient flow to upstream layers on step 1
    # (the gradient w.r.t. their pre-conv2 activation is zero × upstream Jacobian = 0).
    # So on the first backward pass, only `skip` + `conv2` get non-zero gradients (4 of 12).
    # This is the whole point of zero-init — fresh block acts as the skip identity and
    # only "wakes up" upstream params on subsequent steps once conv2 has non-zero weights.
    out.sum().backward()
    n_total = sum(1 for _ in block.parameters())
    n_with_attr = sum(1 for p in block.parameters() if p.grad is not None)
    n_nonzero = sum(1 for p in block.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"      params with .grad attr: {n_with_attr}/{n_total}")
    print(f"      params with non-zero .grad: {n_nonzero}/{n_total}  (zero-init blocks upstream on step 1)")
    assert n_with_attr == n_total, "every param must be reached by autograd"
    assert n_nonzero >= 4, "at least the output conv + skip path must learn on step 1"


def test_resblock_without_t_emb():
    print("\n[4] ResBlock(64, 64) without t_emb_dim — backward-compat with LiDAR VAE")
    block = ResBlock(in_ch=64, out_ch=64)
    x = torch.randn(2, 64, 16, 512)
    out = block(x)  # No t_emb argument — should still work
    _ok("shape", tuple(out.shape), (2, 64, 16, 512))


def test_encoder_level_new_mode():
    print("\n[5] EncoderLevel(t_emb_dim=384, return_skip=True)")
    lvl = EncoderLevel(
        in_ch=96, out_ch=192, kv_channels=10,
        num_res_blocks=2, num_heads=8, do_downsample=True,
        t_emb_dim=384, return_skip=True,
    )
    x = torch.randn(2, 96, 8, 256)
    kv = torch.randn(2, 10, 8, 64)
    t_emb = torch.randn(2, 384)
    out, skip = lvl(x, kv, t_emb=t_emb)
    _ok("downsampled shape", tuple(out.shape), (2, 192, 8, 128))
    _ok("skip shape", tuple(skip.shape), (2, 192, 8, 256))


def test_encoder_level_old_mode():
    print("\n[6] EncoderLevel old mode (M-1 backward-compat: no t_emb, no skip)")
    lvl = EncoderLevel(in_ch=96, out_ch=192, kv_channels=10)
    x = torch.randn(2, 96, 8, 256)
    kv = torch.randn(2, 10, 8, 64)
    out = lvl(x, kv)
    _ok("output shape", tuple(out.shape), (2, 192, 8, 128))


def test_bottleneck():
    print("\n[7] Bottleneck(192 -> 384) at 8x64")
    btl = Bottleneck(in_ch=192, out_ch=384, kv_channels=10, t_emb_dim=384)
    x = torch.randn(2, 192, 8, 64)
    kv = torch.randn(2, 10, 8, 64)
    t_emb = torch.randn(2, 384)
    out = btl(x, kv, t_emb=t_emb)
    _ok("shape", tuple(out.shape), (2, 384, 8, 64))


def test_decoder_level():
    print("\n[8] DecoderLevel: upsample + cat(skip) + 2x blocks (384, skip=192) -> 192 @ 8x128")
    dec = DecoderLevel(
        in_ch=384, skip_ch=192, out_ch=192, kv_channels=10,
        num_res_blocks=2, num_heads=8, do_upsample=True, t_emb_dim=384,
    )
    x = torch.randn(2, 384, 8, 64)
    skip = torch.randn(2, 192, 8, 128)
    kv = torch.randn(2, 10, 8, 64)
    t_emb = torch.randn(2, 384)
    out = dec(x, skip, kv, t_emb=t_emb)
    _ok("shape", tuple(out.shape), (2, 192, 8, 128))


def test_lidar_unet_forward():
    print("\n[9] LiDARUNet — full forward")
    unet = LiDARUNet()  # all defaults
    z = torch.randn(1, 8, 8, 256)
    t = torch.tensor([500])
    kv = torch.randn(1, 10, 8, 64)
    out = unet(z, t, kv)
    _ok("output shape", tuple(out.shape), (1, 8, 8, 256))

    # Zero-init head: first forward output should be approximately zero everywhere
    # (the SiLU-then-zero-conv head can't generate any signal yet).
    max_out = out.abs().max().item()
    print(f"      |out|.max() = {max_out:.2e} (zero-init head → expect near zero)")
    assert max_out < 1e-3, f"zero-init head broken: max |out| = {max_out}"


def test_lidar_unet_backward():
    print("\n[10] LiDARUNet — backward, every param accumulates grad")
    unet = LiDARUNet()
    z = torch.randn(1, 8, 8, 256, requires_grad=True)
    t = torch.tensor([500])
    kv = torch.randn(1, 10, 8, 64)
    out = unet(z, t, kv)
    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    n_total = sum(1 for _ in unet.parameters())
    n_with_grad = sum(1 for p in unet.parameters() if p.grad is not None)
    print(f"      params with grad attribute: {n_with_grad}/{n_total}")
    assert n_with_grad == n_total, f"some params missed: {n_total - n_with_grad}"
    # Loss must be finite.
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    print(f"      loss value: {loss.item():.4f}  (finite ✓)")


def test_lidar_unet_param_count():
    print("\n[11] LiDARUNet — param count")
    unet = LiDARUNet()
    n_m = count_params(unet) / 1e6
    print(f"      trainable params: {n_m:.2f} M")
    # Original plan target was 25-35M based on an estimate that mistakenly assumed
    # kv_dim=384 inside CrossAttention. The actual kv_dim is 10 (raw KV-context
    # channels), so each CrossAttn is ~6× cheaper than estimated. 14-18M is the
    # correct range for full attention at every level. Smaller is better for the 3060.
    assert 10 < n_m < 25, f"param count {n_m:.2f}M out of expected [10, 25] M range"


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("M0: LIDAR U-NET VALIDATION (CPU)")
    print("=" * 70)

    test_timestep_embedding()
    test_timestep_mlp()
    test_resblock_with_t_emb()
    test_resblock_without_t_emb()
    test_encoder_level_new_mode()
    test_encoder_level_old_mode()
    test_bottleneck()
    test_decoder_level()
    test_lidar_unet_forward()
    test_lidar_unet_backward()
    test_lidar_unet_param_count()

    print("\n" + "=" * 70)
    print("M0 U-NET TESTS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
