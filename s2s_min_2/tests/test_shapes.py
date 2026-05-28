"""M-1: tensor-shape sanity check.

Runs cheaply on CPU. Validates:
  1. CrossAttention(query 384, kv 10, heads 8) accepts the planned grids.
  2. SelfAttention is shape-preserving.
  3. One EncoderLevel (ResBlock + SelfAttn + CrossAttn + DownsampleW) maps the
     planned `[B, 96, 8, 256]` input through level-0 channels [96 -> 192] and
     halves W -> `[B, 192, 8, 128]`.
  4. CircularConv2d preserves spatial dims and wraps W correctly (a single
     non-zero pixel at column 0 produces a kernel response at column W-1
     because of the circular pad).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_shapes.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.attention import CrossAttention, SelfAttention
from models.blocks import CircularConv2d, EncoderLevel, ResBlock


def _assert_shape(name: str, actual: torch.Size, expected: tuple) -> None:
    expected = torch.Size(expected)
    if actual != expected:
        raise AssertionError(f"{name}: expected {tuple(expected)}, got {tuple(actual)}")
    print(f"  OK  {name:<45} -> {tuple(actual)}")


def test_cross_attention():
    """Plan M-1 step 1-3."""
    print("\n[1] CrossAttention(q=384, kv=10, heads=8)")
    ca = CrossAttention(q_channels=384, kv_channels=10, num_heads=8)
    # Plan calls for grids matching the bottleneck: Q at 8x64, KV at 8x64.
    q  = torch.randn(2, 384, 8, 64)
    kv = torch.randn(2, 10, 8, 64)
    out = ca(q, kv)
    _assert_shape("output", out.shape, (2, 384, 8, 64))


def test_self_attention():
    print("\n[2] SelfAttention(channels=96)")
    sa = SelfAttention(channels=96, num_heads=8)
    x = torch.randn(2, 96, 8, 256)
    out = sa(x)
    _assert_shape("output", out.shape, (2, 96, 8, 256))


def test_encoder_level():
    """Plan M-1 step 4: full Level-0 — channels 96 -> 192, W: 256 -> 128, H=8 fixed."""
    print("\n[3] EncoderLevel level-0 (96 -> 192, downsample W)")
    lvl = EncoderLevel(
        in_ch=96,
        out_ch=192,
        kv_channels=10,
        num_res_blocks=2,
        num_heads=8,
        do_downsample=True,
    )
    x  = torch.randn(2, 96, 8, 256)
    kv = torch.randn(2, 10, 8, 64)
    out = lvl(x, kv)
    _assert_shape("output", out.shape, (2, 192, 8, 128))

    n_params = sum(p.numel() for p in lvl.parameters())
    print(f"      params: {n_params/1e6:.2f} M")


def test_circular_padding():
    """Sanity check: a kernel that just averages the 3 W-neighbours should
    'see' column W-1 when fed an impulse at column 0 (circular wrap).
    Verifies CircularConv2d implements wrap-around on W.
    """
    print("\n[4] CircularConv2d W-wrap behaviour")
    conv = CircularConv2d(1, 1, kernel_size=3)
    # Manually set weights to a mean filter along W only, zero along H.
    with torch.no_grad():
        conv.conv.weight.zero_()
        conv.conv.bias.zero_()
        # weight shape: [out_ch=1, in_ch=1, kH=3, kW=3]. Set middle row, all cols, to 1/3.
        conv.conv.weight[0, 0, 1, :] = 1.0 / 3.0

    x = torch.zeros(1, 1, 4, 8)
    x[0, 0, 1, 0] = 1.0  # impulse at row 1, col 0
    out = conv(x)        # [1, 1, 4, 8]

    # Expect non-zero response at cols {-1, 0, +1} mod 8 = {7, 0, 1} in the same row.
    nonzero_cols = (out[0, 0, 1].abs() > 1e-6).nonzero(as_tuple=True)[0].tolist()
    expected = sorted([7, 0, 1])
    assert sorted(nonzero_cols) == expected, (
        f"W-wrap broken: expected nonzero at {expected}, got {nonzero_cols}"
    )
    print(f"  OK  impulse at col 0 -> nonzero cols {nonzero_cols} (wrap proven)")


def test_resblock_zero_init():
    """The output conv is zero-init, so a fresh ResBlock starts as identity (+ skip)."""
    print("\n[5] ResBlock zero-init -> initial output equals skip path")
    rb = ResBlock(in_ch=96, out_ch=96)
    x = torch.randn(2, 96, 8, 256)
    out = rb(x)
    # With conv2 = 0, residual contribution is 0, so out == skip(x) == x.
    diff = (out - x).abs().max().item()
    assert diff < 1e-5, f"ResBlock zero-init broken: max diff {diff}"
    print(f"  OK  max |out - x| = {diff:.2e}")


def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("M-1: TENSOR SHAPE SANITY CHECK (CPU)")
    print("=" * 60)

    test_cross_attention()
    test_self_attention()
    test_encoder_level()
    test_circular_padding()
    test_resblock_zero_init()

    print("\n" + "=" * 60)
    print("M-1 PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
