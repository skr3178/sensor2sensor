# LiDAR Diffusion U-Net — current vs paper-match 4-stage scale-up

## Contents

1. [Why this document exists](#why-this-document-exists)
2. [Current architecture — 3-stage, ~14.81 M params](#current-architecture--3-stage-1481-m-params)
3. [Proposed D — 4-stage paper-match, ~125 M params](#proposed-d--4-stage-paper-match-125-m-params)
4. [Side-by-side row-by-row comparison](#side-by-side-row-by-row-comparison)
5. [Module-type catalog — is anything NEW being introduced?](#module-type-catalog--is-anything-new-being-introduced)
6. [Numeric summary table](#numeric-summary-table)
7. [What the 4th stage actually buys](#what-the-4th-stage-actually-buys)
8. [What the structural refactor entails](#what-the-structural-refactor-entails)
9. [Where this fits in the larger plan](#where-this-fits-in-the-larger-plan)

---

## Why this document exists

The current LiDAR diffusion U-Net at [`s2s_min/models/unet.py`](s2s_min/models/unet.py) has **14.81 M params** and produced `CD-3D-raw = 2.486 m` (at CFG `w = 2.5`, 16 held-out samples). The paper's LiDAR-tower sub-module is **~125 M params**. The CFG sweep's modest 7.9% gain and gentle saturation at `w ≈ 3.0` (see [`s2s_min/docs/lidar-unet.md §9`](s2s_min/docs/lidar-unet.md)) strongly suggest the U-Net capacity gap is the dominant remaining bottleneck.

This document records:

1. The exact tensor shapes at every level of the **current** 3-stage U-Net.
2. The exact tensor shapes at every level of the **proposed** 4-stage paper-match variant (`level_channels = (160, 320, 640, 1024)`, ~125 M params, option D).
3. A **row-by-row side-by-side comparison** so you can verify whether the architecture is fundamentally the same (just bigger) or if there are genuinely new module types.
4. The structural refactor sketch and verification protocol.

**Tensor convention:** PyTorch NCHW — `[B, C, H, W]`. LiDAR latent enters at `H = 8` (elevation rows) and `W = 256` (azimuth columns). H is preserved throughout — only W is downsampled. KV cross-attention context is pre-pooled to `[B, 10, 8, 64]` once outside the U-Net and reused at every block.

---

## Current architecture — 3-stage, ~14.81 M params

```text
Input  z_noisy [B, 8, 8, 256]               KV context (pre-pooled, shared)
   │                                         [B, 10, 8, 64]   (4 image + 6 raymap)
   ▼                                         │
┌───────────────────────────────────────┐    │
│ STEM   Conv2d(8 → 96, 3×3, circ-W)    │    │
│        [B, 96, 8, 256]                │    │
└───────────────────────────────────────┘    │
   │                                         │
   ▼                                         │
┌───────────────────────────────────────┐    │
│ ENC L0 — width 96                     │ ◀──┤
│  2× ResBlock(96 → 96) + FiLM(t)       │    │
│  + SelfAttn + CrossAttn               │    │
│  [B, 96, 8, 256]                      │    │
└───────────────────────────────────────┘    │
   │                                         │
   ├─→ skip_0  [B, 96, 8, 256] ──────┐       │
   ▼                                 │       │
   DOWN W (stride 2 on W only)       │       │
   [B, 96, 8, 256] → [B, 96, 8, 128] │       │
   │                                 │       │
   ▼                                 │       │
┌───────────────────────────────────────┐    │
│ ENC L1 — width 192                    │ ◀──┤
│  2× ResBlock(96 → 192) + FiLM         │    │
│  + SelfAttn + CrossAttn               │    │
│  [B, 192, 8, 128]                     │    │
└───────────────────────────────────────┘    │
   │                                 │       │
   ├─→ skip_1  [B, 192, 8, 128] ───┐ │       │
   ▼                               │ │       │
   DOWN W                          │ │       │
   [B, 192, 8, 128] → [B, 192, 8, 64]        │
   │                               │ │       │
   ▼                               │ │       │
┌───────────────────────────────────────┐    │
│ BOTTLENECK — width 384                │ ◀──┤
│  2× ResBlock(192 → 384) + FiLM        │    │
│  + SelfAttn + CrossAttn               │    │
│  [B, 384, 8, 64]                      │    │
└───────────────────────────────────────┘    │
   │                               │ │       │
   ▼                               │ │       │
   UP W                            │ │       │
   [B, 384, 8, 64] → [B, 384, 8, 128]        │
   │                               │ │       │
   concat skip_1 ◀─────────────────┘ │       │
   [B, 576 (= 384+192), 8, 128]    │ │       │
   │                                 │       │
   ▼                                 │       │
┌───────────────────────────────────────┐    │
│ DEC L1 — width 192                    │ ◀──┤
│  2× ResBlock(576 → 192) + FiLM        │    │
│  + SelfAttn + CrossAttn               │    │
│  [B, 192, 8, 128]                     │    │
└───────────────────────────────────────┘    │
   │                                 │       │
   ▼                                 │       │
   UP W                              │       │
   [B, 192, 8, 128] → [B, 192, 8, 256]       │
   │                                 │       │
   concat skip_0 ◀───────────────────┘       │
   [B, 288 (= 192+96), 8, 256]               │
   │                                         │
   ▼                                         │
┌───────────────────────────────────────┐    │
│ DEC L0 — width 96                     │ ◀──┘
│  2× ResBlock(288 → 96) + FiLM         │
│  + SelfAttn + CrossAttn               │
│  [B, 96, 8, 256]                      │
└───────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────┐
│ HEAD                                  │
│  GroupNorm → SiLU                     │
│  Conv2d(96 → 8, 3×3, circ-W, zero)    │
└───────────────────────────────────────┘
   │
   ▼
Output v_pred  [B, 8, 8, 256]
```

### Per-stage shapes and parameter accounting

| Stage | Width | Spatial | Tensor shape | Per-stage params (approx) |
|---|---|---|---|---|
| Stem | 96 | 8×256 | `[B, 96, 8, 256]` | 0.01 M |
| Enc L0 (2 RB + 2 attn) | 96 | 8×256 | `[B, 96, 8, 256]` | 0.5 M |
| Enc L1 (2 RB + 2 attn) | 192 | 8×128 | `[B, 192, 8, 128]` | 1.5 M |
| Bottleneck (2 RB + 2 attn) | 384 | 8×64 | `[B, 384, 8, 64]` | 5.2 M |
| Dec L1 (in-cat 576) | 192 | 8×128 | `[B, 192, 8, 128]` | 2.0 M |
| Dec L0 (in-cat 288) | 96 | 8×256 | `[B, 96, 8, 256]` | 0.9 M |
| Head | 96 → 8 | 8×256 | `[B, 8, 8, 256]` | 0.007 M |
| **Total verified** | | | | **14.81 M** |

- W-only downsample chain: `256 → 128 → 64`. H stays at 8 throughout (8 elevation rows of the LiDAR latent — self-attention sees all 8 globally at every level).
- Skips are taken at each encoder level's resolution **before** the downsample.

---

## Proposed D — 4-stage paper-match, ~125 M params

`level_channels = (160, 320, 640, 1024)`, `stem_channels = 160`.

```text
Input  z_noisy [B, 8, 8, 256]               KV context (pre-pooled, shared)
   │                                         [B, 10, 8, 64]
   ▼                                         │
┌───────────────────────────────────────┐    │
│ STEM   Conv2d(8 → 160, 3×3, circ-W)   │    │
│        [B, 160, 8, 256]               │    │
└───────────────────────────────────────┘    │
   │                                         │
   ▼                                         │
┌───────────────────────────────────────┐    │
│ ENC L0 — width 160                    │ ◀──┤
│  2× ResBlock(160 → 160) + FiLM        │    │
│  + SelfAttn + CrossAttn               │    │
│  [B, 160, 8, 256]                     │    │
└───────────────────────────────────────┘    │
   │                                         │
   ├─→ skip_0  [B, 160, 8, 256] ──────────┐  │
   ▼                                      │  │
   DOWN W   [B, 160, 8, 256] → [B, 160, 8, 128]
   │                                      │  │
   ▼                                      │  │
┌───────────────────────────────────────┐ │  │
│ ENC L1 — width 320                    │ │◀─┤
│  2× ResBlock(160 → 320) + FiLM        │ │  │
│  + SelfAttn + CrossAttn               │ │  │
│  [B, 320, 8, 128]                     │ │  │
└───────────────────────────────────────┘ │  │
   │                                      │  │
   ├─→ skip_1  [B, 320, 8, 128] ─────┐    │  │
   ▼                                 │    │  │
   DOWN W   [B, 320, 8, 128] → [B, 320, 8, 64]
   │                                 │    │  │
   ▼                                 │    │  │
┌───────────────────────────────────────┐ │  │
│ ENC L2 — width 640    ★ NEW STAGE     │◀│──┤
│  2× ResBlock(320 → 640) + FiLM        │ │  │
│  + SelfAttn + CrossAttn               │ │  │
│  [B, 640, 8, 64]                      │ │  │
└───────────────────────────────────────┘ │  │
   │                                 │    │  │
   ├─→ skip_2  [B, 640, 8, 64] ──┐   │    │  │
   ▼                             │   │    │  │
   DOWN W   [B, 640, 8, 64] → [B, 640, 8, 32]  ← W=32 (new low)
   │                             │   │    │  │
   ▼                             │   │    │  │
┌───────────────────────────────────────┐ │  │
│ BOTTLENECK — width 1024               │◀│──┤
│  2× ResBlock(640 → 1024) + FiLM       │ │  │
│  + SelfAttn + CrossAttn               │ │  │
│  [B, 1024, 8, 32]                     │ │  │
└───────────────────────────────────────┘ │  │
   │                             │   │    │  │
   ▼                             │   │    │  │
   UP W    [B, 1024, 8, 32] → [B, 1024, 8, 64]
   │                             │   │    │  │
   concat skip_2 ◀────────────── ┘   │    │  │
   [B, 1664 (= 1024+640), 8, 64]     │    │  │
   │                                 │    │  │
   ▼                                 │    │  │
┌───────────────────────────────────────┐ │  │
│ DEC L2 — width 640    ★ NEW STAGE     │◀│──┤
│  2× ResBlock(1664 → 640) + FiLM       │ │  │
│  + SelfAttn + CrossAttn               │ │  │
│  [B, 640, 8, 64]                      │ │  │
└───────────────────────────────────────┘ │  │
   │                                 │    │  │
   ▼                                 │    │  │
   UP W    [B, 640, 8, 64] → [B, 640, 8, 128]
   │                                 │    │  │
   concat skip_1 ◀───────────────────┘    │  │
   [B, 960 (= 640+320), 8, 128]           │  │
   │                                      │  │
   ▼                                      │  │
┌───────────────────────────────────────┐ │  │
│ DEC L1 — width 320                    │ │◀─┤
│  2× ResBlock(960 → 320) + FiLM        │ │  │
│  + SelfAttn + CrossAttn               │ │  │
│  [B, 320, 8, 128]                     │ │  │
└───────────────────────────────────────┘ │  │
   │                                      │  │
   ▼                                      │  │
   UP W    [B, 320, 8, 128] → [B, 320, 8, 256]
   │                                      │  │
   concat skip_0 ◀───────────────────────┘   │
   [B, 480 (= 320+160), 8, 256]              │
   │                                         │
   ▼                                         │
┌───────────────────────────────────────┐    │
│ DEC L0 — width 160                    │ ◀──┘
│  2× ResBlock(480 → 160) + FiLM        │
│  + SelfAttn + CrossAttn               │
│  [B, 160, 8, 256]                     │
└───────────────────────────────────────┘
   │
   ▼
┌───────────────────────────────────────┐
│ HEAD                                  │
│  GroupNorm → SiLU                     │
│  Conv2d(160 → 8, 3×3, circ-W, zero)   │
└───────────────────────────────────────┘
   │
   ▼
Output v_pred  [B, 8, 8, 256]
```

### Per-stage shapes and parameter accounting (estimated)

| Stage | Width | Spatial | Tensor shape | Per-stage params (est) |
|---|---|---|---|---|
| Stem | 160 | 8×256 | `[B, 160, 8, 256]` | 0.01 M |
| Enc L0 (2 RB + 2 attn) | 160 | 8×256 | `[B, 160, 8, 256]` | 1.4 M |
| Enc L1 (2 RB + 2 attn) | 320 | 8×128 | `[B, 320, 8, 128]` | 4.2 M |
| **Enc L2 ★ NEW** (2 RB + 2 attn) | 640 | 8×64 | `[B, 640, 8, 64]` | 14.5 M |
| **Bottleneck** (2 RB + 2 attn) | 1024 | 8×32 | `[B, 1024, 8, 32]` | 41 M |
| **Dec L2 ★ NEW** (in-cat 1664) | 640 | 8×64 | `[B, 640, 8, 64]` | 22 M |
| Dec L1 (in-cat 960) | 320 | 8×128 | `[B, 320, 8, 128]` | 14 M |
| Dec L0 (in-cat 480) | 160 | 8×256 | `[B, 160, 8, 256]` | 4.2 M |
| Head | 160 → 8 | 8×256 | `[B, 8, 8, 256]` | 0.01 M |
| **Total estimate** | | | | **~125 M** |

- W-only downsample chain: `256 → 128 → 64 → 32`. **Bottleneck is at W=32**, vs current's W=64 — the network has 2× more global context at its deepest level.
- The bottleneck (1024 channels at 8×32) holds ~41 M params on its own — about a third of the entire model.

---

## Side-by-side row-by-row comparison

**Headline answer:** the architecture is the same — same module types, same wiring pattern, same attention recipe. **Only two things change:** (a) every existing stage gets wider channels, and (b) one new encoder level + matching decoder level + skip connection are added between L1 and the bottleneck.

| # | Forward-pass step | Current (3-stage, ~15 M) | Proposed D (4-stage, ~125 M) | Change |
|---|---|---|---|---|
| 1 | Input | `[B, 8, 8, 256]` | `[B, 8, 8, 256]` | — (identical input) |
| 2 | KV context (cross-attn) | `[B, 10, 8, 64]` (image + raymap, pre-pooled) | `[B, 10, 8, 64]` (same) | — (identical, reused) |
| 3 | Stem | `Conv2d(8→96, 3×3, circ-W)` → `[B, 96, 8, 256]` | `Conv2d(8→160, 3×3, circ-W)` → `[B, 160, 8, 256]` | wider out (96 → 160) |
| 4 | Enc L0 | 2×ResBlock(96→96) + SelfAttn + CrossAttn | 2×ResBlock(160→160) + SelfAttn + CrossAttn | wider only |
| 5 | skip_0 stored | `[B, 96, 8, 256]` | `[B, 160, 8, 256]` | wider only |
| 6 | DownW after L0 | `[B, 96, 8, 256] → [B, 96, 8, 128]` | `[B, 160, 8, 256] → [B, 160, 8, 128]` | wider only |
| 7 | Enc L1 | 2×ResBlock(96→192) + SelfAttn + CrossAttn | 2×ResBlock(160→320) + SelfAttn + CrossAttn | wider only |
| 8 | skip_1 stored | `[B, 192, 8, 128]` | `[B, 320, 8, 128]` | wider only |
| 9 | DownW after L1 | `[B, 192, 8, 128] → [B, 192, 8, 64]` | `[B, 320, 8, 128] → [B, 320, 8, 64]` | wider only |
| 10 | **Enc L2 ★ NEW** | (does not exist) | `2×ResBlock(320→640) + SelfAttn + CrossAttn` → `[B, 640, 8, 64]` | **NEW stage** (same module type, same pattern as L0/L1) |
| 11 | **skip_2 stored ★ NEW** | (does not exist) | `[B, 640, 8, 64]` | **NEW skip connection** (same mechanism as skip_0, skip_1) |
| 12 | **DownW after L2 ★ NEW** | (does not exist) | `[B, 640, 8, 64] → [B, 640, 8, 32]` | **NEW downsample step** (same op as the existing two) |
| 13 | Bottleneck | 2×ResBlock(192→384) + SelfAttn + CrossAttn → `[B, 384, 8, 64]` | 2×ResBlock(640→1024) + SelfAttn + CrossAttn → `[B, 1024, 8, 32]` | wider (384→1024) AND deeper position (W=64 → W=32) |
| 14 | **UpW into Dec L2 ★ NEW** | (does not exist) | `[B, 1024, 8, 32] → [B, 1024, 8, 64]` | **NEW upsample** (same op) |
| 15 | **concat skip_2 ★ NEW** | (does not exist) | 1024 + 640 = **1664 ch** at 8×64 | **NEW concat** (same mechanism) |
| 16 | **Dec L2 ★ NEW** | (does not exist) | `2×ResBlock(1664→640) + SelfAttn + CrossAttn` → `[B, 640, 8, 64]` | **NEW decoder stage** |
| 17 | UpW into Dec L1 | `[B, 384, 8, 64] → [B, 384, 8, 128]` | `[B, 640, 8, 64] → [B, 640, 8, 128]` | wider only |
| 18 | concat skip_1 | 384 + 192 = **576 ch** at 8×128 | 640 + 320 = **960 ch** at 8×128 | wider only (mechanism identical) |
| 19 | Dec L1 | 2×ResBlock(576→192) + SelfAttn + CrossAttn | 2×ResBlock(960→320) + SelfAttn + CrossAttn | wider only |
| 20 | UpW into Dec L0 | `[B, 192, 8, 128] → [B, 192, 8, 256]` | `[B, 320, 8, 128] → [B, 320, 8, 256]` | wider only |
| 21 | concat skip_0 | 192 + 96 = **288 ch** at 8×256 | 320 + 160 = **480 ch** at 8×256 | wider only |
| 22 | Dec L0 | 2×ResBlock(288→96) + SelfAttn + CrossAttn | 2×ResBlock(480→160) + SelfAttn + CrossAttn | wider only |
| 23 | Head | `GroupNorm → SiLU → Conv2d(96→8, 3×3, circ-W)` | `GroupNorm → SiLU → Conv2d(160→8, 3×3, circ-W)` | wider in (96 → 160), same out (8) |
| 24 | Output | `[B, 8, 8, 256]` | `[B, 8, 8, 256]` | — (identical output) |

### Quick visual summary

```text
Current:    Stem → L0 → L1 → BN → L1' → L0' → Head      (3 enc, 1 bn, 2 dec, 2 skips)
                  └─ skip_0 ──────────┘
                       └─ skip_1 ─┘

Proposed D: Stem → L0 → L1 → L2 → BN → L2' → L1' → L0' → Head   (4 enc, 1 bn, 3 dec, 3 skips)
                  └─ skip_0 ─────────────────────┘
                       └─ skip_1 ──────────┘
                            └─ skip_2 ─┘
                            ↑
                       ★ NEW stage + NEW skip
```

---

## Module-type catalog — is anything NEW being introduced?

| Module type | Exists today? | Used in current 3-stage? | Used in proposed D? | Multiplicity change |
|---|---|---|---|---|
| `Conv2d` (circular pad on W) | ✓ | stem, head | stem, head | none |
| `ResBlock` (with FiLM timestep) | ✓ | yes | yes | **+4 ResBlocks** (the 2 in Enc L2 + 2 in Dec L2) |
| `SelfAttention` | ✓ | yes | yes | **+2 instances** (one each in Enc L2, Dec L2) |
| `CrossAttention` (Q=LiDAR, KV=context) | ✓ | yes | yes | **+2 instances** (one each in Enc L2, Dec L2) |
| `EncoderLevel` (RB+attn+downsample) | ✓ | 2 instances | **3 instances** | **+1** |
| `Bottleneck` (RB+attn, no down/up) | ✓ | 1 instance | 1 instance | none |
| `DecoderLevel` (upsample+concat+RB+attn) | ✓ | 2 instances | **3 instances** | **+1** |
| `DownsampleW` (stride-2 on W only) | ✓ | 2 instances | **3 instances** | **+1** |
| `UpsampleW` | ✓ | 2 instances | **3 instances** | **+1** |
| `GroupNorm`, `SiLU`, `TimestepMLP` | ✓ | yes | yes | none |

**Verdict: zero new module types.** Every building block already exists in the current codebase. The 4-stage variant is structurally identical to the 3-stage in *kind* — only the *count* of encoder/decoder/skip stages, and the *width* of every layer, change.

This means the structural refactor doesn't introduce algorithmic risk — only plumbing risk (off-by-one in the skip-connection loop). Hence the regression test in the next section: with `level_channels=(96, 192, 384)`, the refactored N-stage code must produce **bitwise-identical output** to the current hardcoded-3-stage code given the same seed and weights.

---

## Numeric summary table

| Property | Current (~15 M) | Proposed D (~125 M) | Ratio |
|---|---|---|---|
| Stages (enc/dec) | 3 / 3 | **4 / 4** | +1 stage |
| Stem width | 96 | 160 | 1.7× |
| Level widths | (96, 192, 384) | (160, 320, 640, 1024) | ~2× wider |
| Bottleneck width | 384 | **1024** | 2.7× |
| Bottleneck spatial | 8×64 | **8×32** | half the W |
| Downsample steps | 2 (256→128→64) | **3** (256→128→64→32) | +1 |
| Total params | 14.81 M | **~125 M** | **~8.4×** |
| Conv RB+attn at the heaviest layer | 384² = 147 k | **1024² = 1.05 M** | 7.1× |
| Skip-concat at largest decoder | 288 ch (192+96) | **480 ch** (320+160) | 1.7× |
| Number of skip connections | 2 | **3** | +1 |
| Code change vs current | none | **~80 LOC structural** in `LiDARUNet.__init__` + `.forward()` | |
| VRAM at eff batch 16 | ~10 GB at bs16 | ~11 GB at bs4×ga4 (estimated) | needs grad-checkpoint risk |
| Wall-clock per epoch | ~2 min | ~9 min (est, ~4.5× slower per step) | |
| 50-epoch full training | ~1 h 45 m | ~7–8 h (est) | |

---

## What the 4th stage actually buys

1. **Deeper hierarchy** — the receptive field at the bottleneck spans almost twice as many azimuth columns (W = 32 sees more, after 3 downsamples, of the original 256-column input).
2. **More compute concentrated where the action is** — the bottleneck (1024 channels at 8×32) holds ~41 M params, the bulk of the entire model. That's where high-level scene structure gets denoised.
3. **Higher-resolution skip information** — the extra skip at L2 (`[B, 640, 8, 64]`) gives the decoder mid-resolution feature information that the current decoder doesn't have access to.
4. **Matches paper architecture** — channels `(160, 320, 640, 1024)` follow the paper's `(320, 640, 1280, 1280)` shape at half the width, preserving the same growth ratios.

---

## What the structural refactor entails

The current [`unet.py:225-276`](s2s_min/models/unet.py#L225) has explicit named members (`self.enc_l0, self.enc_l1, self.bottleneck, self.dec_l1, self.dec_l0`) with hard-coded skip wiring. To support N stages:

```python
# In __init__ (replaces the explicit named members):
self.encoders = nn.ModuleList([
    EncoderLevel(
        in_ch=stem_channels if i == 0 else level_channels[i - 1],
        out_ch=level_channels[i],
        kv_channels=kv_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        do_downsample=True,
        t_emb_dim=self.t_emb_dim,
        return_skip=True,
    )
    for i in range(len(level_channels) - 1)
])
self.bottleneck = Bottleneck(
    in_ch=level_channels[-2],
    out_ch=level_channels[-1],
    kv_channels=kv_channels,
    num_res_blocks=num_res_blocks,
    num_heads=num_heads,
    t_emb_dim=self.t_emb_dim,
)
self.decoders = nn.ModuleList([
    DecoderLevel(
        in_ch=level_channels[i + 1],
        skip_ch=level_channels[i],
        out_ch=level_channels[i],
        kv_channels=kv_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        do_upsample=True,
        t_emb_dim=self.t_emb_dim,
    )
    for i in reversed(range(len(level_channels) - 1))
])

# In forward (replaces the explicit per-level wiring):
x = self.stem(z_noisy)
skips = []
for enc in self.encoders:
    x, s = enc(x, kv_context, t_emb)
    skips.append(s)
x = self.bottleneck(x, kv_context, t_emb)
for dec, skip in zip(self.decoders, reversed(skips)):
    x = dec(x, skip, kv_context, t_emb)
return self.head_conv(F.silu(self.head_norm(x)))
```

Net code change: roughly 60–80 lines (some removed, some added).

### Risks and verification

- **Primary risk:** off-by-one in skip wiring (decoder consumes skips in the wrong order, or channel counts mismatch at concat).
- **Regression test** (mandatory before any 4-stage training): the refactored model with `level_channels=(96, 192, 384)` must produce **bitwise-identical output** to the current implementation given the same random seed and weights. Save weights from the current model, load into the refactored model, feed an identical noise input, and compare outputs with `torch.allclose(rtol=0, atol=0)`.
- **Secondary check:** a 4-stage smoke test with random init — confirm `forward()` runs, output shape is `[B, 8, 8, 256]`, and no NaNs.

---

## Where this fits in the larger plan

This document is the design rationale for option **D (paper-match 4-stage 125 M)** in the U-Net capacity scale-up plan at [`/home/skr/.claude/plans/lets-make-a-plan-staged-church.md`](/home/skr/.claude/plans/lets-make-a-plan-staged-church.md).

The plan recommends a two-step approach:

1. **Phase 1** — Test the cheap **60 M variant** first (`level_channels=(192, 384, 768)`, one-line CLI change). Uses the step-2000 plateau on the baseline bs16 run as the test bar. ~50 min compute.
2. **Phase 4** — If Phase 1 shows STRONG signal (≥ 15 % loss reduction at step 2000), do this 4-stage 125 M refactor + train + eval.

The 60 M test answers *"is capacity the lever?"* cheaply. Only if YES is the 80 LOC structural refactor justified.
