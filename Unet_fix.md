Concrete fix proposals (in order of expected impact)
#	Fix	LOC	Cost	Expected impact
1	Add sinusoidal positional encoding to Q + KV in CrossAttention	~30 in attention.py	0 params, ~0 FLOPs	likely transformative
2	Bump KV pool from 8×64 → 16×128	1 in decode_to_pointcloud.py:39 + matching in train_diffusion.py	~4× attention cost but VRAM has headroom	moderate (6.3 → 8.0 dB only)
3	Drop the pool entirely; let U-Net read all 32×56 = 1792 KV tokens	same 1 LOC	~14× attention cost, may not fit in 12 GB	hits the PSNR ceiling but expensive
4	Multi-scale KV (one pool size per U-Net level)	~100 LOC refactor	medium	paper-faithful, complex to debug


## Consolidated H5 audit findings

Item	Verdict	Headline number	Severity
KV pool → image-latent info	⚠ DESTROYED	6.3 dB PSNR	★★★★★
KV pool → raymap info	✓ Preserved	49.3 dB PSNR	none
Cross-attention positional encoding	⚠ MISSING	0 positional info anywhere	★★★★★
Noise schedule	✓ Clean	α²+σ²=1.000000 exactly	none
Raymap geometric math	✓ Clean	unit length to 6 decimal places	none


---

## Additional fix categories (extensions to the table above)

### A. Cheap diagnostic tests to run BEFORE applying any fix
Do these first — they're free and tell us which fix to pick.

- **A1. SD VAE depth probe** — ✅ **DONE (2026-05-29) — verdict: SD VAE features are depth-IMPOVERISHED.** See "A1 result" below. Skip to fix categories C/D; fix #1 alone is NOT sufficient.
- **A2. Feature-perturbation test** — take the existing 60M checkpoint, replace SD VAE features with random noise of matched statistics, re-run M4 demo. If CD-3D-raw barely degrades → model isn't using image features at all (confirms fix #1 is needed). ~1 h, no training.
- **A3. Oracle-depth conditioning** — replace SD VAE features with LiDAR-projected depth (perfect depth ground truth) as conditioning. Train briefly. Establishes the upper bound on what better encoding could buy. ~3 h.
- **A4. Cross-attention attribution** — plot attention weights for a single batch. Are they uniform (bag-of-features, confirms missing pos-enc) or concentrated (model is doing something)? ~30 min CPU.

#### A1 result — SD VAE depth probe (2026-05-29)

Script: `s2s_min/diagnostics/sdvae_depth_probe.py` · plots/data: `s2s_min/out/depth_probe/`.
Method: froze the cached SD-VAE `image_latent [4,32,56]`, built a camera-plane metric-depth
target by projecting LIDAR_TOP → CAM_FRONT (full extrinsics + ego-pose chain, z-buffered onto
the 32×56 grid), and trained a **per-pixel 1×1 MLP** (no spatial mixing → tests *local* depth
decodability, the regime cross-attention reads pooled features in). 800 samples, ~776k valid
LiDAR cells, 80/20 train/val.

| condition | Pearson(log)↑ | AbsRel↓ | δ<1.25↑ | R²↑ |
|---|---|---|---|---|
| **SD-VAE img** | 0.348 | 0.642 | 0.202 | 0.121 |
| **raymap only** | **0.863** | **0.286** | **0.643** | **0.742** |
| img+ray | 0.845 | 0.308 | 0.608 | 0.712 |
| img (shuffled null) | 0.357 | 0.617 | 0.205 | 0.127 |
| mean floor | 0.000 | 0.704 | 0.165 | 0.000 |

**Verdict: the frozen SD 1.5 VAE latent is depth-impoverished.** The image probe (0.348) is
statistically equal to the shuffled-null control (0.357), and adding the image to the raymap
*hurts* vs raymap alone. Qualitatively the SD-VAE prediction is pure speckle while the raymap
prediction is a coherent ground-plane depth field (`qualitative.png`).

**Consequence for the fix plan:** this disambiguates toward **H2/H3 (image encoder)**, NOT H5
(KV pooling / cross-attn pos-enc). Fix #1 (pos-enc) cannot recover depth the encoder never
captured — it is necessary-at-best, not sufficient. Re-priority: **C-series (encoder swap / C5
depth-channel concat) is now the lead fix.** Pass/fail test for any candidate encoder = rerun
this probe and require `img` to clear the raymap baseline.

#### Encoder gate result (2026-05-29) — ✅ GREEN LIGHT

Script: `s2s_min/diagnostics/encoder_probe.py` · plots/data: `s2s_min/out/encoder_probe/`.
Same depth target + per-pixel probe, comparing encoders head-to-head (500 samples, ~485k cells).
Tested **DINOv2-small** (the exact Depth-Anything-V2 backbone; runs offline from HF cache). The
Depth-Anything-V2 depth head itself could not be pulled — file downloads are network-blocked in
this env — but DINOv2 features are its representation, so the gate is answered (DA = these
features + a depth head, can only do ≥).

| condition | Pearson↑ | AbsRel↓ | δ<1.25↑ | R²↑ | lift vs raymap |
|---|---|---|---|---|---|
| sdvae (current) | 0.358 | 0.645 | 0.199 | 0.127 | — |
| sdvae+ray | 0.834 | 0.302 | 0.627 | 0.691 | **−0.012 (HURTS)** |
| raymap baseline | 0.846 | 0.294 | 0.632 | 0.709 | — |
| **dinov2** | **0.949** | **0.159** | **0.787** | **0.899** | **+0.103** |
| dinov2+ray | 0.949 | 0.163 | 0.789 | 0.898 | +0.103 |

**Verdict: a depth-aware encoder adds large depth residual.** DINOv2 nearly halves AbsRel
(0.294→0.159) and lifts R² to 0.90. Notable: `dinov2 ≈ dinov2+ray` → the features already encode
the viewing geometry the raymap provided (it *subsumes* the prior). Qualitatively the SD-VAE probe
is speckle while DINOv2 produces coherent depth fields (`encoder_qualitative.png`).

#### END-TO-END RESULT (2026-05-29) — ✅ Option B confirmed by decode eval

Trained a 15M U-Net from scratch, Option B (DINOv3 → learned Conv1x1(384→4) → 10ch KV, no SD-VAE),
100-scene cache. Stopped early at 30/50 epochs (step 7135). v-MSE loss ≈ SD-VAE baseline (0.266 vs
0.279 — loss is blind to conditioning), but the decode eval tells the real story:

| metric (60 samples, cfg=1.0, matched noise) | SD-VAE (50ep) | DINOv3 (30ep) | Δ |
|---|---|---|---|
| CD-3D mean | 2.419 m | **2.101 m** | **+13.1%** |
| CD-BEV mean | 1.384 m | **1.063 m** | **+23%** |
| win rate | — | **80%** | — |

DINOv3 wins despite training 40% fewer steps → this understates the real gap. Eval:
`s2s_min/eval/compare_encoders.py`, artifacts in `s2s_min/out/encoder_eval/`. Run:
`s2s_min/out/runs/2026-05-29_204102__m3-unet-15M-dinov3-fromscratch/`.

**Decision: proceed with the encoder change — it is worth the cache-rebuild + retrain.**
Recommended: swap to DINOv2-small features (works offline now) and/or C5 (Depth-Anything-V2 depth
channel — needs network to fetch weights at implementation time), plus B1 pos-enc so cross-attn can
localize the now-rich features. This converts the A1 "skip fix #1" into a concrete go.

**Side-by-side comparison folders** (same target, same 800 samples, same plot layout):
- `s2s_min/out/depth_probe/`        — SD-VAE encoder (img probe r=0.348, pure-speckle qualitative)
- `s2s_min/out/depth_probe_dinov2/` — DINOv2-small swapped in (img r=0.953, coherent depth maps)
- `s2s_min/out/depth_probe_dinov3/` — DINOv3-small swapped in (img r=**0.956**, best on every metric)
Generated by `depth_probe_dinov2.py` / `depth_probe_dinov3.py` (each as the sole image encoder).

Encoder leaderboard (img-only probe, 800 samples, 32×56):

| encoder | params | Pearson↑ | AbsRel↓ | δ<1.25↑ | R²↑ |
|---|---|---|---|---|---|
| SD-VAE (current) | ~34M enc | 0.348 | 0.642 | 0.202 | 0.121 |
| DINOv2-small (ViT-S/14) | ~22M | 0.953 | 0.148 | 0.806 | 0.907 |
| **DINOv3-small (ViT-S/16)** | ~22M | **0.956** | **0.139** | **0.821** | **0.914** |

DINOv3-small is a strict, free upgrade over DINOv2-small (same size/dim, drop-in via timm
`vit_small_patch16_dinov3.lvd1689m`, now cached/offline). The DINOv2↔DINOv3 gap is small (both
saturate the probe); the decisive gap is SD-VAE → any geometric encoder. **Recommended default:
DINOv3-small.** Extractor: `s2s_min/diagnostics/depth_probe_dinov3.py::extract_dinov3` (patch16,
input 224×384 → 14×24 grid, strips 5 prefix tokens incl. 4 registers).

#### Pooled / raymap-baseline / channel-reduction probes (2026-05-29) — all caveats cleared

Script: `s2s_min/diagnostics/pooled_probe.py` · `s2s_min/out/pooled_probe/` (600 samples). Addresses
the three honest objections to the 32×56 linear-probe result:

- **P1 — survives pooling.** dinov2+ray at 32×56 = 0.953 → pooled to the U-Net's actual (8,64) KV =
  **0.958** (loss −0.005, i.e. none). Depth is low-frequency; adaptive-pool preserves it.
- **P2 — honest baseline (vs raymap, pooled).** sdvae+ray = 0.831, **dinov2+ray = 0.958, margin
  +0.127** (AbsRel 0.356→0.157). dinov2-alone (0.958) ≈ dinov2+ray → subsumes raymap. sdvae+ray
  (0.831) is still *below* raymap-alone (0.855) → SD-VAE actively hurts even through the real pipeline.
- **P3 — channel reduction.** A **learned 384→4** projection keeps r=**0.941** (min channels for r≥0.85
  is **4**); PCA needs ~16 (N=4 PCA only 0.776). The concrete combo **[dinov2→4 + sdvae4 + ray6 = 14ch]
  pooled = 0.949**.

**Integration verdict:** the cheap path is real — add a learned `Conv1x1(384→4)` on DINOv2, concat with
existing SD-VAE(4)+raymap(6) → **14-ch KV** (cross-attn input 10→14, a tiny change; no pipeline rebuild).
Remaining unverifiable-by-probe risk: linear-probe decodability ≠ cross-attn learns to use it — only the
retrain settles that, but pooling+baseline+channel risks are now all retired.

### B. Alternative cross-attention pos-encoding flavors (variants of fix #1)
Same one-bit insight, multiple ways to express it.

- **B1. Sinusoidal 2D pos-encoding (recommended)** — port `get_2d_sincos_pos_embed` from `Reference_code/diffusers/src/diffusers/models/embeddings.py`. Fixed buffers, 0 params, 0 FLOPs at runtime. ~30 LOC.
- **B2. Learnable absolute pos-embeddings** — `nn.Parameter` of shape `[H*W, C]` added to KV before flatten. ~5K params. Slightly more flexible than sinusoidal but trains slower.
- **B3. Rotary positional embedding (RoPE)** — applied multiplicatively to Q and K. Better long-range than sinusoidal in modern transformers. ~50 LOC, no extra params.
- **B4. ALiBi (attention with linear biases)** — adds a position-dependent bias to attention logits. No projections, very cheap. ~20 LOC.
- **B5. Raymap-as-positional-encoding** — repurpose the raymap (already in KV) as Q-side positional info too. We'd need a "LiDAR raymap" that maps LiDAR azimuth/elevation grid to ray directions. Geometrically meaningful.

### C. Image encoder swap candidates (H2 / H3 the user mentioned)
Tackles the SD VAE-is-wrong hypothesis directly.

- **C1. Depth-Anything-Small features** — replace 4-ch SD VAE latent with depth-aware encoder. ~80M params (similar to SD VAE), but trained for depth/geometry. Pip-installable. Cache rebuild required (~30 min).
- **C2. DINO v2 small features** — 384-ch self-supervised geometric features. Bigger context but richer signal. Cache rebuild + memory bump.
- **C3. DPT (Dense Prediction Transformer)** — explicit dense depth head. Heavy but most-direct mapping image → 3D.
- **C4. CroCo / DUSt3R features** — 3D-aware cross-view encoders. Built for geometric correspondence.
- **C5. Hybrid: SD VAE + a depth head concatenated** — keep SD VAE for appearance, add Depth-Anything for depth. ~10 channels of conditioning (4 SD + 1 depth + 6 raymap). Cheapest addition that diversifies the conditioning signal.
- **C6. Fine-tune SD VAE on automotive data** — much smaller delta than a full swap but adapts the existing encoder to our distribution.

### D. Conditioning-enrichment fixes (orthogonal to pool + pos-encoding)
Add MORE signal, regardless of how cross-attn reads it.

- **D1. FiLM-style image conditioning** — pool image features to a single global vector per sample, modulate every ResBlock via FiLM. Complements (doesn't replace) cross-attention. Cheap, ~30 LOC.
- **D2. ControlNet-style splatting** — splat the image latent (with raymap) into the LiDAR latent's azimuth/elevation grid directly. No cross-attention needed. Used by RangeLDM.
- **D3. Per-stage raymap injection** — currently raymap is concatenated to KV once and pooled. Instead, inject a fresh raymap (at each level's resolution) at every U-Net stage. ~50 LOC.
- **D4. Explicit depth-via-raymap intersection** — geometrically compute "where does ray (u,v) intersect the predicted scene?" using a learned depth head; condition on that.

### E. U-Net + decoder fixes not yet stress-tested
We have evidence neither helps alone, but they might help once the conditioning is fixed.

- **E1. 4-stage 125M U-Net** — refactor is DONE; ~80 LOC was 0 because we already did it. Training is the only remaining cost (~8 h).
- **E2. LiDAR VAE latent_channels 8 → 16** — paper uses 16. Currently constrained to 8. Cascades into: VAE retrain (~1.5 h), cache rebuild (~30 min), U-Net retrain.
- **E3. More ResBlocks per level (2 → 3)** — adds depth per scale. Small param cost.
- **E4. AdaLN-Zero timestep injection instead of FiLM** — DiT-style. Marginal improvement, ~80 LOC.
- **E5. Cross-view attention (paper's actual design)** — only meaningful if we ever go multi-camera.

### F. Training-time fixes (no architecture change)
Test-cheap, fix-bug-or-confirm-bug-isn't-it.

- **F1. Lower `cond_dropout` from 0.2 → 0.1** — gives the model more conditional samples to learn from; trades off CFG amplitude at inference.
- **F2. Curriculum on noise schedule** — start with low t (easy denoising), gradually expose high t. Helps convergence on hard problems.
- **F3. Loss reweighting at high t** — weight v-loss by `min(SNR, γ)` (Salimans 2022). Easy gradient improvement at high noise.
- **F4. Lower LR + longer warmup** — currently 2e-4 / 500. Going to 1e-4 / 2000 may extract more from the conditioning.

### G. Inference-time fixes (no retraining)
- **G1. More DDIM steps (25 → 50 or 100)** — costs 2-4× wall-clock but may surface model quality that 25 steps misses.
- **G2. DPM-Solver++ or UniPC sampler** — better than DDIM at 10-25 steps. ~50 LOC port from diffusers.
- **G3. Higher CFG scale + temperature scheduling** — already swept; revisit once H5/H2 fix lands.

### H. Bottleneck candidates we've ruled out (cross-off list)
- ❌ Dataset size (H1 ruled out: 4k cache vs 34k cache gave identical loss curves within ±1%)
- ❌ Noise schedule (audit clean)
- ❌ Raymap math (audit clean)
- ❌ Naked model-capacity (60M vs 15M plateaued at same loss)

### I. Recommended attack order (REVISED after A1 result, 2026-05-29)
A1 changed the picture: the SD VAE latent carries no local depth signal, so the lead fix is
now the **encoder**, not pos-enc. Pos-enc (B1) is still worth doing but cannot be the whole
story.
1. ✅ **A1** (depth probe) — DONE. Verdict: encoder is depth-impoverished (H2/H3).
2. **C5** (SD VAE + Depth-Anything depth-channel concat) — cheapest way to inject the missing
   depth signal; keeps the appearance pathway. Rebuild a small cache, then re-run A1 on the new
   features to confirm `img` clears the raymap baseline. **← lead fix now.**
3. **B1** (sinusoidal pos-enc) — still a real bug, ~30 LOC; do alongside C5 so cross-attn can
   actually localize the now-richer features. (A standalone B1 run is in progress for the record.)
4. If C5 underwhelms: full encoder swap **C1 / C2 / C3** (Depth-Anything / DINOv2 / DPT).
5. **A4** (attention attribution) + **D1** (FiLM) — secondary, once the conditioning carries depth.
6. Only after the above plateau: **E1 / E2** (bigger U-Net or richer VAE latent), or
   paper-fidelity multi-camera (#6 in original list).


(Fast, 30 min) Apply Fix #1 (sinusoidal pos encoding from diffusers' utility). Test by warm-starting from the killed H1 checkpoint. If cos sim breaks through 0.32 → bug confirmed, this was probably the whole story.