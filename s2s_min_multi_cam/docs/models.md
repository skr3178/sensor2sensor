# Model build plan — LiDAR U-Net + LiDAR VAE

Focused doc covering the two from-scratch trainable models in the minimum pipeline:

1. **LiDAR U-Net** — the conditional denoiser, trained in M3.
2. **LiDAR VAE** — the encoder + decoder that defines the diffusion latent space, trained in M1 then frozen.

For wider context see [min_pipeline_plan.md](../../min_pipeline_plan.md) (milestones, scope) and [architecture.md](../../architecture.md) (every component including frozen ones). For loss formulas see [equations.md](../../equations.md). Source-of-truth for dimensions is [configs/min.yaml](../configs/min.yaml).

---

## 1. LiDAR U-Net (denoiser backbone)

### 1.1 Purpose

The single trainable backbone in M3. Predicts the noise (v-prediction) added to the LiDAR VAE latent, conditioned on the pre-pooled image + raymap context. Lives inside [`s2s_min/models/unet.py`](../models/unet.py).

### 1.2 Topology (scope option A, committed)

```
Input:  z_lidar_noisy  [B, C=8,  H=8,  W=256]   ← noised LiDAR latent
        t              [B]                       ← diffusion timestep
        kv_context     [B, C=10, H=8,  W=64]    ← pre-pooled image+raymap

Stem        : CircularConv2d(8 → 96)                                  -> [B,  96, 8, 256]

Encoder
  Level 0   : 2× (ResBlock(96)  + SelfAttn + CrossAttn(KV=kv))        -> [B,  96, 8, 256]
              DownsampleW (stride 2 on W only)                         -> [B,  96, 8, 128]
  Level 1   : 2× (ResBlock(96→192) + SelfAttn + CrossAttn)             -> [B, 192, 8, 128]
              DownsampleW                                              -> [B, 192, 8,  64]

Bottleneck  : 2× (ResBlock(192→384) + SelfAttn + CrossAttn)            -> [B, 384, 8,  64]

Decoder (mirror of encoder, with skip-concat from corresponding encoder level)
  Level 1   : UpsampleW                                                -> [B, 384, 8, 128]
              cat(skip_lvl1)                                           -> [B, 576, 8, 128]
              2× (ResBlock(576→192) + SelfAttn + CrossAttn)            -> [B, 192, 8, 128]
  Level 0   : UpsampleW                                                -> [B, 192, 8, 256]
              cat(skip_lvl0)                                           -> [B, 288, 8, 256]
              2× (ResBlock(288→96) + SelfAttn + CrossAttn)             -> [B,  96, 8, 256]

Head        : GroupNorm → SiLU → CircularConv2d(96 → 8)   ← zero-init  -> [B,   8, 8, 256]

Output:  ε̂ or v̂  [B, 8, 8, 256]
```

| Property | Value |
|---|---|
| Trainable params (target) | ~25–35 M |
| Init | Kaiming-normal on convs; **zero-init** on output conv and on every AdaLN-Zero modulation projection |
| Timestep injection | sinusoidal `t` → 2-layer MLP → AdaLN-Zero scale/shift per ResBlock (modulates GN output) |
| Periodic padding | Circular on W, zero on H — every conv that touches W |
| Optional knob | `downsample_h_once` (default off) — collapses H=8→4 at stem for hierarchical elevation features |
| Cross-attn KV | Pre-pooled `[10, 8, 64]`, projected once outside the U-Net, reused at every block |
| Prediction type | `v_prediction` |

### 1.3 Files & status

| File | Status | What it contains |
|---|---|---|
| [`models/attention.py`](../models/attention.py) | ✓ written (M-1) | `MultiHeadAttention`, `SelfAttention`, `CrossAttention` |
| [`models/blocks.py`](../models/blocks.py) | ✓ written (M-1) | `CircularConv2d`, `ResBlock` (no timestep yet), `DownsampleW`, `UpsampleW`, `EncoderLevel` |
| `models/unet.py` | **not yet** | Full `LiDARUNet(nn.Module)` wiring stem → 2 encoder levels → bottleneck → 2 decoder levels → head |
| `models/timestep.py` | **not yet** | Sinusoidal timestep embedding + 2-layer MLP; `AdaLN-Zero` modulation injected into `ResBlock` |

### 1.4 Remaining U-Net work (when we get to M0)

1. **Promote `ResBlock` to be timestep-aware** — add `t_emb` argument, AdaLN-Zero scale/shift on the GN output. Zero-init the (scale, shift) projection.
2. **Write `models/timestep.py`** — sinusoidal embedding `(B,) → (B, dim)` followed by `Linear → SiLU → Linear` → per-block (scale, shift) tuple.
3. **Write `LiDARUNet` in `models/unet.py`** — assemble stem + 2 `EncoderLevel`s + bottleneck + 2 decoder levels (mirror of `EncoderLevel` but with `UpsampleW` first, skip-concat, then the ResBlock stack) + head. ~150 LOC.
4. **Pre-project KV once outside the U-Net** — small helper that does `kv_K = LinearK(kv); kv_V = LinearV(kv)` once and passes the projected tensors into each block. Each `CrossAttention` block accepts pre-projected K/V or projects internally — needs a tiny refactor in `attention.py` to allow both modes.

---

## 2. LiDAR VAE — build plan

Trainable encoder + decoder that compresses the 32×1024 three-channel range image (range, intensity, validity) into an `[B, 8, 8, 256]` latent. Trained in M1 (Phase A), then frozen forever. Lives in [`s2s_min/models/lidar_vae.py`](../models/lidar_vae.py).

> **Channel-count note.** nuScenes LiDAR ships `(x, y, z, intensity, ring_index)` — no elongation channel. The Sensor2Sensor paper includes elongation because it trains on Waymo. We use 3 channels here; re-introduce elongation only if porting to Waymo.

### 2.1 Final architecture

Encoder:

```
range_image [B, 3, 32, 1024]              ← values in [0, 1] per channel: range, intensity, validity

Stem        : CircularConv2d(3 → 32, k=3)                       -> [B,  32, 32, 1024]

Stage 1     : 2× ResBlock(32)                                   -> [B,  32, 32, 1024]
              Downsample2d (stride 2 on H AND W, circ pad W)    -> [B,  64, 16,  512]

Stage 2     : 2× ResBlock(64)                                   -> [B,  64, 16,  512]
              Downsample2d                                       -> [B, 128,  8,  256]

Bottleneck  : 2× ResBlock(128)                                  -> [B, 128,  8,  256]
              SelfAttn(128)                                      -> [B, 128,  8,  256]

Head        : GroupNorm → SiLU → CircularConv2d(128 → 16, k=1)  -> [B,  16,  8,  256]
              split channel dim                                  -> μ, logσ²  each [B, 8, 8, 256]
```

Decoder (mirror):

```
z [B, 8, 8, 256]

Stem        : CircularConv2d(8 → 128, k=3)                      -> [B, 128,  8,  256]

Bottleneck  : SelfAttn(128)                                     -> [B, 128,  8,  256]
              2× ResBlock(128)                                   -> [B, 128,  8,  256]

Stage 2     : Upsample2d (nearest ×2 on H and W)                -> [B, 128, 16,  512]
              CircularConv2d(128 → 64)                          -> [B,  64, 16,  512]
              2× ResBlock(64)                                    -> [B,  64, 16,  512]

Stage 1     : Upsample2d                                        -> [B,  64, 32, 1024]
              CircularConv2d(64 → 32)                           -> [B,  32, 32, 1024]
              2× ResBlock(32)                                    -> [B,  32, 32, 1024]

Head        : GroupNorm → SiLU → CircularConv2d(32 → 3)  ← zero-init  -> [B, 3, 32, 1024]
              per-channel sigmoid                                -> values back in [0, 1]
```

| Property | Value |
|---|---|
| Trainable params (target) | ~10 M |
| Stem channels | 32 |
| Down/up stages | 2 (32→64→128, mirror) |
| Bottleneck self-attn | 1 block |
| Spatial down factor | 4× on **both** H and W (unlike the U-Net which is W-only) |
| Latent | `[B, 8, 8, 256]`, μ + logσ² split |
| Output activation | sigmoid per channel (range, intensity, validity all ∈ [0, 1]) |
| Init | Kaiming-normal on conv weights; **zero-init the decoder head conv** so a fresh decoder outputs ~0.5 everywhere (after sigmoid) instead of garbage |

### 2.2 Loss (4 terms, deliberately no LPIPS for M1)

```
L_VAE = λ_range · L1_masked(x_range,     x̂_range,     mask=x_validity)
      + λ_int   · L1_masked(x_intensity, x̂_intensity, mask=x_validity)
      + λ_valid · BCE(x̂_validity, x_validity)
      + λ_KL    · 0.5 · mean(μ² + σ² - log σ² - 1)
```

| Weight | Value |
|---|---|
| `λ_range`, `λ_intensity`, `λ_validity` | 1.0 |
| `λ_KL` | 1e-6 (X-Drive default; otherwise the posterior collapses to the prior since the L1 signals are on [0,1]) |

**Validity-masked L1:** for range and intensity, only score pixels where ground-truth validity = 1. Otherwise the decoder learns to predict noise at invalid pixels.

```python
mask = x_validity                                              # [B, 1, 32, 1024]
denom = mask.sum().clamp(min=1.0)
loss_range = (mask * (x_range - x̂_range).abs()).sum() / denom
# (same for intensity)
loss_valid = F.binary_cross_entropy(x̂_validity, x_validity)
loss_KL    = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()
```

LPIPS terms (normals, intensity) are re-added **only if** M1 reconstructions look visibly blurry.

### 2.3 Training regime (M1)

| Item | Value |
|---|---|
| Optimizer | `torch.optim.AdamW`, lr=4e-4, betas=(0.9, 0.999), weight_decay=1e-4 |
| Batch | actual 2, grad-accum 4 → effective 8 |
| Mixed precision | `torch.cuda.amp` autocast + GradScaler |
| Gradient checkpoint | Only the bottleneck ResBlocks if VRAM is tight |
| Epochs (10-scene subset) | 50 (~30 min on RTX 3060) |
| Overfit-10 milestone | mean L1-range < 0.05 within ~500 steps on a fixed 10-sample batch |
| EMA | None for the VAE (only the U-Net uses EMA, in M3) |
| Checkpoint out | `s2s_min/out/lidar_vae.pt` (consumed by M2/M3/M4) |

### 2.4 Implementation order (step-by-step)

Each step is independently testable. Don't skip the round-trip sanity check at step 4 — it catches more bugs than any other single test in M1.

| Step | What | Deliverable | Test |
|---|---|---|---|
| **1** | **`data/range_image.py`** — channel normalization helpers `normalize_channels(x)` and `denormalize_channels(x̂)`. Per the [channel table in architecture.md](../../architecture.md#43-channel-normalization-must-match-training-data): range/100 m, intensity/255, validity passthrough. | ~40 LOC | Round-trip identity on a random tensor with channel-specific scales |
| **2** | **`models/lidar_vae.py` skeleton** — class `LiDARVAE(nn.Module)` with stubbed `encode`, `decode`, `reparameterize`, `forward`. All ops return zeros of the right shape. Import `CircularConv2d`, `ResBlock`, `SelfAttention` from existing M-1 modules. | ~50 LOC stub | `python -c "from s2s_min.models.lidar_vae import LiDARVAE; LiDARVAE()"` runs |
| **3** | **Encoder forward** — wire stem → 2 down stages → bottleneck → head producing μ, logσ² of shape `[B, 8, 8, 256]`. | +80 LOC | Shape assertion: `[B, 3, 32, 1024] → [B, 8, 8, 256]` for both μ and logσ² |
| **4** | **Decoder forward** — wire stem → bottleneck → 2 up stages → head producing `[B, 3, 32, 1024]`. Sigmoid per channel. | +80 LOC | **Encode→sample→decode round trip on random input.** Output shape matches input. Per-channel mean of decoded output ≈ 0.5 (zero-init head + sigmoid). |
| **5** | **`reparameterize` + `forward`** — `z = μ + σ · ε` during training, `z = μ` at eval. `forward` returns `(x̂, μ, logσ²)`. | +20 LOC | Encode→reparam→decode round trip; `vae.train()` and `vae.eval()` give different results given a fixed input |
| **6** | **Loss function** — `lidar_vae_loss(x, x̂, mu, logvar, weights)` returns a dict of per-term losses + the total. Validity-masked L1; per-channel BCE; KL. | +60 LOC | Plug in identical x and x̂: range/intensity L1 = 0; validity BCE > 0 (sigmoid never exactly 1 or 0); KL = 0.5·mean(μ²+σ²−logσ²−1). |
| **7** | **`data/nuscenes_mini.py`** — 50-LOC nuScenes loader using nuscenes-devkit; pairs `CAM_FRONT` ↔ `LIDAR_TOP` keyframes; restricted to the 10-scene seed-0 subset; yields `dict(rgb=..., lidar_pc=..., K=..., T_cam2ego=...)`. | ~60 LOC | `next(iter(loader))` returns a dict with right shapes for one sample |
| **8** | **Borrow `point_cloud_to_range_image`** from [X-Drive's pipeline.py:830](../../Reference_code/X-Drive/xdrive/dataset/pipeline.py#L830) into `data/range_image.py`. Override normalization to [0, 1] per architecture.md §4.3. | ~+30 LOC | Round-trip test: `pc → range_image → pc'` recovers ≥95 % of valid points within 0.5 m |
| **9** | **`train/train_vae.py`** — load 10-scene subset, AdamW lr=4e-4, fp16 autocast, log every 25 steps. CLI flag `--overfit 10` clamps the dataset to 10 samples. | ~150 LOC | `python -m s2s_min.train.train_vae --overfit 10 --steps 500` drives mean L1-range below 0.05 |
| **10** | **Full epoch run** | — | `python -m s2s_min.train.train_vae --epochs 50` finishes without OOM; final BEV reconstruction visually matches ground truth |
| **11** | **Save frozen checkpoint** | `s2s_min/out/lidar_vae.pt` | Reload it in a fresh process, run `decode(encode(x))` → matches the training-time output bit-for-bit (within fp16 noise) |

### 2.5 Pass criteria (M1 done if both pass)

1. **Overfit-10:** with 10 fixed samples and a few hundred AdamW steps, mean validity-masked L1 on range drops below **0.05** (= 5 m at 100 m range clamp).
2. **Full-epoch:** after 50 epochs on the 10-scene subset, the eyeball BEV check (decoded range image projected to 3D point cloud, plotted top-down) shows lane lines, vehicles, and ground plane plausibly placed. The reviewer-suggested "LPIPS_normals" terms get added only if this check fails.

### 2.6 What can go wrong (and the first thing to try)

| Symptom | Likely cause | First fix |
|---|---|---|
| KL term explodes | `λ_KL` too high | Confirm `λ_KL=1e-6`, not `1e-2` |
| All reconstructions flat (mean = 0.5) | Decoder head zero-init never updated; output conv frozen accidentally | Verify `requires_grad=True` on head; check that gradient flows |
| Range channel sharp, intensity smeared | Validity mask not applied to intensity | Check `loss_intensity = (mask * ...)`, not unmasked |
| Output range > 1.0 or NaN | Sigmoid skipped on output | Apply sigmoid in decoder head |
| KL=0, μ=0, σ=0 | Posterior collapse; KL weight too high relative to recon | Lower `λ_KL` further (1e-7) or raise recon weights |
| OOM at batch 2 | Gradient checkpoint off; or fp16 not enabled | Enable `torch.utils.checkpoint` on the two bottleneck ResBlocks, ensure autocast active |
| Round-trip pc → range → pc loses many points | Normalization mismatch with X-Drive defaults | Verify clamp 100 m and the [0,1] mapping per architecture.md §4.3 |
