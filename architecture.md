# Sensor2Sensor Minimum Pipeline — Component Architecture

Per-component build spec for the minimum pipeline. Implementation lives under
[s2s_min/](s2s_min/). For the *why* / scoping, see
[min_pipeline_plan.md](min_pipeline_plan.md). For loss equations, see
[equations.md](equations.md). For the trained-vs-frozen schedule across
milestones, see [`min_pipeline_plan.md` §U-Net details](min_pipeline_plan.md).

## Conventions

| Item | Value |
|---|---|
| Tensor layout | PyTorch `[N, C, H, W]` everywhere |
| Range image axes | `H` = elevation (32 rows, **not** periodic), `W` = azimuth (1024 cols, **periodic at 0°/360°**) |
| Periodic padding | All convs on the LiDAR side use **circular pad on W, zero pad on H** ([`CircularConv2d`](s2s_min/models/blocks.py)) |
| Norm | `GroupNorm` (`min(32, channels)` groups), never BatchNorm |
| Activation | `SiLU` (≡ Swish), out-of-place |
| Source of truth for shapes | [`configs/min.yaml`](s2s_min/configs/min.yaml). Never hard-code dims in module files. |

---

## Pipeline data flow

```
CAM_FRONT 256×448 RGB ─▶ [Image VAE encoder (frozen)] ─▶ image_latent [B, 4, 32, 56]
                                                                          │
camera K, T (CAM_FRONT) ─▶ [Raymap builder @ 32×56]   ─▶ raymap     [B, 6, 32, 56]
                                                                          │
                                                       concat on C → kv_full   [B, 10, 32, 56]
                                                                          │
                                                       adaptive avg-pool → kv_context [B, 10, 8, 64]

LIDAR_TOP point cloud ─▶ [point_cloud_to_range_image] ─▶ range_image [B, 3, 32, 1024]
                                                                          │
                                                            ┌─── [LiDAR VAE encoder] ─▶ μ, logσ²  ─▶ z_lidar  [B, 8, 8, 256]
                                                            │                                            │
   (M1: VAE training only)                                  │                       (M3: + noise) ─▶ z_noisy [B, 8, 8, 256]
                                                            │                                            │
                                                            ▼                                            ▼
                                                  [LiDAR VAE decoder]                       [Conditional LiDAR U-Net]
                                                            │                                            │
                                                            ▼                                            ▼
                                                 reconstructed range_image                       ε̂  [B, 8, 8, 256]
```

---

## 1. Image VAE encoder (frozen, off-the-shelf)

**Purpose:** encode the dashcam-stand-in CAM_FRONT frame into a compact latent that the
LiDAR U-Net can attend to as cross-attention KV context.

| Aspect | Value |
|---|---|
| Source | HuggingFace `diffusers.AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")` |
| Status | **Frozen** in all milestones (M0 through M4). `requires_grad_(False)` and `eval()`. |
| Encoder only | We never call `.decode()` |
| Input | `[B, 3, 256, 448]` RGB, in **[-1, 1]** range (the SD VAE expects this, *not* [0,1]) |
| Output | `[B, 4, 32, 56]` (spatial /8) |
| Latent scaling factor | `0.18215` — multiply the encoder output before concat into KV. This matches SD 1.5's training distribution. |
| dtype during use | `fp16` to save VRAM; cast back to `fp32` after encoding |
| File | [`s2s_min/models/image_encoder.py`](s2s_min/models/image_encoder.py) (~30 LOC wrapper) |

**Implementation sketch.** Single class `FrozenSDVAEEncoder(nn.Module)`. `forward(rgb_minus1_to_1) -> latent`. The constructor downloads the VAE on first run and caches it; all weights `.requires_grad_(False)`. No `_init_weights` — we never train this.

---

## 2. Raymap builder

**Purpose:** geometric conditioning input for the U-Net. For every pixel of the
*latent* grid, store the camera ray origin and unit direction in the vehicle frame.

| Aspect | Value |
|---|---|
| Status | Pure-function; no learned parameters |
| Input | Camera intrinsics `K [3,3]`, extrinsics `cam-to-ego T [4,4]`, target grid shape `(H_latent=32, W_latent=56)` |
| Output | `[B, 6, 32, 56]` — `[origin_xyz (3), direction_xyz (3)]` |
| Critical correctness rule | **Compute directly on the latent grid using scaled intrinsics** (`K' = diag(1/s, 1/s, 1) @ K`, where `s=8` is the SD-VAE downsample factor). Do **not** build the raymap on the full 256×448 image and `F.interpolate` down — bilinear-resampling unit direction vectors produces non-unit results and corrupts conditioning. |
| Direction normalization | Each direction vector is normalized to unit length **after** transforming into the ego frame |
| Origin | All pixels share the camera origin in ego frame (`T[:3, 3]`), broadcast to `[B, 3, H, W]` |
| File | [`s2s_min/models/raymap.py`](s2s_min/models/raymap.py) (~50 LOC) |

**Math.**
```
for each (u, v) in the H_latent × W_latent grid:
    pixel_h = (K'⁻¹ @ [u, v, 1]ᵀ)              # ray direction in camera frame
    pixel_h /= ||pixel_h||                       # unit
    dir_world = T[:3,:3] @ pixel_h               # rotate into ego frame
raymap[:, 0:3] = T[:3, 3].broadcast_to(grid)     # origin
raymap[:, 3:6] = dir_world
```

---

## 3. KV context assembly (concat + pool)

**Purpose:** produce a single fixed-size KV context tensor that every U-Net block
cross-attends to. Pooled once outside the U-Net and reused at every level.

| Aspect | Value |
|---|---|
| Inputs | `image_latent [B, 4, 32, 56]`, `raymap [B, 6, 32, 56]` |
| Step 1 | `kv_full = torch.cat([image_latent, raymap], dim=1)` → `[B, 10, 32, 56]` |
| Step 2 | `kv_context = F.adaptive_avg_pool2d(kv_full, (8, 64))` → `[B, 10, 8, 64]` |
| Why pool | Without pooling, KV is `32·56 = 1792` tokens; Q at level 0 is `8·256 = 2048`. Attention matrix dominates VRAM. Pooling to `8·64 = 512` KV tokens saves ~3.5×. |
| Trade-off | Spatial precision of image conditioning is reduced. If M3 generation ignores conditioning, revisit (try `(16, 32)` KV grid or multi-scale KV per U-Net level). |
| File | Inline in [`s2s_min/train/smoke_test.py`](s2s_min/train/smoke_test.py) and later [`s2s_min/train/train_diffusion.py`](s2s_min/train/train_diffusion.py) — no separate module |

### Two-latent-space view — image and LiDAR latents are **independent**

The image VAE's 4-channel latent and the LiDAR VAE's 8-channel latent are **decoupled** dimensions — neither constrains the other. The image latent's 4 channels are fixed by Stable Diffusion 1.5 (baked into the pretrained weights); the LiDAR latent's 8 channels are our deliberate choice (see [LiDAR VAE §4](#4-lidar-vae--detailed-spec)). They only meet in the cross-attention block of the LiDAR U-Net, where pooled image-side features (KV) condition the noisy LiDAR latent (Q).

```
        image side                              LiDAR side
        ───────────                             ──────────
CAM_FRONT (256×448)              LIDAR_TOP point cloud
   │                                  │
   ▼ SD 1.5 VAE (frozen)              ▼ our LiDAR VAE (M1-trained)
image_latent [B, 4, 32, 56]      lidar_latent [B, 8, 8, 256]
   │ 4-ch — fixed by SD              │ 8-ch — our choice
   │                                  │
   └────────► concat + pool ◄────────┘
              [B, 10, 8, 64]
              (image_latent_ch=4 + raymap_ch=6)
                    │
                    ▼ cross-attention KV
       Diffusion U-Net operates on the LiDAR latent (Q)
       and cross-attends to the pooled KV
```

Practical consequence: scaling `lidar.latent_channels` (e.g. 8 → 16 to match the paper) does **not** require changing the image VAE or the SD 1.5 pretrained weights. It only affects the LiDAR VAE head, the cached latents (M2), and the U-Net's input/output convs (`unet.in_channels`/`out_channels` in [configs/min.yaml](s2s_min/configs/min.yaml)).

---

## 4. LiDAR VAE — **detailed spec**

**Purpose:** compress the 32×1024×3 range-image LiDAR representation into a compact
latent space that the diffusion U-Net then operates over. Trained in M1 (Phase A),
then **frozen** for the rest of the pipeline.

> **Channel-count note.** nuScenes LiDAR ships `(x, y, z, intensity, ring_index)` — no
> elongation. The Sensor2Sensor paper includes a 4th elongation channel because it
> targets Waymo. For this nuScenes pipeline we use **3 channels: range, intensity,
> validity**. Re-introduce elongation only if the pipeline is ported to Waymo later.

### 4.1 Source & status

| Aspect | Choice |
|---|---|
| Provenance | **Written from scratch.** RangeLDM-style convolutional VAE — same family as Stable Diffusion's `AutoencoderKL` but with 3-channel input/output, anisotropic spatial downsample, and circular-W padding. |
| Pretrained warm-start | **None.** No public 3-channel-range-image VAE checkpoint exists for nuScenes. RangeLDM ships a 2-channel (range+intensity) VAE on PKU Disk but it requires extraction from a bundled LDM checkpoint and has a 4-d latent (vs our 8-d) — see [min_pipeline_plan.md §"Optional extension"](min_pipeline_plan.md). X-Drive's published VAE uses 2-channel range-only `mean=[50,0], std=[50,255]` — not directly transferable. |
| Reference for channel/stage choices | [`Reference_code/X-Drive/xdrive/networks/blocks_pc_RangeLDM.py`](Reference_code/X-Drive/xdrive/networks/blocks_pc_RangeLDM.py) — cross-reference, do not copy |
| Training | M1 (Phase A). Plain `AdamW` (no 8-bit for the VAE; small enough). |
| Frozen | After M1 completes, for M2/M3/M4. `requires_grad_(False)` and `eval()`. |
| Target params | ~10 M |
| File | [`s2s_min/models/lidar_vae.py`](s2s_min/models/lidar_vae.py) |

### 4.2 I/O

| | Input | Output |
|---|---|---|
| **Encoder** | range image `[B, 3, 32, 1024]`, values normalized to **[0, 1]** per channel | `μ [B, 8, 8, 256]`, `logσ² [B, 8, 8, 256]` |
| **Sample** | `μ, logσ²` | `z = μ + σ · ε`, `ε ~ N(0, I)`, `z [B, 8, 8, 256]` (or `μ` at inference) |
| **Decoder** | `z [B, 8, 8, 256]` | reconstructed range image `[B, 3, 32, 1024]` |

### 4.3 Channel normalization (must match training data)

| Channel | Source range | Mapped to | Notes |
|---|---|---|---|
| 0 — range (m) | `[0, 100]` (clamped) | `[0, 1]` | nuScenes 32-beam max range is 100 m. Paper uses 150 m for Waymo — keep ours at 100. |
| 1 — intensity | `[0, 255]` | `[0, 1]` | nuScenes raw `.bin` intensity is 0–255 |
| 2 — validity | `{0, 1}` | `{0, 1}` | Binary mask of valid returns (1 where a LiDAR return landed in this (row, col) cell, 0 otherwise) |

Reverse mapping at decode time uses the same constants.

### 4.4 Encoder architecture

Goal: 4× spatial downsampling on each axis (H: 32→8, W: 1024→256), reaching 8-channel latent.

```
Input:  range_image [B, 3, 32, 1024]

Stem        : CircularConv2d(3 → 32, k=3)                            -> [B, 32, 32, 1024]

Stage 1     : 2× ResBlock(32)                                        -> [B, 32, 32, 1024]
              Downsample2d (stride 2 on BOTH H and W)                -> [B, 64, 16, 512]

Stage 2     : 2× ResBlock(64)                                        -> [B, 64, 16, 512]
              Downsample2d (stride 2 on BOTH H and W)                -> [B, 128, 8, 256]

Bottleneck  : 2× ResBlock(128)                                       -> [B, 128, 8, 256]
              SelfAttn(128)  (optional, 1 block at bottleneck)       -> [B, 128, 8, 256]

Head        : GroupNorm → SiLU → CircularConv2d(128 → 2·8, k=1)      -> [B, 16, 8, 256]
              Split channel dim                                      -> μ [B, 8, 8, 256], logσ² [B, 8, 8, 256]
```

Notes:
- Encoder downsamples **both H and W** (unlike the diffusion U-Net which is W-only). The VAE is acting on the full range image where H=32 is large enough to pool; the U-Net acts on the already-compressed latent where H=8 is too small.
- `Downsample2d` here uses a stride-2 `CircularConv2d(k=3)` on W and a regular stride-2 conv on H (or a single conv with `stride=2` and the circular-pad-on-W wrapper).
- ResBlock structure: pre-norm `GroupNorm → SiLU → CircConv → GroupNorm → SiLU → CircConv` (zero-init on the second conv) + 1×1 skip if channels change. Same module as the U-Net's [`ResBlock`](s2s_min/models/blocks.py).

### 4.5 Decoder architecture

Mirror of encoder. Upsample by 2 on both axes using nearest-neighbour + circular conv (avoids checkerboard artifacts from transposed convs).

```
Input:  z [B, 8, 8, 256]

Stem        : CircularConv2d(8 → 128, k=3)                          -> [B, 128, 8, 256]

Bottleneck  : SelfAttn(128) (optional)                              -> [B, 128, 8, 256]
              2× ResBlock(128)                                      -> [B, 128, 8, 256]

Stage 2     : Upsample2d (nearest, ×2 on H and W)                   -> [B, 128, 16, 512]
              CircularConv2d(128 → 64, k=3)
              2× ResBlock(64)                                       -> [B, 64, 16, 512]

Stage 1     : Upsample2d (nearest, ×2 on H and W)                   -> [B, 64, 32, 1024]
              CircularConv2d(64 → 32, k=3)
              2× ResBlock(32)                                       -> [B, 32, 32, 1024]

Head        : GroupNorm → SiLU → CircularConv2d(32 → 3, k=3)        -> [B, 3, 32, 1024]   (zero-init)
              Per-channel activation:
                  - range, intensity: sigmoid → [0,1]
                  - validity:         sigmoid → [0,1]   (BCE loss interprets as probability)
```

### 4.6 Loss (4 terms, initial M1 version)

Defined in [equations.md §(1)(3)(4)(7)](equations.md). Full formula:

```
L_VAE = λ_range  · L1_masked(x_range,     x̂_range,     mask=x_validity)
      + λ_inten  · L1_masked(x_intensity, x̂_intensity, mask=x_validity)
      + λ_valid  · BCE(x_validity, x̂_validity)
      + λ_KL     · 0.5 · mean( μ² + σ² − log σ² − 1 )
```

| Weight | Start value | Reason |
|---|---|---|
| `λ_range` | 1.0 | Anchor weight |
| `λ_intensity` | 1.0 | Same magnitude as range after [0,1] normalization |
| `λ_validity` (BCE) | 1.0 | Binary; BCE is on similar scale to L1 here |
| `λ_KL` | **1e-6** | X-Drive's value. KL is much larger raw than L1 on a [0,1]-normalized signal; without this down-weight, posteriors collapse to the prior. |

**Validity-masked L1:** the L1 terms for range/intensity are evaluated **only at pixels where the ground-truth validity is 1**. Invalid pixels carry no geometric/intensity signal — including them in L1 trains the decoder to predict noise. Masking is per-pixel; the BCE term separately learns where validity is 0 or 1.

```python
mask = x_validity                                # [B, 1, 32, 1024]
loss_range = (mask * (x_range - x̂_range).abs()).sum() / mask.sum().clamp(min=1)
# (similar for intensity)
loss_valid = F.binary_cross_entropy(x̂_validity, x_validity)
loss_KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()
```

### 4.7 Deferred loss terms (re-add only if M1 reconstructions blur)

| Term | When to add | Cost |
|---|---|---|
| `LPIPS_normals` (on normals computed by finite differences from `x̂_range`) | If geometric edges are smeared in M1 BEV check | +250 MB VRAM, +30% wall-clock; needs `pip install lpips` |
| `LPIPS_intensity` | If intensity texture is blurred | Same as above |
| `LPIPS_validity` | Paper has it but it's the least impactful — add last | Same as above |

### 4.8 Initialization

- Conv weights: Kaiming-normal (`fan_in`, `relu` nonlinearity).
- Conv biases: zero.
- GroupNorm scales: 1, biases: 0.
- **Decoder head conv: weights zero-init, bias zero-init.** Makes the fresh-decoder output exactly the centre of `[0,1]` after sigmoid (≈0.5 everywhere) instead of garbage — gives the early loss signal a sensible starting point.

### 4.9 Training regime (M1)

| Item | Value |
|---|---|
| Optimizer | `torch.optim.AdamW`, `lr=4e-4`, `betas=(0.9, 0.999)`, `weight_decay=1e-4` |
| Batch | actual 2, grad-accum 4 → effective 8 |
| Mixed precision | `torch.cuda.amp` (fp16 autocast + GradScaler) |
| Gradient checkpoint | Optional — only the bottleneck ResBlocks if VRAM is tight |
| Epochs (10-scene subset) | 50 (~30 min on RTX 3060) |
| Overfit-10 milestone (M1 pass criterion) | mean L1-range < 0.05 within ~500 steps on a fixed 10-sample batch |
| EMA | None for the VAE (only the diffusion U-Net in M3 uses EMA) |
| Output checkpoint | `s2s_min/out/lidar_vae_ema.pt` (consumed by M2/M3/M4) |

---

## 5. LiDAR U-Net (conditional denoiser)

Already specified in [min_pipeline_plan.md §"U-Net details"](min_pipeline_plan.md). Summary:

| Aspect | Value |
|---|---|
| Provenance | Written from scratch |
| Input | `z_noisy [B, 8, 8, 256]`, timestep `t [B]`, `kv_context [B, 10, 8, 64]` |
| Output | `ε̂` or `v̂` (v-prediction) `[B, 8, 8, 256]` |
| Channels per level | `[96, 192, 384]` (stem 96, bottleneck 384) |
| Downsampling | **W-only**, stride 2; H stays at 8 across all levels (config knob `downsample_h_once` available) |
| Self-attn | Every ResBlock |
| Cross-attn | Every ResBlock, KV = `kv_context`, KV projections shared across levels |
| Timestep | Sinusoidal → 2-layer MLP → AdaLN-Zero (scale/shift) per ResBlock |
| Conv | All `CircularConv2d` (circular pad W, zero H) |
| Output head | GroupNorm → SiLU → CircularConv2d(96 → 8), **zero-init** |
| Target params | ~25–35 M |
| Files | [`s2s_min/models/unet.py`](s2s_min/models/unet.py), [`s2s_min/models/attention.py`](s2s_min/models/attention.py), [`s2s_min/models/blocks.py`](s2s_min/models/blocks.py) |

### Important divergence from paper documented in M5 deviations table

The paper defines a single "Cross-sensor Attn" block as **`flatten + concat (camera_tokens, lidar_tokens) → shared-projection self-attention → split`** (Figure 3, right panel; symmetric, bidirectional information flow).

Our `CrossAttention` is **one-way SD-style cross-attention**: distinct Q-projection from LiDAR features and K/V-projection from the pre-pooled image+raymap context. We deliberately do not update image features because we do not generate images.

Cross-view attention (the blue Figure-3 box) is **not implemented** — we have only one output modality (LiDAR), so view consistency is moot.

---

## 6. Diffusion schedule

| Aspect | Value |
|---|---|
| Class | `diffusers.DDPMScheduler` (training noise) + `diffusers.DDIMScheduler` (inference) |
| Train timesteps | 1000 |
| Beta schedule | `scaled_linear` (SD 1.5 default) |
| Prediction type | `v_prediction` (more stable at small batch than `epsilon`) |
| Inference steps | 25 (DDIM) |
| File | [`s2s_min/models/diffusion.py`](s2s_min/models/diffusion.py) (~40 LOC wrapper) |

---

## Component-level shape table (quick reference)

| Stage | Tensor name | Shape | Module |
|---|---|---|---|
| 0 | `rgb` | `[B, 3, 256, 448]` | dataloader |
| 1 | `image_latent` | `[B, 4, 32, 56]` | `FrozenSDVAEEncoder` |
| 2 | `raymap` | `[B, 6, 32, 56]` | `build_raymap` |
| 3 | `kv_full` | `[B, 10, 32, 56]` | `torch.cat` |
| 4 | `kv_context` | `[B, 10, 8, 64]` | `F.adaptive_avg_pool2d` |
| 5 | `range_image` | `[B, 3, 32, 1024]` | `point_cloud_to_range_image` |
| 6 | `μ, logσ²` | `[B, 8, 8, 256]` ×2 | `LiDARVAE.encode` |
| 7 | `z_lidar` | `[B, 8, 8, 256]` | `LiDARVAE.reparameterize` |
| 8 | `z_noisy` | `[B, 8, 8, 256]` | `DDPMScheduler.add_noise` |
| 9 | `ε̂` | `[B, 8, 8, 256]` | `LiDARUNet` |
| 10 | `ẑ` (after DDIM 25 steps at inference) | `[B, 8, 8, 256]` | `DDIMScheduler` loop |
| 11 | `range_image_pred` | `[B, 3, 32, 1024]` | `LiDARVAE.decode` |
| 12 | point cloud `(N, 4)` | variable N | `range_image_to_pc` |




