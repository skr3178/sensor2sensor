# LiDAR VAE — single-source reference

Consolidated reference for everything LiDAR-VAE-related in this project. Pulls from
[`../../architecture.md`](../../architecture.md) §4 (architectural spec),
[`./models.md`](./models.md) §2 (build plan + failure modes),
[`../RESULTS.md`](../RESULTS.md) (experiment record),
[`../../min_pipeline_plan.md`](../../min_pipeline_plan.md) (scope decisions),
[`../../hyperparameters.md`](../../hyperparameters.md) (paper reference values),
and [`../../equations.md`](../../equations.md) (loss formulas).

If a fact here disagrees with one of the source docs, **the source doc is canonical** —
this file is a synthesis, not a replacement.

---

## 0. Quick reference

| Property | Value |
|---|---|
| Status | **trained and frozen** (v3 checkpoint kept) |
| Source code | [`s2s_min/models/lidar_vae.py`](../models/lidar_vae.py) (2.07 M params) |
| Loss code | [`s2s_min/train/losses.py`](../train/losses.py) (4 terms) |
| Trainer code | [`s2s_min/train/train_vae.py`](../train/train_vae.py) |
| Trained on | nuScenes v1.0-trainval, 10-scene seed-0 subset, 401 LIDAR_TOP keyframes |
| Best checkpoint | [`../out/lidar_vae_best.pt`](../out/lidar_vae_best.pt) — EMA weights at step ~900, `l1_range_ema = 0.01116` (training) |
| Channels in | 3: range, intensity, validity (no elongation — Waymo-only signal) |
| Channels out | 3: same |
| Latent shape | `[B, 8, 8, 256]` (μ and logσ² each) |
| Reconstruction quality | training L1_range ≈ 0.011 (normalized) ≈ 1.1 m; held-out expected 2-5× worse |
| End-to-end role | Frozen encoder + decoder. M2 caches its latents; M3 diffuses in latent space; M4 decodes samples back to range images. |

---

## 1. Role in the pipeline

```
nuScenes raw .pcd.bin              [N, 5]  (x, y, z, intensity, ring_index)
       │
       ▼  data.range_image.point_cloud_to_range_image
range image (VAE input)            [3, 32, 1024] in [0, 1]
       │
       ▼  models.lidar_vae.LiDARVAE.encode  ←── M1 trains this end-to-end
posterior params μ, logσ²          [8, 8, 256] each
       │
       ▼  reparameterize (z = μ + σ·ε in .train(); z = μ in .eval())
LiDAR latent z                     [8, 8, 256]                ←── M2 caches this
       │
       ▼  conditional diffusion U-Net (M3 trains this)
       ▼  DDIM sampling (M4 inference)
LiDAR latent ẑ                     [8, 8, 256]
       │
       ▼  LiDARVAE.decode → sigmoid                            ←── M1 trains this too
reconstruction x̂                   [3, 32, 1024] in [0, 1]
       │
       ▼  data.range_image.range_image_to_point_cloud
point cloud (lossy inverse)        [M, 4]  (x, y, z, intensity)
```

After M1 the VAE is **frozen forever** (`requires_grad_(False).eval()`). M2-M4 only consume `encode` and `decode`.

---

## 2. I/O contract

| | Input | Output |
|---|---|---|
| **Encoder** | `[B, 3, 32, 1024]` float32 in `[0, 1]` per channel | `μ, logσ²` each `[B, 8, 8, 256]` |
| **Sample** | `μ, logσ²` | `z = μ + σ·ε`  in `.train()`; `z = μ` in `.eval()`; shape `[B, 8, 8, 256]` |
| **Decoder** | `z [B, 8, 8, 256]` | `[B, 3, 32, 1024]` float32 in `[0, 1]` per channel |
| **Forward** | `[B, 3, 32, 1024]` | `(x̂, μ, logσ²)` |

Spatial downsample factor: **4× on both H and W** (32→8, 1024→256). Unlike the diffusion U-Net which is W-only (since the U-Net operates on the already-compressed latent where H=8 is too small to pool further).

---

## 3. Channels

### 3.1 Layout

| Index | Channel | Source range (raw) | Normalized to | Notes |
|---|---|---|---|---|
| 0 | range (m) | `[0, 100]` clamped | `[0, 1]` | nuScenes 32-beam max useful range ≈ 100 m. Paper uses 150 m for Waymo. |
| 1 | intensity | `[0, 255]` | `[0, 1]` | nuScenes raw `.bin` intensity is uint8 |
| 2 | validity | `{0, 1}` | `{0, 1}` | 1 where a LiDAR return landed; ≈78 % cell fill rate on a typical sample |

### 3.2 Why 3 channels (not 4)

The Sensor2Sensor paper uses **4 channels** (range, intensity, **elongation**, validity) because it targets the Waymo Open Dataset. **nuScenes LiDAR does not measure elongation** — the HDL-32E sensor doesn't expose the waveform spread the way Waymo's Honeycomb LiDAR does. Raw nuScenes `.pcd.bin` ships only `(x, y, z, intensity, ring_index)`.

The plan ([min_pipeline_plan.md](../../min_pipeline_plan.md)) explicitly drops elongation for nuScenes. If the pipeline is ever ported to Waymo, reintroduce a 4th elongation channel and add `λ_elongation` back to the loss.

### 3.3 Cells without a return

The validity channel is the source of truth for "is this a real return". Invalid cells have **all three channels set to 0**, not NaN. The loss masks out range/intensity at invalid cells; only validity sees them.

---

## 4. Architecture

`Reference_code/RangeLDM`-style convolutional VAE. Same family as Stable Diffusion's `AutoencoderKL` but with:
- 3-channel I/O (vs SD's 3-channel RGB)
- Anisotropic downsample factor (4× both axes; SD uses 8×)
- **Circular padding on the W axis** (azimuth wraps at 0°/360°; W is periodic, H is not)
- Smaller base channels (32 vs 64) for the 3060 budget

### 4.1 Encoder

```
Input:  range_image [B, 3, 32, 1024]

Stem        : CircularConv2d(3 → 32, k=3)                            -> [B,  32, 32, 1024]

Stage 1     : 2× ResBlock(32)                                        -> [B,  32, 32, 1024]
              Downsample2d  (stride 2 on H and W, circ-pad W)        -> [B,  64, 16,  512]

Stage 2     : 2× ResBlock(64)                                        -> [B,  64, 16,  512]
              Downsample2d                                            -> [B, 128,  8,  256]

Bottleneck  : 2× ResBlock(128)                                       -> [B, 128,  8,  256]
              SelfAttn(128)                                           -> [B, 128,  8,  256]

Head        : GroupNorm → SiLU → CircularConv2d(128 → 2·8, k=1)      -> [B,  16,  8,  256]
              split channel dim                                      -> μ, logσ²  each [B, 8, 8, 256]
```

### 4.2 Decoder (mirror, nearest-neighbour upsample)

```
Input:  z [B, 8, 8, 256]

Stem        : CircularConv2d(8 → 128, k=3)                           -> [B, 128,  8,  256]

Bottleneck  : SelfAttn(128)                                          -> [B, 128,  8,  256]
              2× ResBlock(128)                                       -> [B, 128,  8,  256]

Stage 2     : Upsample2d  (nearest ×2 on H and W)                    -> [B, 128, 16,  512]
              CircularConv2d(128 → 64, k=3)
              2× ResBlock(64)                                         -> [B,  64, 16,  512]

Stage 1     : Upsample2d                                              -> [B,  64, 32, 1024]
              CircularConv2d(64 → 32, k=3)
              2× ResBlock(32)                                         -> [B,  32, 32, 1024]

Head        : GroupNorm → SiLU → CircularConv2d(32 → 3, k=3)  ← zero-init  -> [B, 3, 32, 1024]
              per-channel sigmoid                                     -> values in [0, 1]
```

Notes:
- **Nearest-neighbor + circular conv** for upsample (avoids checkerboard artifacts of transposed convs).
- **ResBlock**: pre-norm `GroupNorm → SiLU → CircConv → GroupNorm → SiLU → CircConv` (zero-init second conv) + 1×1 skip if channels change. Shared with the U-Net's [`ResBlock`](../models/blocks.py).
- **Padding**: every conv that touches the W axis uses `mode="circular"` on W and `mode="constant", value=0` on H.

### 4.3 Properties summary

| Property | Value |
|---|---|
| Total trainable params | **2.07 M** (0.94 M encoder + 1.13 M decoder) |
| Stem channels | 32 |
| Down/up stages | 2 — channel progression 32 → 64 → 128 |
| Bottleneck self-attn | 1 block (3.5 % of total params) |
| Spatial down factor | **4× on both H and W** (anisotropic from the U-Net which is W-only) |
| Latent shape | `[B, 8, 8, 256]` (μ + logσ² split from a single `[B, 16, 8, 256]` head output) |
| Output activation | per-channel sigmoid → `[0, 1]` |
| Padding | circular on W, zero on H, every conv |

### 4.4 Initialization

- Conv weights: Kaiming-normal (fan_in, ReLU nonlinearity).
- Conv biases: zero.
- GroupNorm scales: 1, biases: 0.
- **Decoder head conv: weight + bias both zero-init.** Fresh decoder outputs exactly `sigmoid(0) = 0.5` everywhere on every channel. Gives the early loss signal a sensible starting point — the model improves *from* the dataset mid-point rather than from random garbage. Verified: at init, the max absolute deviation from 0.5 is < 1e-6 across the full reconstruction.

### 4.5 Design rationale — why 8-d latent and `base_channels=32`

#### Latent dim is decoupled from the image VAE

The image VAE's 4 channels are baked into SD 1.5's pretrained weights; the LiDAR VAE's 8 channels are our choice. Independent dimensions — they only meet at cross-attention KV in the U-Net.

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

**The binding constraint on `latent_channels` is M3 VRAM**, not the image side:

| `latent_channels` | Est. M3 VRAM (3060, batch 1, fp16, grad-ckpt) | Verdict |
|---|---|---|
| 4 (RangeLDM) | ~3 GB | very comfortable |
| **8 (ours)** | **~5 GB** | **50 % headroom for batch>1, optim state, EMA shadow** |
| 16 (paper) | ~7–8 GB | tight on 3060 |
| 32 | ~10–12 GB | won't fit at batch 2 |

The paper uses 16-d on 128 TPUs where VRAM is free; we chose 8 to keep M3 headroom. Scaling later costs `lidar.latent_channels` + `unet.{in,out}_channels` in [`configs/min.yaml`](../configs/min.yaml), one M1 re-train (~22 min), one M2 re-cache. Secondary reasons not to go higher: harder diffusion problem on 401 keyframes; small encoder (~1 M params) gains little from a richer latent.

#### Param count is dominated by `base_channels²`

Conv weights scale as `in_ch × out_ch × k²`. With multipliers `[1, 2, 4]` and base width `b`, **the bottleneck (at `4b` channels) holds ~70 % of all params** — so halving `b` quarters the total.

Encoder param distribution at b=32: stem+stage1 ~4 %, stage2 ~16 %, **bottleneck ~70 %**, head <1 %, downsamples ~10 %.

| | b=32 (ours) | b=64 (RangeLDM-width) | ratio |
|---|---|---|---|
| Encoder | 0.94 M | ~3.75 M | ~4× |
| Decoder | 1.13 M | ~4.5 M | ~4× |
| **Total** | **2.07 M** | **~8–9 M** | **~4×** |

(Earlier docs estimated RangeLDM at "~16 M" — that was an overshoot; ~8–9 M is closer.)

#### Knob-effects summary

| Change | Params | M1 VRAM | M3 VRAM (VAE is frozen) | Reconstruction quality |
|---|---|---|---|---|
| `base_channels`: 32 → 64 | **4×** | ~2–3× | no change | small-but-real improvement |
| `base_channels`: 32 → 48 (intermediate) | ~2.25× | ~+50 % | no change | likely the cheapest capacity bump |
| `latent_channels`: 8 → 16 | small (head only) | ~10 % | **~1.5×** | modest |
| `num_res_blocks`: 2 → 3 | +33 % | +linear | no change | small |
| Enable LPIPS terms (v4) | 0 (frozen VGG) | +750 MB | no change | **large — geometric sharpness** |

Widening `b` is a **one-time M1 cost** — the VAE freezes after M1, so M2/M3/M4 are unaffected. If a held-out val eval (see §11) shows we're capacity-bound, `base_channels=48` is the cheapest knob to claw back capacity.

---

## 5. Loss

### 5.1 Formula (4 terms, M1 version, no LPIPS)

```
L_VAE = λ_range     · L1_masked(x_range,     x̂_range,     mask=x_validity)
      + λ_intensity · L1_masked(x_intensity, x̂_intensity, mask=x_validity)
      + λ_validity  · BCE(x_validity, x̂_validity)
      + λ_KL        · 0.5 · mean(μ² + σ² − log σ² − 1)
```

| Symbol | Meaning |
|---|---|
| λ (lambda) | scalar weight that scales how much each loss term contributes to the gradient |
| L1_masked | mean absolute error, averaged only over pixels where the ground-truth validity = 1 |
| BCE | binary cross-entropy on the validity channel (sigmoid output vs binary target) |
| KL | analytical Gaussian KL between the posterior `N(μ, σ²)` and the prior `N(0, I)` |

### 5.2 λ weights (committed config)

| Weight | Value | Why |
|---|---|---|
| `λ_range` | **50.0** | RangeLDM's nuScenes config uses 50; range is the geometrically important channel. Was 1.0 in v1, raised to 50 in v2/v3 |
| `λ_intensity` | 1.0 | Same magnitude as range after [0,1] normalization |
| `λ_validity` | 1.0 | Binary; BCE is on similar scale to L1 here |
| `λ_KL` | **1e-6** | X-Drive / RangeLDM default. KL is much larger raw than L1 on a [0,1] signal; without this down-weight, posterior collapses to the prior |

### 5.3 Validity masking

Range and intensity L1 terms are computed **only at cells where the ground-truth validity = 1**. Invalid pixels carry no geometric/intensity signal — including them in L1 trains the decoder to predict noise where there's nothing.

```python
mask  = x_validity                              # [B, 1, 32, 1024]
denom = mask.sum().clamp(min=1.0)
loss_range     = (mask * (x_range     - x̂_range    ).abs()).sum() / denom
loss_intensity = (mask * (x_intensity - x̂_intensity).abs()).sum() / denom
loss_validity  = F.binary_cross_entropy(x̂_validity, x_validity)
loss_kl        = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()
```

### 5.4 Deferred terms (re-add only if M1 reconstructions visibly blur)

| Term | When to add | Cost |
|---|---|---|
| `LPIPS_normals` | If geometric edges are smeared in BEV check | +250 MB VRAM, +30 % wall-clock; needs `pip install lpips` |
| `LPIPS_intensity` | If intensity texture is blurred | Same |
| `LPIPS_validity` | Least impactful — add last | Same |

The paper's full 9-term loss adds 4 LPIPS terms on top of the 4 we have plus a 5th L1 for elongation (which we don't have). We chose the 4-term subset for M1 to keep the loop fast and the failure mode debuggable.

---

## 6. Training regime

### 6.1 Hyperparameters (v3 — the committed run)

| Item | Value |
|---|---|
| Optimizer | AdamW, betas=(0.9, 0.999), weight_decay=1e-4 |
| Peak learning rate | 4e-4 |
| LR schedule | **cosine with 200-step linear warmup**, lr_min = 4e-6 |
| Grad-norm clip | 1.0 |
| Mixed precision | fp16 autocast + GradScaler (BCE forced to fp32 inside `autocast(enabled=False)` block) |
| Batch × grad-accum | 2 × 4 = effective 8 |
| EMA decay | 0.999 (shadow weights kept on CPU) |
| Epochs | 50 over 401-keyframe subset (~2500 optimizer steps) |
| Best-checkpoint save | `lidar_vae_best.pt` overwritten when `l1_range_ema` (α=0.99 smoothing) reaches new low after 50-step warmup |

### 6.2 Pass criteria (M1)

1. **Overfit-10**: with 10 fixed samples and ~500 AdamW steps, mean validity-masked L1 on range drops below **0.05** (= 5 m at 100 m range clamp).
2. **Full-epoch**: 50 epochs over the 401-keyframe subset complete without OOM/NaN; final decoded range image visually matches input in the BEV eyeball check.

Both passed in v3. See §8 for the actual results.

---

## 7. Dataset

### 7.1 nuScenes accounting

| | Count | Notes |
|---|---|---|
| Total scenes available | **850** | v1.0-trainval, on disk via the `nuscenes` symlink → `/media/skr/storage/self_driving/S2GO/data/nuscenes/` |
| Total LIDAR_TOP keyframes | **34,149** | mean ~40 per scene, range 32-41 |
| Scenes we picked for M1 | **10** | sampled with `np.random.default_rng(0).choice(850, 10, replace=False)` |
| Keyframes in the subset | **401** | listed in [../out/subset_scene_tokens.txt](../out/subset_scene_tokens.txt) |
| Coverage used | **1.18 %** | (401 / 34,149 keyframes) |

### 7.2 The 10 scenes (seed=0)

| Scene token | Name | `nbr_samples` |
|---|---|---|
| `813213458a214a39a1d1fc77fa52fa34` | scene-0015 | 40 |
| `3dd2be428534403ba150a0b60abc6a0a` | scene-0035 | 40 |
| `30e77a3561ce4f1bb4fd90e322ea29d7` | scene-0066 | 39 |
| `f9e460f092c94466b1211704b5a8859d` | scene-0191 | 40 |
| `26540bfbab79463cb1ba76b52ec6013b` | scene-0285 | 40 |
| `93608f0d57794ba6b014314c488e2b4a` | scene-0332 | 40 |
| `788c5502523f4d01b3a8de47ec3dadfb` | scene-0541 | 41 |
| `62d9a581c7114d90a48bacfb2a47b104` | scene-0684 | 41 |
| `d3b86ca0a17840109e9e049b3dd40037` | scene-0905 | 40 |
| `a04daf2d0f194b2ab2ff2a47dfebc1d7` | scene-0930 | 40 |
| **Total** | | **401** |

Reproduce with `python s2s_min/scripts/select_subset.py`. Scale up by passing `--n 700` (full nuScenes train split).

---

## 8. M1 training results

### 8.1 Three runs comparison

| | v1 | v2 | **v3 (committed)** |
|---|---|---|---|
| `λ_range` | 1 | 50 | 50 |
| LR schedule | constant | constant | **cosine + 200-step warmup** |
| EMA(0.999) shadow | none | enabled | enabled |
| Best-ckpt save | none | enabled | enabled |
| Wall-clock | 9.1 min | 9.4 min | **9.4 min** |
| Peak VRAM | 446 MiB | 446 MiB | **446 MiB** |
| Best `l1_range_ema` | ~0.015 (not saved) | 0.01147 | **0.01116** (step ~900) |
| Divergence step | ~2000 (epoch 40) | ~1000 (epoch 20) | ~1100 (epoch 21) |
| Final `l1_range_ema` | 0.188 | 0.030 | 0.066 |

Full loss curves: [../out/loss_comparison.png](../out/loss_comparison.png). Re-render with `python s2s_min/scripts/plot_train_runs.py`. Per-step logs in [`../out/train_vae_epoch50{,_v2,_v3}.log`](../out).

### 8.2 The divergence pattern (a bug we didn't fix)

All three runs hit a "convergence then explosion" pattern, but with different timing:

| Run | Symptom |
|---|---|
| v1 (`λ_range=1`) | Slow drift over ~500 steps. Stable at L1_range ≈ 0.013–0.018 from step 700–1800, then BCE_validity climbs (0.006 → 0.077), L1_range follows (0.035 → 0.165) over the next 300 steps. |
| v2 (`λ_range=50`, constant LR) | Single-shot explosion. Step 900: L1_range = 0.013, BCE = 0.171. Step 950: L1_range = **0.273**, BCE = **2.665** (21× and 16× jumps in 50 steps). Partial recovery, never returns to the good basin. |
| v3 (cosine LR) | Multi-bump precursors then collapse. Steps 850–1000 show BCE climbing (0.153 → 0.258 → 0.470) while L1_range stays small. Step 1050: L1_range jumps 7.7× (0.017 → 0.130). Cosine LR was at 2.68e-4 — not low enough to dampen. |

**Diagnostic finding:** in all three runs, **BCE_validity climbs *before* L1_range jumps**. This is the saturation precursor of a sigmoid output paired with `F.binary_cross_entropy`:

- Once the decoder is well-trained, validity logits push past `|logit| > 10`.
- Sigmoid is in its saturated tail; loss for a misprediction = `-log(eps)` ≈ huge.
- Gradient through saturated sigmoid is ≈ 0 — model can't recover; the loss spike just propagates.
- The huge gradient backprop destabilizes shared conv weights, taking L1_range down with it.

`λ_range=50` accelerates this because pushing range fidelity harder pushes parameters into saturation faster — explaining why v1 (λ=1) took 2000 steps and v2/v3 (λ=50) took 1000.

**The underlying cause is not gradient noise (more data alone won't fix it), nor LR (cosine didn't fix it).** It's a numerical fragility in the loss formulation. See §11.

### 8.3 What the EMA + best-ckpt rescue captures

`lidar_vae_best.pt` keeps the EMA(0.999) shadow weights at the timestep with the lowest training-set `l1_range_ema`. For v3 that's step ~900 (epoch ~17), `l1_range_ema = 0.01116`, which is:

- Below the M1 overfit-gate pass criterion of 0.05 (3.4× margin)
- ≈ 1.12 m mean range error on **training** data
- Expected to be 2–5× worse on **held-out** data (no val set used)

---

## 9. Checkpoints on disk

```
s2s_min/out/lidar_vae.pt        live final weights (v3, diverged region — DO NOT USE)
s2s_min/out/lidar_vae_ema.pt    EMA final weights (smoothed, but still in diverged region)
s2s_min/out/lidar_vae_best.pt   EMA at the best l1_range_ema basin (step ~868)  ← what M2/M3/M4 load
```

### 9.1 Inspecting any checkpoint

```python
import torch
ckpt = torch.load("s2s_min/out/lidar_vae_best.pt", map_location="cpu")
print("step:", ckpt["step"])
print("config:", ckpt["config"])
print("l1_range_ema:", ckpt.get("l1_range_ema"))     # only set on _best.pt
```

### 9.2 Loading for downstream use

```python
import torch
from s2s_min.models.lidar_vae import LiDARVAE

ckpt = torch.load("s2s_min/out/lidar_vae_best.pt", map_location="cuda")
vae = LiDARVAE(**{k: ckpt["config"][k]
                  for k in ("in_channels", "latent_channels", "base_channels")}).cuda().eval()
vae.load_state_dict(ckpt["state_dict"])
vae.requires_grad_(False)
```

After this the VAE is frozen and ready for M2 caching / M3 conditioning / M4 inference.

---

## 10. Reference implementations

| Source | Local path | What's there | How it compares to ours |
|---|---|---|---|
| **RangeLDM** (ECCV 2024) | [`Reference_code/RangeLDM/vae/`](../../Reference_code/RangeLDM/vae/) | Full PyTorch Lightning + SGM VAE trainer. `main.py` + `configs/nuscenes.yaml` + `sgm/models/autoencoder.py` | Closest cousin. 2-channel (range+intensity), 4-d latent. Same SD-family encoder/decoder, base channels 64 (2× wider). Optional LPIPS + adversarial discriminator. |
| **X-Drive** | [`Reference_code/X-Drive/`](../../Reference_code/X-Drive/) | Consumes RangeLDM's pretrained VAE — does not retrain. The `*RangeLDM*` files are the conditional U-Net (M3 reference). | Useful for M3 design, not for M1. |
| **OpenDWM** | HF `wzhgba/opendwm-models` (only config local) | `VAEPointCloud` — voxel-based 640×640, geometry-only | Different family. Not range-image. Not a drop-in. |
| **Sensor2Sensor paper** | no code | 4 channels, 16-d latent, 9-term loss, trained on 128 TPUs | Specifies design only; we mirror what's described and skip what isn't (LPIPS, discriminator, elongation). |
| **Ours** | [`s2s_min/models/lidar_vae.py`](../models/lidar_vae.py) | 3 channels, 8-d latent, 4-term loss, 2.07 M params | The minimum-pipeline cousin of RangeLDM. |

### Specific things worth borrowing from RangeLDM if we ever re-train

1. `GeneralLPIPSWithDiscriminator` loss class (`sgm/modules/autoencoding/losses.py`) — drops in LPIPS + optional discriminator with one yaml change.
2. PyTorch Lightning `ModelCheckpoint` callback — replaces our hand-rolled best-ckpt logic with version history and topk tracking.
3. SGM's `Encoder` / `Decoder` with `circular=True` flag — slightly more thorough circular-padding than ours.

---

## 11. Known limitations & open follow-ups

### Confirmed bottleneck for end-to-end pipeline

The current VAE is the dominant source of error in the M4 end-to-end Chamfer distance:

```
CD-3D-raw (image → DDIM → decode → cloud)   =  6.135 m
  └─ CD-VAE-only (raw cloud → range image
                  → encode → decode → cloud) =  5.583 m   ← 91 % of total error
  └─ diffusion delta                          = +0.552 m   ←  9 %
```

The diffusion U-Net (M3) adds only 9 % on top. Fixing the VAE would close ~91 % of the gap before touching the U-Net.

### Fix paths, ordered by leverage

| Path | What | Cost | Expected CD-VAE improvement | Confidence |
|---|---|---|---|---|
| **A. Fix BCE saturation** (architectural) | Drop sigmoid from decoder validity output; use `binary_cross_entropy_with_logits`; label-smooth validity targets to {0.05, 0.95}; bump weight_decay 1e-4 → 1e-2 | ~30 LOC, 1 re-train (~9 min) | Stops divergence, gets unbroken 50-epoch convergence | high |
| **B. Scale data 10 → 100 scenes** | `--n 100`, retrain | ~45 min | ~2-3× (5.58 → ~2 m), delays saturation by exposure | medium |
| **C. Scale data 10 → 700 scenes** | Full nuScenes train | ~5 hours | ~5-10× (5.58 → ~0.5-1 m, near RangeLDM published) | medium |
| **D. Bigger model** | base_channels 32 → 64 (~4× params) | minor wall-clock, +VRAM | small alone — doesn't address root cause | low |
| **E. Add LPIPS** | LPIPS_normals + LPIPS_intensity | +30 % wall-clock, +250 MB VRAM | sharpness improvement, not capacity | low |

**My recommendation order: A → B → C → E**. A fixes the actual numerical mechanism we diagnosed; B/C give clean data-scale wins on a stable trainer; E is polish.

### Validation gaps to close

1. **Held-out evaluation** — split the 10 scenes into 9 train + 1 val; report val L1, round-trip Chamfer Distance, MMD on range histograms. Turns "we think it works" into apples-to-apples vs the RangeLDM table.
2. **VAE-only quality benchmark** — round-trip Chamfer on a held-out scene, compared to RangeLDM's published numbers.

### Scale-up decision criteria (when 401 keyframes → 28 k or 34 k)

We have **34,149 LIDAR_TOP keyframes** (~23 GB) on disk; the M1 run uses 401 of them (1.18 %). Three categories of signal determine whether scaling pays off:

#### A. Architecture-validation signals — **must all be green** before any scale-up

Tell you: *the architecture itself can absorb more data productively.*

| Signal | Threshold | Why it matters |
|---|---|---|
| Training loss curve smooth, no divergence | No L1_range spike >2× | If the architecture diverges at 10 scenes it will diverge at 700 too — fix architecture first |
| Overfit-10 reaches near-zero | L1_range < 0.05 in ≤500 steps | Model has enough capacity to fit any data |
| Per-loss term magnitudes balanced | No single term > 10× others on log scale | Lambdas are sized correctly; more data won't fix bad weighting |
| VRAM well below budget | ≤6 GB at batch 2×4 | Headroom for larger batch / more epochs at scale |
| Per-step wall-clock predictable | Linear with sample count | Lets you size the scale-up budget honestly |
| EMA + best-ckpt save logic works | Best ckpt captured at the right basin | Safety net survives at scale |

#### B. Decision driver — **measure once, sets the scale-up size**

| Metric | What it tells you | How to use it |
|---|---|---|
| **Held-out val L1_range** (split 9 train / 1 val scene, retrain v4 recipe, eval on val) | Whether model is data-bound vs architecture-bound | The val/train ratio is the actual decision rule (below) |

Decision rule on val/train ratio:

```
val_L1 / train_L1   →   diagnosis                        →   next move
─────────────────────────────────────────────────────────────────────────────
≤ 1.3×                  already generalizing from 9 scenes    skip 100, go to 700
1.3× – 3×               moderate memorization                  scale to 100 first
> 3× (current likely)   heavy memorization                    scale to 100, verify, then 700
NaN / val collapses     bug in held-out loader                 debug before scaling
```

A 9/1 split costs ~10–12 min of extra wall-clock per data point. Cheap insurance against a wasted 5-hour 700-scene run.

#### C. Sanity-check signals — informative, not blocking

| Signal | What it is | When to use |
|---|---|---|
| Round-trip Chamfer Distance | decode → unproject → CD vs raw cloud, in meters | Track improvement before vs after scaling — not a fixed pre-scaling threshold |
| BEV visual sanity | `visualize_lidar_vae.py` output looks like real LiDAR | Catches catastrophic silent failures (mode collapse, flat plane) that metrics might hide |

Both should pass anyway if the architecture-validation signals are green. They're confirmation, not gates.

#### Summary — the practical sequence

1. **Read training log** of the latest run → confirm all "A" signals are green
2. **Add a held-out scene split + val loop** to `train_vae.py` (~50 LOC)
3. **Re-train current recipe on 9 train scenes**, log val L1 every epoch (~10–12 min)
4. **Read off val/train ratio** → pick 100-scene vs 700-scene scale-up
5. **Generate a BEV plot** of the best checkpoint (cheap insurance against silent failure)

You **only need step 3's result to unblock the scaling decision.** Steps 5 is sanity insurance.

---

## 12. Failure modes (quick lookup)

| Symptom | Likely cause | First fix |
|---|---|---|
| KL term explodes | `λ_KL` too high | Confirm `λ_KL=1e-6`, not `1e-2` |
| All reconstructions flat (mean ≈ 0.5) | Decoder head zero-init never updated | Verify `requires_grad=True` on head; check gradients flow |
| Range channel sharp, intensity smeared | Validity mask not applied to intensity | Check `loss_intensity = (mask * ...)` |
| Output > 1.0 or NaN | Sigmoid skipped on output | Apply per-channel sigmoid in `decode()` head |
| KL=0, μ=0, σ=0 | Posterior collapse; KL weight too high relative to recon | Lower `λ_KL` (try 1e-7) or raise `λ_range`/`λ_intensity` |
| OOM at batch 2 | Gradient checkpoint off or fp16 not enabled | Wrap bottleneck ResBlocks in `torch.utils.checkpoint`; verify `autocast` active |
| Round-trip `pc → range → pc` loses many points | Normalization mismatch | Verify 100 m clamp and `/100` mapping match X-Drive defaults |
| BCE_validity climbing before L1_range explodes | Sigmoid saturation in decoder head | Switch to `binary_cross_entropy_with_logits` (no sigmoid in decode); label-smooth targets |
| Loss diverges around step ~1000 | Same as above; gradient noise at small batch | Path A in §11 (BCE-with-logits) or Path B (more data) |

---

## 13. Build plan (how M1 was assembled, in order)

| Step | What | File | Test |
|---|---|---|---|
| 1 | Channel normalization helpers | [`data/range_image.py`](../data/range_image.py) | Round-trip on random tensor |
| 2 | `LiDARVAE` skeleton (stubs for encode/decode/reparameterize/forward) | [`models/lidar_vae.py`](../models/lidar_vae.py) | Imports cleanly |
| 3 | Encoder forward | same | Shape assert `[B, 3, 32, 1024] → [B, 8, 8, 256]` |
| 4 | Decoder forward + zero-init head | same | Round trip on random; recon ≈ 0.5 at init |
| 5 | `reparameterize` + `forward` | same | `.train()` vs `.eval()` differ |
| 6 | Loss function | [`train/losses.py`](../train/losses.py) | Identity inputs → range/intensity L1 = 0 |
| 7 | nuScenes loader (LIDAR_TOP keyframes only) | [`data/nuscenes_mini.py`](../data/nuscenes_mini.py) | `next(iter(loader))` returns `[3, 32, 1024]` |
| 8 | Range-image converter (point cloud → range image) | [`data/range_image.py`](../data/range_image.py) | Round-trip `pc → range_img → pc'` near-identity |
| 9 | Trainer with `--overfit` / `--epochs` | [`train/train_vae.py`](../train/train_vae.py) | `--overfit 10 --steps 500` drops L1_range < 0.05 |
| 10 | Full epoch run | same | 50 epochs without OOM/NaN |
| 11 | Save + reload checkpoint | same | Reload reproduces decoder output |

---

## 14. Cross-references

| If you want to know… | Source doc |
|---|---|
| Why we picked this architecture | [`../../architecture.md`](../../architecture.md) §4 |
| The full project plan + scope decisions | [`../../min_pipeline_plan.md`](../../min_pipeline_plan.md) |
| Paper's hyperparameters (Sensor2Sensor) | [`../../hyperparameters.md`](../../hyperparameters.md) |
| Paper's loss equation | [`../../equations.md`](../../equations.md) |
| All milestone results (M-1 through M5) | [`../RESULTS.md`](../RESULTS.md) |
| The build steps with per-file deliverables | [`./models.md`](./models.md) §2 |
| Image VAE counterpart | [`./image_vae_choice.md`](./image_vae_choice.md), [`./pretrained_vae_integration.md`](./pretrained_vae_integration.md) |

This document is the synthesis — when those source docs change, this one should be regenerated or updated to reflect them.
