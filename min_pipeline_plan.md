# Minimum Sensor2Sensor Test Pipeline — Implementation Plan

## Context

**Why this plan exists.** [implementation.md](/media/skr/storage/self_driving/sensor2sensor/implementation.md) describes a full small-scale reproduction (weeks of training, Phase 0 → Phase 3). The user wants to go one step earlier: get the *thinnest viable version* of the paper's core methodology — conditional monocular‑camera → LiDAR latent diffusion — running end-to-end on a single 12 GB RTX 3060, on nuScenes mini, so the architecture itself can be validated before any serious training budget is spent.

**Goal.** Within ~1–2 days of work produce a repo that:
1. Smoke-tests the whole pipeline on **one paired sample** (gradients flow, no shape errors).
2. Can **overfit ~10 paired samples** to prove the architecture can learn at all.
3. Runs **one full epoch over nuScenes mini** (~400 paired samples) to confirm the data loader, training loop, optimizer, and inference path all hold together.

Quantitative quality is **not** in scope. The deliverable is a known-good implementation that any later training run can plug into.

**Confirmed answers from the questionnaire** ([Questionaire/test_answered.md](/media/skr/storage/self_driving/sensor2sensor/Questionaire/test_answered.md)) and the follow-up clarification:
- Input: single AV front camera (`CAM_FRONT`) as a dashcam stand-in.
- Output: LiDAR range image → unprojected point cloud only. No multi-view image generation.
- Code split: borrow X-Drive's nuScenes loader + range-image conversion; write the model from scratch following the Sensor2Sensor paper, referencing X-Drive only where the paper is silent.
- Dataset: subset of nuScenes trainval, **already on disk locally** — see below.

## Local dataset

| Item | Value |
|---|---|
| Root path | `/media/skr/storage/self_driving/S2GO/data/nuscenes/` |
| Total size on disk | 56 GB |
| Version string for nuScenes-devkit | `v1.0-trainval` (the full split — `v1.0-mini` is **not** present, and we don't need it) |
| Scenes available | **850 / 850** trainval scenes |
| `samples/CAM_FRONT/` | 34,149 keyframes ✓ |
| `samples/LIDAR_TOP/` | 34,149 keyframes (synchronized to CAM_FRONT) ✓ |
| Other 5 cameras + radar | Present (unused by the minimum pipeline; available if the 6-cam variant is ever attempted) |
| Metadata under `v1.0-trainval/` | `calibrated_sensor.json`, `ego_pose.json`, `sample_data.json`, `sample.json`, `scene.json`, etc. — all present |
| `sweeps/` (non-keyframe LiDAR/cam) | Missing. **Fine** — we use keyframes only. |
| `maps/` | Present (unused by the minimum pipeline) |

**Subset definition.** Rather than downloading `v1.0-mini`, take 10 scenes from `v1.0-trainval/scene.json` sampled with a **fixed random seed** (`np.random.default_rng(0).choice(850, 10, replace=False)`) — taking the first 10 scenes biases the subset toward whatever ordering nuScenes happened to use (often clustered by location/time-of-day). Persist the chosen scene tokens to `s2s_min/scripts/select_subset.py` output so the subset is reproducible across runs. Promote to the full 850 scenes by changing one config line once M0–M4 pass.

**Required code change vs the original plan:** swap any reference to `v1.0-mini` for a hard-coded `dataroot='/media/skr/storage/self_driving/S2GO/data/nuscenes/'` + `version='v1.0-trainval'` + a `scene_indices=range(10)` filter. No download step.

---

## Architecture (minimum spec)

```
CAM_FRONT image (256×448) ──▶ [Frozen SD 1.5 VAE encoder] ──▶ image latent [C=4, H=32, W=56]
                                                                       │
camera intrinsics + extrinsics ─▶ [raymap builder @ 32×56 grid] ─▶ raymap [C=6, H=32, W=56] ─┐
                                                                       │                    │
                                                                       ▼                    ▼
                                                          concat on C → kv_full [C=10, 32, 56]
                                                                       │
                                                                       ▼ adaptive avg-pool
                                                          kv_context  [C=10, H=8, W=64]   ← pooled once, reused per block

LiDAR range image [C=4, H=32, W=1024] ─▶ [LiDAR VAE encoder, 4× spatial down] ─▶ z_lidar [C=8, H=8, W=256]
                                            │
        (training) z_lidar + noise ─────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │  Single LiDAR U-Net      │
              │  channels [96, 192, 384] │
              │  3 down / 3 up           │
              │  ResNet + GroupNorm      │
              │  Self-Attn on LiDAR      │
              │  Cross-Attn (LiDAR ← KV) │
              │  Circular conv on W      │  ← panoramic continuity
              │  ~25–35 M params         │
              │  gradient checkpoint ON  │
              └──────────────────────────┘
                          │
                          ▼
                ε̂ (predicted noise)

(inference) DDIM 25 steps ─▶ ẑ_lidar ─▶ [LiDAR VAE decoder] ─▶ range image ─▶ point cloud
```

**What is in:** LiDAR VAE, single conditional U-Net, cross-attention to image features, raymap, range-image LiDAR (4-channel), DDIM sampler, frozen SD 1.5 image VAE encoder.

**What is out (deferred from `implementation.md`):** multi-view image generation, image U-Net tower, cross-view attention, image VAE decoder, temporal/previous-frame conditioning, DAgger, autoregressive rollout, 4DGS data synthesis, real dashcam input.

Justified by [Questionaire/test_answered.md](/media/skr/storage/self_driving/sensor2sensor/Questionaire/test_answered.md) §"Summary: Your Ideal Pipeline Spec".

---

## U-Net details — initialization, training regime, freeze policy

### Source & initialization

| Aspect | Choice |
|---|---|
| Provenance | **Written from scratch.** No pretrained checkpoint loaded. |
| Pretrained alternatives considered | (a) SD 1.5 U-Net: rejected — expects 4-ch RGB latent at `64×64`, our LiDAR latent is 8-ch at `8×256`; adapting input/output convs breaks transferable features. (b) copilot4D world model: rejected — discrete-token MaskGIT, see [Optional extension](#optional-extension--reuse-copilot4ds-backbone-deferred-not-in-m0m5). |
| Weight init | Kaiming-normal on conv layers; zero-init on the final output conv and on AdaLN-Zero scale/shift projections (stabilizes early diffusion training). |
| Parameter count target | ~25–35 M. Knob is the channel multiplier in [`configs/min.yaml`](#repository-layout); start at `[96, 192, 384]` and adjust if VRAM is tight. |

### Detailed block stack

PyTorch tensor convention throughout: `[N, C, H, W]`. The LiDAR latent is `H=8 elevation rows × W=256 azimuth columns × C=8 channels` — **azimuth resolution is preserved**, only the H/W spatial dims of the range image are 4× downsampled by the VAE.

```
Input:  z_lidar_noisy [B, C=8,  H=8,  W=256]   ← noised LiDAR latent (from LiDAR VAE)
        t              [B]                       ← diffusion timestep
        kv_context     [B, C=10, H=8, W=64]     ← pre-pooled image-latent (4) + raymap (6), see below

Stem:    Conv2d(8 → 96, k=3, circular pad on W)

Encoder
  Level 0  : 2× ResBlock(96)   + SelfAttn + CrossAttn(KV=kv_context)   @ 8×256
             Downsample W only (stride 2 on W, keep H=8) → 8×128
  Level 1  : 2× ResBlock(192)  + SelfAttn + CrossAttn(KV=kv_context)   @ 8×128
             Downsample W only                                          → 8×64

Bottleneck: 2× ResBlock(384) + SelfAttn + CrossAttn(KV=kv_context)     @ 8×64

Decoder (mirror of encoder with skip-concat)
  Level 1  : Upsample W → 8×128, skip-concat, 2× ResBlock(192) + SelfAttn + CrossAttn
  Level 0  : Upsample W → 8×256, skip-concat, 2× ResBlock(96)  + SelfAttn + CrossAttn

Head:    GroupNorm → SiLU → Conv2d(96 → 8, k=3, circular pad)   ← zero-init

Output:  ε̂ [B, C=8, H=8, W=256]   ← predicted noise (v-prediction parametrization)
```

Notes:
- **Default downsampling is W-only.** The LiDAR latent enters the U-Net at H=8 already; self-attention sees all 8 elevation rows globally at every level, so the network has full elevation context without H pooling. Config knob `downsample_h_once: bool` in [`configs/min.yaml`](#repository-layout) enables an optional 8→4 H downsample at the stem, giving hierarchical elevation features at the cost of halving elevation resolution. Default off; flip on if M1/M3 evaluation shows ground/object confusion.
- Every conv that touches the W axis uses **circular padding on W, zero padding on H** — handcrafted wrapper, ~10 LOC.
- **Cross-attention KV is pre-pooled to a fixed `[8, 64]` grid outside the U-Net** and reused at every block. Without pooling, KV is `32×56 = 1792` tokens, Q at level 0 is `8×256 = 2048` tokens; the attention matrix dominates VRAM. Adaptive average pool `(32,56) → (8,64)` shrinks KV to 512 tokens (3.5× saving) at the cost of spatial precision in the image conditioning. The pool is applied **after** the image-latent + raymap concat so both are downsampled together.
- `K, V` projections are computed once from `kv_context` outside the U-Net (the same KV is reused at every block).
- Timestep `t` enters via sinusoidal embedding → 2-layer MLP → per-block **AdaLN-Zero**: the per-block linear that produces (scale, shift) modulation has its **output weights and bias both zero-initialised**, so each block starts as the identity function (matches DiT / Peebles 2023). No separate class/text conditioning.
- All ResBlocks avoid in-place ops so they remain safe under `torch.utils.checkpoint`.

### Training regime & freeze policy across milestones

| Milestone | U-Net status | LiDAR VAE status | SD 1.5 VAE encoder | LPIPS-VGG |
|---|---|---|---|---|
| M0 (smoke test) | Random init, **trainable**, one forward+backward step | Random init, trainable | Frozen | Not loaded |
| M1 (VAE train) | **Not loaded.** Not involved. | **Trainable** (the target of M1) | Frozen | Frozen |
| M2 (latent cache) | Not loaded | **Frozen** (eval-mode inference only) | Frozen (eval-mode inference) | Not loaded |
| M3 (diffusion train) | **Trainable + EMA tracked** (decay 0.999) | **Frozen** (cached latents) | Frozen (cached latents) | Not loaded |
| M4 (inference + viz) | **Frozen** — load EMA weights | **Frozen** — decoder only | Frozen — encoder only | Not loaded |

Key consequences:
- The **only** component that ever sees diffusion gradients is the U-Net (during M3). The LiDAR VAE was already frozen at the end of M1; cached latents in M2 mean the SD VAE encoder also doesn't run during M3 training steps.
- This gives a clean optimizer state: ~30 M params × (params + grad + 8-bit Adam state) ≈ ~0.5 GB optimizer state. Activations dominate VRAM, which is why gradient checkpointing is mandatory.
- EMA shadow weights are kept on CPU between optimizer steps to save VRAM; copied to GPU only for the periodic eval pass and the final M4 checkpoint.

### What we are explicitly *not* doing

- **No LoRA / QLoRA on the U-Net.** Full gradients on a from-scratch 30 M model are cheaper and simpler than LoRA on a randomly-initialized backbone. LoRA only becomes interesting if we later scale to the paper's 250 M and need to fit it on the 3060.
- **No pretrained U-Net warm start.** No SD checkpoint, no copilot4D checkpoint, no MagicDrive / BEVGen checkpoint. The architectural mismatches (input channels, periodic axis, latent shape) make warm-starting more trouble than it's worth at this scale.
- **No partial-freeze schedules** (e.g. freeze attention, train conv). Either fully trainable (M3) or fully frozen (M4). Simpler to reason about.

---

## Repository layout

Create a new sibling directory: `/media/skr/storage/self_driving/sensor2sensor/s2s_min/`.

```
s2s_min/
├── README.md
├── requirements.txt
├── configs/
│   └── min.yaml                  # all knobs (image size, LiDAR size, channels, lr, batch, steps)
├── data/
│   ├── nuscenes_mini.py          # thin wrapper over X-Drive's NuScenesDatasetM
│   └── range_image.py            # imports point_cloud_to_range_image from X-Drive
├── models/
│   ├── image_encoder.py          # frozen SD 1.5 VAE encoder
│   ├── raymap.py                 # build raymap from intrinsics+extrinsics
│   ├── lidar_vae.py              # encoder/decoder + 7-term loss (paper Eq.1, trimmed)
│   ├── attention.py              # self-attn + cross-attn blocks (circular-aware)
│   ├── unet.py                   # the single LiDAR U-Net
│   └── diffusion.py              # DDPM schedule, v-prediction, DDIM sampler
├── train/
│   ├── train_vae.py              # Phase A
│   ├── train_diffusion.py        # Phase B
│   └── smoke_test.py             # M0: one sample, one step
├── eval/
│   ├── decode_to_pointcloud.py
│   ├── bev_viz.py
│   └── chamfer.py
└── scripts/
    └── select_subset.py          # writes the 10-scene subset token list (no download)
```

### Borrow vs. write-from-scratch

| Concern | Source | Action |
|---|---|---|
| nuScenes loading + sync of CAM_FRONT / LIDAR_TOP | `nuscenes-devkit` directly, against local `/media/skr/storage/self_driving/S2GO/data/nuscenes/` (`v1.0-trainval`) | **Write thin loader from scratch** (~50 LOC). X-Drive's `nuscenes_dataset.py` depends on mmdet3d + pre-generated `.pkl` info files, which is not worth the install pain for a 10-scene subset. Use `NuScenes(version='v1.0-trainval', dataroot=...)` and walk `sample.json` to pair `CAM_FRONT` ↔ `LIDAR_TOP` keyframes. |
| Range-image conversion | `Reference_code/X-Drive/xdrive/dataset/pipeline.py` line 830 `point_cloud_to_range_image` | **Borrow** the class directly. **Override** X-Drive's normalization (`mean=[50,0], std=[50,255]` per [`Nuscenes_lidar_rangeldm.yaml:17`](Reference_code/X-Drive/configs/dataset/Nuscenes_lidar_rangeldm.yaml#L17)) — use the paper's linear-to-[0,1] mapping with range clamp 100 m, intensity & elongation clipped to their natural ranges. Verify with a round-trip test: `range_img → point_cloud → range_img` should be near-identity on a held-out sample. |
| LiDAR VAE | X-Drive `xdrive/networks/blocks_pc_RangeLDM.py` is RangeLDM-style | **Write from scratch** following paper Eq. (1)–(4)+(6)+(7), but verify channel-count choices against X-Drive's `SDv2.1pc_RangeLDM_box.yaml`. |
| U-Net + cross-attn + circular conv | X-Drive `xdrive/networks/unet_pc_condition_RangeLDM.py`, `circular_modules.py` | **Write from scratch** (smaller). Reference X-Drive's `circular_modules.py` for the circular-padding pattern on the azimuth axis. |
| Diffusion training loop, DDIM sampler | none of X-Drive's runners are clean enough | **Write from scratch** using `diffusers.schedulers.DDPMScheduler` + `DDIMScheduler` (HuggingFace `diffusers`). |
| Raymap | not in X-Drive | **Write from scratch**, ~30 LOC. Origin + direction per pixel, normalized to camera frame, concatenated to image latent on channel axis (paper Sec. 3.2.4). **Compute the raymap directly at the latent grid** (`32×56`) using intrinsics scaled by the SD-VAE downsample factor (`K' = diag(1/8, 1/8, 1) @ K`). Do **not** naively `F.interpolate` a full-res raymap — bilinearly resampling unit ray directions yields non-unit vectors and corrupts the geometric conditioning. |
| SD 1.5 VAE | HuggingFace `diffusers.AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")` | **Reuse.** Freeze. Encoder only. |

The deliberate split keeps the parts that are tedious-but-solved (mmdet3d-style data pipeline) borrowed, and the parts that are the **actual paper contribution** (cross-sensor conditioning, LiDAR-VAE objective, circular range-image U-Net) written by hand so the user actually exercises the methodology.

---

## Milestones

### M-1 — Tensor-shape sanity check (~30 minutes, before M0)

Before instantiating any real weights or loading any data, write `tests/test_shapes.py` that:
1. Builds a single `CrossAttention(query_dim=384, kv_dim=10, heads=8)` block.
2. Pushes a random `[B=2, Q=8*64, 384]` query and a random `[B=2, KV=8*64, 10]` context through it.
3. Asserts output shape matches input query shape.
4. Repeats for one full encoder level (ResBlock + SelfAttn + CrossAttn + Downsample) with the planned channel counts.

Cheaper to find a shape bug here than after a full smoke-test stack is assembled. Five-minute insurance.

### M0 — Smoke test (half a day)

Goal: `train/smoke_test.py` runs without error on one sample.

1. Stub all modules with the right input/output shapes; weights random-init.
2. Load one nuScenes-mini sample → image tensor `[1,3,256,448]` + lidar range image `[1,4,32,1024]`.
3. Encode image with frozen SD VAE → `[1,4,32,56]`. Build raymap **directly at the 32×56 grid** using scaled intrinsics → `[1,6,32,56]`. Concat → `[1,10,32,56]`. **Adaptive-pool to `[1,10,8,64]`** as `kv_context`.
4. Encode LiDAR through (random-init) LiDAR VAE → `[1,8,8,256]`.
5. Add noise per a `DDPMScheduler` for `t=500`. Run U-Net forward (Q=LiDAR latent, KV=pooled image+raymap). Compute MSE on predicted noise. Backward. Adam step.
6. **Pass criterion:** finite, non-NaN loss; no shape errors; <8 GB peak VRAM.

### M1 — Phase A: LiDAR VAE training (1 day)

Goal: VAE reconstructs nuScenes LiDAR cleanly enough to be a useful target space.

- **Initial loss (5 terms): `L1_range + L1_intensity + L1_elongation + BCE_validity + KL`.** LPIPS terms are deferred — computing LPIPS-on-normals requires finite-difference normal estimation from the predicted range channel, which is fiddly around invalid returns, and LPIPS-VGG adds ~250 MB VRAM and ~30 % wall-clock per VAE step. Add `LPIPS_normals` and `LPIPS_intensity` back **only if** the 5-term VAE produces visibly blurry reconstructions in M1's eyeball-BEV check.
- Equations: paper (1), (3), (4), (7) for the 5-term variant — see [equations.md](/media/skr/storage/self_driving/sensor2sensor/equations.md). Add (6) only when LPIPS terms are re-introduced.
- Loss weights start at `1.0` except `λ_KL = 1e-6` (X-Drive value).
- Range clamp 100 m (nuScenes; smaller than the paper's 150 m for Waymo); intensity & elongation linearly mapped to [0,1].
- **Overfit milestone:** 10 samples to <0.05 mean L1 on range within ~500 steps. Save the checkpoint.
- **Epoch milestone:** one full pass over the ~400 mini samples. Save final checkpoint, freeze.
- **Pass criterion:** decoded range image looks like the input range image (eyeball BEV).

### M2 — Latent caching (~1 hour)

For the diffusion phase, pre-encode all nuScenes-mini images (SD VAE) and all LiDAR range images (Phase-A VAE) to `.npz` on disk. Saves ~5× VRAM and ~3× wall-clock vs. encoding in the inner training loop. This is the single most important memory optimization for a 3060.

### M3 — Phase B: conditional diffusion (1–2 days)

Goal: `train/train_diffusion.py` overfits 10 cached samples, then completes 1 epoch over the full mini set without instability.

- Objective: MSE on predicted noise, v-prediction parametrization (better stability than ε at small batch).
- Optimizer: 8-bit AdamW (`bitsandbytes`), lr `1e-4`, cosine schedule, 200-step warmup.
- Batch: 1 actual × 4 grad-accum = effective 4.
- Mixed precision (fp16) + gradient checkpointing on U-Net.
- EMA decay 0.999 on U-Net weights (paper).
- Gradient clip 1.0 (paper).
- Conditioning dropout 0.2 on the image-side KV (paper Sec. B.2).
- **Overfit milestone:** loss drops monotonically over ~1k steps on 10 samples.
- **Epoch milestone:** 1 epoch (~400 batches × 1 grad-accum step = ~100 optimizer steps).
- **Pass criterion:** loss finite; DDIM 25-step inference produces a non-trivial range image when fed a held-out image.

### M4 — End-to-end inference + visualization (~2 hours)

`eval/decode_to_pointcloud.py`:
1. Pick a held-out nuScenes-mini frame.
2. Encode image → KV context.
3. DDIM 25-step sample → ẑ_lidar.
4. LiDAR VAE decode → range image → spherical unproject → `(N,4)` point cloud.
5. `eval/bev_viz.py` renders side-by-side BEV of generated vs ground-truth LiDAR.
6. `eval/chamfer.py` computes Chamfer Distance against ground truth.

**Pass criterion:** the generated BEV looks geometrically plausible (road plane present, vehicle returns roughly where the input camera sees them). Chamfer is whatever it is — quantitative quality is out of scope.

### M5 — Document failure modes (~1 hour)

Append a short `s2s_min/RESULTS.md` containing:
- VRAM peaks per phase.
- Wall-clock per milestone.
- What the M4 output actually looks like.
- A **deviations-from-paper** table (template):

| Paper | Minimum pipeline | Reason |
|---|---|---|
| Two U-Net towers + bidirectional cross-sensor attn | Single LiDAR U-Net + one-way cross-attn from LiDAR to image | LiDAR-only output; halves params |
| 8-view multi-view image generation | None | LiDAR-only output |
| Dashcam as 9th view | CAM_FRONT as input, no 9th view | No paired dashcam data without 4DGS |
| Auto-regressive + DAgger | Single frame | No temporal scope in minimum pipeline |
| LiDAR VAE 9-term loss | 5 terms (drop LPIPS×4) initially | Simplicity; revisit if blurry |
| Full-resolution image KV in cross-attn | Pooled `(32,56)→(8,64)` KV | VRAM bound |
| Range clamp 150 m (Waymo) | 100 m (nuScenes 32-beam) | Sensor max range |
| ~250 M params, 128 TPU | ~30 M params, 1× RTX 3060 | Hardware reality |

This is what gets read before any real training run is launched.

---

## Optional extension — reuse copilot4D's backbone (deferred, not in M0–M5)

If at any point after M3 the hand-written U-Net needs richer hierarchical attention or temporal modeling, the existing [skr3178/copilot4D](https://github.com/skr3178/copilot4D) implementation already provides drop-in building blocks. These are explicitly **not** part of the minimum path and should only be considered after the paper-faithful pipeline is verified end-to-end.

### What to potentially borrow

| copilot4D module | File | Reuse for |
|---|---|---|
| `SpatioTemporalBlock` | [`copilot4d/world_model/spatio_temporal_block.py`](https://github.com/skr3178/copilot4D/blob/main/copilot4d/world_model/spatio_temporal_block.py) | Swin-window spatial self-attention (set `T=1` to collapse the temporal axis) **or** future temporal conditioning if M3 is later extended to multi-frame |
| `WorldModelPatchMerging` | [`copilot4d/world_model/patch_merging.py`](https://github.com/skr3178/copilot4D/blob/main/copilot4d/world_model/patch_merging.py) | Encoder downsampling between U-Net levels |
| `LevelMerging` | [`copilot4d/world_model/level_merging.py`](https://github.com/skr3178/copilot4D/blob/main/copilot4d/world_model/level_merging.py) | Decoder upsampling + skip-connection wiring |
| U-Net topology | [`copilot4d/world_model/world_model.py`](https://github.com/skr3178/copilot4D/blob/main/copilot4d/world_model/world_model.py) | Reference for three-level (128² → 64² → 32², dims 256/384/512) hierarchy and skip-connection structure |

### Required modifications before drop-in

These blocks were written for **discrete-token MaskGIT** on a **non-periodic BEV grid with a temporal axis**, not continuous Gaussian diffusion on a periodic range image. Reusing them implies:

1. **Set `T=1`** in `SpatioTemporalBlock` so the temporal attention collapses and the block behaves as pure spatial Swin. The temporal axis is out of scope for the minimum pipeline.
2. **Replace input embedding** — copilot4D embeds discrete token IDs via `nn.Embedding`. For Sensor2Sensor, use `Conv2d(in=lidar_latent_ch + raymap_ch, out=dim_0)` on continuous latents.
3. **Replace output head** — swap `LayerNorm → tied Linear → 1025-class logits` for `LayerNorm → Conv2d(dim_0 → lidar_latent_ch)` to predict continuous ε (or v).
4. **Add timestep conditioning** — copilot4D has none of the continuous-DDPM machinery. Inject sinusoidal-timestep embedding via AdaLN-Zero (shift/scale per block).
5. **Add image cross-attention** — copilot4D conditions only on a 16-d action vector. Add a `CrossAttention(Q=lidar_tokens, K=V=image_VAE_latent + raymap)` after each Swin self-attn.
6. **Handle azimuth periodicity** — Swin window partitioning assumes both spatial axes are non-periodic. The W axis of the 32×1024 range image wraps at 0°/360°. Either pad cyclically on W before partitioning windows, or accept the seam artifact at the boundary.

### When to actually do this

- **Skip entirely** if the from-scratch U-Net in M3 trains stably and the M4 BEV output is plausible.
- **Consider** if (a) the from-scratch U-Net hits a wall on capacity per VRAM (Swin's window attention is more memory-efficient than dense self-attention at higher resolution), or (b) the project later expands to multi-frame conditioning, at which point copilot4D's temporal axis becomes a real advantage instead of dead weight.

---

## Critical references

| File | Why |
|---|---|
| [Sensor2Sensor.pdf](/media/skr/storage/self_driving/sensor2sensor/Sensor2Sensor.pdf), §3 + §B (supplemental) | Architecture spec |
| [equations.md](/media/skr/storage/self_driving/sensor2sensor/equations.md) | Loss formulas |
| [hyperparameters.md](/media/skr/storage/self_driving/sensor2sensor/hyperparameters.md) | Optimizer/schedule defaults |
| [Questionaire/test_answered.md](/media/skr/storage/self_driving/sensor2sensor/Questionaire/test_answered.md) | All scoping decisions |
| `Reference_code/X-Drive/xdrive/dataset/nuscenes_dataset.py` | Borrow as data loader |
| `Reference_code/X-Drive/xdrive/dataset/pipeline.py:830` (`point_cloud_to_range_image`) | Borrow as range-image converter |
| `Reference_code/X-Drive/xdrive/networks/circular_modules.py` | Reference for circular conv on azimuth |
| `Reference_code/X-Drive/configs/dataset/Nuscenes_lidar_rangeldm.yaml` | Reference for sensible nuScenes LiDAR settings (32 × 1024, 100 m range) |

---

## Verification (end-to-end)

Each milestone has a single concrete pass criterion (above). The full pipeline is considered verified when:

```bash
# 1. Plumbing
python -m s2s_min.train.smoke_test                # M0 passes

# 2. VAE
python -m s2s_min.train.train_vae --overfit 10    # M1 overfit
python -m s2s_min.train.train_vae --epochs 1      # M1 epoch

# 3. Diffusion
python -m s2s_min.train.train_diffusion --overfit 10 --vae_ckpt out/vae.pt
python -m s2s_min.train.train_diffusion --epochs 1 --vae_ckpt out/vae.pt

# 4. Inference + viz
python -m s2s_min.eval.decode_to_pointcloud --ckpt out/diffusion.pt --sample 0
```

All four invocations complete on the 3060 in <12 GB VRAM and produce a BEV plot.

---

## Open risks / things to watch

1. **X-Drive's mmdet3d dependency chain.** `nuscenes_dataset.py` needs `mmdet3d` + `mmcv-full` **and** pre-generated `nuscenes_infos_{train,val}.pkl` files (referenced in [`Reference_code/X-Drive/configs/dataset/Nuscenes_lidar_rangeldm.yaml:196`](Reference_code/X-Drive/configs/dataset/Nuscenes_lidar_rangeldm.yaml#L196)). Since the raw data is already at `/media/skr/storage/self_driving/S2GO/data/nuscenes/`, the cheaper path is to skip X-Drive's loader entirely and write a ~50-LOC `nuscenes-devkit`-based loader inside `s2s_min/data/nuscenes_mini.py`. Borrow only X-Drive's `point_cloud_to_range_image` class from [`pipeline.py:830`](Reference_code/X-Drive/xdrive/dataset/pipeline.py#L830) for the range-image conversion.
2. **SD 1.5 VAE expects 512×512-ish inputs.** Encoding 256×448 works but spatial latent becomes 32×56 — fine, just verify with a forward pass during M0.
3. **Circular convolution on the azimuth axis.** Standard `nn.Conv2d` zero-pads, which corrupts the 0°↔360° seam. The U-Net must wrap on the W axis only (height = elevation isn't periodic). Pattern is in X-Drive's `circular_modules.py`.
4. **Range-image LiDAR is forgiving but lossy.** Spherical unprojection from a 32×1024 grid won't recover 32-beam nuScenes returns exactly — that is expected and noted in [implementation.md](/media/skr/storage/self_driving/sensor2sensor/implementation.md) §10.
5. **nuScenes mini has only 10 scenes.** Diversity is awful; the model **will** overfit and **won't** generalize. That's acceptable because the goal is pipeline validation, not paper-quality results.
6. **Cross-attn KV pooling loses image spatial precision.** Adaptive-pooling `(32,56) → (8,64)` saves ~3.5× attention VRAM but blurs the conditioning signal at the level of small distant objects. If M3 output shows that conditioning is ignored (LiDAR drifts independent of input image), revisit: try `(16,32)` KV grid, or switch to multi-scale image features (one KV grid per U-Net level).
7. **In-place ops break `torch.utils.checkpoint`.** When writing the ResBlock + AdaLN-Zero modulation, avoid `tensor.add_()`, `F.silu(x, inplace=True)`, `nn.ReLU(inplace=True)`, etc. Each checkpointed forward must re-run cleanly with the same inputs.
