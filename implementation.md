# Sensor2Sensor Reproduction — Implementation Plan

A small-scale, learning-focused reproduction of the core ideas from *Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving*, scoped for limited compute (single NVIDIA 3060, 12GB VRAM).

---

## 1. Goal

Build an end-to-end pipeline that takes monocular camera input and generates LiDAR point cloud output, demonstrating the core technical contribution of Sensor2Sensor (conditional LiDAR diffusion from monocular camera). The project is a **proof-of-concept / learning exercise**, not a paper reproduction at the published scale.


### Explicit scope

- **In scope:** Architecture reproduction, end-to-end working pipeline, validation on public AV data

- Input: DashCam video sourced from the internet, etc. 
- Output: 4D gaussian Lidar of it. Perhaps even a real time video build based of the dashcam video real time rendering. 



- **Out of scope:** Matching paper's quantitative numbers, training at 100K-scene scale, full in-the-wild dashcam generalization

### Reframed project statement

> A small-scale reproduction of Sensor2Sensor's conditional LiDAR diffusion architecture, trained on public AV data, demonstrating the core technical approach. In-the-wild dashcam generalization is acknowledged as a known limitation requiring follow-up work.

---

## 2. Hardware & Compute Constraints

| Resource | Available | Paper used |
|---|---|---|
| GPU | 1× NVIDIA RTX 3060, 12GB | 128 TPUs |
| Training data | ~1000 scenes (nuScenes) | 100,000 scenes (Waymo internal) |
| Wall-clock budget | 4–6 weeks calendar time | Not specified |

### Forced design constraints

These are non-negotiable consequences of the 12GB GPU:

- Image resolution capped at ~256×256 (paper likely uses 1280×960)
- LiDAR range image at 32×512 or 32×1024 (matching nuScenes 32-beam LiDAR)
- Batch size 4–8 with gradient accumulation
- Mixed precision (fp16) mandatory
- Gradient checkpointing mandatory
- Model size capped at ~30–60M parameters (vs 250M in paper)
- Single-frame generation only (skip temporal/DAgger stages for v1)
- No multi-view image generation (LiDAR output only)

### Optional cloud spend

Renting an A100 on RunPod for one weekend (~$50) is reasonable if 4DGS augmentation becomes necessary in Phase 2. Skip for v1.

---

## 3. Dataset

### Primary: nuScenes

**Why:** Public, well-documented, has both camera and LiDAR, most accessible SDK.

**Files needed:**
- `samples/CAM_FRONT/*.jpg` — front camera (monocular input)
- `samples/LIDAR_TOP/*.pcd.bin` — LiDAR point clouds (training target)
- `v1.0-trainval/calibrated_sensor.json` — intrinsics/extrinsics
- `v1.0-trainval/sample_data.json` — synchronization metadata
- `v1.0-trainval/ego_pose.json` — vehicle pose

**Sizes:**
- nuScenes mini (10 scenes, ~4 GB) — for code development
- nuScenes trainval (1000 scenes, ~340 GB) — for actual training

### Optional secondary datasets

For multi-dataset training in Phase 2 (better generalization):

- **KITTI** — small, forward-facing camera closer to dashcam geometry
- **Waymo Open (mini)** — same sensor family as the paper

### For in-the-wild inference testing

- **Comma2k19** or YouTube driving clips — qualitative inference only, no ground truth

### Data access tooling

**Consider `py123d`** (123D framework) if planning multi-dataset training. It provides a unified API across nuScenes, Waymo, Argoverse 2, KITTI-360, PandaSet, and others. Skip if using nuScenes only — `nuscenes-devkit` is sufficient.

---

## 4. Architectural Simplifications vs Paper

| Paper component | Decision | Reason |
|---|---|---|
| 4DGS data pairing pipeline | **Skip entirely** | Public datasets give free paired data; 4DGS is too expensive |
| Multi-view image generation (8 cameras) | **Remove** | Memory constraint; LiDAR-only output |
| Cross-view attention | **Remove** | No multi-view output |
| Image VAE decoder | **Remove** | No image output |
| Cross-sensor attention (bidirectional) | **Simplify to one-way cross-attention** | Image conditions LiDAR only |
| Temporal conditioning (previous frame) | **Skip for v1** | Add in Phase 3 if needed |
| DAgger fine-tuning | **Skip entirely** | Requires temporal model first |
| LiDAR range-image representation | **Keep** | Core technical insight |
| LiDAR VAE (9-term loss) | **Keep** | Recipe transfers directly |
| Raymap conditioning | **Keep** | Standard, simple |
| Stable Diffusion VAE for images | **Keep, frozen** | Reuse pretrained, save weeks |

### Architecture diagram (simplified)

```
Dashcam image (256×256)
       ↓
[Frozen SD 1.5 VAE Encoder]
       ↓
Image latent + raymap conditioning
       ↓                ┌─ Noisy LiDAR latent (training: ground truth + noise;
       │                │                       inference: pure noise → denoise)
       ↓                ↓
┌──────────────────────────────────┐
│   Conditional Diffusion U-Net    │
│                                  │
│   Self-Attn (on LiDAR tokens)    │
│   Cross-Attn (LiDAR ← Image)     │
│   Conv + ResNet blocks           │
│   (~30-60M params)               │
└──────────────────────────────────┘
       ↓
Denoised LiDAR latent
       ↓
[Trained LiDAR VAE Decoder, frozen after Phase 1]
       ↓
LiDAR range image [32 × 1024 × 4]
       ↓
[Spherical unprojection → point cloud]
       ↓
Output: LiDAR point cloud
```

---

## 5. Hyperparameters

The Sensor2Sensor paper discloses limited hyperparameters. Use these starting values, derived from sensible diffusion training defaults and X-DRIVE (which is fully reproducible and trained on the same dataset).

### LiDAR VAE (Phase 1)

- Optimizer: AdamW
- Learning rate: 4e-4 (from X-DRIVE)
- Batch size: 8 effective (2 actual × 4 grad accum)
- Epochs: 50–100 (reduced from paper)
- Range image dimensions: 32 × 1024
- Latent dimensions: 8 × 256 × 8 (from X-DRIVE)
- Loss weights (start equal): λ_range = λ_intensity = λ_elongation = λ_BCE = λ_LPIPS_* = 1.0, λ_KL = 1e-6
- Range clamp: 150m (from paper)
- Circular convolutions for panoramic continuity

### Diffusion U-Net (Phase 2)

- Optimizer: 8-bit AdamW (via `bitsandbytes`)
- Learning rate: 1e-4
- Schedule: Cosine with 1000-step warmup
- Batch size: 4 effective (1 actual × 4 grad accum)
- Training steps: 30,000–50,000 (reduced from paper's 80K)
- Noise schedule: DDPM with v-prediction
- Inference steps: 25–50 (DDIM or DPM-Solver)
- EMA decay: 0.999
- Gradient clipping: 1.0
- Conditioning dropout: 0.2 (matches paper)
- Mixed precision: fp16
- Gradient checkpointing: enabled

### Memory optimizations (mandatory)

- `xformers` memory-efficient attention
- `bitsandbytes` 8-bit AdamW
- Pre-encode and cache image/LiDAR latents to disk (run frozen encoders once)
- fp16 training
- Gradient checkpointing on U-Net

---

## 6. Implementation Phases

### Phase 0: Baseline (1 week)

**Goal:** End-to-end pipeline using only pretrained models. No training required.

1. Install Depth Anything V2 (Apache 2.0)
2. Set up nuScenes mini loader
3. Pipeline: Image → metric depth → unproject to 3D → bin into LiDAR range image
4. Validation harness: Chamfer Distance, BEV visualization
5. Test on 50 nuScenes scenes and a few YouTube dashcam clips

**Outcome:** Working pipeline. Quantifies the baseline that any trained model must beat. May be "good enough" for some use cases.

**Why first:** Cheap insurance. Even if Phase 1+ fails, you have something working. The eval harness built here is reused throughout.

### Phase 1: LiDAR VAE (1–2 weeks)

**Goal:** Train a VAE that can compress and reconstruct nuScenes LiDAR range images.

1. Implement range image conversion (point cloud ↔ range image)
2. Implement VAE architecture (3 encoder + 3 decoder blocks, channels [32, 64, 128])
3. Implement 9-term loss (L1 × 3, BCE, LPIPS × 4, KL)
4. Train on nuScenes LiDAR_TOP for 50–100 epochs
5. Validate: reconstruction quality on held-out LiDAR

**Outcome:** Frozen VAE ready for Phase 2. This is a self-contained component you can debug independently.

### Phase 2: Conditional Diffusion (3–4 weeks)

**Goal:** Train the monocular-camera-to-LiDAR diffusion model.

1. Pre-encode all images (frozen SD VAE) and LiDAR (Phase 1 VAE) to latents, save to disk
2. Implement raymap encoder for camera conditioning
3. Implement diffusion U-Net with cross-attention to image features
4. Training loop: standard diffusion denoising loss in LiDAR latent space
5. Train for 30K–50K steps
6. Implement DDIM inference sampler

**Outcome:** Working monocular → LiDAR pipeline. End-to-end test on nuScenes validation set.

### Phase 3 (optional): Improvements

Only after Phase 2 produces reasonable results. Pick based on observed failure modes:

- **3a — Augmentation for generalization** (1 week): Add aggressive geometric/photometric augmentations to simulate dashcam variation
- **3b — Intrinsic conditioning** (3–5 days): Feed camera intrinsics as additional input so model can adapt to varied cameras
- **3c — Multi-dataset training** (2 weeks): Add KITTI + Waymo via py123d for cross-dataset diversity
- **3d — Temporal conditioning** (2 weeks): Add previous-frame conditioning for video coherence
- **3e — DAgger fine-tuning** (1 week): Reduce drift in long autoregressive rollouts

Each phase is independent. Skip what isn't needed.

---

## 7. Validation Strategy

### Quantitative (on nuScenes held-out scenes)

- **Chamfer Distance** — geometric similarity between generated and ground-truth point clouds
- **Range distribution** — histogram comparison of range values
- **Intensity distribution** — sanity check that intensity statistics look realistic
- **Downstream 3D detection mAP** — feed generated LiDAR into a pretrained CenterPoint detector and compare to mAP on real LiDAR

### Qualitative

- BEV (bird's-eye view) side-by-side plots of generated vs real point clouds
- 3D rendering with Open3D for visual inspection
- Inference on YouTube dashcam clips — no ground truth, just sanity check that output is plausible

### Comparison baselines

- Phase 0 baseline (Depth Anything V2 → LiDAR)
- Sensor2Sensor paper numbers (for context only — not directly comparable due to different dataset)
- X-DRIVE paper numbers (more comparable since both use nuScenes)

---

## 8. Open Source Tools

| Component | Tool | License |
|---|---|---|
| Monocular depth (Phase 0) | Depth Anything V2 | Apache 2.0 |
| Diffusion training | HuggingFace `diffusers` | Apache 2.0 |
| Image VAE (frozen) | Stable Diffusion 1.5 VAE | CreativeML Open RAIL-M |
| Dataset loading | `nuscenes-devkit` or `py123d` | Apache 2.0 |
| 3D detection validation | OpenPCDet or MMDetection3D | Apache 2.0 |
| Point cloud operations | Open3D, PyTorch3D | MIT / BSD |
| Memory optimization | `xformers`, `bitsandbytes` | BSD / MIT |
| Reference implementation | X-DRIVE on GitHub | Released by authors |

### Code to read before starting

- **X-DRIVE** (https://github.com/yichen928/X-Drive) — Full hyperparameters disclosed, code released, trained on nuScenes. Closest reproducible reference for the LiDAR diffusion architecture.
- **RangeLDM, LiDARGen** — Prior LiDAR diffusion work. Useful for the VAE design.
- **OmniRe / DriveStudio** — If Phase 3 4DGS work is ever attempted (not recommended on 3060).

---

## 9. Realistic Time Estimate

Assuming part-time work (10–20 hours/week):

| Phase | Calendar time | Cumulative |
|---|---|---|
| Phase 0 (baseline) | 1 week | 1 week |
| Phase 1 (LiDAR VAE) | 1–2 weeks | 2–3 weeks |
| Phase 2 (Diffusion) | 3–4 weeks | 5–7 weeks |
| Phase 3 (selected improvements) | 1–4 weeks | 6–11 weeks |

GPU-time only (24/7 training): roughly 2 weeks total. Calendar time is dominated by code, debug, iteration cycles.

---

## 10. Known Limitations & Honest Caveats

These should be stated explicitly in any writeup:

1. **Not a faithful reproduction.** Architecture is significantly simplified vs the paper (no multi-view output, no 4DGS, no temporal model).

2. **Domain gap for real dashcams.** Training uses nuScenes' AV-grade roof camera as "dashcam stand-in." Real YouTube dashcams have different intrinsics, mounting, and image statistics. Inference quality on real dashcams will be degraded.

3. **Scale gap.** Training on ~1000 scenes vs paper's 100,000. Expect lower quality across all metrics.

4. **No published hyperparameters in the paper.** Specifically, 9 LiDAR VAE loss weights, batch sizes, LR schedules, and noise schedule are not disclosed. Values used here are educated guesses based on X-DRIVE and standard practice.

5. **Single-frame, no temporal consistency.** Video outputs will flicker. Phase 3d addresses this if needed.

6. **No quantitative comparison to paper.** The paper uses a proprietary evaluation set (1000 Waymo bumper-camera sequences). Cannot directly compare numbers.

7. **3060 hardware bottleneck.** Even with all optimizations, training time per phase is days, not hours. Iteration is slow.

---

## 11. Decision Points

Pause and reassess after each phase:

- **After Phase 0:** Is the baseline good enough for the downstream use case? If yes, ship it.
- **After Phase 1:** Does the LiDAR VAE reconstruct cleanly? If no, fix loss weights / architecture before proceeding.
- **After Phase 2:** Are generated point clouds plausible? If Chamfer Distance is no better than Phase 0 baseline, something is wrong — debug before adding Phase 3 complexity.
- **Throughout:** Is the project still worth the time investment? Be willing to declare partial success and stop.

---

## 12. References

- **Sensor2Sensor** — *Cross-Embodiment Sensor Conversion for Autonomous Driving* (the paper being reproduced)
- **X-DRIVE** — *Cross-modality Consistent Multi-Sensor Data Synthesis for Driving Scenarios* (closest reproducible reference, code available)
- **123D** — *Unifying Multi-Modal Autonomous Driving Data at Scale* (optional dataset tooling)
- **RangeLDM, LiDARGen** — prior LiDAR diffusion work
- **OmniRe / DriveStudio** — 4DGS for driving scenes (only if Phase 3 4DGS attempted)
- **Depth Anything V2** — for Phase 0 baseline
- **MagicDrive, BEVGen** — multi-view image generation references
