# Sensor2Sensor — Project Questions & Answers

## Pipeline Testing & Architecture Simplification

### Q: Can I test the pipeline with one single image and dashcam video coupled?

**Yes.** A single paired sample `(dashcam frame, AV LiDAR scan)` is enough for an end-to-end forward pass test. This is the right way to debug: get one sample flowing through the entire pipeline before scaling to a real dataset. You'll catch shape mismatches, device errors, and data format bugs immediately.

For a proper *training* run, you need many more pairs (hundreds at minimum), but a single pair is perfect for pipeline validation.

---

### Q: Generated using the 4DGS?

**Not needed for testing.** Skip 4DGS entirely. Use a real paired sample from nuScenes:
- Input: `CAM_FRONT` image (treated as "dashcam")
- Target: `LIDAR_TOP` point cloud

The 4DGS pipeline exists in the paper only to synthesize *training-time* dashcam-style inputs from AV data. For your prototype, real AV-camera data substitutes fine.

---

### Q: I can eliminate the Image VAE decoder. Can I also eliminate the LiDAR VAE decoder?

**No — you need the LiDAR VAE decoder.** Here's why:

- The **Image VAE decoder** can be removed because you're not generating images. The image VAE *encoder* is still needed at the input side (to turn the dashcam into a latent for conditioning), but the decoder isn't.
- The **LiDAR VAE decoder** is required at inference time. The diffusion model produces a LiDAR *latent*, not a usable point cloud. The decoder converts that latent into the 4-channel range image, which you then project into a 3D point cloud.

```
Dashcam → [Image VAE encoder] → image latent (conditioning) → KEEP encoder
                                                              REMOVE decoder

Diffusion U-Net → LiDAR latent → [LiDAR VAE decoder] → range image → KEEP decoder
                                                       → point cloud
```

Without the LiDAR VAE decoder, the U-Net output is meaningless — it's just a tensor in latent space.

---

### Q: Do we need 2 diffusion U-Net models? If we aren't using the image decoder, or does the cross-attn model need this?

**You need only 1 U-Net.** The paper uses two parallel "towers" because it generates both modalities simultaneously. Since you're only generating LiDAR, you need:

- **One U-Net** operating on the LiDAR latent
- **No image-side U-Net** (because you're not denoising image latents)
- **Cross-attention from LiDAR queries to image features** (one-way conditioning, replacing the paper's bidirectional cross-sensor attention)

The cross-attention doesn't require a parallel U-Net. It just needs image *features* as keys/values, which you get from running the dashcam through the frozen Image VAE encoder once.

```
Paper:           Your simplified version:
                 
[Image U-Net]    [Frozen Image VAE encoder]
     ↕ cross-           ↓ (features used as keys/values)
[LiDAR U-Net] →  [LiDAR U-Net with cross-attn to image features]
```

This is a major simplification — roughly halves your parameter count and compute.

---

### Q: Can I recreate a simpler smaller architecture which emulates this and fits in 12GB?

**Yes, this is exactly the right approach.** Target architecture for a 3060:

| Component | Size |
|---|---|
| Image VAE (frozen, SD 1.5) | 84M params, but no gradients needed |
| LiDAR VAE (trained from scratch) | ~10M params |
| Diffusion U-Net | ~30-50M params (vs paper's 250M) |
| **Trainable total** | **~40-60M params** |

Reductions vs paper:
- Single LiDAR tower (no image generation tower) → -60% params
- Smaller U-Net channels: [128, 256, 512] instead of larger
- Lower input resolution: 256×256 image, 32×512 LiDAR range image
- Fewer attention heads, smaller embedding dim
- Single-frame (no temporal conditioning) → -20% memory

Combined with fp16, gradient checkpointing, and gradient accumulation, this fits in 12GB.

---

### Q: Can I create a pipeline and test it with a single pass of sensor value + dashcam for training?

**Yes — and you should.** This is exactly the right development workflow:

```python
# Pseudocode for sanity-check training step
sample = dataset[0]  # one paired (dashcam, lidar) sample
loss = model(sample)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item()}")  # Should be a finite, non-NaN number
```

If this works without crashing or producing NaN, your pipeline is structurally correct. Then move to a batch of 4, then a full epoch, then full training.

---

### Q: Which 3DGS variant for 4DGS pipeline supporting dynamic rigid + deformable objects?

**Open-source options ranked:**

| System | Rigid dynamic | Deformable | Notes |
|---|---|---|---|
| **OmniRe / DriveStudio** | ✓ | ✓ (SMPL pedestrians) | Best fit; ICLR 2025 Spotlight |
| **Street Gaussians** | ✓ | ✗ | Simpler, vehicles only |
| **DrivingGaussian** | ✓ | Limited | Older, less polished |
| **PAGS** | ✓ | ? | Cited by Sensor2Sensor itself |

**For your project: skip 4DGS entirely.** None of these run on a 3060 for non-trivial driving scenes. If you absolutely need it, rent an A100 on RunPod for a weekend (~$50) and reconstruct 10-20 nuScenes scenes with OmniRe.

---

### Q: Which loss terms to consider for LiDAR-only case?

**Two separate loss sets:**

**For the LiDAR VAE (Phase 1 training):** Keep most of the paper's 9 terms, but you can simplify:

```
Essential (7 terms):
- L1_range
- L1_intensity  
- L1_elongation
- BCE_validity
- LPIPS_normals      ← most important LPIPS term (geometry)
- LPIPS_intensity    ← preserves appearance
- KL                 ← latent regularization

Optional (can drop initially):
- LPIPS_elongation
- LPIPS_validity
```

**For the diffusion model (Phase 2 training):** Standard diffusion denoising loss only:

```
L = MSE(predicted_noise, actual_noise)
```

That's it. The diffusion loss is much simpler than the VAE loss. All the perceptual quality work happens in the VAE.

---

### Q: U-Net architecture (default 2x). Can we reduce to 1x since we don't need video and LiDAR-only?

**Yes.** Since you're:
- LiDAR-only (drop image tower)
- Single-frame (drop previous-frame conditioning)
- No autoregressive video (drop DAgger)

You need exactly **one U-Net**. The "2x" in the paper refers to the dual-tower design (image tower + LiDAR tower running in parallel). Your simplified pipeline has just the LiDAR tower.

Additionally, you can reduce within that single U-Net:
- Fewer downsampling blocks (3 instead of 4)
- Smaller channel multipliers
- Fewer attention layers (only at middle resolutions, not at every level)

---

### Q: Say we have 4DGS-generated coupled dashcam + AV logs. What model params next?

If you have paired data ready, the model spec becomes:

```
Image VAE:
  - Pretrained Stable Diffusion 1.5 VAE
  - Frozen, used only as encoder
  - Output latent: 4 channels at 1/8 spatial resolution

LiDAR VAE (train first, Phase 1):
  - Encoder: 3 blocks, channels [32, 64, 128]
  - Decoder: 3 blocks, channels [128, 64, 32]
  - Latent: 8 channels at 1/4 spatial resolution
  - Range image input: 32 × 512 × 4
  
Diffusion U-Net (train second, Phase 2):
  - Input channels: 8 (LiDAR latent) + raymap channels
  - Cross-attention dim: matches image VAE latent dim (4 × spatial × spatial)
  - Channels: [128, 256, 512]
  - Attention at middle 2 blocks
  - ~30-40M params total
  
Training:
  - AdamW (8-bit via bitsandbytes), lr=1e-4
  - Batch 4 with grad accumulation
  - fp16 + gradient checkpointing
  - 30-50k steps
```

---

### Q: Explore alternative more efficient algorithms

**Strong alternatives, ranked by relevance to your goal:**

| Paper | Why consider | Code? |
|---|---|---|
| **Veila** | First diffusion model for monocular RGB → panoramic LiDAR. Most directly aligned with your goal. | Check arXiv 2508.03690 |
| **LiDAR-Diffusion** (CVPR 2024) | Range-view LiDAR diffusion with pretrained checkpoints. Strong foundation. | github.com/hancyran/LiDAR-Diffusion |
| **RangeLDM** | Fast LiDAR generation via latent diffusion. Foundational. | Available |
| **R3DPA** (Valeo) | Leverages RGB pretrained priors for LiDAR generation. Pretrained models available. | github.com/valeoai/R3DPA |
| **X-DRIVE** | Multi-modal but fully reproducible, nuScenes-trained, code released. | github.com/yichen928/X-Drive |

**My recommendation: Start with LiDAR-Diffusion as the unconditional baseline, then add monocular image conditioning following Veila's approach.** This is more tractable than reproducing Sensor2Sensor.

---

### Q: Say I don't want to add dashcam video, but first test pipeline with 8 AV cameras + LiDAR logs. What changes?

This is **essentially what X-DRIVE does** in its zero-shot "Camera-to-LiDAR" mode. Concrete changes:

1. **Input becomes 6 or 8 surround-view cameras** instead of one dashcam
2. **Add cross-view attention on the input side** to fuse the 6/8 input views
3. **No need for raymap with dashcam-specific intrinsics** — use known AV camera calibration
4. **No domain gap concerns** — input and output are from the same sensor rig

Pipeline:
```
6 AV cameras → [Image VAE encoders, shared frozen] → 6 image latents
                                                          ↓
                                                   [Cross-view fusion]
                                                          ↓
                                                   Fused image conditioning
                                                          ↓
                                          [Diffusion U-Net with cross-attn]
                                                          ↓
                                                   LiDAR latent
                                                          ↓
                                                   [LiDAR VAE decoder]
                                                          ↓
                                                   LiDAR point cloud
```

**This is significantly easier than monocular conditioning** and is a great Phase 0 milestone before attempting the monocular case.

---

## Video Generation

> No autoregressive video generation. DAgger algorithm mentioned in the paper does this.

**Confirmed correct interpretation.** DAgger fine-tuning in the paper exists specifically to make autoregressive video rollouts stable. Since you're not doing video, you skip:
- Previous-frame conditioning
- DAgger rollout data generation
- DAgger fine-tuning stage

This eliminates training Stages 2, 3, and 4 from the paper. You only need to run Stage 1 (single-frame training).

---

## Architectural Concepts

### Q: Cross-view Attn — replacing 2D attention modules with 3D (1D cross views + 2D in spatial)

**What it does:** In standard Latent Diffusion, attention operates within a single image's spatial grid (each pixel attends to other pixels in the same image — that's "2D" attention because it's over a 2D spatial grid).

For multi-view generation, you have V views, each with H×W spatial dimensions. You need pixels in view 1 to be able to attend to pixels in view 2, 3, ..., V — otherwise each view would generate independently and they wouldn't be consistent with each other.

**The fix:** Flatten all views together, run attention across the entire `(V × H × W)` sequence, then split back.

```
Standard 2D attention:        New 3D attention:
[H × W × C]                   [V × H × W × C]
flatten spatial dims          flatten ALL dims
attend over H·W tokens        attend over V·H·W tokens
reshape back                  reshape back
```

The naming "1D cross views + 2D in spatial" is just describing the dimensions being attended over: 1 dimension across views (V) + 2 dimensions across spatial (H, W) = 3D total.

**For your LiDAR-only pipeline:** You don't need cross-view attention if you only have one LiDAR output (no multi-view generation). But if you use the 6-AV-camera input variant (above), you'd need cross-view attention to fuse the 6 input cameras.

---

### Q: Cross-sensor Attn — generate consistent images and LiDAR within each block of U-Net

**What it does:** This is the mechanism that makes generated LiDAR spatially align with generated images. Without it, the LiDAR might show a car at position X while the images show it at position Y.

**How it works:** Image latents and LiDAR latent are flattened into a combined token sequence, self-attention runs over the combined sequence, then they're split back to their separate towers.

```
Image latent + LiDAR latent → flatten + concat → self-attention → split → back to towers
```

This happens at every U-Net block, so information exchange is continuous through the network.

### Q: Can we replace cross-view with cross-sensor or vice versa?

**No, they serve different purposes:**

- **Cross-view attention:** Within a single modality (images), across multiple views. Job: make the 8 generated camera views consistent with each other.
- **Cross-sensor attention:** Across modalities (images ↔ LiDAR). Job: make generated LiDAR consistent with generated images.

They're orthogonal — you need both for the full paper's pipeline. **For your LiDAR-only pipeline, you only need cross-sensor attention** (simplified to one-way: image → LiDAR). Cross-view attention is irrelevant since you have only one output modality.

### Q: Is cross-sensor attention what causes the LiDAR to be consistent 360-degree rebuilding?

**Partly yes, but not the whole story.** The 360° LiDAR reconstruction comes from two sources:

1. **The LiDAR VAE itself** — it learned to encode/decode complete 360° spin images during Phase 1 training
2. **The diffusion model** — learned that "given any input image, the output LiDAR should be a complete 360° spin image"
3. **Cross-sensor attention** — ensures the LiDAR's 360° output is *consistent* with what the input image shows (in the front-facing portion)

The 360° generation happens because the model was trained to output complete LiDAR scans, even though the input only shows the front view. The model essentially **hallucinates** what's behind and to the sides, conditioned on the limited frontal information. The cross-sensor attention makes the front-facing portion accurate; the back-facing portion is plausible but not verifiable.

This is why **in-the-wild generalization is hard** — the model is making educated guesses about unseen regions based on training distribution, not actually reconstructing them.

---

## Dataset for Testing

> 3 sec long paired Fixed camera to AV log sequences
> Training steps: Step 1: 80k, Step 2: 40k, Step 4: 20k

**Note:** These are the paper's training schedule numbers. For your 3060 reproduction:

| Stage | Paper | Your version |
|---|---|---|
| LiDAR VAE | (unspecified) | 50-100 epochs on nuScenes |
| Stage 1 (single-frame) | 80k steps | 30-50k steps |
| Stage 2 (temporal) | 40k steps | SKIP — no temporal |
| Stage 3 (DAgger gen) | — | SKIP |
| Stage 4 (DAgger fine-tune) | 20k steps | SKIP — no autoregressive |

Your pipeline reduces to **two training phases: VAE training + single-frame diffusion**. Total time on a 3060: ~2-3 weeks of GPU time.

**For data:** The paper's "Fixed-Camera-to-AV" 3-second sequences are proprietary Waymo data. Substitute with **nuScenes samples** (keyframes at 2Hz, no need for 3-second sequences since you're not doing video).

---

## Why Won't 250M Params Fit on a 12GB GPU?

**The model parameters themselves do fit, but training requires much more memory than just weights.** Here's the breakdown:

For a 250M-parameter model in fp32:

```
Model weights:                  250M × 4 bytes  = 1.0 GB
AdamW optimizer state:          250M × 8 bytes  = 2.0 GB  (momentum + variance)
Gradients:                      250M × 4 bytes  = 1.0 GB
                                                  ━━━━━━━
Fixed cost just for training:                   = 4.0 GB

Activations (depends on batch size + resolution):
  - For batch 1 at low res:                    ~3-4 GB
  - For batch 4 at moderate res:                ~6-8 GB
  - For batch 8 at paper's res:                ~15-20 GB

Total: well over 12 GB for any meaningful batch size
```

**Mitigations to fit 250M on 12GB (theoretically possible but painful):**

| Technique | Memory savings |
|---|---|
| fp16 mixed precision | ~50% on weights, gradients, activations |
| Gradient checkpointing | ~40% on activations |
| 8-bit AdamW (bitsandbytes) | ~75% on optimizer state |
| Gradient accumulation | Allows tiny batches |
| xformers attention | ~30% on attention activations |

**Even with all of these,** training a 250M-parameter multi-modal diffusion model at the paper's resolution on a 12GB GPU is impractical — you'd be doing batch size 1 with all the tricks, taking weeks.

**This is why we scoped your architecture down to ~40-60M params.** That's the realistic range for your hardware.

---

## Comparison Pipeline — X-DRIVE

### Q: Can I make this pipeline lean enough to run X-DRIVE's dataset for training on 12GB 3060?

**Yes, with modifications.** X-DRIVE's official setup uses A6000 GPUs (48GB) with batch sizes 96 and 24. For a 3060:

| X-DRIVE setting | A6000 (paper) | 3060 (yours) |
|---|---|---|
| LiDAR VAE batch | 96 | 4 with grad accum → effective 16 |
| LiDAR VAE epochs | 200 | 100 |
| LDM batch | 96 | 2 with grad accum → effective 8 |
| LDM epochs | 2000 | 500 |
| Joint training batch | 24 | 1 with grad accum → effective 4 |
| Joint training epochs | 250 | 80 |
| Mixed precision | (not specified) | fp16 required |
| Gradient checkpointing | (not specified) | required |

**Important:** You'd also want to drop the multi-view image generation tower if your goal is LiDAR-only. X-DRIVE has both branches by default; stripping the image branch saves significant memory.

### Q: Compare with X-DRIVE output?

**Yes, this is a strong comparison strategy:**

1. Train your simplified pipeline on nuScenes
2. Run X-DRIVE's released pretrained model on the same nuScenes validation scenes
3. Compare:
   - MMD and JSD (point cloud quality, from X-DRIVE's paper)
   - Chamfer Distance
   - Downstream 3D detection mAP using a pretrained CenterPoint detector
   - Visual side-by-side BEV renders

**Expected outcome:** X-DRIVE will likely outperform your simplified version (more params, more compute, multi-view conditioning). That's fine — your contribution is showing the architecture works at smaller scale, not beating X-DRIVE.

---

## Summary: Your Ideal Pipeline Spec

> - no autoregressive component
> - no video VAE head at the end
> - no support for dashcam video

**Final simplified architecture:**

```
INPUT
  └─ Single front-view AV camera image (256×256 RGB)
      OR 6 surround AV cameras (for Phase 0 simpler test)

PREPROCESSING
  └─ Image → [Frozen SD 1.5 VAE Encoder] → image latent
  └─ Camera calibration → raymap

DIFFUSION (the only trainable part)
  └─ LiDAR latent (noise) + raymap conditioning
  └─ → [Single U-Net, ~30-50M params]
      └─ Self-attention on LiDAR tokens
      └─ Cross-attention to image features (one-way conditioning)
      └─ Conv + ResNet blocks
  └─ → Denoised LiDAR latent

DECODING
  └─ [Frozen LiDAR VAE Decoder] → LiDAR range image [32 × 512 × 4]
  └─ Spherical unprojection → 3D point cloud

OUTPUT
  └─ LiDAR point cloud
```

**What's NOT in this pipeline:**
- ❌ Multi-view image generation (no image output)
- ❌ Image VAE decoder (encoder only)
- ❌ Cross-view attention (single-view input)
- ❌ Temporal conditioning
- ❌ Autoregressive rollout
- ❌ DAgger fine-tuning
- ❌ 4DGS data synthesis
- ❌ Real dashcam input (use AV camera as stand-in)

**What IS in this pipeline:**
- ✅ LiDAR VAE (trained first, then frozen)
- ✅ Single diffusion U-Net
- ✅ Cross-attention from LiDAR queries to image features
- ✅ Stable Diffusion VAE for image encoding (frozen, reused)
- ✅ Range-image LiDAR representation with 4 channels
- ✅ Standard DDPM/DDIM diffusion sampler

This is a focused, tractable project for a 12GB 3060.
