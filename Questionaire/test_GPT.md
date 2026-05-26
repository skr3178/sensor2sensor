# Sensor2Sensor / 4DGS / LiDAR-only Architecture Notes

## Recommended short answer

For a 12GB RTX 3060, do **not** try to reproduce the full paper architecture first. The practical version should be:

```text
single RGB frame or AV camera frame
→ frozen/small image encoder
→ small conditional LiDAR generator
→ LiDAR range image [H, W, 4]
→ point cloud projection
```

Start with **LiDAR-only**, **single-frame**, **non-autoregressive** generation. Remove the 8-view image generation branch, remove cross-view attention, remove DAgger, and only keep the pieces needed to map camera conditioning to a LiDAR range-view output.

Best first target:

```text
front camera / fixed camera image
→ generated front/360 LiDAR range image
```

not:

```text
dashcam video
→ 8 AV cameras + full LiDAR + temporal rollout
```

---

## Q1. Can I test the pipeline with one single image and dashcam video coupled?

Yes, but only as a **smoke test**, not as a real evaluation.

A single paired sample can test:

- data loading
- camera/LiDAR alignment
- tensor shapes
- VAE encode/decode
- U-Net forward pass
- loss computation
- one backward pass
- whether the model can overfit one example

It cannot prove the method generalizes.

Minimum useful test levels:

| Test level | Data | What it proves |
|---|---:|---|
| Forward-pass test | 1 image + 1 LiDAR frame | Pipeline does not crash |
| Overfit test | 1–10 paired frames | Model/loss can learn something |
| Tiny sequence test | 3 sec, about 30–60 frames | Temporal/data alignment sanity |
| Small real experiment | 100–1,000 paired frames | Some generalization signal |

If you have one dashcam/fixed-camera frame coupled with one AV LiDAR frame, you can use it to overfit a tiny model. If you have a dashcam video coupled to AV logs, sample aligned frames and treat each frame independently first.

---

## Q2. Should the dashcam/fixed-camera input be generated using 4DGS?

For the paper-style setup, yes.

The paper uses 4DGS as a **data-pairing engine**:

```text
real AV logs
→ reconstruct dynamic scene with 4DGS
→ render synthetic third-party/dashcam/fixed-camera views
→ train model on synthetic third-party view → real AV sensors
```

This creates paired data even when real dashcam-to-AV pairs do not exist.

But for your 3060 experiment, you do **not** need 4DGS at first. Start with:

```text
AV front camera frame → AV LiDAR range image
```

or:

```text
AV multi-view cameras → AV LiDAR range image
```

Then add 4DGS later only if you need third-party dashcam viewpoints.

---

## Q3. If I eliminate the Image VAE decoder, can I also eliminate the LiDAR VAE decoder?

You can eliminate the **Image VAE decoder** if you are not generating images.

But you usually should **not eliminate the LiDAR VAE decoder** if your model operates in LiDAR latent space and you want an actual range image or point cloud at the end.

There are two valid choices:

### Option A — keep LiDAR VAE

```text
LiDAR range image
→ LiDAR VAE encoder
→ latent diffusion
→ LiDAR VAE decoder
→ generated range image
→ point cloud
```

This is closer to the paper.

### Option B — remove LiDAR VAE completely

```text
camera image
→ small U-Net / diffusion model
→ LiDAR range image directly
→ point cloud
```

This is simpler and better for a first 3060 prototype.

So:

| Decoder | Can remove? | When |
|---|---|---|
| Image VAE decoder | Yes | If not generating target images |
| LiDAR VAE decoder | Only if using direct range-image prediction | Keep it if using latent diffusion |

For your case, I would start with **direct LiDAR range-image generation** and skip the LiDAR VAE until the rest of the pipeline works.

---

## Q4. Do we need two diffusion U-Nets? If we are not using the image decoder, does cross-attention require two U-Nets?

No.

For LiDAR-only generation, you need only **one conditional LiDAR U-Net**.

The full paper has image and LiDAR generation branches because it jointly generates:

```text
8 camera images + LiDAR
```

If you remove image generation, the image side becomes only a **conditioning encoder**.

Recommended simplified structure:

```text
RGB image
→ image encoder
→ condition tokens

noisy LiDAR range latent / noisy LiDAR range image
→ LiDAR U-Net
→ denoised LiDAR
```

The U-Net can use camera conditioning through:

- cross-attention
- feature concatenation
- FiLM/AdaGN conditioning
- ControlNet-style conditioning

You do **not** need a second diffusion U-Net unless you also generate images.

---

## Q5. Can I recreate a simpler smaller architecture that emulates the pipeline and fits in 12GB GPU?

Yes. A good 3060-friendly architecture is:

```text
Input:
  RGB image: 256x448 or 384x640

Condition encoder:
  ResNet-18 / ResNet-34 / small ConvNeXt / frozen DINO-small
  output: camera tokens or feature pyramid

Target:
  LiDAR range image:
    start with 32x512 or 64x512
    channels = range, intensity, validity
    optionally elongation if dataset has it

Generator:
  small conditional U-Net
  base channels: 32 or 64
  channel multipliers: [1, 2, 4]
  attention only at low resolution
  no cross-view attention
  no temporal module
  no image decoder
  no DAgger

Training:
  fp16 / mixed precision
  batch size 1–4
  gradient accumulation
  gradient checkpointing if needed
```

Suggested first model size:

| Component | Size target |
|---|---:|
| Image encoder | 10M–25M params |
| LiDAR U-Net | 20M–60M params |
| Total trainable | 30M–80M params |
| Output resolution first | 32x512 or 64x512 |
| Later output resolution | 64x1024 |

This is much more realistic than the paper’s full 250M-parameter multi-modal diffusion model.

---

## Q6. Can I create a pipeline and test it with a single pass of a sensor value and dashcam for training?

Yes.

You should separate this into two tests:

### 1. Single forward/backward test

Goal: confirm the code works.

```text
image, lidar_range = one paired sample

pred = model(image, noisy_lidar, timestep)
loss = diffusion_loss or lidar_loss
loss.backward()
optimizer.step()
```

This proves the training loop is wired correctly.

### 2. One-sample overfit test

Goal: prove the model can learn.

Train on the same 1 sample for 500–5,000 iterations. The loss should drop strongly and the generated LiDAR should start matching the target.

If it cannot overfit one sample, the issue is likely:

- wrong LiDAR projection
- wrong normalization
- invalid mask bug
- conditioning not connected
- U-Net output shape mismatch
- loss applied to invalid pixels
- timestep/noise scheduler bug

---

## Q7. Which 3DGS / 4DGS variant should we consider for the 4DGS pipeline?

For autonomous-driving data, prefer **driving-scene Gaussian Splatting** over generic object-centric 4DGS.

Recommended order:

| Candidate | Use case |
|---|---|
| **SplatAD** | Strong candidate if you care about camera + LiDAR rendering for AV scenes |
| **DrivingGaussian / Composite Gaussian Splatting** | Good for static background + dynamic vehicles |
| **Street Gaussians-style methods** | Good for decomposing street scenes into background and actors |
| **HUST-VL 4DGaussians** | Good general 4DGS baseline, but not AV-specialized |
| **Deformable 3D Gaussians** | Good for monocular dynamic/deformable reconstruction |
| **ST-4DGS / other spatiotemporal variants** | Better temporal consistency, usually heavier |

For your stated requirement:

```text
dynamic rigid vehicles + deformable pedestrians
```

the best conceptual design is:

```text
static world/background:
  3D Gaussians

vehicles:
  object-level dynamic Gaussians with rigid SE(3) transforms

pedestrians/cyclists:
  dynamic/deformable Gaussians or residual deformation field
```

Do **not** start with full deformable 4DGS for everything. It is heavier and less stable.

Best practical first choice:

```text
static 3DGS background + rigid dynamic actor Gaussians
```

Then add deformable support later.

---

## Q8. Which loss terms should we consider for LiDAR-only?

For LiDAR-only range-view generation, use these losses.

### Core losses

| Loss | Purpose |
|---|---|
| Diffusion noise MSE | Standard DDPM/latent diffusion training |
| Masked range L1 / Huber | Accurate range where LiDAR return is valid |
| Validity BCE / focal loss | Predict whether each range-view pixel has a return |
| Intensity L1 / Huber | If intensity channel is used |
| Elongation L1 / Huber | If elongation channel exists |

### Helpful geometry losses

| Loss | Purpose |
|---|---|
| Range gradient loss | Sharper object boundaries |
| Smoothness loss | Reduces noisy floating points |
| Chamfer distance | Point-cloud-level alignment |
| Occupancy / BEV IoU | Useful if converted to voxel/BEV grid |
| Depth/range scale loss | Stabilizes long-range predictions |

### Recommended first loss

For a simple non-diffusion baseline:

```text
loss =
  1.0 * masked_L1(log_range_pred, log_range_gt)
+ 1.0 * BCE(validity_pred, validity_gt)
+ 0.2 * range_gradient_loss
+ 0.1 * intensity_L1
```

For latent diffusion:

```text
loss =
  diffusion_noise_MSE
```

plus VAE reconstruction losses if training the VAE.

Use **log range** or normalized inverse range. Raw range values can make far points dominate training.

---

## Q9. U-Net architecture default is 2x. Can we reduce it to 1x since we do not need video and only need LiDAR?

Yes.

If `2x` means width multiplier or larger U-Net scale, reduce it to `1x`.

For your first LiDAR-only model:

```text
base_channels = 32 or 64
channel_mult = [1, 2, 4]
num_res_blocks = 1 or 2
attention_resolutions = only lowest 1–2 resolutions
temporal_blocks = disabled
cross_view_attention = disabled
cross_sensor_attention = disabled
```

Suggested starting config:

```yaml
lidar_range_shape: [64, 512, 3]
base_channels: 64
channel_mult: [1, 2, 4]
num_res_blocks: 1
attention: low_res_only
condition: image_cross_attention_or_concat
precision: fp16
batch_size: 1-2
```

If you still get OOM:

```text
64x512 → 32x512
base_channels 64 → 32
attention → remove
batch size → 1
use gradient checkpointing
```

---

## Q10. Suppose we have 4DGS-generated coupled dashcam and AV logs. What model parameters are needed next?

Once you have paired data:

```text
dashcam/fixed-camera image x_t
AV LiDAR range image y_t
optional AV camera images C_t
calibration/raymaps
timestamps
```

you need to define these model/data parameters.

### Data parameters

```yaml
input_camera_resolution: [256, 448] or [384, 640]
lidar_range_resolution: [32, 512] or [64, 1024]
lidar_channels: [range, intensity, validity]
range_normalization: log_range
max_range_m: 80 or 100
min_range_m: 1
validity_mask: true
camera_intrinsics: required if using raymaps/projection
camera_extrinsics: required for geometric conditioning
lidar_calibration: required for point-cloud projection
```

### Model parameters

```yaml
image_encoder: ResNet18 or frozen DINO-small
condition_dim: 256 or 512
generator: conditional_lidar_unet
base_channels: 32 or 64
channel_mult: [1, 2, 4]
num_res_blocks: 1
attention: low_resolution_only
diffusion_timesteps_train: 1000
diffusion_timesteps_infer: 20-50
prediction_type: epsilon or v_prediction
```

### Training parameters

```yaml
batch_size: 1-2
gradient_accumulation: 4-16
optimizer: AdamW
lr: 1e-4 for small model, 1e-5 to 5e-5 for diffusion
precision: fp16
ema: optional
steps_first_test: 500-5000 overfit
steps_tiny_train: 10000-50000
```

---

## Q11. Explore alternative more efficient algorithms

Before training diffusion, build baselines.

### Baseline 1 — deterministic range regression

```text
image → U-Net → range + validity
```

Pros:

- easiest
- fastest
- fits 12GB
- good debugging baseline

Cons:

- blurry/plausible average LiDAR
- weak uncertainty modeling

### Baseline 2 — monocular depth → pseudo-LiDAR

```text
image → monocular depth model → projected pseudo point cloud
```

Pros:

- very easy
- can use pretrained depth models
- good sanity baseline

Cons:

- not true spinning LiDAR
- no intensity/elongation
- sparse LiDAR pattern must be simulated

### Baseline 3 — conditional VAE

```text
image → encoder
LiDAR → VAE latent
decoder(condition, latent) → LiDAR
```

Pros:

- much cheaper than diffusion
- can model some uncertainty

Cons:

- quality usually lower than diffusion

### Baseline 4 — small latent diffusion

```text
image condition → latent diffusion → LiDAR VAE latent → LiDAR
```

Pros:

- closest to paper
- better multimodality

Cons:

- more complex
- needs VAE

### Baseline 5 — BEV/occupancy generation

```text
image → BEV occupancy/depth grid → point cloud sampling
```

Pros:

- easier to evaluate for driving scenes
- lower resolution than range-view LiDAR

Cons:

- loses raw LiDAR scan structure

Recommended order:

```text
1. deterministic LiDAR range regression
2. monocular-depth pseudo-LiDAR baseline
3. conditional VAE
4. small latent diffusion
5. 4DGS-generated dashcam conditioning
```

---

## Q12. If I do not want to add dashcam video and first test with 8 AV cameras + LiDAR logs, what changes?

Then the problem becomes easier:

```text
8 AV cameras → AV LiDAR
```

or even:

```text
front AV camera → AV LiDAR
```

Changes to the paper pipeline:

| Paper component | Change for your first test |
|---|---|
| 4DGS dashcam rendering | Remove |
| third-party camera condition | Replace with AV front camera or 8 AV cameras |
| image generation branch | Remove |
| Image VAE decoder | Remove |
| cross-view attention for generated images | Remove |
| cross-sensor attention between generated image/LiDAR | Remove |
| previous-frame conditioning | Remove |
| DAgger | Remove |
| LiDAR branch | Keep |
| calibration/raymaps | Keep if using geometry-aware conditioning |

For 8-camera input, use lightweight fusion:

```text
each camera → shared image encoder
camera features + camera ray/camera id
→ concatenate or small transformer
→ condition LiDAR U-Net
```

For the simplest first version:

```text
front camera only → LiDAR range image
```

Then upgrade to:

```text
8 cameras → LiDAR range image
```

---

## Video generation decision

Do **not** consider autoregressive video generation for the first version.

Remove:

```text
P(C_t, L_t | x_t, C_{t-1}, L_{t-1})
```

Use:

```text
P(L_t | x_t)
```

or:

```text
P(L_t | C_t)
```

where `C_t` can be one camera or 8 AV cameras.

DAgger is mainly useful when the model rolls out over time and conditions on its own previous generated outputs. Since you are not doing autoregressive video generation, you do not need DAgger.

---

## Explain 1 — Cross-view attention

Cross-view attention is for consistency across multiple generated camera views.

The paper has 8 target AV cameras. A car may appear in more than one view. If each view is generated independently, the car can appear with inconsistent location, color, shape, or even disappear.

Cross-view attention allows tokens from different views to interact:

```text
view 1 tokens
view 2 tokens
...
view 8 tokens
→ attention across views and spatial positions
→ consistent multi-view generation
```

It replaces plain 2D image-only attention with attention across:

```text
spatial dimensions + view dimension
```

For your LiDAR-only pipeline, cross-view attention is unnecessary unless you use 8 cameras as input and want to fuse them. Even then, you need only a small **multi-view input fusion module**, not the full paper cross-view generation module.

---

## Explain 2 — Cross-sensor attention

Cross-sensor attention is for consistency between generated image features and generated LiDAR features.

In the paper:

```text
image U-Net features
LiDAR U-Net features
→ flatten
→ concatenate
→ self-attention
→ split back into image and LiDAR branches
```

This lets information move between the two sensors while denoising.

It helps the model learn things like:

```text
object visible in generated image
↔ corresponding LiDAR returns should exist
```

Can cross-view attention and cross-sensor attention replace each other?

No.

They solve different problems:

| Attention type | Purpose |
|---|---|
| Cross-view attention | Consistency across camera views |
| Cross-sensor attention | Consistency between camera and LiDAR modalities |

For LiDAR-only generation, you do not need full cross-sensor attention because there is only one generated modality. You need **camera-to-LiDAR conditioning**, for example:

```text
LiDAR U-Net queries attend to camera encoder tokens
```

This is asymmetric conditioning, not full cross-sensor generation.

Also, cross-sensor attention alone does not magically create correct 360° LiDAR. The 360° structure comes from:

- training data
- LiDAR range-view representation
- vehicle/sensor calibration
- learned scene priors
- multi-view/fixed-camera information if available

---

## Dataset for testing

A 3-second paired fixed-camera-to-AV-log sequence is enough for a pipeline test.

Approximate frame counts:

| Sensor rate | Frames in 3 sec |
|---:|---:|
| 10 Hz | 30 frames |
| 20 Hz | 60 frames |
| 30 Hz | 90 frames |

Use it like this:

```text
train split: 80%
validation split: 20%
```

But with only 3 seconds, validation is mostly a sanity check. It is not a real generalization test.

### Training steps

The paper-level steps like:

```text
Step 1: 80k
Step 2: 40k
Step 4: 20k
```

are too large for a tiny dataset and 3060 prototype.

Use this instead:

| Stage | Purpose | Tiny-data steps |
|---|---|---:|
| Step 0 | one-sample overfit | 500–5,000 |
| Step 1 | LiDAR VAE, if used | 5k–20k |
| Step 2 | LiDAR conditional U-Net/diffusion | 10k–50k |
| Step 3 | optional refinement | skip first |
| Step 4 | DAgger/video fine-tuning | skip |

If you are not using a LiDAR VAE, skip Step 1.

---

## Why will 250M parameters not fit my 12GB GPU?

The problem is not just parameter count.

250M parameters in fp16 is about:

```text
250M * 2 bytes = 500 MB
```

So inference weights alone may fit.

Training is different. You also need memory for:

- gradients
- Adam optimizer states
- fp32 master weights
- U-Net activations
- attention key/query/value tensors
- 8 camera views
- LiDAR feature maps
- cross-view attention
- cross-sensor attention
- previous-frame conditioning
- batch dimension
- diffusion timestep activations

With Adam-style training, parameter-related memory can easily become several GB:

```text
weights + gradients + optimizer states + master weights
```

But the real killer is usually **activation memory**, especially in attention blocks. Cross-view and cross-sensor attention flatten many spatial/view/sensor tokens, and attention memory can grow very quickly.

So:

```text
250M params may fit for inference
250M params usually does not fit comfortably for training on 12GB
```

especially with multi-view + LiDAR + diffusion + attention.

---

## X-Drive / comparison pipeline

Assuming X-Drive means a dataset or pipeline with camera/LiDAR driving logs, you can make a lean version fit on the RTX 3060 by using:

```text
single-frame
LiDAR-only
low-resolution range image
small U-Net
no video
no image generation
no DAgger
```

Recommended X-Drive-style lean training setup:

```yaml
input:
  camera: front only first, 8 cameras later
  resolution: 256x448

output:
  lidar_range: 64x512
  channels: range, validity, intensity

model:
  image_encoder: ResNet18
  generator: small conditional U-Net
  base_channels: 32 or 64
  attention: low-res only
  diffusion: optional after regression baseline

training:
  batch_size: 1
  fp16: true
  grad_accum: 8
  checkpointing: true
```

Comparison metrics:

| Metric | What it checks |
|---|---|
| masked range MAE | range accuracy on valid returns |
| RMSE / log RMSE | depth/range quality |
| validity IoU | whether return/no-return structure matches |
| Chamfer distance | point-cloud geometry |
| F-score at distance threshold | point-cloud overlap |
| BEV occupancy IoU | driving-scene occupancy correctness |
| visual overlay | qualitative alignment with camera |

For a fair comparison, evaluate:

```text
same input frame
same LiDAR coordinate frame
same range-view resolution
same max range
same valid mask convention
```

---

## Final ideal pipeline for your requirements

Your ideal 3060-friendly pipeline should be:

```text
No autoregressive component
No video VAE head
No image generation decoder
No DAgger
No 8-view output generation
LiDAR-only output
Single-frame conditioning
Small U-Net or direct regression baseline
```

### Version 0 — fastest sanity baseline

```text
AV front camera
→ ResNet18 encoder
→ U-Net decoder
→ LiDAR range image [32/64, 512, range+validity]
```

### Version 1 — better LiDAR-only model

```text
AV front or fixed camera
→ image encoder tokens
→ conditional LiDAR U-Net
→ range + intensity + validity
→ point cloud projection
```

### Version 2 — latent diffusion version

```text
LiDAR VAE trained on range images
camera encoder
conditional latent diffusion U-Net
LiDAR VAE decoder
point cloud projection
```

### Version 3 — paper-inspired version

```text
4DGS-generated fixed/dashcam view
→ conditional LiDAR latent diffusion
→ generated AV LiDAR
```

Do not attempt the full paper model until Version 1 or Version 2 works.

---

## My recommended next implementation target

Build this first:

```text
Input:
  one AV front camera image, 256x448

Target:
  LiDAR range image, 64x512x3
  channels = log_range, intensity, validity

Model:
  ResNet18 image encoder
  small conditional U-Net
  direct range prediction first, not diffusion

Loss:
  masked L1 on log_range
  BCE on validity
  optional L1 on intensity
  optional range-gradient loss

Goal:
  overfit 1 sample
  then overfit 30 frames
  then train on 1k+ frames
```

Only after this works should you add:

```text
multi-view camera input
LiDAR VAE
latent diffusion
4DGS-generated dashcam/fixed-camera views
```
