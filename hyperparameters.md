# Sensor2Sensor — Hyperparameters

## Compute & Optimization

| Item | Value |
|---|---|
| Hardware | 128 TPUs |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Global gradient norm clip | 1.0 |
| EMA decay on model weights | 0.999 |
| Total model parameters | ~250 M |

## Model Architecture (multi-stream UNet)

| Item | Value |
|---|---|
| UNet backbone | Latent Diffusion, multi-stream (camera + LiDAR) |
| UNet block output channels | (320, 640, 1280, 1280) |
| Image VAE latent | 8-channel |
| LiDAR VAE latent | 16-channel (UNet LiDAR stream has 16 in/out channels to match) |
| Camera views processed jointly | 8 surrounding + 1 dashcam conditioning view (9 total in latent stack) |
| 2D attention → 3D attention | replaces image-LDM 2D attn with 1D cross-view + 2D spatial |
| Cross-sensor module | self-attention over concatenated `[T_C; T_L]` tokens, applied after each UNet conv block |

## Conditioning

| Item | Value |
|---|---|
| Dashcam conditioning input | concatenated in **view dimension** as 9th view, with binary "noise-free" mask |
| Camera-parameter conditioning | raymaps (origin + direction per pixel), normalized to camera 1, concatenated channel-wise |
| 9th (dashcam) view | excluded from loss computation |
| Spatial-mask drop probability on dashcam frames (train) | 0.2 |
| Temporal conditioning (prev-frame latents) drop probability | 0.5 |

## LiDAR Representation

| Item | Value |
|---|---|
| Tensor shape | `[H_L, W_L, D_L]` with `D_L = 4` (range, intensity, elongation, validity) |
| Range clamp | 150 m |
| Range / intensity / elongation normalization | linear to [0, 1] |

## Four-Stage Training Pipeline

| Step | Task | Steps |
|---|---|---|
| 1 | Base single-frame generation, conditioned on dashcam | 80 k |
| 2 | Fine-tune w/ dense previous-frame conditioning (prev camera + LiDAR latents + prev dashcam) | 40 k |
| 3 | DAgger data generation (model unrolls to produce drifted rollouts) | — (data gen, not gradient steps) |
| 4 | DAgger fine-tuning on Step-3 rollouts | 20 k |

Each step fine-tunes from the previous step's checkpoint.

## DAgger Fine-tuning

| Item | Value |
|---|---|
| Rollout horizon during DAgger fine-tune | 6 steps (~35 frames per training segment) |
| Probability of training on original GT context | 0.2 (to retain robustness) |

## Diffusion Inference / Auto-regressive Generation

| Item | Value |
|---|---|
| Auto-regressive conditioning window | single previous frame `(t-1)` |
| Initial frame (`t = 0`) conditioning | dashcam frame `x_0` only |

## VAE Loss Weights (LiDAR VAE)

Loss = sum of L1 (range, elongation, intensity) + BCE (validity) + LPIPS (normals, elongation, intensity, validity) + KL.

Per-signal scalar weights `λ_signal`, `λ_BCE`, `λ_KL` are present but exact numeric values are not stated in the paper.

## Evaluation Settings

| Metric | Direction | Used for |
|---|---|---|
| FID | ↓ | image realism |
| FVD | ↓ | video realism |
| PSNR | ↑ | paired GT fidelity |
| SSIM | ↑ | paired GT fidelity |
| LPIPS | ↓ | paired GT fidelity |
| Chamfer Distance | ↓ | LiDAR geometry fidelity |
| Human Eval | ↑ | top-rank + pair-wise preferences (26 raters, 40×3 samples) |
