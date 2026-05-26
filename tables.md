# Sensor2Sensor — Tables

## Table 1. Multi-view Image Generation (Fixed-Camera-to-AV)

Per-frame quality from a fixed front-left bumper camera input. **VC** = view-concatenation of the dashcam input.

| Method | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| VGGT | 250.93 | 14.73 | 0.433 | 0.491 |
| π³ | 246.27 | 14.93 | 0.470 | 0.458 |
| X-Drive | 8.30 | 18.61 | 0.536 | 0.345 |
| Ours without VC | 6.88 | 18.69 | 0.531 | 0.346 |
| **Ours** | **6.47** | **19.06** | **0.539** | **0.316** |

---

## Table 2. Multi-view Video Generation (Fixed-Camera-to-AV)

Front-view metrics only (X-Drive excluded — image-only model).

| Method | FVD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| VGGT | 2373.15 | 14.73 | 0.433 | 0.491 |
| π³ | 2007.35 | 14.93 | 0.470 | 0.458 |
| Ours without VC | 293.73 | 22.07 | 0.622 | 0.204 |
| **Ours** | **278.12** | **22.42** | **0.623** | **0.186** |

---

## Table 3. LiDAR Generation Accuracy

Chamfer Distance between predicted and ground-truth LiDAR point clouds.

| Method | Chamfer Distance ↓ | Improvement (%) |
|---|---|---|
| X-Drive | 10.02 | — |
| **Ours** | **8.68** | **13.37 %** |

---

## Table 4. Human Evaluation — In-the-Wild Generation

26 participants evaluated 40 × 3 image + LiDAR samples; top-rank rates (top half) and pair-wise win rates (bottom half).

| Method | Dashcam Image ↑ | Dashcam LiDAR ↑ | Internet Image ↑ | Internet LiDAR ↑ |
|---|---|---|---|---|
| X-Drive | 3.08 % | 8.08 % | 1.54 % | 7.69 % |
| Ours without VC | 13.46 % | 23.85 % | 13.85 % | 33.85 % |
| **Ours** | **83.46 %** | **68.08 %** | **84.62 %** | **58.46 %** |
| Ours without VC > X-Drive | 67.69 % | 69.62 % | 84.62 % | 73.46 % |
| Ours > Ours without VC | 85.77 % | 73.46 % | 85 % | 63.46 % |
| Ours > X-Drive | 94.62 % | 87.31 % | 95.38 % | 85 % |

---

## Table 5. Ablation — Model Architecture (Fixed-Camera-to-AV)

CC = channel concatenation of the dashcam input. VC = view concatenation.

| Method | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| CAT3D + CC (image-only) | 6.63 | 18.91 | 0.542 | 0.314 |
| CAT3D + VC (image-only) | **6.20** | **19.12** | **0.543** | **0.307** |
| CAT3D + CC + LiDAR | 6.88 | 18.69 | 0.531 | 0.346 |
| **CAT3D + VC + LiDAR (Ours)** | 6.47 | 19.06 | 0.539 | 0.316 |

Take-aways:
- VC > CC in the image-only setting (lower FID).
- Adding joint LiDAR training keeps LPIPS competitive (0.316 vs. image-only 0.307) while enabling LiDAR generation.

---

## Table 6. Ablation — DAgger Fine-tuning (Video Generation)

Front-view metrics on the Fixed-Camera-to-AV dataset.

| Method | Front-view FVD ↓ | Front-view FID ↓ |
|---|---|---|
| Without DAgger | 288.90 | 24.65 |
| **With DAgger (Ours)** | **278.12** | **21.54** |
