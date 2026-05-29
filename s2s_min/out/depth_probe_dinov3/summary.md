# DINOv3 depth probe (vit_small_patch16_dinov3.lvd1689m)

- 800 samples, 775,882 cells

| condition | AbsRel | RMSE | δ<1.25 | Pearson | R² |
|---|---|---|---|---|---|
| DINOv3 img | 0.139 | 6.68 | 0.821 | 0.956 | 0.914 |
| DINOv3+ray | 0.139 | 6.79 | 0.817 | 0.955 | 0.911 |
| raymap only | 0.286 | 10.72 | 0.643 | 0.863 | 0.742 |
| DINOv3 (shuffled) | 0.339 | 11.49 | 0.606 | 0.824 | 0.669 |
| mean floor | 0.704 | 16.10 | 0.165 | 0.000 | -0.000 |

Compare against depth_probe/ (SD-VAE) and depth_probe_dinov2/ (DINOv2).