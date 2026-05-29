# DINOv2 depth probe (encoder swapped in for SD-VAE)

- samples 800, valid cells 775,882

| condition | AbsRel↓ | RMSE↓ | δ<1.25↑ | Pearson↑ | R²↑ |
|---|---|---|---|---|---|
| DINOv2 img | 0.148 | 6.87 | 0.806 | 0.953 | 0.907 |
| DINOv2+ray | 0.150 | 6.81 | 0.808 | 0.953 | 0.909 |
| raymap only | 0.286 | 10.72 | 0.643 | 0.863 | 0.742 |
| DINOv2 (shuffled) | 0.320 | 11.24 | 0.607 | 0.831 | 0.687 |
| mean floor | 0.704 | 16.10 | 0.165 | 0.000 | -0.000 |

## Verdict: **DINOv2 FEATURES CARRY DEPTH**

=> Encoder swap is justified: DINOv2 supplies the depth signal SD-VAE lacked.
   Proceed to cache-rebuild + U-Net retrain on DINOv2 conditioning (+ B1 pos-enc).

Compare against `s2s_min/out/depth_probe/` (SD-VAE).