# Encoder decode eval — DINOv3 vs SD-VAE

- 60 samples, cfg_scale=1.0, matched noise seeds, target = VAE-decode(μ)
- DINOv3 ckpt: `s2s_min/out/runs/2026-05-29_204102__m3-unet-15M-dinov3-fromscratch/lidar_unet_best.pt` (step 7135)
- SD-VAE ckpt: `s2s_min/out/runs/2026-05-28_161242__m3-unet-v5cache-50ep-bs16/lidar_unet_best.pt` (step 12510)

| metric | SD-VAE | DINOv3 | Δ |
|---|---|---|---|
| CD-3D mean (m) | 2.419 | 2.101 | +13.1% |
| CD-3D median (m) | 2.437 | 2.129 | |
| CD-BEV mean (m) | 1.384 | 1.063 | |
| DINOv3 win rate | | 80% | |

_In-distribution relative comparison; lower CD = generated LiDAR closer to the conditioned scene._