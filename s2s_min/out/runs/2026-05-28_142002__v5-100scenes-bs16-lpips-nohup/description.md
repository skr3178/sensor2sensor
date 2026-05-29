# Training run: `v5-100scenes-bs16-lpips-nohup`

- **Started**: 2026-05-28T15:14:57
- **Wall-clock**: 3287.9 s
- **Git commit**: `230d12249daea9cefaffce4d05a3b09f84d22c2d`
- **Final step**: 4024
- **Best l1_range_ema**: 0.00664
- **Final l1_range_ema**: 0.00673

## Recipe

- λ_range / intensity / validity / kl = 50.0 / 1.0 / 1.0 / 1e-06
- λ_lpips_normals / intensity / validity = 1.0 / 1.0 / 1.0 (on, net=vgg)
- Optimizer: AdamW lr=0.0004, wd=0.0001, schedule=cosine (warmup=200, lr_min=4e-06)
- Batch × grad_accum: 8 × 2
- EMA decay: 0.999
- Mixed precision: True

## Notes (write your observations here)

_TODO: what was different about this run? what worked / didn't?_
