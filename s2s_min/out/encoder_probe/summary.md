# Encoder gate probe

- samples 500, valid cells 484,769, Depth-Anything available: False

| condition | Pearson↑ | AbsRel↓ | δ<1.25↑ | R²↑ | lift vs raymap |
|---|---|---|---|---|---|
| sdvae | 0.358 | 0.645 | 0.199 | 0.127 | -0.487 |
| sdvae+ray | 0.834 | 0.302 | 0.627 | 0.691 | -0.012 |
| raymap | 0.846 | 0.294 | 0.632 | 0.709 | — |
| dinov2 | 0.949 | 0.159 | 0.787 | 0.899 | +0.103 |
| dinov2+ray | 0.949 | 0.163 | 0.789 | 0.898 | +0.103 |
| mean | -0.000 | 0.697 | 0.161 | -0.000 | — |

## GREEN LIGHT — a depth-aware encoder adds real depth residual.

=> Worth the cache-rebuild + retrain. Recommend: / DINOv2 swap, plus B1 pos-enc.