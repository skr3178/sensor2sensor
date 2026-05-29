# SD-VAE depth probe

- samples: **800**, valid cells: **775,882**, device: cuda
- cache: `s2s_min/out/cached_latents_v5_100scenes`

| condition | AbsRel↓ | RMSE(m)↓ | δ<1.25↑ | Pearson(log)↑ | R²↑ |
|---|---|---|---|---|---|
| img | 0.642 | 15.36 | 0.202 | 0.348 | 0.121 |
| img+ray | 0.308 | 11.21 | 0.608 | 0.845 | 0.712 |
| ray | 0.286 | 10.72 | 0.643 | 0.863 | 0.742 |
| img_shuf | 0.617 | 15.48 | 0.205 | 0.357 | 0.127 |
| mean | 0.704 | 16.10 | 0.165 | 0.000 | -0.000 |

## Verdict: **SD-VAE FEATURES ARE DEPTH-IMPOVERISHED**

=> Image probe ~ position prior; features add little depth signal.
   Disambiguates toward H2/H3.  ACTION: swap the image encoder (C-series).

Plots: `metrics_bar.png`, `scatter_pred_vs_gt.png`, `training_curves.png`, `qualitative.png`, `decision_summary.png`.