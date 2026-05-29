# SD-VAE depth probe

- samples: **120**, valid cells: **117,860**, device: cuda
- cache: `s2s_min/out/cached_latents_v5_100scenes`

| condition | AbsRel↓ | RMSE(m)↓ | δ<1.25↑ | Pearson(log)↑ | R²↑ |
|---|---|---|---|---|---|
| img | 0.640 | 15.59 | 0.205 | 0.312 | 0.097 |
| img+ray | 0.491 | 13.92 | 0.256 | 0.706 | 0.423 |
| ray | 0.444 | 13.40 | 0.281 | 0.772 | 0.512 |
| img_shuf | 0.632 | 15.81 | 0.194 | 0.268 | 0.071 |
| mean | 0.687 | 16.14 | 0.180 | 0.000 | -0.000 |

## Verdict: **SD-VAE FEATURES ARE DEPTH-IMPOVERISHED**

=> Image probe ~ position prior; features add little depth signal.
   Disambiguates toward H2/H3.  ACTION: swap the image encoder (C-series).

Plots: `metrics_bar.png`, `scatter_pred_vs_gt.png`, `training_curves.png`, `qualitative.png`, `decision_summary.png`.