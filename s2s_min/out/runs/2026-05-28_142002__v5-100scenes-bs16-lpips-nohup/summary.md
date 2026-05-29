# Run summary — `v5-100scenes-bs16-lpips-nohup`

- **Generated**: 2026-05-28T15:32:18
- **Run folder**: `2026-05-28_142002__v5-100scenes-bs16-lpips-nohup`
- **Git commit**: `230d12249daea9cefaffce4d05a3b09f84d22c2d`
- **Wall-clock**: 3287.9 s (54.8 min)

## Headline metrics

- best `l1_range_ema`: **0.00664** → ≈ 0.66 m mean per-pixel range error
- final `l1_range_ema`: **0.00673** (after 4024 optimizer steps)

## VAE-only Chamfer ([chamfer.json](eval/chamfer.json))

| group     | CD_floor (m) | CD_roundtrip (m) | VAE_delta (m) |
|-----------|-------------:|-----------------:|--------------:|
| train     | 5.826 | 5.944 | 0.118 |
| held_out  | 5.547 | 5.717 | 0.170 |

_`VAE_delta` = `CD_roundtrip − CD_floor` — what the VAE adds on top of the projection floor. This is the apples-to-apples number to compare against RangeLDM's published ~0.01–0.02 m VAE-only Chamfer._

## BEV reconstruction sanity ([eval/samples.png](eval/samples.png))

```
  n015-2018-08-01-15-10-21+0800__LIDAR_TOP__1533107659547179.p       1.146     0.0457     0.1286      0.945
  MEAN                                                               1.051     0.0340     0.1171      0.950
```

_4 LIDAR_TOP keyframes pulled from the 10-scene subset (training samples)._

## Recipe ([metadata.json](metadata.json) has the full args dict)

- `epochs` = `16`
- `batch_size` = `8`
- `grad_accum` = `2`
- `lr` = `0.0004`
- `lr_min` = `4e-06`
- `lr_schedule` = `cosine`
- `lr_warmup_steps` = `200`
- `weight_decay` = `0.0001`
- `lam_range` = `50.0`
- `lam_intensity` = `1.0`
- `lam_validity` = `1.0`
- `lam_kl` = `1e-06`
- `lam_lpips_normals` = `1.0`
- `lam_lpips_intensity` = `1.0`
- `lam_lpips_validity` = `1.0`
- `ema_decay` = `0.999`
- `subset_file` = `s2s_min/out/subset_100scenes.txt`

## Full run history (v1 → this run)

| | v1 | v2 | v3 | v4 | v5-100scenes-bs16-lpips-nohup |
|---|---|---|---|---|---|
| Scenes | 10 | 10 | 10 | 10 | 100 |
| Keyframes | 401 | 401 | 401 | 401 | 4 023 |
| Effective batch | 8 | 8 | 8 | 8 | 16 |
| LPIPS terms | ✗ | ✗ | ✗ | ✓ | ✓ |
| LR schedule | constant | constant | cosine | cosine | cosine |
| Total optim steps | 2513 | 2513 | 2513 | 2513 | 4024 |
| Wall-clock | 9.1 min | 9.4 min | 9.4 min | 21.0 min | 54.8 min |
| Peak VRAM | 446 MiB | 446 MiB | 446 MiB | 1.2 GB | ? |
| Best `l1_range_ema` | ~0.015 (lost) | 0.01147 | 0.01116 | 0.00656 | 0.00664 |
| Final `l1_range_ema` | 0.188 | 0.030 | 0.066 | 0.00667 | 0.00673 |
| Divergence step | ~2 000 | ~950 | ~1 050 | none | none |
| Final ckpt usable? | ✗ | ✗ | ✗ | ✓ | ✓ |
| VAE_delta on held-out | — | — | — | — | 0.170 m |

_v1-v4 numbers are hardcoded in `s2s_min/scripts/eval_after.py:HISTORICAL_RUNS` since those runs pre-date the run-folder layout. Keep in sync with [`docs/lidar_vae.md` §8.1](../../docs/lidar_vae.md)._

## Sibling runs in this folder (run-folder layout only)

| run | description | best l1_range_ema | final l1_range_ema | step | wall-clock (s) |
|---|---|---|---|---|---|
| `2026-05-28_114044__smoke-test-ru…` | smoke-test-runfolder | inf | 0.38873 | 20 | 6.5 |
| `2026-05-28_142002__v5-100scenes-…` | v5-100scenes-bs16-lpips-nohup ← **this run** | 0.00664 | 0.00673 | 4024 | 3287.9 |

## Eval artifacts (all in this folder)

- [eval/loss_comparison.png](eval/loss_comparison.png) — 8-panel loss-term comparison vs prior runs
- [eval/loss_plots/](eval/loss_plots/) — standalone PNG per loss term
- [eval/samples.png](eval/samples.png) — BEV/range visualization on the best ckpt
- [eval/stats.txt](eval/stats.txt) — per-sample reconstruction stats
- [eval/chamfer.json](eval/chamfer.json) — VAE-only round-trip Chamfer + projection-floor baseline
- [lidar_vae_best.pt](lidar_vae_best.pt) — what M2/M3 should load
- [metadata.json](metadata.json) — full CLI args + final losses
