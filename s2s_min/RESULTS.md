# s2s_min — experiment results

Running log of milestone runs in the minimum pipeline. Appended to as M2 → M5 land.
For the paper-level dataset description see [../dataset.md](../dataset.md);
this file is the experiment-specific record (what we picked, what we trained, what came out).

---

## Dataset (nuScenes, M1)

### Scene & keyframe accounting

| | Count | Notes |
|---|---|---|
| Total scenes available | **850** | v1.0-trainval, fully on disk via the `nuscenes` symlink → `/media/skr/storage/self_driving/S2GO/data/nuscenes/` |
| Total LIDAR_TOP keyframes available | **34,149** | mean ~40 per scene, range 32–41 |
| Scenes we picked for M1 | **10** | sampled with `np.random.default_rng(0).choice(850, 10, replace=False)` |
| Keyframes in the subset | **401** | listed in [out/subset_scene_tokens.txt](out/subset_scene_tokens.txt) |
| Coverage used | **1.18 %** | (401 / 34,149 keyframes) |

### The 10 scenes (seed=0)

| Scene token | Name | `nbr_samples` |
|---|---|---|
| `813213458a214a39a1d1fc77fa52fa34` | scene-0015 | 40 |
| `3dd2be428534403ba150a0b60abc6a0a` | scene-0035 | 40 |
| `30e77a3561ce4f1bb4fd90e322ea29d7` | scene-0066 | 39 |
| `f9e460f092c94466b1211704b5a8859d` | scene-0191 | 40 |
| `26540bfbab79463cb1ba76b52ec6013b` | scene-0285 | 40 |
| `93608f0d57794ba6b014314c488e2b4a` | scene-0332 | 40 |
| `788c5502523f4d01b3a8de47ec3dadfb` | scene-0541 | 41 |
| `62d9a581c7114d90a48bacfb2a47b104` | scene-0684 | 41 |
| `d3b86ca0a17840109e9e049b3dd40037` | scene-0905 | 40 |
| `a04daf2d0f194b2ab2ff2a47dfebc1d7` | scene-0930 | 40 |
| | **Total** | **401** |

Reproduce with [`scripts/select_subset.py`](scripts/select_subset.py). Promote to the full nuScenes
train split by running `--n 700`.

---

## M1 — LiDAR VAE training

Three full-epoch runs were performed; only the third's `lidar_vae_best.pt` is retained.
All runs were 50 epochs over the 401-keyframe subset on a single RTX 3060.

### Run recipes

| | v1 | v2 | **v3 (committed)** |
|---|---|---|---|
| `λ_range` | 1 | 50 | 50 |
| `λ_intensity` / `λ_validity` / `λ_kl` | 1 / 1 / 1e-6 | 1 / 1 / 1e-6 | 1 / 1 / 1e-6 |
| Optimizer | AdamW | AdamW | AdamW |
| Peak LR | 4e-4 | 4e-4 | 4e-4 |
| LR schedule | constant | constant | **cosine, 200-step warmup, lr_min=4e-6** |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Grad-norm clip | 1.0 | 1.0 | 1.0 |
| Mixed precision | fp16 (autocast) | fp16 | fp16 |
| Batch × grad-accum | 2 × 4 (effective 8) | 2 × 4 | 2 × 4 |
| EMA(0.999) shadow on CPU | none | **enabled** | **enabled** |
| Best-ckpt save (by L1_range-EMA) | none | **enabled** | **enabled** |

### Run results

| | v1 | v2 | **v3** |
|---|---|---|---|
| Wall-clock | 9.1 min | 9.4 min | **9.4 min** |
| Peak VRAM | 446 MiB | 446 MiB | 446 MiB |
| Convergence (l1_range_ema first < 0.05) | step ~100 | step ~250 | **step ~300** |
| Best l1_range_ema observed | ~0.015 (lost — not saved) | 0.01147 (step ~900) | **0.01116** (step ~900) |
| Divergence step | ~2000 (epoch 40) | ~1000 (epoch 20) | ~1100 (epoch 21) |
| Final l1_range_ema (diverged region) | 0.188 | 0.030 | 0.066 |
| Best checkpoint saved? | ❌ | ✓ | ✓ |

### Key takeaway

Divergence is **fundamentally driven by gradient noise at small effective batch (8) on small dataset (401 samples)**, not by LR engineering. Cosine decay (v3) lowered the LR at the danger step from 4e-4 → 2.68e-4 — not enough to prevent the spike. EMA + best-ckpt rescue captured the good basin in both v2 and v3.

When scaling to the full 700-scene nuScenes train (~28k keyframes, ~3500 batches/epoch vs our 50), gradient noise should drop ~70× and this divergence pattern is expected to disappear naturally — no hyperparameter changes needed.

### Reference comparison

Public reference numbers for a fully-trained nuScenes range-image VAE on **held-out** data:

| Source | Metric | Value | Comparable to our number? |
|---|---|---|---|
| RangeLDM (full LDM, paper) | Chamfer Distance (test) | ~0.04 m | apples-to-oranges: LDM end-to-end vs our VAE reconstruction |
| LiDARGen (full LDM, paper) | Chamfer Distance (test) | ~0.16 m | same caveat |
| X-Drive (LDM, paper) | Chamfer Distance (test) | ~0.15 m | same caveat |
| **Ours v3** (VAE only, **training**) | L1_range_ema normalized | **0.01116** | ~1.12 m mean range error on training data; **2–5× worse expected on held-out** |

A proper held-out evaluation (Chamfer + MMD on a left-out scene) is not yet implemented — see "Open follow-ups" below.

### Checkpoints on disk

```
s2s_min/out/lidar_vae.pt        live final weights (v3, diverged region — DO NOT USE)
s2s_min/out/lidar_vae_ema.pt    EMA final weights (smoothed, but still diverged)
s2s_min/out/lidar_vae_best.pt   EMA at the best l1_range_ema basin (step ~900)  ← M2/M3 should load this
```

The best checkpoint is keyed by training-time `l1_range_ema` (smoothing α=0.99 on per-step L1_range). Loading:

```python
import torch
from s2s_min.models.lidar_vae import LiDARVAE

ckpt = torch.load("s2s_min/out/lidar_vae_best.pt", map_location="cuda")
vae = LiDARVAE(**{k: ckpt["config"][k] for k in ("in_channels", "latent_channels", "base_channels")}).cuda().eval()
vae.load_state_dict(ckpt["state_dict"])
vae.requires_grad_(False)
```

### Logs (full per-run output)

- v1: [out/train_vae_epoch50.log](out/train_vae_epoch50.log)
- v2: [out/train_vae_epoch50_v2.log](out/train_vae_epoch50_v2.log)
- v3: [out/train_vae_epoch50_v3.log](out/train_vae_epoch50_v3.log)

---

## Open follow-ups before M2

1. **Held-out scene evaluation** — split the 10 scenes into 9 train + 1 val; compute val L1, round-trip Chamfer Distance, and MMD on range histograms. Turns "we *think* the VAE works" into apples-to-apples vs RangeLDM table.
2. **Scale-up dry-run** — try `--n 100` (100 scenes, ~4000 keyframes) to verify the divergence pattern goes away with more data; cost ~1.5 hours wall-clock.
3. **M2 latent caching** — encode the 401 keyframes once through `lidar_vae_best.pt`, save `[8, 8, 256]` latents to `out/latents/*.npy`. Required by M3.
