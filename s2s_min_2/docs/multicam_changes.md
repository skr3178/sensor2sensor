# Single-camera → multi-camera (6-view nuScenes) — change summary

This doc lists the deltas applied to `s2s_min_2` (started as a copy of `s2s_min`)
to make the LiDAR diffusion U-Net condition on all six nuScenes surround
cameras instead of just `CAM_FRONT`.

The U-Net architecture itself is unchanged. Everything happens on the *input*
side: encode 6 cameras with the frozen SD VAE → build 6 per-camera raymaps →
pool → run a new `CrossViewFusion` module that mixes views via shared-projection
self-attention → token-concat the 6 grids along the W axis → feed the result to
the existing cross-attention path.

## Shape contract

```
6 × RGB                    [B, 6, 3, 256, 448]   (-1, 1)
6 × SD-VAE encode          [B, 6, 4, 32, 56]     (scaled by 0.18215)
6 × per-camera raymap      [B, 6, 6, 32, 56]
concat on channels         [B, 6, 10, 32, 56]
adaptive_avg_pool          [B, 6, 10, 8, 64]
CrossViewFusion            [B, 6, 10, 8, 64]     (shape preserved; views fused)
token-concat along W       [B, 10, 8, 384]        ← UNet kv_context
```

`unet.cross_attn_kv_channels` stays at **10**. Only the KV grid grew (8×64 → 8×384).

## Workspace fixes

- `s2s_min_2` started missing `models/`, `train/`, `scripts/`, `docs/`, `eval/`,
  `tests/` — rsynced from `s2s_min`.
- Added two symlinks pointing at the existing nuScenes root
  `/home/satya/skr/S2GO/S2GO/data/nuscenes` (full `v1.0-trainval`, 56 GB, all 6
  cams + LIDAR_TOP already present — no download needed):
  - [data/nuscenes_root](../data/nuscenes_root) — used by the new multi-cam loader
    and `configs/min.yaml`'s `nuscenes.root` field.
  - [nuscenes](../nuscenes) — matches the original `s2s_min` convention so legacy
    scripts that hardcode `Path("nuscenes")` (e.g. `train/smoke_test.py`,
    `scripts/benchmark_raymap.py`) keep working.
- Copied [out/subset_scene_tokens.txt](../out/subset_scene_tokens.txt) (10 scenes)
  and [out/lidar_vae_ema.pt](../out/lidar_vae_ema.pt) (frozen LiDAR VAE).
- Patched [scripts/download_image_vae.py](../scripts/download_image_vae.py) to
  resolve its output dir relative to *this* repo (`parents[1]` instead of `parents[2]`),
  then ran it to fetch the SD 1.5 VAE weights into
  [checkpoints/sd15_vae/](../checkpoints/sd15_vae/).
- Env deltas in `selfocc` conda env: `pip install diffusers>=0.30 safetensors huggingface_hub`.

## Files touched

### NEW

| File | What it does |
|---|---|
| [data/nuscenes_mini_paired.py](../data/nuscenes_mini_paired.py) | Paired LiDAR + 6-camera nuScenes loader. Walks `v1.0-trainval` metadata once, returns per-sample `{range_image, cams [V,3,256,448], cam_K [V,3,3] (scaled), cam_T_cam2ego [V,4,4], sample_token}`. Camera order fixed in `CAMERA_ORDER`. |
| [models/cross_view_fusion.py](../models/cross_view_fusion.py) | `CrossViewFusion(nn.Module)`. Up-projects `[B,V,10,H,W]` → `hidden_dim=64`, runs N pre-norm transformer blocks over the flattened `V*H*W` tokens (paper's flatten-concat-selfattn-split pattern), projects back to 10. Output projection is **zero-init** so the module is exactly identity at step 0 and a stable starting point for training. |
| [scripts/encoder_smoke_test_multicam.py](../scripts/encoder_smoke_test_multicam.py) | End-to-end shape verification: loads 2 paired samples → SD VAE encode → batched raymaps → pool → CrossViewFusion → token-concat → U-Net forward. Asserts every intermediate shape and confirms fusion-at-init is exact identity. |

### MODIFIED

| File | Change |
|---|---|
| [configs/min.yaml](../configs/min.yaml) | Added `nuscenes.{root,cameras}` block (6-camera list in canonical order). Restructured `kv_context` to `{channels:10, height:8, per_view_width:64, num_views:6, width:384}`. Added `cross_view_fusion: {num_layers:2, num_heads:4, hidden_dim:64, mlp_ratio:4}`. `unet.cross_attn_kv_channels` unchanged (still 10). |
| [models/image_encoder.py](../models/image_encoder.py) | Added `encode_views(rgb [B,V,3,H,W]) -> [B,V,4,H/8,W/8]` — flattens views into the batch dim for a single VAE call. Pre-imported `safetensors.torch` to dodge a diffusers lazy-import bug on this env. |
| [train/cache_latents.py](../train/cache_latents.py) | Rebuilt around `NuScenesPairedKeyframes`. Per sample now writes `img_latents [6,4,32,56]`, `raymaps [6,6,32,56]`, `mu [8,8,256]`, plus a `camera_order` array. Single VAE call encodes all 6 cams via `encode_views`; batched `build_raymap` produces all 6 raymaps in one shot (the existing `build_raymap` already supported a batch dim, so no change to `models/raymap.py`). Path roots switched from `s2s_min/out/...` to `out/...`. |
| [data/cached_latents.py](../data/cached_latents.py) | Reader returns `img_latents`/`raymaps` (plural) instead of `image_latent`/`raymap`. Default `cache_dir` switched from `s2s_min/out/cached_latents` to `out/cached_latents`. |
| [train/train_diffusion.py](../train/train_diffusion.py) | • `_collate` and `_train_one_batch` consume the new plural keys. • `_build_kv_context(img_latents, raymaps, fusion)` does `concat → pool per-view → CrossViewFusion → token-concat along W`. • Instantiates a `CrossViewFusion` alongside `LiDARUNet` and wraps both in an `nn.ModuleDict({"unet": ..., "fusion": ...})` so optimizer / EMA / state_dict / grad-clip cover both with no extra plumbing. • Probes `V` from the cache so different camera counts work without code changes. • New CLI flags: `--fusion_hidden_dim`, `--fusion_num_layers`, `--fusion_num_heads`, `--fusion_mlp_ratio`. • `_common_meta` records `num_views` + fusion config in the checkpoint. • Path roots updated from `s2s_min/out/...` to `out/...`. |

### UNCHANGED (worth noting)

- [models/unet.py](../models/unet.py) — `LiDARUNet` consumes the new
  `[B, 10, 8, 384]` KV context with no architecture change.
  `kv_channels=10` is unchanged.
- [models/raymap.py](../models/raymap.py) — already batched; the data loader and
  cache script call it with a flattened `[B*V, 3, 3]` / `[B*V, 4, 4]` batch.

## Verification (passed)

1. **Shape smoke test** — `PYTHONPATH=. python scripts/encoder_smoke_test_multicam.py` prints:
   ```
   range_image     [2, 3, 32, 1024]
   cams            [2, 6, 3, 256, 448]
   cam_K           [2, 6, 3, 3]
   cam_T_cam2ego   [2, 6, 4, 4]
   img_latents     [2, 6, 4, 32, 56]
   raymaps         [2, 6, 6, 32, 56]
   kv_full         [2, 6, 10, 32, 56]
   kv_pooled       [2, 6, 10, 8, 64]
   kv_fused        [2, 6, 10, 8, 64]
   kv_context      [2, 10, 8, 384]
   v_pred          [2, 8, 8, 256]
   fusion-at-init max|kv_fused - kv_pooled| = 0.00e+00
   ```

2. **Cache smoke test (3 samples)** — `PYTHONPATH=. python train/cache_latents.py --limit 3`:
   ```
   tensor_shapes: { img_latents: [6,4,32,56], raymaps: [6,6,32,56], mu: [8,8,256] }
   total_cache_mb: 0.84   (~0.28 MB / sample — ~6× the single-cam ~0.045 MB)
   ```

3. **One training step** — `PYTHONPATH=. python train/train_diffusion.py --overfit 3 --steps 1 --grad_accum 1 --batch_size 2`:
   ```
   views (V)              : 6  (kv_context W = 384)
   U-Net params           : 14.81 M
   CrossViewFusion params : 0.10 M
   [step 1 ...  vram=2252MiB  lr=5.00e-07]  mse=0.90705
   ```

## Scope C — Cross-Sensor Attention (paper §3.2.3)

Added in a follow-up pass. Opt-in via `--cross_sensor` on
`train/train_diffusion.py`. Default is still Scope B.

### Code changes

| File | Change |
|---|---|
| [models/attention.py](../models/attention.py) | New `CrossSensorSelfAttn(q_channels, kv_channels, num_heads)`. Drop-in replacement for `CrossAttention`: LayerNorm + `kv_to_d: Linear(kv_channels → q_channels)` to lift camera tokens to `d_i`, concat with LiDAR tokens, run shared-QKV self-attention over the unified sequence, slice back the LiDAR-token portion, residual. Output projection is zero-init so the block starts as exact identity. |
| [models/blocks.py](../models/blocks.py) | `EncoderLevel` accepts `use_cross_sensor`; picks `CrossSensorSelfAttn` vs `CrossAttention` at construction. |
| [models/unet.py](../models/unet.py) | `Bottleneck`, `DecoderLevel`, and `LiDARUNet` accept `use_cross_sensor` and propagate it to every level. |
| [train/train_diffusion.py](../train/train_diffusion.py) | New `--cross_sensor` flag; recorded in checkpoint meta; logged in the run header. |
| [scripts/encoder_smoke_test_multicam.py](../scripts/encoder_smoke_test_multicam.py) | Asserts both Scope B and C UNets produce `[B, 8, 8, 256]` and agree to 0 at init. |

### Mechanics (paper §3.2.3)

```
T_L^i = LiDAR feature map [B, d_i, H_L, W_L]   -> flatten ->            [B, K_L, d_i]
T_C^i = camera KV         [B, 10,  H_C, W_C]   -> flatten -> kv_to_d -> [B, K_C, d_i]
T_U^i = concat([T_L^i, T_C^i], dim=1)                                    # [B, K_L+K_C, d_i]
out   = self_attn(T_U^i, T_U^i)                                          # shared QKV
x    += out_proj(out[:, :K_L]).reshape_as(x)                             # LiDAR slice + residual
```

The camera-token slice is discarded after the attention — cameras stay frozen
conditioning. The paper's "shared `d_i` for both modalities" assumption is
approximated by the per-block `kv_to_d` linear, which lifts the 10-channel fused
KV grid to the LiDAR feature width at each level.

### Cost vs Scope B (1-step training smoke, B=2, V=6)

| | Scope B | Scope C |
|---|---|---|
| UNet params | 14.81 M | 16.23 M (+1.42 M) |
| VRAM        | 2.25 GiB | 6.57 GiB (~3×) |
| Step-1 MSE  | 0.90705 | 0.90720 (≈ identical — both zero-init) |

### Running

```
# Scope B (default):
PYTHONPATH=. python train/train_diffusion.py --overfit 3 --steps 1 --grad_accum 1

# Scope C:
PYTHONPATH=. python train/train_diffusion.py --overfit 3 --steps 1 --grad_accum 1 \
    --cross_sensor --checkpoint out/lidar_unet_scopeC.pt
```

Scope B and C checkpoints are not interchangeable (extra
`cross_attns.*.kv_to_d.*`, `norm_u.*`, `out_proj.*` keys per level in C).

## Open follow-ups (out of scope for this change)

- **Inference scripts** (`scripts/m3{1,2}_ddim_sanity.py`, `scripts/run_m4_demo.py`,
  `scripts/visualize_unet_forward.py`) still reference single-cam keys and load
  just the UNet state dict. They need: instantiate `CrossViewFusion`, load the
  new ModuleDict state dict, read `use_cross_sensor` from the checkpoint config
  and build the matching UNet, then assemble the multi-view KV context.
- **CFG dropout** currently zeros the *final* kv_context (post-fusion). Per-camera
  dropout pre-fusion is worth trying as an ablation.
- **Camera-order** — `CrossViewFusion` is permutation-equivariant, so the fixed
  `CAMERA_ORDER` doesn't matter for the math, but downstream visualisation
  scripts assume `[0]==CAM_FRONT`. Don't reshuffle without updating those.
- **Faithful Scope C** would have a learnable image-side tower whose features
  grow with `d_i` per level; ours approximates that with the per-block
  `kv_to_d` adapter.
