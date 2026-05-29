# Classifier-free guidance (CFG) — implementation plan

Source of this doc: discussion on 2026-05-29 about how to push `CD-3D-raw`
below 2 m without retraining. CFG was option **A** from the four
post-training items at that point.

## TL;DR

> **No retraining required.** The training-time investment was already made
> via `--cond_dropout 0.2` during the M3 bs16 run. CFG is a pure
> inference-time change to [`s2s_min/eval/decode_to_pointcloud.py`](../eval/decode_to_pointcloud.py),
> ~30 LOC of work.

## Why no retraining is needed

CFG requires the U-Net to have learned two distributions simultaneously:

- `p(z | conditioning)` — the conditional distribution
- `p(z)`               — the unconditional distribution

Our M3 bs16 training was launched with `--cond_dropout 0.2`. From
[`s2s_min/train/train_diffusion.py`](../train/train_diffusion.py) lines 249-252:

```python
# Classifier-free-guidance hook: per-sample, zero kv_context with prob cond_dropout.
if args.cond_dropout > 0:
    drop = (torch.rand(B, device=device) < args.cond_dropout)   # [B]
    kv_context[drop] = 0
```

For 20 % of training steps, the `kv_context` (image latent + raymap) was zeroed
before the U-Net forward. Across 50 epochs × 252 steps/epoch × ~20 % drop rate
≈ 2,500 unconditional steps. The trained weights know both modes already.

Verified in [`out/runs/2026-05-28_161242__m3-unet-v5cache-50ep-bs16/train.log`](../out/runs/2026-05-28_161242__m3-unet-v5cache-50ep-bs16/train.log):

```
cond dropout            : 0.2  (per-sample CFG hook)
```

## The CFG formula

```
ε_cond   = unet(z_t, t, kv_context)        # 80% mode the net learned
ε_uncond = unet(z_t, t, zeros)             # 20% mode the net learned
ε_cfg    = ε_uncond + w · (ε_cond − ε_uncond)
                       └── w = guidance scale; w > 1 amplifies the conditional pull
```

Typical `w` values across published work:

| `w` | Effect |
|---|---|
| 1.0 | vanilla conditional sampling (no guidance) |
| 1.5 | subtle sharpening — safe default if quality unknown |
| 3.0 | LDM / SD default; strong but rarely over-saturates |
| 5.0 | aggressive; useful when conditioning is weak |
| 7.5+ | over-saturation territory; artifacts in object regions |

We'll sweep `w ∈ {1.0, 1.5, 3.0, 5.0}` on first run and pick visually best.

## Two implementation options

### Option 1 — "Naive" (two sequential forward passes per DDIM step)

```python
with torch.no_grad():
    eps_cond   = unet(z, t, kv_context)
    eps_uncond = unet(z, t, torch.zeros_like(kv_context))
    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

- Wall-clock per step: ~1.0 s (2× over no-CFG, ~0.5 s)
- VRAM: unchanged
- Simplest to implement; lowest GPU efficiency

### Option 2 — "Batched" (one forward pass on a 2× batch)

Pattern cribbed from [Reference_code/LiDAR-Diffusion/lidm/models/diffusion/ddim.py:175-179](../../Reference_code/LiDAR-Diffusion/lidm/models/diffusion/ddim.py#L175-L179):

```python
with torch.no_grad():
    z_in  = torch.cat([z, z])                          # [2B, ...]
    t_in  = torch.cat([t, t])
    kv_in = torch.cat([torch.zeros_like(kv_context), kv_context])
    eps_uncond, eps_cond = unet(z_in, t_in, kv_in).chunk(2)
    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
```

- Wall-clock per step: ~0.7 s (1.4× over no-CFG)
- VRAM: ~+15 % (2× batch in U-Net)
- One kernel launch instead of two — better SM utilization at our small batch size

**Both produce numerically identical outputs.** The choice is purely a
CPU/GPU/VRAM trade-off. Recommendation: **Option 2** (batched) — same
quality, ~30 % faster on RTX 3060.

## Reference implementations across our `Reference_code/` clones

| Reference | Has CFG? | Lines | Notes |
|---|---|---|---|
| **LiDAR-Diffusion** | ✓ | [`ddim.py:175-179`](../../Reference_code/LiDAR-Diffusion/lidm/models/diffusion/ddim.py#L175) | Closest match (same modality). Recommended template. |
| **MVDream** | ✓ | [`ddim.py:200`](../../Reference_code/MVDream/mvdream/ldm/models/diffusion/ddim.py#L200) | Same batched pattern. |
| **diffusers** | ✓ | [`pipeline_stable_diffusion.py:1055`](../../Reference_code/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1055) | Canonical one-liner. |
| **X-Drive** | ✓ (via diffusers) | — | Uses diffusers internally. |
| **RangeLDM** | ✗ | — | Public code only has training; inference + CFG unreleased. |
| **diffusion** (Ho 2020 DDPM) | ✗ | — | **Pre-dates CFG.** CFG was Ho 2021. |
| **guided-diffusion** | ✗ | — | Uses **classifier guidance** (CG, predecessor). CFG simplified CG away. |

## Implementation checklist

- [ ] Add `ddim_sample_cfg(unet, shape, kv_context, cfg_scale, ...)` to
      [`s2s_min/models/diffusion.py`](../models/diffusion.py) — ~25 LOC, batched variant
- [ ] Add `--cfg_scale` flag to [`s2s_min/eval/decode_to_pointcloud.py`](../eval/decode_to_pointcloud.py),
      plumb through to the sampler — ~5 LOC
- [ ] Extend [`s2s_min/scripts/run_m4_demo.py`](../scripts/run_m4_demo.py) to optionally
      run a sweep `cfg_scale ∈ {1.0, 1.5, 3.0, 5.0}` → 4 oblique grids per call,
      each in its own m4_eval timestamped folder
- [ ] Pick visually best `w` and update the canonical `run_m4_demo.py` default
- [ ] Re-run on the 16-sample held-out set → measure new `CD-3D-raw`

## Expected impact (predictions to verify)

| Metric | Current (no CFG) | After CFG | Predicted Δ |
|---|---|---|---|
| `CD-3D-raw` | 2.698 ± 0.318 m | **~2.0–2.4 m** likely | -10 to -30 % |
| `CD-3D-oracle` | 2.540 m | ~1.8–2.2 m | similar reduction |
| `cos(z_pred, μ)` | 0.317 | ~0.40–0.55 | sharper alignment |
| Per-scene differentiation in oblique | present but blurry | crisper, more distinct | qualitative |
| Wall-clock per scan | 0.5 s | ~0.7 s (batched) / 1.0 s (naive) | — |
| Risk | n/a | over-saturation at high w (5+) | mitigated by w sweep |

If CFG produces 2.0 m `CD-3D-raw` and we still want lower:
- Combine with **C** (warm-start +25 epochs) → likely sub-1.8 m
- Implement Scope-B (6-camera input + paper-faithful cross-sensor attn) → likely sub-1 m
