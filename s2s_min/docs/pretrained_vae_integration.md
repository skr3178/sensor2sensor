# Building the pipeline around a pretrained LiDAR VAE

This doc covers the case where M1 is **skipped** because a working pretrained LiDAR VAE for nuScenes is already available. The rest of the pipeline (M2 → M5) is built to consume whatever the VAE produces, regardless of provenance.

If you're training the VAE from scratch instead, see [`models.md`](models.md) §2.

---

## 1. Anchor the pipeline on the VAE's contract, not the other way around

The minimum pipeline has a *spec* in [`configs/min.yaml`](../configs/min.yaml):

| Field | Current spec value | Authority |
|---|---|---|
| `lidar.channels` | 3 (range, intensity, validity) | aspirational |
| `lidar.range_clamp_m` | 100 | aspirational |
| `lidar.latent_channels` | 8 | aspirational |
| Latent spatial size | `8 × 256` | aspirational |

When a pretrained VAE arrives, **the VAE is the authority** — these spec values get overwritten by what the checkpoint actually produces. The pipeline then conforms.

The reason: the VAE's latent space *is* the diffusion space. Mis-matching it would force you to retrain anyway.

---

## 2. Inspect the checkpoint before writing anything else

First step on day-one: write `scripts/probe_lidar_vae.py` that loads the checkpoint and prints its contract. Without this you risk wiring the U-Net to assumptions that don't hold.

```python
# scripts/probe_lidar_vae.py — ~30 LOC
"""Loads the pretrained LiDAR VAE and prints everything downstream needs to know."""
import torch
from pathlib import Path

CKPT = Path("/path/to/your/lidar_vae.pt")            # set by user
state = torch.load(CKPT, map_location="cpu")

# Discover via probe with a dummy 3-, 2-, or 1-channel input:
for n_ch in (1, 2, 3, 4):
    x = torch.zeros(1, n_ch, 32, 1024)
    try:
        z = vae.encode(x).latent_dist.sample()       # diffusers convention
        print(f"  in_channels={n_ch}  -> latent {tuple(z.shape)}")
        x_hat = vae.decode(z).sample
        print(f"                   <- recon {tuple(x_hat.shape)}")
        break
    except Exception as e:
        print(f"  in_channels={n_ch}  failed: {e}")
```

**Required output before proceeding:**
- input channel count `C_in` (typically 2 for RangeLDM, 3 if it's a custom retrained one)
- latent shape `(C_lat, H_lat, W_lat)` (e.g. `(4, 4, 128)` or `(8, 8, 256)`)
- expected input normalization (range mean/std, intensity scaling)
- whether the VAE uses circular padding (matters if we wrap our own data preprocessing)

Persist that as `s2s_min/out/lidar_vae_contract.yaml` so every downstream module reads from one place.

---

## 3. Adapter interface

Wrap whatever the checkpoint is in a single class with this exact API, so M2/M3/M4 stay agnostic.

```python
# s2s_min/models/lidar_vae_adapter.py
class LiDARVAEAdapter(nn.Module):
    """Frozen wrapper around a pretrained LiDAR VAE.

    Hides:
      - normalization of inputs into whatever range the underlying VAE expects
      - de-normalization of outputs back into the canonical [range_m, intensity, validity] space
      - the shape of the latent (exposed as .latent_shape for the U-Net to query)
    """
    in_channels: int                   # set from probe
    latent_shape: tuple[int, int, int] # (C_lat, H_lat, W_lat), e.g. (4, 4, 128)

    def encode(self, range_image: torch.Tensor) -> torch.Tensor:
        """range_image : [B, C_in, 32, 1024] in canonical [0, range_clamp_m] units, validity ∈ {0,1}
        returns z : [B, C_lat, H_lat, W_lat]"""

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z : [B, C_lat, H_lat, W_lat] -> range_image [B, C_in, 32, 1024] in canonical units"""

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...
    @torch.no_grad()
    def denormalize(self, x: torch.Tensor) -> torch.Tensor: ...
```

Backends to implement against this interface:

| Backend | Class | When to use |
|---|---|---|
| RangeLDM (2-ch, X-Drive-style) | `RangeLDMAdapter` | If using the [WoodwindHu/RangeLDM](https://github.com/WoodwindHu/RangeLDM) checkpoint, loaded via X-Drive's recipe in [`pc_ldm_box_RangeLDM_runner.py:76`](../../Reference_code/X-Drive/xdrive/runner/pc_ldm_box_RangeLDM_runner.py#L76) |
| User-supplied custom | `CustomVAEAdapter` | Generic loader; the user passes a checkpoint path and probed contract |
| From-scratch (M1 trained) | `FromScratchAdapter` | Wraps the [`LiDARVAE`](../models/lidar_vae.py) we would have trained in M1 |

All three implement `encode`, `decode`, `normalize`, `denormalize`. The rest of the pipeline only ever sees the adapter.

---

## 4. Downstream changes per milestone

### M0 — smoke test

Already designed to use whatever VAE is present. Change:
- In `train/smoke_test.py`, instantiate `LiDARVAEAdapter` (probed shape) instead of a randomly-initialized stub.
- Update `LiDARUNet` construction to read `latent_shape` from the adapter at runtime, not from a hard-coded YAML.

### M1 — VAE training

**Skipped.** Replace with `M1-load`:
1. Load checkpoint into the adapter.
2. Run `vae.eval(); vae.requires_grad_(False)`.
3. Sanity check: encode a known sample, decode, compute Chamfer between the original point cloud and the round-tripped one. Acceptable threshold: < 0.5 m mean per-point error on a held-out frame. If this fails, the checkpoint is unusable as-is and we fall back to either:
   - re-train (return to original M1 plan), or
   - fine-tune just the input/output convs to bridge the channel mismatch (sub-path 2 from the earlier discussion).
4. Persist `lidar_vae_contract.yaml`.

### M2 — latent caching

This is where the adapter pays off. For every sample in the 10-scene subset:
```
range_image = point_cloud_to_range_image(pc)
z_lidar     = adapter.encode(range_image)          # whatever shape it is
np.savez(out_dir / f"{sample_token}.npz", z=z_lidar.cpu().numpy(), ...)
```
The cached `.npz` files carry the adapter's latent shape; the rest of the pipeline reads from these without ever calling the VAE encoder again. Saves wall-clock and VRAM in M3.

### M3 — conditional diffusion

The U-Net is built from `adapter.latent_shape`:
```python
unet = LiDARUNet(
    in_channels  = adapter.latent_shape[0],
    h_latent     = adapter.latent_shape[1],
    w_latent     = adapter.latent_shape[2],
    ...
)
```
Two things may need to change vs. the current U-Net spec (`[B, 8, 8, 256]`):

- **Latent channel count.** If the adapter produces e.g. 4 channels instead of 8, the stem and head convs change; everything in between is unchanged. Trivial.
- **Latent spatial size.** If the adapter produces e.g. `4 × 128` instead of `8 × 256`, the U-Net's W-only downsample chain needs one fewer level (`128 → 64` only, no `64 → 32` because that's the bottleneck). The 25–35 M param target shrinks proportionally. Re-cross-check VRAM after the adjustment.

A `LiDARUNet.from_adapter(adapter, cfg)` constructor that auto-sizes itself is worth ~30 LOC and removes hard-coded values from `configs/min.yaml`.

### M4 — inference / visualization

```
z_pred = ddim_sample(unet, noise, image_kv, steps=25)
range_image = adapter.decode(z_pred)                # canonical units
pc = range_image_to_pc(range_image)                 # see eval/decode_to_pointcloud.py
```
Unchanged logic — just talks to the adapter.

### M5 — documentation

The deviations table now must record:

| Deviation | Reason | Impact |
|---|---|---|
| VAE not trained from scratch | Pretrained checkpoint reused | M1 wall-clock saved (~1 day) |
| LiDAR channel count = `<probed value>` (likely 2) | Pretrained model dictates | Validity-as-latent-channel lost if `C_in < 3`. Validity must be re-introduced as auxiliary supervision or dropped entirely. |
| Latent shape = `<probed value>` | Pretrained model dictates | U-Net topology adjusted to match. |
| Normalization scheme | Inherited from pretrained model | Range/intensity reconstructed in adapter's native units, denormalized at the boundary. |

---

## 5. Plumbing checklist (do this in order)

1. [ ] Place the checkpoint at a known path. Note: HuggingFace `diffusers.AutoencoderKL`-format expects `vae/config.json` + `vae/diffusion_pytorch_model.safetensors` (the X-Drive layout); plain `torch.save({"state_dict": ...})` works too if loader knows about it.
2. [ ] Write `scripts/probe_lidar_vae.py` (~30 LOC) → produces `out/lidar_vae_contract.yaml`.
3. [ ] Write `models/lidar_vae_adapter.py` with the abstract base class + the specific backend matching the checkpoint (~120 LOC for one backend).
4. [ ] Round-trip Chamfer test on 10 held-out samples; pass criterion mean error < 0.5 m.
5. [ ] Update `configs/min.yaml` to pull `lidar.latent_channels`, `lidar.latent_h`, `lidar.latent_w` from the probed contract (or remove them and read at runtime).
6. [ ] Refactor `LiDARUNet` to accept latent shape from constructor args (no hard-codes).
7. [ ] Re-run the M-1 [`test_shapes.py`](../tests/test_shapes.py) with the probed shape to confirm cross-attention dims still work. Likely need to update test constants.
8. [ ] Proceed to M2 (caching), M3 (diffusion training), M4 (inference).

---

## 6. Watch-outs

| Risk | What to check |
|---|---|
| Checkpoint expects different range normalization than our data pipeline produces | Compare `adapter.normalize(known_input)` output stats against the VAE's training data stats (often documented in the source repo). Off-by-50m bias is a common bug. |
| Checkpoint expects circular conv that our preprocessing doesn't preserve | If `point_cloud_to_range_image` zero-pads or wraps wrongly, the seam at column 0 will reconstruct poorly. Visualize a known sample's seam region. |
| Latent scaling factor (cf. SD VAE's 0.18215) | RangeLDM-style VAEs often have a `scaling_factor` in `vae/config.json` that you must multiply the latent by before feeding to the U-Net (and divide after sampling). Read it from config. |
| Latent values are extremely large or tiny | If μ values are O(100) after encode, the diffusion noise schedule won't behave. Either rescale via `scaling_factor` (preferred) or normalize the latent to ~unit variance before diffusion. |
| KL term in the checkpoint's training drift | If the pretrained VAE is poorly regularized, the latent will not look like a standard normal. The diffusion model will still learn, but `v_prediction` may need a different `beta_schedule`. |

---

## 7. If you don't have a specific checkpoint in mind

Concrete plug-and-play option: **RangeLDM pretrained nuScenes VAE**, exactly the way X-Drive uses it.

- Download from the link in [Reference_code/X-Drive/README.md:81](../../Reference_code/X-Drive/README.md#L81) (Google Drive folder `1rP0_YNgBn-...`)
- Place at `/media/skr/storage/self_driving/sensor2sensor/Reference_code/X-Drive/pretrained/RangeLDM-nuScenes/vae/`
- The folder layout will be `vae/config.json` + `vae/diffusion_pytorch_model.safetensors` (HF diffusers format)
- Probed contract (expected, to be verified after download):
  - input channels: 2 (range + intensity)
  - latent shape: likely `(4, 4, 128)` based on SD-style `block_out_channels=(128, 256, 512, 512)` at 32×1024 input with 8× downsample
  - normalization: `range = (range - 50) / 50`, `intensity = intensity / 255`
  - scaling_factor: in `vae/config.json`, typically `0.18215` (inherited from SD) — verify after download
- Backend class: `RangeLDMAdapter`, copy the loader pattern from [`pc_ldm_box_RangeLDM_runner.py:76`](../../Reference_code/X-Drive/xdrive/runner/pc_ldm_box_RangeLDM_runner.py#L76)
- Validity channel: not provided. Either drop validity from the pipeline entirely or carry it as an auxiliary mask alongside the latent (X-Drive's approach).

If you have a *different* checkpoint, replace step 7 with the actual filename + step 2 (probe) tells you everything else.
