# Image VAE encoder — candidate review and channel-count analysis

The paper specifies an "Image VAE [10]" (CAT3D-family, 8-channel latent) but never released a checkpoint. This doc captures the candidates we evaluated for the minimum pipeline, the analysis of how much the channel count actually matters in our use case, and the final choice with a clear swap path.

Decision: **stay on SD 1.5 VAE (4-channel)** for v1. FlexTok c8 is the recommended upgrade target if M3 conditioning quality turns out to be the bottleneck.

---

## 1. Candidates evaluated

All evaluated against requirements:
- HuggingFace diffusers loadable (one-line swap with the current wrapper)
- 8× spatial compression (to land at `[32, 56]` from `[256, 448]`)
- Encoder-only use (never call `.decode()`)
- Permissive enough license for research

| Checkpoint | Channels | Domain | License | Diffusers? | Verdict |
|---|---|---|---|---|---|
| **SD 1.5 VAE** ([`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)) | **4** | LAION (incl. driving photos) | CreativeML Open RAIL-M | ✅ | **Current default.** Battle-tested, broad LAION coverage including driving imagery, ~84 MB. |
| **FlexTok c8** ([`EPFL-VILAB/flextok_vae_c8`](https://huggingface.co/EPFL-VILAB/flextok_vae_c8)) | **8** ✓ | undocumented but the FlexTok paper targets natural-image tokenization → almost certainly ImageNet/photographic | Apple Model License (research only) | ✅ | **Best paper-faithful drop-in.** Channel count matches paper; domain plausibly photographic; license is research-only which fits our scope. |
| **LibreVAE f8-d8** ([`scrumptious/librevae-f8-d8`](https://huggingface.co/scrumptious/librevae-f8-d8)) | **8** ✓ | anime/artwork (e621-2024, danbooru2023) | CC0 | ✅ | **Domain mismatch is fatal.** Encoder learned features for line art, flat shading, character poses — not photographs of roads. Channel match doesn't compensate. |
| **ostris vae-kl-f8-d16** ([`ostris/vae-kl-f8-d16`](https://huggingface.co/ostris/vae-kl-f8-d16)) | 16 | mix incl. photos | MIT | ✅ | Right domain, wrong channel count. Closer to SD 3 than the paper. Doubles cross-attn KV input dim. |
| **WF-VAE-S** | 8 | undocumented | undocumented | ? | "S" variant doesn't appear to be on HuggingFace; only WF-VAE-L-16Chn is up. Couldn't evaluate. |
| **SD 3 / SD 3.5 VAE** ([`stabilityai/stable-diffusion-3-medium`](https://huggingface.co/stabilityai/stable-diffusion-3-medium)) | 16 | LAION-quality | research/commercial mix | ✅ | Same shape problem as ostris d16. Slicing 16 → 8 channels breaks the latent distribution. |
| **Flux VAE** | 16 | LAION-quality | non-commercial | ✅ | Same as SD 3 — 16-channel, latent space designed for that width. |
| **CAT3D VAE** (paper ref [10]) | 8 | undisclosed | **not released** | ❌ | Project page at [cat3d.github.io](https://cat3d.github.io/) has no code or weights. No DeepMind GitHub repo. Currently unobtainable. |
| **Sensor2Sensor's actual VAE** | 8 | Waymo internal | **not released** | ❌ | Proprietary. |

---

## 2. Where the channel count actually enters our pipeline

```
image_encoder.encode(rgb)  → [B, C_img, 32, 56]            C_img ∈ {4, 8}
                                    │
              cat with raymap       │  (raymap is always 6 channels)
                                    ▼
                          kv_full   [B, C_img + 6, 32, 56]   = [B, 10 or 14, 32, 56]
                                    │
              adaptive_avg_pool     ▼
                          kv_context [B, C_img + 6, 8, 64]   = [B, 10 or 14, 8, 64]

   inside every U-Net CrossAttention block:
       K = Linear(C_img + 6 → 384) @ kv_context
       V = Linear(C_img + 6 → 384) @ kv_context

   from this point on, every tensor is in 384-dim attention space.
```

After the K/V `Linear → 384` projection, the U-Net has no notion of how many channels the image-side latent had. The bottleneck dim is 384 either way.

---

## 3. Why the paper needs 8 channels but we don't

| Paper | Our minimum pipeline |
|---|---|
| Image VAE encodes **8 generated views simultaneously** — tokens must carry rich content for downstream reconstruction | Single CAM_FRONT, one viewpoint, much less content per latent |
| Image VAE **decoder also runs** (image-generation tower outputs images) → reconstruction fidelity matters → 8 channels helps the decoder | We **never call `.decode()`** — reconstruction fidelity is irrelevant |
| Image latents are full-rank **tokens in cross-sensor self-attention** (`flatten + concat + shared-proj self-attn`) | Image latents are **dim-projected to 384** inside our one-way `CrossAttention` before the U-Net sees them |
| Trained at Google-DeepMind scale, can empirically measure gains from extra channels | Trained on 10 scenes / ~400 samples on a 3060 — every gain has a cost |

---

## 4. Ranked impact on output quality

What actually limits the quality of our generated LiDAR point clouds, in descending order:

| Rank | Factor | Estimated impact | Cheap to fix in this pipeline? |
|---|---|---|---|
| 1 | **LiDAR VAE reconstruction quality** | Hard ceiling — generated LiDAR can't be better than what the VAE can decode from a real-source latent | requires retraining or scaling the VAE |
| 2 | **Cross-attn KV spatial pooling** (`(32, 56) → (8, 64)`) | Loses small/distant object information from the image | revisit pooling grid in M3 if conditioning is weak |
| 3 | **U-Net capacity** (~30 M params) | Caps what the model can learn from the latents | scale up if VRAM allows |
| 4 | **Training data volume** (10 scenes / ~400 samples) | Limits generalization, not the architecture's ceiling | promote to all 850 scenes |
| 5 | **Image VAE training-domain match** (LAION photos vs anime vs unspecified) | Determines whether the channels we get are *useful* channels for driving imagery | swap VAE checkpoint |
| 6 | **Cross-attn embedding dim** (384) | Caps info-per-token through attention | configurable, costs VRAM |
| **7** | **Image VAE channel count** (4 vs 8 vs 16) | **Smallest effect** — washed out by the K/V linear projection | trivial config swap |

**Domain match (#5) likely matters more than channel count (#7).** A 4-channel SD VAE trained on driving imagery beats an 8-channel anime VAE for our use case.

---

## 5. Estimated end-to-end impact of going 4 → 8 channels

Held constant: same training data, same U-Net, same training budget.

| Metric | Estimated change | Why |
|---|---|---|
| **Chamfer distance** | ±2–5 % (likely within training-seed variance) | The K/V projection bottlenecks information either way |
| **Conditioning fidelity** at small distant objects | Slightly better, but spatial pooling dominates | More channels carry more info per spatial token |
| **FID-equivalent for LiDAR** | Negligible | LiDAR quality is gated by the LiDAR VAE, not the conditioning |
| **Generalization to unseen scenes** | Negligible | Driven by training data volume, not encoder channels |
| **VRAM** | +~10 MB activation memory | Trivial |
| **Wall-clock** | <1 % change in M3 step time | Trivial |

This is a structural argument, not an empirical measurement. The paper does not ablate channel count.

---

## 6. Decision

**Stay on SD 1.5 VAE (4-channel) for v1.**

| Reason | Detail |
|---|---|
| **Domain match** | LAION has extensive driving imagery; SD 1.5's encoder produces well-distributed features on dashcam frames out of the box |
| **Channel count is the smallest knob** | Per §4, ranked #7 — below the noise floor of the other limitations we've already accepted (~30 M params, 10 scenes, `8×64` KV pooling) |
| **Production-friendly license** | CreativeML Open RAIL-M is more permissive than Apple Research |
| **Community-validated** | Thousands of downstream uses; well-understood pitfalls |
| **One-line swap path stays open** | If M3 hits a wall attributable to conditioning quality, switch to FlexTok c8 in ~5 LOC |

---

## 6.5 Caveat — the calculus changes under scope (B)

The "channel count is ranked #7" argument in §4 leans on a specific bottleneck: in scope (A), the image latent passes through **one** linear projection (`Linear(10 → 384)` inside `CrossAttention`) before everything becomes 384-dim. That projection sits right next to the image-VAE output, so channel count is one matmul away from being absorbed.

Under [scope (B)](../../min_pipeline_plan.md#b-6-camera-input--paper-faithful-cross-sensor-self-attn-lidar-only-output--deferred-follow-on-after-m3-passes) the bottleneck moves later:

```
6 cameras → 6 image latents [B, C_img, 32, 56]
   │
   ▼ flatten + concat across views
   │
   ▼ CROSS-VIEW FUSION (self-attn, SHARED QKV projection)
   │   ← Q, K, V all derived from C_img-dim tokens.
   │     Inter-camera info flow is gated by C_img, not by a downstream 384-dim cap.
   │
   ▼ flatten + concat with LiDAR tokens
   │
   ▼ CROSS-SENSOR SELF-ATTN (shared QKV, paper-faithful)
   │   ← still operating at full input channel width
   │
   ▼ ...eventually a 384-dim projection
```

In scope (A), `C_img ∈ {4, 8}` matters only through the single `Linear(10 → 384)` projection — one matmul.

In scope (B), `C_img` matters through **two attention blocks** as the token width before any 384-dim projection. The information that flows between cameras during cross-view fusion is at `C_img` width; the information that flows between modalities during cross-sensor self-attn is at `max(C_img, C_lidar)` width.

### How the ranking from §4 shifts under scope (B)

| Rank in scope (A) | Factor | New rank in scope (B) |
|:-:|---|:-:|
| 1 | LiDAR VAE quality | 1 (unchanged) |
| 2 | KV spatial pooling | 2 (unchanged) |
| 3 | U-Net capacity | 3 (unchanged) |
| 4 | Training data volume | 4 (unchanged) |
| 5 | Image VAE domain match | 5 (unchanged) |
| 6 | Cross-attn embedding dim | 6 (unchanged) |
| **7** | **Image VAE channel count** | **moves up to ~5, on par with domain match** |

### Practical implication

**Do not swap the VAE before adopting scope (B).** In scope (A) the swap is wasted effort — the 4 vs 8 channel difference is washed out by the K/V projection. In scope (B) the swap is properly motivated — channel count materially gates information flow through two attention blocks.

The two changes are naturally bundled:

- Same motivation (paper fidelity)
- Same swap path (the §7 diff applies unchanged)
- One M5 deviations-table update covers both

If/when scope (B) is committed, the recipe is: (1) wire the 6-camera data loader, (2) add the cross-view-fusion and cross-sensor-self-attn blocks, (3) swap SD 1.5 → FlexTok c8 in the same commit, (4) re-run M-1 shape tests with `C_kv = 14`.

### Worthless? No

SD 1.5 + 6 cameras still works — the encoder still produces good photographic features per view, cross-view fusion still learns inter-view consistency at 4 channels per token. You just give up real (though still not dominant) capacity. Calling it "worthless" overstates the gap; "suboptimal in scope (B)" is the honest framing.

---

## 7. Swap path (if we ever switch)

To migrate from SD 1.5 (4-channel) to FlexTok c8 (8-channel):

```diff
# s2s_min/configs/min.yaml
 image:
   height: 256
   width: 448
   sd_vae_downsample: 8
-  latent_channels: 4
+  latent_channels: 8

 kv_context:
-  channels: 10        # image_latent.channels + raymap.channels = 4 + 6
+  channels: 14        # image_latent.channels + raymap.channels = 8 + 6

 unet:
-  cross_attn_kv_channels: 10
+  cross_attn_kv_channels: 14
```

```diff
# s2s_min/models/image_encoder.py
- vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
+ vae = AutoencoderKL.from_pretrained("EPFL-VILAB/flextok_vae_c8")
```

Then re-run the M-1 [`test_shapes.py`](../tests/test_shapes.py) with the new KV channel count to confirm `CrossAttention` dims still work.

If the swap is made, update the M5 deviations table in [`min_pipeline_plan.md`](../../min_pipeline_plan.md) accordingly.

---

## 8. Decision triggers — when to revisit

Revisit the choice if any of these emerge in M3 or M4:

| Trigger | Why it points at channel count / domain |
|---|---|
| Generated LiDAR mostly ignores the input image (objects in different spatial positions vs. ground truth) | Conditioning signal too weak — could be channels, could be pooling, could be image-VAE domain mismatch. Test pooling grid first (cheaper), then VAE swap. |
| Specific object classes consistently missed (e.g. distant cars, pedestrians) | Spatial resolution issue dominates channel-count issue. Address pooling first. |
| Training loss plateaus suspiciously high despite adequate data | Encoder features may not be informative enough. Try FlexTok c8. |
| Per-class quality varies a lot (cars great, pedestrians awful) | Likely a U-Net capacity or data issue, not the image VAE. |

If none of these surface, the SD 1.5 choice was correct.
