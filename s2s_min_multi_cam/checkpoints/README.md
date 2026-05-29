# Project-local pretrained checkpoints

This folder holds external pretrained model checkpoints that the minimum pipeline depends on. Kept here (rather than in `~/.cache/huggingface/`) so the project is self-contained — clone the repo, run the download script, work.

Binary weight files are **git-ignored** (see project root `.gitignore`) to keep the repo small. Re-download them with the scripts under [`s2s_min/scripts/`](../scripts/).

## Contents

| Folder | Source | Size | Re-download command |
|---|---|---|---|
| `sd15_vae/` | `runwayml/stable-diffusion-v1-5` (vae subfolder only, fp16.safetensors variant only) | ~160 MB | `env/bin/python s2s_min/scripts/download_image_vae.py` |

## sd15_vae layout

```
sd15_vae/
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.fp16.safetensors   # ← git-ignored
```

Loaded by `s2s_min/models/image_encoder.py` (when written in M0) via:

```python
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "s2s_min/checkpoints/sd15_vae",
    subfolder="vae",
    variant="fp16",
    torch_dtype=torch.float16,
)
```

All three of those paths/strings live in [`configs/min.yaml`](../configs/min.yaml) under the `image:` block — never hard-code them in module files.

## Why local instead of `~/.cache/huggingface/`

- The repo becomes self-contained after one `download_*.py` run. No dependency on the user's global HF cache state.
- Easier to nuke and re-fetch a single checkpoint without affecting other projects.
- Path conventions are explicit and live in version control (via `configs/min.yaml`).
- Cache-hit behaviour is deterministic for our pipeline.

## What's NOT here

- The **trained LiDAR VAE** checkpoint (`s2s_min/out/lidar_vae.pt`) lives under `out/`, not here. `out/` is for *artifacts the project produces* (training outputs); `checkpoints/` is for *external dependencies*.
- The trained diffusion U-Net checkpoint (M3 output) will also go to `out/`.
