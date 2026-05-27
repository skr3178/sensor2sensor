"""Download the Stable Diffusion 1.5 VAE checkpoint into the project-local
`s2s_min/checkpoints/sd15_vae/` directory, instead of the global HuggingFace
cache (`~/.cache/huggingface/`).

This keeps the checkpoint co-located with the code and makes the project
self-contained on a fresh clone (after re-running this script).

Run once:
    env/bin/python s2s_min/scripts/download_image_vae.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = REPO_ROOT / "s2s_min" / "checkpoints" / "sd15_vae"

# Primary: original SD 1.5 repo. Only the fp16 safetensors variant + config —
# that's all we need (we run encoder in fp16 anyway). Skip the redundant fp32 and
# pickle .bin files (~840 MB saved).
# Fallback: stabilityai standalone VAE (same kl-f8-d4 family, MIT license, fully
# on HF; works identically for us since we never call .decode()).
PRIMARY = (
    "runwayml/stable-diffusion-v1-5",
    ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"],
    "vae",
)
FALLBACK = (
    "stabilityai/sd-vae-ft-mse",
    ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
    None,
)


def _try_download(repo_id: str, allow_patterns: list[str] | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=str(LOCAL_DIR),
    )


def main() -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"target: {LOCAL_DIR}")

    repo, patterns, _ = PRIMARY
    try:
        print(f"trying primary: {repo}  patterns={patterns}")
        _try_download(repo, patterns)
        print("primary OK")
        return 0
    except Exception as e:
        print(f"primary failed: {e}")
        print(f"trying fallback: {FALLBACK[0]}")
        _try_download(FALLBACK[0], FALLBACK[1])
        print("fallback OK — checkpoint layout is flat (no 'vae/' subfolder)")
        print("NOTE: image_encoder.py must load via")
        print("      AutoencoderKL.from_pretrained(<dir>)  # no subfolder kwarg")
        return 0


if __name__ == "__main__":
    sys.exit(main())
