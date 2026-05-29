"""Pre-encode DINOv3-small features for every cached sample (companion to cache_latents.py).

The SD-VAE depth probe showed SD-VAE features carry no depth; DINOv3-small features do
(img-probe r=0.956). This builds a companion cache of frozen DINOv3 features keyed by the
SAME sample tokens as an existing latent cache, so the diffusion dataloader can load
DINOv3 features alongside the existing raymap + LiDAR `mu` (no re-encoding of those).

Stored per sample: `feat [384, 14, 24]` float16 — the DINOv3 ViT-S/16 patch grid (224×384
input → 14×24 patches), prefix tokens (CLS + 4 registers) stripped. Upsample to the 32×56
latent grid at train time (cheap). float16 + patch-grid keeps the cache ~1 GB for 4023 samples.

Run:
    HF_HUB_OFFLINE=1 env/bin/python s2s_min/train/cache_dinov3.py \
        --ref_cache s2s_min/out/cached_latents_v5_100scenes \
        --out_dir   s2s_min/out/cached_dinov3_v5_100scenes
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

NUSCENES_ROOT = Path("nuscenes")
META = NUSCENES_ROOT / "v1.0-trainval"
DINOV3_MODEL = "vit_small_patch16_dinov3.lvd1689m"
HP, WP = 224, 384            # /16 -> 14x24 patch grid, aspect ~0.583 (≈256/448)
GH, GW = HP // 16, WP // 16


def cam_filenames_for(tokens):
    """Map each sample_token -> CAM_FRONT keyframe filename (lean nuScenes index, no ego_pose)."""
    cs = {c["token"]: c for c in json.loads((META / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((META / "sensor.json").read_text())}
    want = set(tokens)
    out = {}
    sd = json.loads((META / "sample_data.json").read_text())
    for r in sd:
        if not r["is_key_frame"] or r["sample_token"] not in want:
            continue
        if sensor[cs[r["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT":
            out[r["sample_token"]] = r["filename"]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cache", type=Path, default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/cached_dinov3_v5_100scenes"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokens = sorted(p.stem for p in args.ref_cache.glob("*.npz"))
    if args.limit:
        tokens = tokens[:args.limit]
    print(f"ref cache: {args.ref_cache}  ({len(tokens)} tokens)")

    print("indexing nuScenes for CAM_FRONT filenames ..."); t = time.time()
    fn = cam_filenames_for(tokens)
    print(f"  matched {len(fn)}/{len(tokens)} ({time.time()-t:.0f}s)")

    import timm
    model = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0).to(device).eval()
    npfx = getattr(model, "num_prefix_tokens", 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    print(f"  {DINOV3_MODEL}  prefix_tokens={npfx}  dim={model.embed_dim}")

    todo = [tk for tk in tokens if tk in fn and (args.overwrite or not (args.out_dir / f"{tk}.npz").exists())]
    print(f"to encode: {len(todo)}")
    t0 = time.time(); n_written = 0
    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        ims = []
        for tk in chunk:
            im = Image.open(NUSCENES_ROOT / fn[tk]).convert("RGB").resize((WP, HP), Image.BICUBIC)
            ims.append(torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
        x = (torch.stack(ims).to(device) - mean) / std
        tok = model.forward_features(x)                                  # [B, npfx+GH*GW, 384]
        patch = tok[:, npfx:, :].transpose(1, 2).reshape(-1, tok.shape[-1], GH, GW)  # [B,384,14,24]
        patch = patch.cpu().numpy().astype(np.float16)
        for tk, f in zip(chunk, patch):
            np.savez_compressed(args.out_dir / f"{tk}.npz", feat=f, sample_token=np.array(tk))
            n_written += 1
        if (s + args.batch) % (args.batch * 10) == 0 or s + args.batch >= len(todo):
            print(f"  {n_written}/{len(todo)}  ({n_written/max(time.time()-t0,1e-6):.0f}/s)")

    total = sum(p.stat().st_size for p in args.out_dir.glob("*.npz"))
    manifest = dict(model=DINOV3_MODEL, ref_cache=str(args.ref_cache), n_tokens=len(tokens),
                    n_written=n_written, feat_shape=[model.embed_dim, GH, GW], dtype="float16",
                    input_hw=[HP, WP], prefix_tokens=npfx, total_mb=round(total / 1e6, 1),
                    note="upsample feat 14x24 -> 32x56 (bilinear) at train time")
    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("\n" + json.dumps(manifest, indent=2))
    print(f"\nwrote {n_written} files to {args.out_dir}/ ({total/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
