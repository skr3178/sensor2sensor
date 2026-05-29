"""End-to-end shape verification for the multi-camera UNet conditioning path.

Loads B=2 paired samples from the 10-scene subset, runs them through:

    paired dataset            -> cams [B, V, 3, 256, 448]
                              -> K   [B, V, 3, 3]
                              -> T   [B, V, 4, 4]
                              -> range_image [B, 3, 32, 1024]
    FrozenSDVAEEncoder        -> img_latents [B, V, 4, 32, 56]
    build_raymap (batched V)  -> raymaps     [B, V, 6, 32, 56]
    concat on channels        -> kv_full     [B, V, 10, 32, 56]
    adaptive_avg_pool         -> kv_pooled   [B, V, 10, 8, 64]
    CrossViewFusion           -> kv_fused    [B, V, 10, 8, 64]
    token-concat along W      -> kv_context  [B, 10, 8, 384]
    LiDARUNet(z_noisy, t, kv) -> v_pred      [B, 8, 8, 256]

Run:
    PYTHONPATH=. python scripts/encoder_smoke_test_multicam.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from data.nuscenes_mini_paired import NuScenesPairedKeyframes, load_subset_tokens
from models.cross_view_fusion import CrossViewFusion
from models.image_encoder import FrozenSDVAEEncoder
from models.raymap import build_raymap
from models.unet import LiDARUNet

NUSCENES_ROOT = Path("data/nuscenes_root")
SUBSET_TOKENS = Path("out/subset_scene_tokens.txt")
IMG_VAE_DIR   = Path("checkpoints/sd15_vae")
B = 2
V = 6
SD_DOWNSAMPLE = 8
H_LAT, W_LAT  = 32, 56          # SD VAE latent grid for 256x448
H_KV, W_KV    = 8, 64           # per-view KV grid after adaptive pool
KV_CHANNELS   = 10              # 4 image + 6 raymap
LIDAR_LAT_H, LIDAR_LAT_W = 8, 256


def _assert_shape(name: str, t: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(t.shape) != expected:
        raise AssertionError(f"{name}: expected {expected}, got {tuple(t.shape)}")
    print(f"  {name:<14} {list(t.shape)}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    # ---- dataset ----
    scene_tokens = load_subset_tokens(SUBSET_TOKENS)
    print(f"subset scenes: {len(scene_tokens)}")
    ds = NuScenesPairedKeyframes(NUSCENES_ROOT, scene_tokens=scene_tokens)
    print(f"paired samples in subset: {len(ds)}")
    assert len(ds) >= B, f"need at least {B} samples, got {len(ds)}"

    items = [ds[i] for i in range(B)]
    range_image    = torch.stack([it["range_image"]   for it in items], dim=0).to(device)
    cams           = torch.stack([it["cams"]          for it in items], dim=0).to(device)
    cam_K          = torch.stack([it["cam_K"]         for it in items], dim=0).to(device)
    cam_T_cam2ego  = torch.stack([it["cam_T_cam2ego"] for it in items], dim=0).to(device)
    print("\n== dataset shapes ==")
    _assert_shape("range_image",    range_image,   (B, 3, 32, 1024))
    _assert_shape("cams",           cams,          (B, V, 3, 256, 448))
    _assert_shape("cam_K",          cam_K,         (B, V, 3, 3))
    _assert_shape("cam_T_cam2ego",  cam_T_cam2ego, (B, V, 4, 4))

    # ---- image encoder ----
    print("\nloading SD VAE ...")
    img_enc = FrozenSDVAEEncoder(local_dir=IMG_VAE_DIR).to(device)
    img_latents = img_enc.encode_views(cams)
    print("\n== image latents ==")
    _assert_shape("img_latents", img_latents, (B, V, 4, H_LAT, W_LAT))

    # ---- raymap (batched: flatten B*V into the build_raymap batch dim) ----
    K_flat = cam_K.reshape(B * V, 3, 3)
    T_flat = cam_T_cam2ego.reshape(B * V, 4, 4)
    raymap_flat = build_raymap(K_flat, T_flat, H_LAT, W_LAT, downsample=SD_DOWNSAMPLE)
    raymaps = raymap_flat.view(B, V, 6, H_LAT, W_LAT)
    print("\n== raymaps ==")
    _assert_shape("raymaps", raymaps, (B, V, 6, H_LAT, W_LAT))

    # ---- per-view (img_latent || raymap) → pool → fuse → token-concat ----
    kv_full = torch.cat([img_latents, raymaps], dim=2)     # [B, V, 10, 32, 56]
    print("\n== kv assembly ==")
    _assert_shape("kv_full", kv_full, (B, V, KV_CHANNELS, H_LAT, W_LAT))

    kv_pooled = F.adaptive_avg_pool2d(
        kv_full.flatten(0, 1), (H_KV, W_KV)
    ).view(B, V, KV_CHANNELS, H_KV, W_KV)
    _assert_shape("kv_pooled", kv_pooled, (B, V, KV_CHANNELS, H_KV, W_KV))

    fusion = CrossViewFusion(channels=KV_CHANNELS, hidden_dim=64, num_layers=2, num_heads=4).to(device)
    kv_fused = fusion(kv_pooled)
    _assert_shape("kv_fused", kv_fused, (B, V, KV_CHANNELS, H_KV, W_KV))

    # Token-concat along W: [B,V,C,H,W] -> [B,C,H,V*W].
    kv_context = kv_fused.permute(0, 2, 3, 1, 4).reshape(B, KV_CHANNELS, H_KV, V * W_KV)
    _assert_shape("kv_context", kv_context, (B, KV_CHANNELS, H_KV, V * W_KV))

    # ---- UNet forward, Scope B (one-way cross-attention) ----
    unet_b = LiDARUNet(in_channels=8, out_channels=8, kv_channels=KV_CHANNELS).to(device)
    z_noisy = torch.randn(B, 8, LIDAR_LAT_H, LIDAR_LAT_W, device=device)
    t       = torch.randint(0, 1000, (B,), device=device)
    with torch.no_grad():
        v_pred_b = unet_b(z_noisy, t, kv_context)
    print("\n== UNet, Scope B (one-way LiDAR->camera cross-attn) ==")
    _assert_shape("v_pred", v_pred_b, (B, 8, LIDAR_LAT_H, LIDAR_LAT_W))

    # ---- UNet forward, Scope C (cross-sensor self-attention) ----
    unet_c = LiDARUNet(
        in_channels=8, out_channels=8, kv_channels=KV_CHANNELS,
        use_cross_sensor=True,
    ).to(device)
    with torch.no_grad():
        v_pred_c = unet_c(z_noisy, t, kv_context)
    print("\n== UNet, Scope C (symmetric self-attn over [lidar; cam]) ==")
    _assert_shape("v_pred", v_pred_c, (B, 8, LIDAR_LAT_H, LIDAR_LAT_W))
    from models.attention import CrossSensorSelfAttn
    ca_kinds = {type(ca).__name__ for ca in unet_c.enc_l0.cross_attns}
    assert ca_kinds == {"CrossSensorSelfAttn"}, f"Scope C UNet still has {ca_kinds}"
    print(f"  enc_l0 cross-attn class : {ca_kinds.pop()}")

    # Identity checks at init:
    # 1. CrossViewFusion(x) = x (zero-init out_proj + residual)
    delta_fusion = (kv_fused - kv_pooled).abs().max().item()
    print(f"\nfusion-at-init      max|kv_fused - kv_pooled| = {delta_fusion:.2e}  (expect ~0)")
    assert delta_fusion < 1e-5, "CrossViewFusion is not identity at init — out_proj zero-init broken?"

    # 2. UNet head is zero-init too, so v_pred_b and v_pred_c are both ~0 at init.
    #    They will NOT be identical (different param init), but Scope C should differ
    #    from Scope B only by the new CrossSensorSelfAttn pathway — both are zero-init,
    #    so on the first forward they should agree to within float noise.
    delta_unet = (v_pred_b - v_pred_c).abs().max().item()
    print(f"unet-head-zero-init max|v_pred_b - v_pred_c| = {delta_unet:.2e}  (expect ~0; head is zero-init)")
    assert delta_unet < 1e-5, "Scope B / C disagree at init — head zero-init or CSSA out_proj zero-init broken?"

    print("\nOK — both Scope B and Scope C produce the expected shapes; modules are identity at init.")


if __name__ == "__main__":
    main()
