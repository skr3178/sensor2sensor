"""Camera-swap test — demonstrate single-cam model's FoV-locked behavior.

Empirical proof of the §13.4 "outside-FoV" finding. For one held-out sample,
we run two inferences through the SAME single-cam DINOv3 ckpt:

  A. CONTROL  — CAM_FRONT image + CAM_FRONT raymap   (the trained setup)
  B. SWAP     — CAM_BACK  image + CAM_FRONT raymap   (lie to the model)

In setup B the raymap says "I'm looking forward (+x)" but the image actually
shows what's BEHIND the vehicle. The model has been trained to put visible
structure in the FRONT region of the LiDAR — so its prediction's forward
direction will show what's actually BEHIND the vehicle. Visual evidence that:
  - the model doesn't "understand" the camera's orientation, it just
    associates whatever image it sees with the forward LiDAR direction
  - 92 % of the LiDAR target (everything outside CAM_FRONT FoV) gets the
    same learned-marginal blob regardless of input

Output: a 2×3 figure (rows = GT vs CAM_BACK image; cols = BEV GT / CONTROL pred
/ SWAP pred) showing the asymmetry. Plus per-sample stats.

Reuses canonical helpers (no duplicated logic):
- DINOv3 encode helper from dinov3_heldout_demo.py
- raymap build from cache_latents.py
- BEV viz + projection helpers from existing scripts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.range_image import range_image_to_point_cloud
from eval.bev_viz import bev_scatter
from eval.chamfer import chamfer_distance
from eval.decode_to_pointcloud import KV_POOL_H, KV_POOL_W, load_lidar_vae, load_unet
from models.diffusion import DiffusionWrapper
from models.dinov3_proj import DINOv3Proj
from models.raymap import build_raymap
from run_m4_demo import raw_lidar_for_sample
from dinov3_heldout_demo import encode_dinov3
from train.cache_latents import scale_intrinsics, make_T, SD_DOWNSAMPLE

NUSCENES = Path("nuscenes")
D3_CACHE = Path("s2s_min/out/cached_dinov3_v5_100scenes")  # training-set sentinel
VAE_CKPT = Path("s2s_min/out/lidar_vae_best.pt")
RANGE_M  = 60.0


def index_cameras():
    """Build {sample_token: {channel: (sample_data_rec, cs_rec)}} index.
    Cheaper than the existing load_meta() which only covers CAM_FRONT + LIDAR_TOP.
    """
    sd = json.loads((NUSCENES / "v1.0-trainval" / "sample_data.json").read_text())
    cs = {c["token"]: c for c in json.loads((NUSCENES / "v1.0-trainval" / "calibrated_sensor.json").read_text())}
    sn = {s["token"]: s for s in json.loads((NUSCENES / "v1.0-trainval" / "sensor.json").read_text())}
    out: dict[str, dict[str, tuple[dict, dict]]] = {}
    for r in sd:
        if not r["is_key_frame"]:
            continue
        ch = sn[cs[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if ch not in ("CAM_FRONT", "CAM_BACK", "LIDAR_TOP"):
            continue
        tok = r["sample_token"]
        out.setdefault(tok, {})[ch] = (r, cs[r["calibrated_sensor_token"]])
    return out


@torch.no_grad()
def gen_pc(unet, vae, diffusion, kv_context, seed: int, cfg_scale: float, device: str):
    torch.manual_seed(seed)
    z = diffusion.ddim_sample_cfg(
        unet=unet, shape=(1, 8, 8, 256), kv_context=kv_context,
        device=torch.device(device), cfg_scale=cfg_scale,
    )
    rng = vae.decode(z)[0].cpu().numpy().clip(0, 1)
    return range_image_to_point_cloud(rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinov3_ckpt", type=Path, default=None,
                    help="Defaults to .last_dinov3_run")
    ap.add_argument("--cfg_scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--token", type=str, default=None,
                    help="Specific sample_token to test. Default = random held-out.")
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/cam_swap_test"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    if args.dinov3_ckpt is None:
        args.dinov3_ckpt = Path(open("s2s_min/out/.last_dinov3_run").read().strip()) / "lidar_unet_best.pt"

    print("=" * 70)
    print("Camera-swap test — single-cam model's FoV-locked behavior")
    print("=" * 70)
    print(f"  ckpt        : {args.dinov3_ckpt}")
    print(f"  cfg_scale   : {args.cfg_scale}")

    # ─── Load models ───
    unet, ckpt = load_unet(args.dinov3_ckpt, device)
    vae = load_lidar_vae(VAE_CKPT, device)
    diffusion = DiffusionWrapper()
    proj = DINOv3Proj(
        ckpt["dinov3_proj"]["mean"].squeeze().tolist(),
        ckpt["dinov3_proj"]["std"].squeeze().tolist(),
    ).to(device).eval()
    proj.load_state_dict(ckpt["dinov3_proj"])
    proj.requires_grad_(False)
    print(f"  K1 clip_sample : {diffusion.inference_scheduler.config.clip_sample}")
    print(f"  step / loss_ema: {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}")

    # ─── Index cameras ───
    cams = index_cameras()
    # Need a sample with BOTH CAM_FRONT and CAM_BACK and LIDAR_TOP, NOT in training cache.
    training_tokens = {p.stem for p in D3_CACHE.glob("*.npz")}
    candidates = sorted([
        t for t, ch_dict in cams.items()
        if "CAM_FRONT" in ch_dict and "CAM_BACK" in ch_dict and "LIDAR_TOP" in ch_dict
        and t not in training_tokens
    ])
    print(f"  held-out tokens with all 3 sensors: {len(candidates)}")

    if args.token:
        assert args.token in cams, f"token {args.token} not in nuScenes"
        tok = args.token
    else:
        rng = np.random.RandomState(args.seed)
        tok = candidates[rng.randint(len(candidates))]
    print(f"  picked token: {tok}")

    cam_front_rec, cam_front_cs = cams[tok]["CAM_FRONT"]
    cam_back_rec,  _            = cams[tok]["CAM_BACK"]
    print(f"  CAM_FRONT image : {cam_front_rec['filename']}")
    print(f"  CAM_BACK  image : {cam_back_rec['filename']}")

    # ─── Build CAM_FRONT raymap (the trained-distribution geometry) ───
    K_scaled = scale_intrinsics(np.array(cam_front_cs["camera_intrinsic"], dtype=np.float32))
    T_cam2ego = make_T(cam_front_cs["translation"], cam_front_cs["rotation"])
    raymap = build_raymap(
        torch.from_numpy(K_scaled), torch.from_numpy(T_cam2ego),
        H_latent=32, W_latent=56, downsample=SD_DOWNSAMPLE,
    ).to(device)  # [1, 6, 32, 56]

    # ─── Encode CAM_FRONT and CAM_BACK images via the SAME DINOv3 ───
    img_front = Image.open(NUSCENES / cam_front_rec["filename"])
    img_back  = Image.open(NUSCENES / cam_back_rec["filename"])
    feat_front = encode_dinov3(img_front, device)  # [1, 384, 14, 24]
    feat_back  = encode_dinov3(img_back,  device)

    # ─── Build kv_context for both setups (SAME raymap, swap features) ───
    def build_kv(feat):
        p4 = proj(feat)
        p4 = F.interpolate(p4, size=raymap.shape[-2:], mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(torch.cat([p4, raymap], dim=1), (KV_POOL_H, KV_POOL_W))

    kv_control = build_kv(feat_front)   # A: trained setup
    kv_swap    = build_kv(feat_back)    # B: lie to model

    # ─── Run both DDIM inferences with identical seed ───
    t0 = time.time()
    pc_control = gen_pc(unet, vae, diffusion, kv_control, args.seed, args.cfg_scale, device)
    pc_swap    = gen_pc(unet, vae, diffusion, kv_swap,    args.seed, args.cfg_scale, device)
    pc_raw     = raw_lidar_for_sample(tok)[:, :3]
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  inference   : {time.time()-t0:.2f}s")

    # ─── Chamfer ───
    cd_control = chamfer_distance(pc_control, pc_raw)["cd"]
    cd_swap    = chamfer_distance(pc_swap,    pc_raw)["cd"]
    cd_ctrl_bev = chamfer_distance(pc_control, pc_raw, use_xy_only=True)["cd"]
    cd_swap_bev = chamfer_distance(pc_swap,    pc_raw, use_xy_only=True)["cd"]
    print()
    print(f"  CD-3D-raw (CONTROL: CAM_FRONT)        : {cd_control:.3f} m")
    print(f"  CD-3D-raw (SWAP   : CAM_BACK as front): {cd_swap:.3f} m   "
          f"Δ={cd_swap - cd_control:+.3f} m")
    print(f"  CD-BEV-raw (CONTROL)                  : {cd_ctrl_bev:.3f} m")
    print(f"  CD-BEV-raw (SWAP)                     : {cd_swap_bev:.3f} m   "
          f"Δ={cd_swap_bev - cd_ctrl_bev:+.3f} m")

    # ─── Plot 2×3 figure ───
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Top row: BEV plots
    # In nuScenes LiDAR sensor frame, +x = forward, +y = left.
    # bev_scatter uses x,y so +x in figure should be forward.
    # Match orientation: rotate so forward is UP (matches camera-image "forward = top").
    def bev_with_forward_up(ax, pc, color, title):
        # Rotate +90°: new_x = -y, new_y = x → forward (+x) goes up
        if len(pc) > 0:
            rotated = np.column_stack([-pc[:, 1], pc[:, 0], pc[:, 2] if pc.shape[1] > 2 else np.zeros(len(pc))])
        else:
            rotated = pc
        bev_scatter(ax, rotated, color=color, range_m=RANGE_M)
        ax.set_title(title, fontsize=10)
        # Add a "FORWARD" arrow at the top
        ax.annotate("FORWARD", xy=(0, RANGE_M * 0.92), ha="center", fontsize=9,
                    color="black", weight="bold")
        ax.annotate("BACK",    xy=(0, -RANGE_M * 0.92), ha="center", fontsize=9,
                    color="black", weight="bold")

    bev_with_forward_up(axes[0, 0], pc_raw,     "tab:blue",
                        f"raw nuScenes GT (N={len(pc_raw)})")
    bev_with_forward_up(axes[0, 1], pc_control, "tab:green",
                        f"CONTROL: CAM_FRONT → pred\nCD={cd_control:.2f}m   (front-region should match)")
    bev_with_forward_up(axes[0, 2], pc_swap,    "tab:red",
                        f"SWAP: CAM_BACK as forward → pred\nCD={cd_swap:.2f}m   (forward should show BACK scene)")

    # Bottom row: the camera images
    axes[1, 0].imshow(img_front); axes[1, 0].set_title("CAM_FRONT (control)", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(img_front); axes[1, 1].set_title("(same — fed in CONTROL)", fontsize=10)
    axes[1, 1].axis("off")
    axes[1, 2].imshow(img_back);  axes[1, 2].set_title("CAM_BACK (fed in SWAP)", fontsize=10)
    axes[1, 2].axis("off")

    fig.suptitle(
        f"Camera-swap test — token {tok[:8]}…   ckpt step {ckpt.get('step')}, cfg={args.cfg_scale}\n"
        f"Tests §13.4 outside-FoV finding: model is FoV-locked, doesn't know which way camera is pointing.",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path = args.out_dir / f"swap_{tok[:8]}_cfg{args.cfg_scale:g}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    (args.out_dir / f"stats_{tok[:8]}_cfg{args.cfg_scale:g}.txt").write_text(
        f"Camera-swap test — token {tok}\n"
        f"ckpt: {args.dinov3_ckpt}\n"
        f"step / loss_ema : {ckpt.get('step')} / {ckpt.get('loss_ema'):.4f}\n"
        f"cfg_scale       : {args.cfg_scale}\n\n"
        f"CAM_FRONT image : {cam_front_rec['filename']}\n"
        f"CAM_BACK  image : {cam_back_rec['filename']}\n\n"
        f"CONTROL (CAM_FRONT image, CAM_FRONT raymap):\n"
        f"  CD-3D-raw : {cd_control:.4f} m\n"
        f"  CD-BEV    : {cd_ctrl_bev:.4f} m\n"
        f"  N_pred    : {len(pc_control)}\n\n"
        f"SWAP (CAM_BACK image fed as if it were front, CAM_FRONT raymap):\n"
        f"  CD-3D-raw : {cd_swap:.4f} m   (Δ {cd_swap - cd_control:+.4f} from control)\n"
        f"  CD-BEV    : {cd_swap_bev:.4f} m   (Δ {cd_swap_bev - cd_ctrl_bev:+.4f})\n"
        f"  N_pred    : {len(pc_swap)}\n\n"
        f"N_raw GT     : {len(pc_raw)}\n\n"
        f"Interpretation:\n"
        f"  If CD-SWAP > CD-CONTROL by ≳0.5m  →  model's forward-region prediction degraded.\n"
        f"  Visual: SWAP's forward region should resemble what's BEHIND the vehicle in GT.\n"
    )
    print()
    print(f"  saved plot  : {plot_path}")
    print(f"  saved stats : {args.out_dir / f'stats_{tok[:8]}_cfg{args.cfg_scale:g}.txt'}")


if __name__ == "__main__":
    main()
