"""Tier-1 diagnostic: does the frozen SD 1.5 VAE latent contain depth information?

We freeze the SD VAE features (already cached as `image_latent [4,32,56]`) and
train a tiny PER-PIXEL 1x1 probe to predict camera-plane metric depth. The probe
has NO spatial receptive field (1x1 convs only), so it can only succeed if the
depth signal lives in the *local* feature vector at each latent cell. That is the
honest test of "are the SD encoder features depth-bearing".

Depth ground truth is built here (not cached): project the LIDAR_TOP point cloud
into CAM_FRONT using the full nuScenes transform chain
    p_cam = inv(T_cam2ego) @ inv(T_ego2glob_cam) @ T_ego2glob_lidar @ T_lidar2ego @ p
then pinhole-project with the latent-resolution intrinsics and z-buffer the
nearest return into each 32x56 cell. Sparse (LiDAR is 32 beams) but unbiased.

Conditions (all probes identical capacity, per-pixel 1x1 MLP):
    img      : SD VAE image_latent (4ch)            <- the test
    ray      : raymap (6ch, ray origin+dir)         <- viewing-direction / position prior
    img+ray  : both (10ch)
    img_shuf : image_latent paired with a DIFFERENT sample's depth  <- null / capacity control
    mean     : constant = train-mean log-depth      <- trivial floor

Verdict:
    img clearly beats ray AND beats img_shuf  -> SD VAE features CARRY depth.
                                                  Bottleneck is downstream (cross-attn
                                                  pooling / missing pos-enc) -> apply Fix #1 (H5).
    img ~= ray (no lift over the position prior) -> features are depth-impoverished
                                                  -> swap the image encoder (H2/H3).

Run:
    env/bin/python s2s_min/diagnostics/sdvae_depth_probe.py --limit 800
Outputs (under s2s_min/out/depth_probe/):
    results.json, summary.md, and PNG plots.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.range_image import load_nuscenes_lidar_bin

NUSCENES_ROOT = Path("nuscenes")
META = NUSCENES_ROOT / "v1.0-trainval"
NATIVE_W, NATIVE_H = 1600, 900
IMG_W, IMG_H = 448, 256
SD_DOWNSAMPLE = 8
H_LAT, W_LAT = 32, 56


# --------------------------- geometry helpers ----------------------------
def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def make_T(translation, rotation_quat_wxyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rotmat(rotation_quat_wxyz)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def latent_intrinsics(K_native):
    """nuScenes native (1600x900) intrinsics -> 256x448 -> /8 latent grid."""
    K = np.asarray(K_native, dtype=np.float64).copy()
    sx = IMG_W / NATIVE_W
    sy = IMG_H / NATIVE_H
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    K[0, :] /= SD_DOWNSAMPLE
    K[1, :] /= SD_DOWNSAMPLE
    return K


def build_depth_target(pc_xyz, T_lidar2ego, T_eg_lid, T_eg_cam, T_cam2ego, K_lat):
    """Project LiDAR points into the camera latent grid; z-buffer nearest depth.

    Returns depth[H,W] (meters, 0 where no return) and valid[H,W] bool.
    """
    N = pc_xyz.shape[0]
    p = np.concatenate([pc_xyz, np.ones((N, 1))], axis=1).T            # [4,N]
    p_cam = (np.linalg.inv(T_cam2ego) @ np.linalg.inv(T_eg_cam)
             @ T_eg_lid @ T_lidar2ego @ p)                            # [4,N]
    X, Y, Z = p_cam[0], p_cam[1], p_cam[2]
    front = Z > 0.5
    X, Y, Z = X[front], Y[front], Z[front]
    u = (K_lat[0, 0] * X + K_lat[0, 2] * Z) / Z
    v = (K_lat[1, 1] * Y + K_lat[1, 2] * Z) / Z
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    inb = (ui >= 0) & (ui < W_LAT) & (vi >= 0) & (vi < H_LAT)
    ui, vi, Z = ui[inb], vi[inb], Z[inb]

    depth = np.full((H_LAT, W_LAT), np.inf, dtype=np.float64)
    # nearest-Z wins: sort far->near so nearest is written last
    order = np.argsort(-Z)
    flat = vi[order] * W_LAT + ui[order]
    depth.reshape(-1)[flat] = Z[order]
    valid = np.isfinite(depth)
    depth[~valid] = 0.0
    return depth.astype(np.float32), valid


# --------------------------- nuScenes index ------------------------------
def build_index(sample_tokens):
    """For the given set of sample tokens, return per-token cam/lidar records
    with the calibration + ego-pose dicts needed for projection."""
    t0 = time.time()
    cs = {c["token"]: c for c in json.loads((META / "calibrated_sensor.json").read_text())}
    sensor = {s["token"]: s for s in json.loads((META / "sensor.json").read_text())}
    print(f"  calibrated_sensor + sensor loaded ({time.time()-t0:.1f}s)")

    want = set(sample_tokens)
    cam_rec, lid_rec = {}, {}
    needed_ego = set()
    t1 = time.time()
    sample_data = json.loads((META / "sample_data.json").read_text())
    print(f"  sample_data.json parsed: {len(sample_data)} records ({time.time()-t1:.1f}s)")
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] not in want:
            continue
        chan = sensor[cs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if chan == "CAM_FRONT":
            cam_rec[sd["sample_token"]] = sd; needed_ego.add(sd["ego_pose_token"])
        elif chan == "LIDAR_TOP":
            lid_rec[sd["sample_token"]] = sd; needed_ego.add(sd["ego_pose_token"])
    del sample_data

    t2 = time.time()
    ego = {}
    for e in json.loads((META / "ego_pose.json").read_text()):
        if e["token"] in needed_ego:
            ego[e["token"]] = e
    print(f"  ego_pose.json parsed, {len(ego)} poses kept ({time.time()-t2:.1f}s)")
    return cs, cam_rec, lid_rec, ego


# --------------------------- dataset build -------------------------------
def assemble(cache_dir, limit):
    npz_paths = sorted(p for p in cache_dir.glob("*.npz"))
    if limit:
        npz_paths = npz_paths[:limit]
    tokens = [p.stem for p in npz_paths]
    print(f"cache: {cache_dir}  ({len(tokens)} samples requested)")

    cs, cam_rec, lid_rec, ego = build_index(tokens)

    imgs, rays, depths, valids, rgb_paths, keep_tokens = [], [], [], [], [], []
    n_fail = 0
    t0 = time.time()
    for i, p in enumerate(npz_paths):
        tok = p.stem
        if tok not in cam_rec or tok not in lid_rec:
            n_fail += 1; continue
        try:
            d = np.load(p)
            cam, lid = cam_rec[tok], lid_rec[tok]
            cs_cam, cs_lid = cs[cam["calibrated_sensor_token"]], cs[lid["calibrated_sensor_token"]]
            T_cam2ego = make_T(cs_cam["translation"], cs_cam["rotation"])
            T_lidar2ego = make_T(cs_lid["translation"], cs_lid["rotation"])
            eg_cam = ego[cam["ego_pose_token"]]; eg_lid = ego[lid["ego_pose_token"]]
            T_eg_cam = make_T(eg_cam["translation"], eg_cam["rotation"])
            T_eg_lid = make_T(eg_lid["translation"], eg_lid["rotation"])
            K_lat = latent_intrinsics(cs_cam["camera_intrinsic"])

            pc = load_nuscenes_lidar_bin(str(NUSCENES_ROOT / lid["filename"]))[:, :3].astype(np.float64)
            depth, valid = build_depth_target(pc, T_lidar2ego, T_eg_lid, T_eg_cam, T_cam2ego, K_lat)
            if valid.sum() < 20:        # too few returns in-frame to be useful
                n_fail += 1; continue

            imgs.append(d["image_latent"]); rays.append(d["raymap"])
            depths.append(depth); valids.append(valid)
            rgb_paths.append(str(NUSCENES_ROOT / cam["filename"])); keep_tokens.append(tok)
        except Exception as e:
            n_fail += 1
            if n_fail < 5:
                print(f"  [skip] {tok}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  built {len(imgs)}/{i+1}  ({(i+1)/(time.time()-t0):.0f} samp/s)")

    print(f"assembled {len(imgs)} usable samples ({n_fail} skipped)")
    return (np.stack(imgs), np.stack(rays),
            np.stack(depths)[:, None], np.stack(valids)[:, None],
            rgb_paths, keep_tokens)


# --------------------------- probe model ---------------------------------
class PixelProbe(nn.Module):
    """Per-pixel 1x1 MLP. No spatial mixing -> reads depth only from local feature."""
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def metrics(pred_log, gt_log, mask):
    """pred_log/gt_log: [N,1,H,W] log-depth; mask bool. Returns metric-space stats."""
    m = mask.reshape(-1).bool()
    p = pred_log.reshape(-1)[m].clamp(np.log(0.5), np.log(120.0))
    g = gt_log.reshape(-1)[m]
    pd = torch.exp(p); gd = torch.exp(g)                  # back to meters
    absrel = (torch.abs(pd - gd) / gd).mean().item()
    rmse = torch.sqrt(((pd - gd) ** 2).mean()).item()
    ratio = torch.maximum(pd / gd, gd / pd)
    d1 = (ratio < 1.25).float().mean().item()
    # Pearson r in log space
    pc = p - p.mean(); gc = g - g.mean()
    r = (pc * gc).sum() / (pc.norm() * gc.norm() + 1e-9)
    # R^2 vs predicting the mean (log space)
    ss_res = ((p - g) ** 2).sum(); ss_tot = ((g - g.mean()) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + 1e-9)).item()
    return dict(absrel=absrel, rmse=rmse, delta1=d1, pearson=r.item(), r2=r2)


def train_probe(Xtr, Ytr, Mtr, Xva, Yva, Mva, in_ch, device, epochs=60, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model = PixelProbe(in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]; bs = 64
    val_curve = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            pred = model(Xtr[idx])
            m = Mtr[idx]
            loss = (((pred - Ytr[idx]) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva)
            vl = (((pv - Yva) ** 2) * Mva).sum() / Mva.sum().clamp(min=1)
        val_curve.append(vl.item())
    model.eval()
    with torch.no_grad():
        pv = model(Xva)
    return model, metrics(pv, Yva, Mva), val_curve, pv.cpu()


# --------------------------- plotting ------------------------------------
def plot_all(results, val_curves, qual, out_dir, n_samples, n_cells):
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["img", "img+ray", "ray", "img_shuf", "mean"]
    colors = {"img": "#2ca02c", "img+ray": "#1f77b4", "ray": "#ff7f0e",
              "img_shuf": "#9467bd", "mean": "#7f7f7f"}
    labels = {"img": "SD-VAE img", "img+ray": "img+ray", "ray": "raymap only",
              "img_shuf": "img (shuffled)", "mean": "mean floor"}
    conds = [c for c in order if c in results]

    # 1) metrics bars
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, title, hi_good in [
        (axes[0], "absrel", "AbsRel  (lower=better)", False),
        (axes[1], "rmse", "RMSE m  (lower=better)", False),
        (axes[2], "pearson", "Pearson r log-depth  (higher=better)", True),
        (axes[3], "delta1", "δ<1.25  (higher=better)", True)]:
        vals = [results[c][key] for c in conds]
        bars = ax.bar([labels[c] for c in conds], vals, color=[colors[c] for c in conds])
        ax.set_title(title, fontsize=11); ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"SD-VAE depth probe — {n_samples} samples, {n_cells:,} valid cells "
                 f"(per-pixel 1x1 probe, frozen features)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_dir / "metrics_bar.png", dpi=130); plt.close(fig)

    # 2) training curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in conds:
        if c in val_curves:
            ax.plot(val_curves[c], label=labels[c], color=colors[c])
    ax.set_xlabel("epoch"); ax.set_ylabel("val masked MSE (log-depth)")
    ax.set_title("Probe validation loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png", dpi=130); plt.close(fig)

    # 3) scatter pred vs gt (img and ray)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, c in zip(axes, ["img", "ray"]):
        if c not in qual: continue
        gt, pr = qual[c]
        ax.scatter(gt, pr, s=2, alpha=0.15, color=colors[c])
        lim = [0, np.percentile(gt, 99)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("GT depth (m)"); ax.set_ylabel("predicted depth (m)")
        ax.set_title(f"{labels[c]}   r={results[c]['pearson']:.3f}  AbsRel={results[c]['absrel']:.3f}")
    fig.suptitle("Predicted vs ground-truth depth (held-out cells)")
    fig.tight_layout(); fig.savefig(out_dir / "scatter_pred_vs_gt.png", dpi=130); plt.close(fig)

    # 4) qualitative panels
    if "panels" in qual:
        panels = qual["panels"]  # list of dicts: rgb, gt, valid, pred_img, pred_ray
        nrow = len(panels)
        fig, axes = plt.subplots(nrow, 4, figsize=(14, 3.0 * nrow))
        if nrow == 1: axes = axes[None, :]
        vmax = np.percentile(np.concatenate([p["gt"][p["valid"]] for p in panels]), 95)
        for r, pn in enumerate(panels):
            axes[r, 0].imshow(pn["rgb"]); axes[r, 0].set_title("CAM_FRONT" if r == 0 else "")
            gt_disp = np.ma.masked_where(~pn["valid"], pn["gt"])
            axes[r, 1].imshow(gt_disp, cmap="turbo", vmin=0, vmax=vmax)
            axes[r, 1].set_title("GT depth (LiDAR, sparse)" if r == 0 else "")
            axes[r, 2].imshow(pn["pred_img"], cmap="turbo", vmin=0, vmax=vmax)
            axes[r, 2].set_title("probe: SD-VAE img" if r == 0 else "")
            axes[r, 3].imshow(pn["pred_ray"], cmap="turbo", vmin=0, vmax=vmax)
            axes[r, 3].set_title("probe: raymap only" if r == 0 else "")
            for c in range(4): axes[r, c].axis("off")
        fig.suptitle("Qualitative depth probing (dense pred shown; GT is sparse LiDAR)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_dir / "qualitative.png", dpi=130); plt.close(fig)

    # 5) decision summary card
    fig = plt.figure(figsize=(11, 6)); ax = fig.add_subplot(111); ax.axis("off")
    r_img, r_ray, r_shuf = results["img"]["pearson"], results["ray"]["pearson"], results["img_shuf"]["pearson"]
    lift_over_pos = r_img - r_ray
    lift_over_null = r_img - r_shuf
    carries = (lift_over_pos > 0.05) and (lift_over_null > 0.10)
    if carries:
        verdict = "SD-VAE FEATURES CARRY DEPTH"
        action = ("=> Bottleneck is DOWNSTREAM (KV pooling / missing cross-attn pos-enc).\n"
                  "   Disambiguates toward H5.  ACTION: apply Fix #1 (sinusoidal pos-enc).")
        vcolor = "#2ca02c"
    else:
        verdict = "SD-VAE FEATURES ARE DEPTH-IMPOVERISHED"
        action = ("=> Image probe ~ position prior; features add little depth signal.\n"
                  "   Disambiguates toward H2/H3.  ACTION: swap the image encoder (C-series).")
        vcolor = "#d62728"
    lines = [
        ("SD-VAE depth probe — verdict", 18, "black"),
        (verdict, 16, vcolor),
        ("", 6, "black"),
        (f"Pearson r (log-depth):   img={r_img:.3f}   ray={r_ray:.3f}   img_shuf={r_shuf:.3f}", 12, "black"),
        (f"AbsRel:                  img={results['img']['absrel']:.3f}   ray={results['ray']['absrel']:.3f}"
         f"   floor={results['mean']['absrel']:.3f}", 12, "black"),
        (f"img lift over position prior (r):  {lift_over_pos:+.3f}   (need > +0.05)", 12, "black"),
        (f"img lift over null/shuffle (r):    {lift_over_null:+.3f}   (need > +0.10)", 12, "black"),
        ("", 6, "black"),
        (action, 13, vcolor),
        ("", 6, "black"),
        ("Pairs with Fix #1 result to fully separate H2/H3 (encoder) vs H5 (cross-attn).", 10, "#555555"),
    ]
    y = 0.95
    for txt, sz, col in lines:
        ax.text(0.02, y, txt, fontsize=sz, color=col, family="monospace",
                transform=ax.transAxes, va="top")
        y -= 0.045 + sz * 0.0016 + txt.count("\n") * 0.05
    fig.savefig(out_dir / "decision_summary.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return carries, verdict, action


# --------------------------- main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("s2s_min/out/cached_latents_v5_100scenes"))
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", type=Path, default=Path("s2s_min/out/depth_probe"))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    imgs, rays, depths, valids, rgb_paths, tokens = assemble(args.cache, args.limit)
    N = imgs.shape[0]
    total_valid = int(valids.sum())
    print(f"valid cells: {total_valid:,}  ({total_valid/N:.0f}/sample avg)")

    # standardize image features per channel (train stats); raymap left as-is (metric)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    img_mean = imgs[tr_idx].mean((0, 2, 3), keepdims=True)
    img_std = imgs[tr_idx].std((0, 2, 3), keepdims=True) + 1e-6
    imgs_n = (imgs - img_mean) / img_std

    # log-depth target, masked
    log_depth = np.zeros_like(depths)
    log_depth[valids] = np.log(np.clip(depths[valids], 0.5, 100.0))

    to = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
    img_t, ray_t = to(imgs_n), to(rays)
    y_t, m_t = to(log_depth), to(valids.astype(np.float32))

    feats = {
        "img": img_t,
        "ray": ray_t,
        "img+ray": torch.cat([img_t, ray_t], dim=1),
    }
    # shuffled-image null: pair each sample's image with another sample's depth/mask
    shuf = torch.from_numpy(rng.permutation(N)).to(device)

    tr = torch.from_numpy(tr_idx).to(device); va = torch.from_numpy(va_idx).to(device)
    results, val_curves, scat = {}, {}, {}

    for name, X in feats.items():
        print(f"training probe: {name} ({X.shape[1]}ch)")
        _, met, curve, pv = train_probe(
            X[tr], y_t[tr], m_t[tr], X[va], y_t[va], m_t[va],
            in_ch=X.shape[1], device=device, epochs=args.epochs)
        results[name] = met; val_curves[name] = curve
        print(f"   {met}")
        if name in ("img", "ray"):
            mm = m_t[va].reshape(-1).bool().cpu()
            gt = torch.exp(y_t[va].reshape(-1).cpu()[mm]).numpy()
            pr = torch.exp(pv.reshape(-1)[mm].clamp(np.log(0.5), np.log(120.0))).numpy()
            scat[name] = (gt, pr)

    # shuffled null
    print("training probe: img_shuf (null control)")
    Xs = img_t[shuf]
    _, met, curve, _ = train_probe(
        Xs[tr], y_t[tr], m_t[tr], Xs[va], y_t[va], m_t[va],
        in_ch=img_t.shape[1], device=device, epochs=args.epochs)
    results["img_shuf"] = met; val_curves["img_shuf"] = curve
    print(f"   {met}")

    # constant-mean floor
    mean_log = (y_t[tr] * m_t[tr]).sum() / m_t[tr].sum()
    pv_mean = torch.full_like(y_t[va], mean_log.item())
    results["mean"] = metrics(pv_mean, y_t[va], m_t[va])
    val_curves["mean"] = [float(((pv_mean - y_t[va]) ** 2 * m_t[va]).sum()
                                / m_t[va].sum())] * args.epochs
    print(f"   mean floor: {results['mean']}")

    # qualitative panels: retrain img & ray quickly to get dense maps for a few val samples
    pick = va_idx[:4]
    panels = []
    # train fresh img & ray probes to get dense per-cell maps for a few val samples
    def fit(X):
        model = PixelProbe(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for ep in range(args.epochs):
            perm2 = torch.randperm(tr.numel(), device=device)
            for s in range(0, tr.numel(), 64):
                idx = tr[perm2[s:s+64]]
                pred = model(X[idx]); mm = m_t[idx]
                loss = (((pred - y_t[idx]) ** 2) * mm).sum() / mm.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); return model
    mdl_img, mdl_ray = fit(img_t), fit(ray_t)
    from PIL import Image
    with torch.no_grad():
        for gi in pick:
            di = np.exp(mdl_img(img_t[gi:gi+1]).cpu().numpy()[0, 0].clip(np.log(0.5), np.log(120.0)))
            dr = np.exp(mdl_ray(ray_t[gi:gi+1]).cpu().numpy()[0, 0].clip(np.log(0.5), np.log(120.0)))
            rgb = np.asarray(Image.open(rgb_paths[gi]).convert("RGB").resize((W_LAT * 8, H_LAT * 8)))
            panels.append(dict(rgb=rgb, gt=depths[gi, 0], valid=valids[gi, 0].astype(bool),
                               pred_img=di, pred_ray=dr))
    scat["panels"] = panels

    carries, verdict, action = plot_all(results, val_curves, scat, args.out_dir, N, total_valid)

    out = dict(n_samples=N, n_valid_cells=total_valid, device=str(device),
               cache=str(args.cache), results=results, verdict=verdict,
               action=action.replace("\n", " "))
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2))

    md = [f"# SD-VAE depth probe\n",
          f"- samples: **{N}**, valid cells: **{total_valid:,}**, device: {device}",
          f"- cache: `{args.cache}`\n",
          "| condition | AbsRel↓ | RMSE(m)↓ | δ<1.25↑ | Pearson(log)↑ | R²↑ |",
          "|---|---|---|---|---|---|"]
    for c in ["img", "img+ray", "ray", "img_shuf", "mean"]:
        r = results[c]
        md.append(f"| {c} | {r['absrel']:.3f} | {r['rmse']:.2f} | {r['delta1']:.3f} "
                  f"| {r['pearson']:.3f} | {r['r2']:.3f} |")
    md += [f"\n## Verdict: **{verdict}**\n", action, "",
           "Plots: `metrics_bar.png`, `scatter_pred_vs_gt.png`, `training_curves.png`, "
           "`qualitative.png`, `decision_summary.png`."]
    (args.out_dir / "summary.md").write_text("\n".join(md))

    print("\n" + "=" * 70)
    print(verdict); print(action)
    print(f"\nwrote results + 5 plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
