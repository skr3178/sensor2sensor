# Sensor2Sensor — Dataset Details

## Training Data

- **Source:** Proprietary AV log dataset (Waymo).
- **Volume:** ~100,000 clips of ~10s duration.
- **Sensor suite per clip:** 8 surrounding cameras + 1 top-mounted LiDAR (used for 4DGS reconstruction).
- **Synthetic pairing pipeline:** Each AV log is reconstructed via 4D Gaussian Splatting (4DGS, with rigid/deformable object support), then re-rendered through virtual dashcam-style cameras. This produces paired (synthetic dashcam input, real AV-log ground truth) tuples.
- **Rendering style:** Ray-tracing-based rendering (instead of the standard 3DGS rasterizer) to better support fish-eye dashcam optics.

## Evaluation Datasets

### (a) Fixed-Camera-to-AV (paired, quantitative)
- 1,000 paired log sequences, each 3 seconds long.
- **Input:** Fixed bumper camera mounted at the front-left bumper of the AV.
- **Target:** 8-view surrounding cameras + top LiDAR on the AV.
- Used for FID / PSNR / SSIM / LPIPS / FVD / Chamfer Distance.

### (b) In-the-Wild (qualitative + human eval)
- Manually collected, uncurated third-party footage.
- Sources: internet driving videos, phone (smartphone) recordings, ADAS recordings, manually captured dashcams (e.g., Nexar).
- Used to test generalization across unseen intrinsics, extrinsics, weather, and content.

### Human Evaluation
- 26 participants.
- 40 × 3 generated image + LiDAR samples ranked as *best / middle / worst* (vs. X-Drive and "Ours w/o VC").
- Yields top-rank and pair-wise preference rates.

## Dashcam Parameter Distribution (synthesis pipeline)

Two-stage sampling for virtual third-party cameras:

1. **Extrinsics (p_e = [R | t], 6-DoF, relative to vehicle frame)**
   - Vehicle category sampled first (e.g., Sedan, SUV).
   - Category-specific pose distributions, e.g. for Sedan:
     - Height: 1.1 – 1.3 m
     - Forward translation: 2.0 – 2.5 m
     - Pitch: ± 10°
   - Small rotational perturbations (θ_p, θ_y, θ_r) to simulate imperfect installation.

2. **Intrinsics (p_i, κ)**
   - Focal length, principal point, and distortion coefficients drawn from a calibrated bank of real-world dashcams (e.g., Nexar, VIOFO).
   - Uniform noise augmentation, e.g. ± 5 % focal length jitter.

3. **Post-processing**
   - Exposure compensation.
   - Gamma correction for lighting normalization.

## LiDAR Representation

- **Native format:** range-view spin image, shape `[H_L, W_L, D_L]` with `D_L = 4` channels:
  1. Range (depth, meters)
  2. Intensity (amount of light reflected)
  3. Elongation (extent to which the waveform has been "flattened")
  4. Validity (1 = return, 0 = otherwise)
- **Rows ↔ elevation angle; columns ↔ azimuth angle.**
- `(row, col, range)` ↔ 3D Euclidean `(x, y, z)` via vehicle trajectory + sensor calibration.
- **Normalization:** range clamped at 150 m, then linearly scaled to [0, 1]. Intensity and elongation are similarly normalized to [0, 1].

## Baselines Used for Comparison

- **Reconstruction-based:** VGGT, π³ (feedforward 3D scene reconstruction).
- **Generative:**
  - X-Drive (image–LiDAR co-generation conditioned on dashcam via attention).
  - CAT3D adapted to: (i) LiDAR via shared VAE, (ii) channel-concatenation (CC) instead of view-concatenation (VC) — referred to as "Ours w/o VC".

## Downstream / Sim-to-Real Tasks

- **LiDAR detection:** vehicle-detection model run on real vs. generated LiDAR.
- **Image segmentation:** Panoptic-DeepLab applied to real vs. generated images.
- Both used to verify that perception models trained on real data transfer to generated data without finetuning.
