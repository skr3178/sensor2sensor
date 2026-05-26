# Sensor2Sensor — Equations

## (1) LiDAR VAE Total Loss

Jointly encodes depth (range), intensity, elongation, and validity for spin images:

$$
\mathcal{L}^{\text{TOTAL}} = \mathcal{L}^{\text{L1}}_{\text{range}} + \mathcal{L}^{\text{L1}}_{\text{elongation}} + \mathcal{L}^{\text{L1}}_{\text{intensity}} + \mathcal{L}^{\text{BCE}}_{\text{validity}} + \mathcal{L}^{\text{LPIPS}}_{\text{normals}} + \mathcal{L}^{\text{LPIPS}}_{\text{elongation}} + \mathcal{L}^{\text{LPIPS}}_{\text{intensity}} + \mathcal{L}^{\text{LPIPS}}_{\text{validity}} + \mathcal{L}^{\text{KL}}
$$

- `L1` losses on continuous signals (range, elongation, intensity).
- `BCE` on the binary validity mask.
- `LPIPS` on surface normals (derived from range), elongation, intensity, validity.
- `KL` regularizes the latent space.

The normals term uses `f^L_normals = ComputeNormals(f^L_range)` computed via finite differences on projected 3D LiDAR points.

---

## (2) Auto-regressive Video Generation

Given the third-party dashcam frame `x_t` at time step `t > 0`, the model predicts:

$$
P(C_t, L_t \mid x_t, C_{t-1}, L_{t-1})
$$

- `C_t = {c_i^t}_{i=1}^N`: multi-view images at time `t` (N = 8 surrounding cameras).
- `L_t`: LiDAR point cloud (range-image latent) at time `t`.
- When `t = 0`, sensor data is generated conditioning only on `x_0`.

---

## (3) L1 Reconstruction Loss (per signal)

For signal ∈ {range, elongation, intensity}:

$$
\mathcal{L}^{\text{L1}}_{\text{signal}} = \lambda_{\text{signal}} \, \lVert f^L_{\text{signal}} - \hat{f}^L_{\text{signal}} \rVert_1
$$

- `f^L_signal`: ground-truth LiDAR feature map.
- `\hat{f}^L_signal`: VAE reconstruction.
- `λ_signal`: per-signal scalar weight.

---

## (4) Binary Cross-Entropy on Validity Mask

$$
\mathcal{L}^{\text{BCE}}_{\text{validity}} = -\lambda_{\text{BCE}} \big[\, f^L_{\text{valid}} \, \log(\hat{f}^L_{\text{valid}}) + (1 - f^L_{\text{valid}}) \, \log(1 - \hat{f}^L_{\text{valid}}) \,\big]
$$

- `f^L_valid ∈ {0,1}` is the ground truth validity mask.
- `\hat{f}^L_valid` is the predicted validity probability map.

---

## (5) LPIPS Perceptual Loss

Pre-trained network (e.g., VGG) features compared layer-wise:

$$
\mathcal{L}_{\text{LPIPS}}(x, \hat{x}) = \sum_i \frac{1}{H_i W_i} \sum_{h,w} \big\lVert w_i \odot (y^i_{hw} - \hat{y}^i_{hw}) \big\rVert_2^2
$$

- `i` indexes network layers used for the comparison.
- `y^i_{hw}` and `\hat{y}^i_{0,hw}`: unit-normalized feature activation vectors at spatial position `(h, w)` of images `x` and `x_0` respectively.
- `H_i, W_i`: spatial dimensions of layer `i`'s feature map.
- `w_i`: learned channel-wise weight vector matched to human perceptual judgments.
- `⊙`: element-wise product.

---

## (6) LPIPS Loss on LiDAR Signal Maps

For each LiDAR signal type (normals, elongation, intensity, validity):

$$
\mathcal{L}^{\text{LPIPS}}_{\text{signal}} = \lambda_{\text{signal}} \, \mathcal{L}_{\text{LPIPS}}(f^L_{\text{signal}}, \hat{f}^L_{\text{signal}})
$$

`λ_signal` is the per-signal weight.

---

## (7) KL Divergence Regularization

Latent space regularized to a standard normal prior:

$$
\mathcal{L}^{\text{KL}} = \frac{1}{2} \, \lambda_{\text{KL}} \sum_{j=1}^{D} \left( \mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1 \right)
$$

- Standard KL between encoder posterior `N(μ, σ²)` and prior `N(0, I)`.
- `D`: latent dimensionality.
- Encoder outputs per-dimension mean `μ_j` and variance `σ_j²`.
- `λ_KL`: balances regularization vs. reconstruction.
