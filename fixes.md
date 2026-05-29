Each row classified
Row	Diff	Meaningful for our single-frame, single-camera task?
U-Net params (250M vs 15M)	~17×	Partly. Roughly half of the paper's 250M is the camera-side U-Net tower (for joint image generation). The LiDAR-side alone is ~125M, so the "comparable" gap is ~8-9×, not 17×. Still real.
Channels per stage	3.3× narrower	Yes. Direct consequence of the param gap.
LiDAR latent dim (16 vs 8)	2× thinner	Yes — independent of everything else. The LiDAR VAE's representational ceiling. No temporal/multi-view involvement.
Multi-stream architecture	dual-tower vs single	✗ No — half the towers exist to generate the 8 surrounding camera views. We don't generate any images. Scope-deliberately-excluded in our (A) scope.
Cross-sensor attn (bidirectional vs one-way)	—	✗ No — bidirectional only needed because the camera tower is generating and needs LiDAR signal back. One-way (LiDAR ← image) is correct for image-conditioning-only.
Hardware	128 TPUs vs 1× 3060	—
LR	5e-5 vs 2e-4	—
Train data (28k vs 4k)	~7×	Yes — real diversity gap. Secondary signal: no overfit gap → adding data alone won't help until capacity expands.
Training steps (120k vs 12.6k)	~10×	Partly. Of the paper's 120k, 40k is temporal fine-tuning (dense previous-frame conditioning — irrelevant to us). The 80k base-stage is the only comparable number. So ~6×, not 10×. And we plateaued, so more steps alone are diminishing returns.
4DGS synthetic data (not in table above)	—	✗ No — exists to support DAgger rollouts (Phase 3). Pure temporal/autonomous-driving concern.
