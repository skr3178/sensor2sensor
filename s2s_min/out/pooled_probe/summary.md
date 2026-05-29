# Pooled / raymap / channel-reduction probes

- 600 samples, pooled to (8,64)

## Probes 1&2 (pooled 8×64)
| condition | Pearson | AbsRel | δ<1.25 | R² |
|---|---|---|---|---|
| sdvae_p | 0.409 | 0.673 | 0.202 | 0.166 |
| sdvae+ray_p | 0.831 | 0.356 | 0.543 | 0.691 |
| raymap_p | 0.855 | 0.329 | 0.582 | 0.730 |
| dinov2_p | 0.958 | 0.159 | 0.784 | 0.918 |
| dinov2+ray_p | 0.958 | 0.157 | 0.786 | 0.918 |
| combo14_p | 0.949 | 0.174 | 0.747 | 0.901 |
| mean_p | -0.000 | 0.753 | 0.175 | -0.000 |

## Probe 3 (channels → Pearson @32×56)
| N | learned | PCA |
|---|---|---|
| 4 | 0.941 | 0.776 |
| 8 | 0.943 | 0.876 |
| 16 | 0.945 | 0.918 |
| 32 | 0.946 | 0.928 |
| 64 | 0.949 | 0.937 |
| 384 | 0.951 | — |

- pooling loss (dinov2+ray): -0.005
- dinov2 margin over sdvae (pooled): +0.127
- min learned channels for r≥0.85: 4