# Step 3 results (FALCON H1)

### Within-session (5-fold CV on train session)

| Metric | Value |
|--------|------:|
| 32-D PCA + Ridge — CV R² primary (`variance_weighted`) | 0.440 |
| Per-fold primary R² | 0.440, 0.452, 0.449, 0.437, 0.423 |
| 176-D Ridge (no PCA) — same CV (reference) | 0.609 |
| Per-fold reference R² | 0.619, 0.618, 0.601, 0.607, 0.597 |
| Train in-sample R² (primary / uniform) | 0.448 / 0.349 |
| Chosen Ridge α (32-D model) | 38.75 |
| PCA | 32 components, `whiten=True` |
| Neural lag | 5 bins × 20 ms (≈100 ms) |
| Neurons × kinematic dims | 176 × 7 |

### Train session (`train_session_refit_eval`)

| session_id | days from train | R² primary | R² uniform | n samples |
|------------|----------------:|-----------:|-----------:|----------:|
| S0_set_1 | 0 | 0.448 | 0.349 | 5769 |

### Held-out sessions (mean R² primary if multiple NWBs share `session_id` + day)

| session_id | days from train | R² primary | n samples |
|------------|----------------:|-----------:|----------:|
| S0_set_1 | 0 | 0.555 | 780 |
| S0_set_2 | 0 | 0.163 | 4117 |
| S1_set_3 | 7 | 0.147 | 4256 |
| S1_set_2 | 7 | 0.174 | 4603 |
| S1_set_1 | 7 | 0.060 | 7363 |
| S2_set_2 | 12 | −0.097 | 6972 |
| S2_set_1 | 12 | 0.006 | 6653 |
| S3_set_2 | 14 | −0.207 | 7332 |
| S3_set_1 | 14 | −0.227 | 6123 |
| S4_set_1 | 18 | −0.226 | 7118 |
| S4_set_2 | 18 | −0.090 | 7077 |
| S5_set_1 | 19 | −0.222 | 7063 |
| S5_set_2 | 19 | −0.221 | 6583 |
| S6_set_2 | 25 | −0.660 | 1236 |
| S6_set_1 | 25 | −0.602 | 1202 |
| S7_set_2 | 26 | −0.573 | 1376 |
| S7_set_1 | 26 | −0.416 | 1272 |
| S8_set_2 | 28 | −1.018 | 1189 |
| S8_set_1 | 28 | −1.012 | 1297 |
| S9_set_2 | 32 | −1.081 | 1324 |
| S9_set_1 | 32 | −0.866 | 1261 |
| S10_set_1 | 33 | −1.454 | 1204 |
| S10_set_2 | 33 | −0.621 | 1233 |
| S11_set_1 | 36 | −0.116 | 1278 |
| S11_set_2 | 36 | −0.094 | 1265 |
| S12_set_2 | 39 | −0.175 | 1378 |
| S12_set_1 | 39 | −0.275 | 1392 |

Held-out (all CSV rows, unaggregated): mean R² primary ≈ −0.25, min −1.45, max 0.55. Figure: `r2_vs_session_distance.png`.

### Per-dimension train R² (in-sample, 32-D model)

0.241, 0.205, 0.188, 0.438, 0.478, 0.473, 0.417 (dims 1–7).
