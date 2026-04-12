# Step 3 results (real FALCON H1)

This directory is a copy under `models/step3_results/` on branch `main` for browsing next to [`linear_decoder.py`](https://github.com/HelloUniversePyC/Falcon-Neuro-Geometry/blob/main/models/linear_decoder.py). The pipeline that writes these artifacts lives on branch `diya` (or your Raindrop checkout); `RESULTS_DIR` there defaults to repo-root `step3_results/`, not this path.

Outputs from `python run_step3.py` with `FALCON_H1_ROOT` set (or `--data-dir` for exported `.nwb`-compatible `.npz`).

## Latest saved results (FALCON H1)

From the last `python run_step3.py` on dandiset 000954 (default train S0_set_1). Re-run Step 3 to refresh; numbers match `summary.json` / `metrics_by_session.csv`.

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

---

## Alignment with your Step 3 write-up

| Proposal item | What the code does |
|----------------|-------------------|
| 32-dim PCA on train spikes | `PCA(n_components=32, whiten=True)` fit only on the training session’s smoothed, still-masked spikes (`models/linear_decoder.py`, `step3_code/config.py`). |
| Project all sessions | Same `StandardScaler` + PCA transform for every session before Ridge. |
| Ridge (latents → kinematics) | `RidgeCV` on 32-D latents → full kinematic vector (default 7 H1 dims after Step 1 preprocessing). |
| Within-session decoding | 5-fold shuffled CV on the train session: refit Ridge each fold; PCA fixed. Primary score uses `variance_weighted` R² (standard in multi-output regression when dimensions have different scales). |
| Across held-out sessions | One Ridge model (fit on full train session); evaluated on every other session; CSV stores `r2_primary` and `r2_uniform`. |
| R² vs session distance | `r2_vs_session_distance.png`: held-out points aggregated by `(session_id, days_from_train)` (mean if multiple NWB map to the same tag), optional linear trend overlay, horizontal lines for 32-D PCA vs 176-D reference (below). |

### Why “~0.5–0.7 within-session” matches full-dimensional Ridge, not always 32-D PCA

32-D PCA is intentionally lossy: it keeps only 32 directions of population variance, so a linear readout on latents typically scores below a Ridge model on all 176 smoothed channels (same preprocessing). On your H1 download, after Step 1’s Falcon-style smoothing + still masking, the pipeline also reports a reference metric: the same CV protocol, but Ridge on z-scored 176-D spikes (no PCA). That reference is usually in the ~0.5–0.7 band and is the right quantity to compare to the literature / Karpowicz-style expectations, while the 32-D PCA + Ridge line is the one required for Step 5+ (FM on latents).

Check `summary.json`:

- `within_session_cv_r2_primary_mean` — 32-D PCA path (headline for the latent pipeline).
- `reference_full_dim_ridge_cv_primary_mean` — 176-D path (closer to “strong linear decoding” in papers).

### Step 1 preprocessing (aligned with `data_demos/h1.ipynb`)

Targets match the notebook’s linear decoder cells: `OpenLoopKinematics` → Gaussian smooth (490 ms) → `np.gradient` (`step1_code/h1_notebook_targets.py`), not raw `OpenLoopKinematicsVelocity`. Spikes use `bin_units` (20 ms) then a causal exponential filter (τ ≈ 240 ms, `step1_code/h1_filter.py`). Masks: `eval_mask`, then still bins where all target components are near zero. See `load_h1_session` in `step1_code/h1_dataset.py`.

### Cross-session “drop-off with distance”

Held-out sessions use calendar-day distance from the train session (from NWB filenames). The plot aggregates duplicate `session_id` rows and draws a least-squares trend through aggregated held-out R² vs days to visualize drift (not guaranteed monotone session-by-session).

## How to regenerate

1. `pip install -r step3_requirements.txt`
2. `export FALCON_H1_ROOT=/path/to/000954` (or parent of all `.nwb`)
3. `python run_step1_verify.py` (optional manifest)
4. `python run_step3.py`

## Output files

| File | Description |
|------|-------------|
| `metrics_by_session.csv` | `r2_primary` (variance-weighted), `r2_uniform`, days from train |
| `summary.json` | CV folds, per-dim R², reference 176-D Ridge CV |
| `scaler.joblib`, `pca.joblib`, `ridge.joblib` | Fitted 32-D pipeline |
| `r2_vs_session_distance.png` | Drift figure (aggregated held-out + trend + reference line) |

Tuning knobs live in `step3_code/config.py` (`NEURAL_LAG_BINS`, `PCA_WHITEN`, `KINEMATIC_DIM_INDICES`, `RIDGE_ALPHAS`, `R2_MULTIOUTPUT_PRIMARY`).
