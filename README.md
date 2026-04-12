# Improving Motor BCI Kinematics Decoding Accuracy in the Presence of Neuronal Drift

**NEURO 120 Final Project** | Harvard University, Spring 2026

Alliyah Steele, Rishab Jain, & Diya Sreedhar

## Overview

This project investigates whether representational drift in the motor cortex can be explained as a geometric transformation of a stable low-dimensional neural manifold, and whether decoders that capture geometry (flow matching) are more robust to drift than standard linear decoders.

We compare four decoder models on the [FALCON H1 dataset](https://dandiarchive.org/dandiset/000954) (Utah array recordings from human motor cortex, 176 channels, 13 sessions over ~40 days, 7 DoF kinematics):

| Model | Input | Decoding |
|-------|-------|----------|
| **Linear (176-ch)** | Full 176-channel binned spikes | Ridge regression |
| **Linear (PCA-32)** | 32-dim PCA latent | Ridge regression |
| **FM-Vanilla** | 128-dim PCA latent | Flow matching (ODE integration) |
| **FM-Raw** | Raw spikes via learned encoder | Flow matching (ODE integration) |
| **FM-Ablated** | 128-dim PCA latent | Direct MLP (no flow matching) |

## Key Results So Far

- **Neural drift is real**: Median per-neuron coefficient of variation = 0.46 across sessions; pairwise session correlations decay at ~-0.005 r/day
- **Population geometry is preserved**: PCA projections colored by reach direction show consistent directional clustering across all 13 sessions, consistent with [Gallego et al. (2020)](https://www.nature.com/articles/s41593-019-0555-4)
- **Linear decoder**: Strong within-session decoding but sharp cross-day degradation
- **FM-Vanilla**: Within-day r-squared = 0.46 (>2x FALCON linear baseline of 0.195), but also struggles across sessions

## Repository Structure

```
.
├── drift_characterization.ipynb   # Neural drift analysis (Rishab)
├── figures/                       # Exported analysis figures
│   ├── fig_drift_composite_3panel.png  # Main composite figure
│   ├── fig_heatmap.png
│   ├── fig_corr_matrix.png
│   ├── fig_pca_directions.png
│   └── ...
├── models/
│   ├── fm_decoder.ipynb           # Flow matching decoder (Alliyah)
│   ├── linear_decoder.py          # Linear decoder (Diya)
│   └── step3_results/             # Linear decoder evaluation outputs
├── falcon_challenge/              # FALCON benchmark utilities
│   ├── config.py                  # Task configs, session mappings
│   ├── dataloaders.py             # NWB loading, spike binning
│   └── interface.py               # Decoder interface spec
├── data_demos/                    # Dataset exploration notebooks
├── decoder_demos/                 # FALCON baseline decoders
└── data/                          # Downloaded NWB files (not tracked)
```

## Setup

### Requirements

```bash
pip install falcon-challenge hydra-core pynwb dandi
pip install torch scikit-learn matplotlib seaborn
```

### Data Download

```bash
cd data && dandi download https://dandiarchive.org/dandiset/000954/draft
```

This downloads ~102 MB of NWB files (27 calibration files across 13 sessions).

### Running the Drift Characterization

```bash
jupyter notebook drift_characterization.ipynb
```

Run all cells top-to-bottom. Requires data in `data/000954/` (DANDI layout) or `data/h1/` (FALCON layout).

### Running the Linear Decoder

```bash
python models/linear_decoder.py
```

### Running the Flow Matching Decoder

Open `models/fm_decoder.ipynb` in Jupyter and run all cells.

## References

- Gallego, J. A., et al. (2020). Long-term stability of cortical population dynamics underlying consistent behavior. *Nature Neuroscience*.
- Karpowicz, B. M., et al. (2024). Few-shot algorithms for consistent neural decoding (FALCON) benchmark. *bioRxiv*.
- Wang, P., et al. (2025). Flow Matching for Few-Trial Neural Adaptation with Stable Latent Dynamics. *ICML 2025*.
- Natraj, N., et al. (2025). Sampling representational plasticity of simple imagined movements across days. *Cell*.

## Original FALCON Benchmark

This repo is forked from the [FALCON benchmark repository](https://snel-repo.github.io/falcon/). See the original documentation below for challenge submission details.

<details>
<summary>FALCON Challenge Submission Instructions</summary>

### Docker Submission
```bash
docker build -t sk_smoke -f ./decoder_demos/sklearn_sample.Dockerfile .
bash test_docker_local.sh --docker-name sk_smoke
```

### EvalAI Submission
```bash
evalai push mysubmission:latest --phase few-shot-<test/minival>-2319 --private
```

See the [EvalAI submission tab](https://eval.ai/web/challenges/challenge-page/2319/submission) for full instructions.

</details>
