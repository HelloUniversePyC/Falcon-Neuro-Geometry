"""Paths and hyperparameters for Step 3."""
from pathlib import Path

import numpy as np

# Repo root (parent of step3_code/)
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "step3_results"

N_PCA_COMPONENTS = 32
RANDOM_SEED = 42

# Ridge grid (log-spaced, similar spirit to falcon sklearn GridSearchCV)
RIDGE_ALPHAS = tuple(float(x) for x in np.logspace(-1.0, 3.0, 18))

# Causal alignment: neural at t predicts kinematics at t + lag (20 ms bins; ~100 ms is typical).
NEURAL_LAG_BINS = 5

# PCA on z-scored spikes (whitening often helps linear readouts)
PCA_WHITEN = True

# Primary headline R² (proposal / plots). ``variance_weighted`` upweights high-variance dims.
R2_MULTIOUTPUT_PRIMARY = "variance_weighted"
R2_MULTIOUTPUT_UNIFORM = "uniform_average"

# None = all H1 outputs (7). Set e.g. (0, 1, 2) to decode a subset only.
KINEMATIC_DIM_INDICES: tuple[int, ...] | None = None

# Optional: directory of .npz session files (see io_numpy_sessions.py)
SESSION_DATA_DIR = REPO_ROOT / "step3_session_data"

CV_SPLITS = 5
