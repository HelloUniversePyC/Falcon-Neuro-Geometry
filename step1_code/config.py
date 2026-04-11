"""Step 1 configuration: data roots and H1 constants."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STEP1_RESULTS_DIR = REPO_ROOT / "step1_results"

# Match falcon_challenge H1 preprocessing
H1_BIN_SIZE_S = 0.02
H1_N_KINEMATIC_DIMS = 7


def resolve_h1_data_root(explicit: Path | None = None) -> Path | None:
    """Return directory tree to search for ``*.nwb``, or None if not configured."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_dir() else None
    env = os.environ.get("FALCON_H1_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    guess = REPO_ROOT / "data" / "h1"
    if guess.is_dir() and any(guess.rglob("*.nwb")):
        return guess.resolve()
    return None
