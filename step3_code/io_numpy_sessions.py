"""Load sessions from .npz files (contract for Step 1 → Step 3 handoff).

Each file: one session, named ``session_<id>.npz`` with arrays:
  X : (n_timepoints, n_neurons) float — binned spike counts or rates
  Y : (n_timepoints, n_kinematic_dims) float — aligned kinematics
  day_offset : scalar float — days from training session (train file should be 0.0)
  session_id : str (optional) — saved via allow_pickle if using np.savez

Alternatively provide metadata CSV (see load_sessions_from_dir).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from step3_code.session_bundle import SessionBundle


def load_sessions_from_dir(data_dir: Path, train_session_id: str | None = None) -> Tuple[List[SessionBundle], int]:
    """Load all ``*.npz`` in ``data_dir``. If train_session_id is None, pick lexicographically first as train.

    Expected keys per npz: X, Y, day_offset (float). Optional: session_id (0-d array str).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Session data directory not found: {data_dir}")

    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    sessions: List[SessionBundle] = []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float64)
        Y = np.asarray(z["Y"], dtype=np.float64)
        day = float(np.asarray(z["day_offset"]).reshape(-1)[0])
        sid = str(z["session_id"]) if "session_id" in z.files else p.stem
        sessions.append(SessionBundle(session_id=sid, day_offset=day, X=X, Y=Y))

    if train_session_id is None:
        train_idx = 0
    else:
        ids = [s.session_id for s in sessions]
        if train_session_id not in ids:
            raise ValueError(f"train_session_id={train_session_id!r} not in {ids}")
        train_idx = ids.index(train_session_id)

    return sessions, train_idx


def write_session_example_npz(out_dir: Path) -> None:
    """Write a tiny example npz for documentation (not used in pipeline by default)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.poisson(0.5, size=(100, 20)).astype(np.float64)
    Y = rng.standard_normal((100, 4)).astype(np.float64)
    np.savez(
        out_dir / "session_example.npz",
        X=X,
        Y=Y,
        day_offset=np.float64(0.0),
        session_id=np.array("example"),
    )
