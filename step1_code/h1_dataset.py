"""
FALCON H1 — minimal Step 1 pipeline: discover sessions, load NWB, bin spikes, align kinematics.

Downstream steps (e.g. Step 3 PCA+Ridge) should import from here rather than calling ``load_nwb`` directly.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from step1_code.config import H1_BIN_SIZE_S, H1_N_KINEMATIC_DIMS, STEP1_RESULTS_DIR
from pynwb import NWBHDF5IO

from step1_code.h1_filter import apply_exponential_filter
from step1_code.h1_notebook_targets import notebook_decoder_targets

try:
    from falcon_challenge.config import FalconConfig, FalconTask, H1_NEW_TO_OLD
    from falcon_challenge.dataloaders import bin_units
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Step 1 requires falcon-challenge (and h5py). Install: pip install -r step3_requirements.txt"
    ) from exc


def nwb_stem(path: Path) -> str:
    return path.stem


def session_hash_from_path(path: Path) -> str:
    """Benchmark session id (e.g. ``S6_set_1``) from NWB filename."""
    handle = nwb_stem(path)
    cfg = FalconConfig(task=FalconTask.h1)
    if "sub-HumanPitt" in handle and "_ses-" in handle:
        raw = handle.split("_ses-")[-1]
        token = raw.split("_")[0]
        if token not in H1_NEW_TO_OLD:
            raise KeyError(f"Unknown H1 session token {token!r} in {path.name}")
        return H1_NEW_TO_OLD[token]
    try:
        return cfg.hash_dataset(handle)
    except (ValueError, KeyError):
        return handle


def calendar_date_from_path(path: Path) -> dt.date:
    """Anonymized calendar date from DANDI-style H1 filename (or fallback from session index)."""
    handle = nwb_stem(path)
    if "sub-HumanPitt" in handle and "_ses-" in handle:
        raw = handle.split("_ses-")[-1]
        token = raw.split("_")[0]
        ymd = token[:8]
        return dt.datetime.strptime(ymd, "%Y%m%d").date()
    m = re.search(r"S(\d+)_", session_hash_from_path(path))
    if m:
        return dt.date(2000, 1, 1) + dt.timedelta(days=int(m.group(1)) * 7)
    return dt.date(2000, 1, 1)


def day_index_from_hash(session_hash: str) -> int:
    m = re.match(r"S(\d+)_", session_hash)
    return int(m.group(1)) if m else 0


def is_held_in_session(session_hash: str) -> bool:
    """Held-in = S0–S5 in the public H1 split."""
    return day_index_from_hash(session_hash) <= 5


def discover_h1_nwb_files(root: Path) -> List[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"H1 data root is not a directory: {root}")
    files = sorted({p.resolve() for p in root.rglob("*.nwb")})
    if not files:
        raise FileNotFoundError(f"No .nwb files under {root}.")
    return files


def align_neuron_columns(X: np.ndarray, n_target: int) -> np.ndarray:
    """Pad or truncate neuron dimension to ``n_target`` (reference = train session)."""
    _t, c = X.shape
    if c == n_target:
        return X
    if c < n_target:
        return np.pad(X, ((0, 0), (0, n_target - c)), mode="constant", constant_values=0.0)
    return X[:, :n_target].copy()


@dataclass
class H1Session:
    """One session: spikes + targets aligned with ``data_demos/h1.ipynb`` linear decoder.

    - **Spikes:** ``bin_units`` on 20 ms grid, causal exponential filter (τ=240 ms).
    - **Targets:** ``OpenLoopKinematics`` → Gaussian smooth (490 ms kernel) → ``np.gradient``
      (same as notebook ``create_targets``), **not** raw ``OpenLoopKinematicsVelocity``.
    - **Masks:** ``eval_mask`` then still-time removal ``|Y| < 1e-3`` on all dims (notebook rule).
    """

    nwb_path: Path
    session_hash: str
    calendar_date: dt.date
    X: np.ndarray  # (n_bins, n_units)
    Y: np.ndarray  # (n_bins, n_kin)
    bin_size_s: float = H1_BIN_SIZE_S

    @property
    def n_bins(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_units(self) -> int:
        return int(self.X.shape[1])

    def to_manifest_row(self) -> Dict[str, Any]:
        return {
            "nwb_stem": nwb_stem(self.nwb_path),
            "nwb_path": str(self.nwb_path),
            "session_hash": self.session_hash,
            "calendar_date": self.calendar_date.isoformat(),
            "held_in_split": is_held_in_session(self.session_hash),
            "n_bins": self.n_bins,
            "n_units": self.n_units,
            "n_kinematic_dims": int(self.Y.shape[1]),
            "bin_size_s": self.bin_size_s,
        }


def load_h1_session(nwb_path: Path) -> H1Session:
    """Load one H1 NWB using the same ingredients as ``notebooks/h1.ipynb`` decoder section."""
    nwb_path = Path(nwb_path)
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwbfile = io.read()
        units = nwbfile.units.to_dataframe()
        kin_pose = np.asarray(nwbfile.acquisition["OpenLoopKinematics"].data[:], dtype=np.float64)
        ts = nwbfile.acquisition["OpenLoopKinematics"].offset + np.arange(
            kin_pose.shape[0]
        ) * float(nwbfile.acquisition["OpenLoopKinematics"].rate)
        eval_mask = np.asarray(nwbfile.acquisition["eval_mask"].data[:]).astype(bool)

    X = np.asarray(bin_units(units, bin_size_s=H1_BIN_SIZE_S, bin_timestamps=ts), dtype=np.float64)
    Y = notebook_decoder_targets(kin_pose)
    if X.shape[0] != Y.shape[0] or X.shape[0] != len(eval_mask):
        n = min(X.shape[0], Y.shape[0], len(eval_mask))
        X, Y, eval_mask = X[:n], Y[:n], eval_mask[:n]

    m = eval_mask
    X, Y = X[m], Y[m]
    still = np.all(np.abs(Y) < 0.001, axis=1)
    X, Y = X[~still], Y[~still]
    X = apply_exponential_filter(X)

    if Y.shape[1] != H1_N_KINEMATIC_DIMS:
        pass
    return H1Session(
        nwb_path=nwb_path.resolve(),
        session_hash=session_hash_from_path(nwb_path),
        calendar_date=calendar_date_from_path(nwb_path),
        X=X,
        Y=Y,
    )


def choose_default_train_nwb(paths: List[Path]) -> Path:
    """Earliest calendar date among held-in (S0–S5); if none, earliest overall."""
    held_in = [p for p in paths if is_held_in_session(session_hash_from_path(p))]
    pool = held_in if held_in else paths
    return min(pool, key=lambda p: (calendar_date_from_path(p), nwb_stem(p)))


def write_dataset_manifest(sessions: List[H1Session], out_path: Optional[Path] = None) -> Path:
    """Write JSON manifest of loaded sessions (counts, paths, split flags)."""
    out_path = out_path or (STEP1_RESULTS_DIR / "dataset_manifest.json")
    STEP1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_sessions": len(sessions),
        "bin_size_s": H1_BIN_SIZE_S,
        "preprocessing": "h1.ipynb parity: bin_units 20ms; OpenLoopKinematics → smooth(490ms) → np.gradient; eval_mask; still |Y|<1e-3; exp filter τ=240ms on spikes",
        "sessions": [s.to_manifest_row() for s in sessions],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
