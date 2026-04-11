"""Shared types for Step 3 pipelines (real or exported sessions)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SessionBundle:
    session_id: str
    day_offset: float  # ordinal or day metric; differences used for "distance"
    X: np.ndarray  # (n_timepoints, n_neurons)
    Y: np.ndarray  # (n_timepoints, n_kin)


def corpus_to_arrays(sessions: List[SessionBundle], train_idx: int) -> Tuple[SessionBundle, List[SessionBundle]]:
    train = sessions[train_idx]
    holdouts = [sessions[i] for i in range(len(sessions)) if i != train_idx]
    return train, holdouts
