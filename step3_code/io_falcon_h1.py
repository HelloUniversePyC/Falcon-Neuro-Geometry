"""Bridge Step 3 to Step 1: build ``SessionBundle`` lists from ``step1_code.h1_dataset``."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from step1_code.h1_dataset import (
    align_neuron_columns,
    calendar_date_from_path,
    choose_default_train_nwb,
    discover_h1_nwb_files,
    load_h1_session,
    nwb_stem,
)
from step3_code.session_bundle import SessionBundle


def load_h1_corpus(
    root: Path,
    *,
    train_nwb_stem: Optional[str] = None,
) -> Tuple[List[SessionBundle], int]:
    """Load all H1 sessions under ``root`` (Step 1 loader), return Step 3 bundles + train index.

    Default train session: earliest held-in (S0–S5) by date (``choose_default_train_nwb``).
    ``day_offset`` is ``calendar_date.toordinal()`` for day-distance in Step 3 plots.
    """
    paths = discover_h1_nwb_files(root)

    if train_nwb_stem is not None:
        train_path = next((p for p in paths if nwb_stem(p) == train_nwb_stem or p.name == train_nwb_stem), None)
        if train_path is None:
            raise ValueError(f"No NWB with stem/name {train_nwb_stem!r} under {root}")
    else:
        train_path = choose_default_train_nwb(paths)

    train_rec = load_h1_session(train_path)
    train_n = train_rec.n_units

    ordered = sorted(paths, key=lambda p: (calendar_date_from_path(p), nwb_stem(p)))
    sessions: List[SessionBundle] = []
    train_idx: Optional[int] = None
    for i, p in enumerate(ordered):
        rec = load_h1_session(p)
        Xa = align_neuron_columns(rec.X, train_n)
        sessions.append(
            SessionBundle(
                session_id=rec.session_hash,
                day_offset=float(rec.calendar_date.toordinal()),
                X=Xa,
                Y=rec.Y,
            )
        )
        if p.resolve() == train_path.resolve():
            train_idx = i
    if train_idx is None:  # pragma: no cover
        raise RuntimeError("Train path not found in ordered session list.")
    return sessions, train_idx
