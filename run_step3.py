#!/usr/bin/env python3
"""
Run Step 3 from repo root on **real FALCON H1** (or your own ``.npz`` sessions).

Requires downloaded H1 NWBs. No synthetic/demo mode — outputs are for real data only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _discover_default_h1_root() -> Path | None:
    from step1_code.config import resolve_h1_data_root

    return resolve_h1_data_root(None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: 32-D PCA (train session only) + Ridge, R² vs session distance (real data)."
    )
    parser.add_argument(
        "--h1-root",
        type=Path,
        default=None,
        help="Directory tree containing FALCON H1 *.nwb (e.g. parent of held_in_calib/).",
    )
    parser.add_argument(
        "--train-nwb-stem",
        type=str,
        default=None,
        help="Filename stem of the training NWB (default: earliest held-in S0–S5 session by date).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder of session *.npz (X, Y, day_offset) if you export sessions yourself.",
    )
    parser.add_argument(
        "--train-session-id",
        type=str,
        default=None,
        help="With --data-dir: which session_id is train (default: first file).",
    )
    args = parser.parse_args()

    if args.data_dir is not None:
        from step3_code.step3_pipeline import main_from_npz

        main_from_npz(args.data_dir.resolve(), train_session_id=args.train_session_id)
        return

    h1 = args.h1_root.resolve() if args.h1_root is not None else _discover_default_h1_root()
    if h1 is None or not h1.is_dir():
        sys.exit(
            "No FALCON H1 data found. This project records Step 3 results only on real data.\n"
            "  1) Download H1 NWBs (FALCON / DANDI 000954), then:\n"
            "       export FALCON_H1_ROOT=/path/to/folder/containing/nwb/files\n"
            "       python run_step1_verify.py\n"
            "  2) Run Step 3:\n"
            "       python run_step3.py\n"
            "     (or: python run_step3.py --h1-root /path/to/h1)\n"
            "  If you use exported .npz sessions instead:\n"
            "       python run_step3.py --data-dir /path/to/npz\n"
        )

    from step3_code.step3_pipeline import main_from_falcon_h1

    main_from_falcon_h1(h1, train_nwb_stem=args.train_nwb_stem)


if __name__ == "__main__":
    main()
