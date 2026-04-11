#!/usr/bin/env python3
"""
Step 1 (minimal) verification: discover H1 NWBs, load each session, write a manifest.

Run this once you have data on disk so Step 3 (and later steps) share the same loader contract.

  export FALCON_H1_ROOT=/path/to/h1
  python run_step1_verify.py

Or:  python run_step1_verify.py --h1-root /path/to/h1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: verify FALCON H1 loading and write manifest.")
    parser.add_argument("--h1-root", type=Path, default=None, help="Directory tree containing H1 *.nwb")
    args = parser.parse_args()

    from step1_code.config import resolve_h1_data_root, STEP1_RESULTS_DIR
    from step1_code.h1_dataset import (
        choose_default_train_nwb,
        discover_h1_nwb_files,
        load_h1_session,
        write_dataset_manifest,
    )

    root = resolve_h1_data_root(args.h1_root)
    if root is None:
        sys.exit(
            "No H1 data root. Set FALCON_H1_ROOT or place NWBs under ./data/h1, or pass --h1-root.\n"
            "Then re-run: python run_step1_verify.py"
        )

    paths = discover_h1_nwb_files(root)
    train_path = choose_default_train_nwb(paths)
    sessions = [load_h1_session(p) for p in paths]
    out = write_dataset_manifest(sessions, STEP1_RESULTS_DIR / "dataset_manifest.json")

    train_rec = load_h1_session(train_path)
    print(f"H1 root: {root}")
    print(f"NWBs found: {len(paths)}")
    print(f"Default train session (earliest held-in): {train_rec.session_hash} — {train_rec.nwb_path.name}")
    print(f"Train shape: X {train_rec.X.shape}, Y {train_rec.Y.shape}")
    print(f"Wrote manifest: {out}")


if __name__ == "__main__":
    main()
