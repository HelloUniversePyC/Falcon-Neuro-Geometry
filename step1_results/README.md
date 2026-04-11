# Step 1 outputs (generated)

Run `python run_step1_verify.py` after setting `FALCON_H1_ROOT` (or `--h1-root`). This writes:

- `dataset_manifest.json` — one entry per NWB: path, benchmark `session_hash`, date, held-in flag, bin counts, neuron count, kinematic dims.

The loader implementation lives in `step1_code/h1_dataset.py`; Step 3 imports it via `step3_code/io_falcon_h1.py`.
