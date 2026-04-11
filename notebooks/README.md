# FALCON H1 notebooks

## `h1.ipynb`

This is the official FALCON [`data_demos/h1.ipynb`](https://github.com/snel-repo/falcon-challenge/blob/main/data_demos/h1.ipynb) (copied into this repo for convenience). Open it in Jupyter / VS Code to explore:

- how NWBs are discovered (`*calib*.nwb`),
- a notebook-local `load_nwb` that bins spikes with `OpenLoopKinematics` timestamps and builds a blacklist from `~eval_mask`,
- raster / kinematic plots and helper utilities.

### How this relates to Raindrop Step 1 / Step 3

Step 1 / Step 3 now follow the same decoder-facing recipe as the “Training and Evaluating a Linear Decoder” section of `h1.ipynb` (see that notebook’s `prepare_train_test`):

| Piece | `h1.ipynb` | Raindrop `step1_code/h1_dataset.py` |
|--------|------------|--------------------------------------|
| Spike binning | `bin_units` on `OpenLoopKinematics` timestamps | Same. |
| Neural smoothing | `apply_exponential_filter` (from `decoder_demos.filtering`) | Same kernel in `step1_code/h1_filter.py`. |
| Targets | `create_targets(OpenLoopKinematics)` = Gaussian smooth (490 ms) + `np.gradient` | Reproduced in `step1_code/h1_notebook_targets.py` (`notebook_decoder_targets`). Not raw `OpenLoopKinematicsVelocity`. |
| Masks | `still_times \| blacklist` with `blacklist = ~eval_mask` | `eval_mask` then still `|Y| < 0.001` on all dims. |

Use `h1.ipynb` for plots and exploration; `run_step1_verify.py` / `run_step3.py` use the same preprocessing logic in Python modules so offline R² matches what your team expects from that notebook’s linear baseline (modulo train/test split: notebook uses chronological 80/20; Step 3 uses PCA + Ridge + CV as in your project spec).

To run the notebook, set paths in the first cells to your DANDI download (e.g. `~/000954/...`) and use the same conda env as `pip install -r step3_requirements.txt`.
