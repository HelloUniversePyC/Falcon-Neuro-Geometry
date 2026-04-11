# Session `.npz` format (for `--data-dir`)

Each file is one session. Required arrays:

| Key | Shape | dtype | Meaning |
|-----|--------|--------|---------|
| `X` | `(n_time, n_neurons)` | float | Binned spikes (or rates) |
| `Y` | `(n_time, n_kinematic_dims)` | float | Kinematics aligned to `X` |
| `day_offset` | scalar | float | **Days relative to the training session** (use `0.0` for the train file) |

Optional:

| Key | Meaning |
|-----|---------|
| `session_id` | 0-d `np.str_` or array; else the filename stem is used |

The training session is either the first file when sorted by path or the one passed as `--train-session-id`.
