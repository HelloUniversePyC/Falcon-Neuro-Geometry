"""H1 neural preprocessing aligned with falcon-challenge ``decoder_demos/filtering.py``."""
from __future__ import annotations

import numpy as np
from scipy import signal

# Same defaults as snel-repo/falcon-challenge decoder_demos/filtering.py
NEURAL_TAU_MS = 240.0
H1_BIN_SIZE_MS = 20.0


def apply_exponential_filter(
    x: np.ndarray,
    *,
    tau_ms: float = NEURAL_TAU_MS,
    bin_size_ms: float = H1_BIN_SIZE_MS,
    extent: float = 1.0,
) -> np.ndarray:
    """Causal exponential smoothing per channel (time × units)."""
    if x.ndim != 2:
        raise ValueError("apply_exponential_filter expects (time, channels)")
    t = np.arange(0, extent * tau_ms, bin_size_ms)
    kernel = np.exp(-t / tau_ms)
    kernel /= np.sum(kernel)
    xf = x.astype(np.float64, copy=False)
    out = np.empty_like(xf, dtype=np.float64)
    for ch in range(xf.shape[1]):
        out[:, ch] = signal.convolve(xf[:, ch], kernel, mode="full")[: xf.shape[0]]
    return out


def mask_h1_still_bins(Y: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """True where the arm is moving (same rule as falcon ``sklearn_decoder.prepare_train_test``)."""
    still = np.all(np.abs(Y) < threshold, axis=1)
    return ~still
