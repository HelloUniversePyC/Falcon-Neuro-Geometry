"""
Kinematic targets aligned with ``data_demos/h1.ipynb`` linear-decoder section.

The notebook does **not** regress on raw ``OpenLoopKinematicsVelocity``; it uses
``OpenLoopKinematics`` (pose), Gaussian-smooths it, then ``np.gradient`` along time
to form smoothed pseudo-velocity targets — same recipe as ``create_targets`` there.

``decoder_demos`` is not shipped in the ``falcon_challenge`` PyPI package, so the
Gaussian smoothing kernel is reproduced here (see snel-repo/falcon-challenge
``decoder_demos/filtering.py``).
"""
from __future__ import annotations

import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    size = int(size)
    x = np.linspace(-size // 2, size // 2, size, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-9)) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_channels(x: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Replicate ``decoder_demos.filtering.smooth`` (replicate pad + conv1d per channel)."""
    k = gaussian_kernel(kernel_size, sigma)
    pad_left = (len(k) - 1) // 2
    pad_right = len(k) - 1 - pad_left
    out = np.empty_like(x, dtype=np.float64)
    xf = x.astype(np.float64, copy=False)
    for j in range(xf.shape[1]):
        padded = np.pad(xf[:, j], (pad_left, pad_right), mode="edge")
        out[:, j] = np.convolve(padded, k, mode="valid")
    assert out.shape == xf.shape
    return out


def notebook_decoder_targets(
    kin_pose: np.ndarray,
    *,
    target_smooth_ms: float = 490.0,
    bin_size_ms: float = 20.0,
    sigma: int = 3,
) -> np.ndarray:
    """``h1.ipynb`` ``create_targets``: smooth pose, then temporal gradient (per column)."""
    kernel_size = int(target_smooth_ms / bin_size_ms)
    kernel_sigma = target_smooth_ms / (sigma * bin_size_ms)
    sm = smooth_channels(kin_pose, kernel_size, kernel_sigma)
    return np.gradient(sm, axis=0)
