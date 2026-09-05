"""Apply the GL fractional operator to a series.

    D^d x_t = sum_{k=0}^{K} w_k x_{t-k}

Values are NaN until all K lags exist (spec section 6: no fractional feature is
valid until every required lag observation exists).  The operation is strictly
causal: element t depends only on x_{t-K..t}.
"""

from __future__ import annotations

import numpy as np

from .weights import build_weights


def fractional_transform(series: np.ndarray, d: float, threshold: float = 1e-5,
                         threshold_run: int = 10, max_lags: int = 500,
                         weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if x.ndim != 1:
        raise ValueError("series must be one-dimensional")
    w = build_weights(d, threshold, threshold_run, max_lags) if weights is None else np.asarray(weights, dtype=float)
    k_max = len(w) - 1
    n = len(x)
    out = np.full(n, np.nan)
    if n <= k_max:
        return out
    # Causal convolution: out[t] = sum_k w[k] * x[t-k]; "valid" mode yields t = k_max..n-1.
    full = np.convolve(x, w, mode="full")
    out[k_max:] = full[k_max:n]
    # Any NaN inside the window must propagate (np.convolve does propagate NaN, but be explicit).
    if np.isnan(x).any():
        bad = np.isnan(x).astype(float)
        bad_conv = np.convolve(bad, np.ones(k_max + 1), mode="full")[k_max:n]
        out[k_max:][bad_conv > 0] = np.nan
    return out


def fractional_latest(series: np.ndarray, d: float, threshold: float = 1e-5,
                      threshold_run: int = 10, max_lags: int = 500,
                      weights: np.ndarray | None = None) -> float:
    """Latest value only: dot product of the kernel with the trailing K+1 observations."""
    x = np.asarray(series, dtype=float)
    w = build_weights(d, threshold, threshold_run, max_lags) if weights is None else np.asarray(weights, dtype=float)
    k = len(w)
    if len(x) < k:
        return float("nan")
    tail = x[-k:][::-1]
    if np.isnan(tail).any():
        return float("nan")
    return float(np.dot(w, tail))


__all__ = ["fractional_transform", "fractional_latest"]
