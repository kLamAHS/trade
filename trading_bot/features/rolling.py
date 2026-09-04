"""Causal rolling statistics implemented with numpy sliding windows.

Every function returns an array aligned with its input where element t depends
only on x_{t-N+1..t}; positions without a full window are NaN.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    return sliding_window_view(x, n)


def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.median(_windows(x, n), axis=1)
    return out


def rolling_mad(x: np.ndarray, n: int) -> np.ndarray:
    """Median absolute deviation over a trailing window."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = _windows(x, n)
        med = np.median(w, axis=1, keepdims=True)
        out[n - 1:] = np.median(np.abs(w - med), axis=1)
    return out


def robust_zscore(x: np.ndarray, n: int, eps: float = 1e-12) -> np.ndarray:
    """Z_t(X) = (X_t - median_N(X)) / (1.4826 * MAD_N(X) + eps)  (spec section 9)."""
    x = np.asarray(x, dtype=float)
    med = rolling_median(x, n)
    mad = rolling_mad(x, n)
    return (x - med) / (1.4826 * mad + eps)


def rolling_std(x: np.ndarray, n: int, ddof: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.std(_windows(x, n), axis=1, ddof=ddof)
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.mean(_windows(x, n), axis=1)
    return out


def rolling_corr_with_index(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling Pearson correlation between x and the bar index over the last n bars."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = _windows(x, n)
    idx = np.arange(n, dtype=float)
    idx_c = idx - idx.mean()
    w_c = w - w.mean(axis=1, keepdims=True)
    num = (w_c * idx_c).sum(axis=1)
    den = np.sqrt((w_c ** 2).sum(axis=1) * (idx_c ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0, num / den, 0.0)
    out[n - 1:] = corr
    return out


def lag(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if k == 0:
        return x.copy()
    if len(x) > k:
        out[k:] = x[:-k]
    return out


def ewma_variance(returns: np.ndarray, lam: float, window: int) -> np.ndarray:
    """Finite-kernel exponentially weighted variance.

        v_t = (1-lam) * sum_{j=0}^{W-2} lam^j r_{t-j}^2  +  lam^{W-1} r_{t-W+1}^2

    This is the recursion v_t = lam v_{t-1} + (1-lam) r_t^2 seeded W-1 bars back
    with r^2, truncated so that batch and streaming computations are identical.
    NaN until W observations exist.
    """
    r2 = np.asarray(returns, dtype=float) ** 2
    j = np.arange(window)
    kernel = (1.0 - lam) * lam ** j
    kernel[-1] = lam ** (window - 1)
    out = np.full(len(r2), np.nan)
    if len(r2) >= window:
        full = np.convolve(r2, kernel, mode="full")
        out[window - 1:] = full[window - 1:len(r2)]
        # propagate NaNs from the input window
        nan_conv = np.convolve(np.isnan(r2).astype(float), np.ones(window), mode="full")[window - 1:len(r2)]
        out[window - 1:][nan_conv > 0] = np.nan
    return out


__all__ = ["rolling_median", "rolling_mad", "robust_zscore", "rolling_std", "rolling_mean",
           "rolling_corr_with_index", "lag", "ewma_variance"]
