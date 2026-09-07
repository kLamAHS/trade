"""Experimental VPIN / flow-toxicity features (market-state spec section 31).

Bulk volume classification at bar resolution: the buy fraction of a bar's volume is
Phi(r_t / sigma_t).  Volume is bucketed over ``bucket_bars`` bars and VPIN is the rolling mean of
|buy - sell| / volume over ``n_buckets`` buckets.  The family is disabled (all NaN, flagged) when
the classification confidence -- the mean |2 * buy_fraction - 1| over the training window -- is
below ``min_classification_confidence``: a coin-flip classifier must not masquerade as toxicity.
Research feature; off by default.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .rolling import lag, robust_zscore


def vpin_feature_arrays(returns: np.ndarray, sigma: np.ndarray, volume: np.ndarray, bucket_bars: int = 5,
                        n_buckets: int = 50, z_window: int = 250, eps: float = 1e-12) -> tuple[dict[str, np.ndarray], float]:
    z = returns / (sigma + eps)
    buy_frac = norm.cdf(np.where(np.isfinite(z), z, 0.0))
    buy = volume * buy_frac
    sell = volume - buy
    n = len(volume)
    # bucket sums (trailing, non-overlapping alignment is not required for a rolling estimate)
    from numpy.lib.stride_tricks import sliding_window_view
    imb = np.full(n, np.nan)
    tot = np.full(n, np.nan)
    if n >= bucket_bars:
        imb[bucket_bars - 1:] = np.abs(sliding_window_view(buy - sell, bucket_bars).sum(axis=1))
        tot[bucket_bars - 1:] = sliding_window_view(volume, bucket_bars).sum(axis=1)
    vpin = np.full(n, np.nan)
    w = n_buckets * bucket_bars
    if n >= w:
        vpin[w - 1:] = sliding_window_view(imb, w)[:, ::bucket_bars].sum(axis=1) / (sliding_window_view(tot, w)[:, ::bucket_bars].sum(axis=1) + eps)
    pct = np.full(n, np.nan)
    if n >= z_window:
        win = sliding_window_view(vpin, z_window)
        with np.errstate(invalid="ignore"):
            pct[z_window - 1:] = np.nanmean(win <= win[:, -1:], axis=1)
    confidence = float(np.nanmean(np.abs(2.0 * buy_frac - 1.0))) if n else 0.0
    return {"vpin": vpin, "vpin_zscore": robust_zscore(vpin, z_window, eps), "vpin_change": vpin - lag(vpin, bucket_bars),
            "vpin_percentile": pct}, confidence


VPIN_NAMES = ("vpin", "vpin_zscore", "vpin_change", "vpin_percentile")

__all__ = ["vpin_feature_arrays", "VPIN_NAMES"]
