"""Grünwald-Letnikov fractional differencing weights (spec sections 5-6).

    w_0 = 1,   w_k = -w_{k-1} * (d - k + 1) / k

Generation stops when |w_k| < threshold for ``threshold_run`` consecutive terms
or when k reaches ``max_lags``, whichever occurs first.  The returned kernel
contains every generated weight (index 0..K), so the transform is deterministic
for a given (d, threshold, threshold_run, max_lags).
"""

from __future__ import annotations

from functools import lru_cache
from math import gamma, lgamma
import math

import numpy as np


@lru_cache(maxsize=256)
def _cached_weights(d: float, threshold: float, threshold_run: int, max_lags: int) -> tuple[float, ...]:
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"fractional order must lie in [0, 1], got {d}")
    if max_lags < 1:
        raise ValueError("max_lags must be >= 1")
    weights = [1.0]
    small_run = 0
    k = 1
    while k <= max_lags:
        w = -weights[-1] * (d - k + 1.0) / k
        weights.append(w)
        if abs(w) < threshold:
            small_run += 1
            if small_run >= threshold_run:
                break
        else:
            small_run = 0
        k += 1
    return tuple(weights)


def build_weights(d: float, threshold: float = 1e-5, threshold_run: int = 10, max_lags: int = 500) -> np.ndarray:
    """Return the truncated GL weight kernel ``w_0..w_K`` for fractional order ``d``.

    Weights are computed once per (d, threshold, run, max_lags) and cached.
    """
    return np.asarray(_cached_weights(float(d), float(threshold), int(threshold_run), int(max_lags)), dtype=float)


def kernel_length(d: float, threshold: float = 1e-5, threshold_run: int = 10, max_lags: int = 500) -> int:
    """K(d): the largest lag index in the truncated kernel."""
    return len(build_weights(d, threshold, threshold_run, max_lags)) - 1


def weight_via_binomial(d: float, k: int) -> float:
    """Direct generalized-binomial evaluation ``(-1)^k C(d, k)`` for unit tests."""
    if k == 0:
        return 1.0
    # (-1)^k * Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1)); use the product form to stay finite for d - k + 1 <= 0.
    prod = 1.0
    for j in range(1, k + 1):
        prod *= (d - j + 1.0) / j
    return ((-1.0) ** k) * prod


__all__ = ["build_weights", "kernel_length", "weight_via_binomial"]
