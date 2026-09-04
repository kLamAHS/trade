"""Volatility, fractional-volatility and range features (spec sections 8, 14, 15)."""

from __future__ import annotations

import numpy as np

from ..fractional.engine import FractionalEngine
from .rolling import ewma_variance, robust_zscore, rolling_std


def volatility_features(log_close: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        engine: FractionalEngine, vol_windows=(10, 50, 200), ewma_lambda: float = 0.94,
                        ewma_window: int = 450, volatility_order: float = 0.40, z_window: int = 250,
                        eps: float = 1e-12) -> dict[str, np.ndarray]:
    r = np.full(len(log_close), np.nan)
    r[1:] = np.diff(log_close)
    out: dict[str, np.ndarray] = {}
    sig: dict[int, np.ndarray] = {}
    for w in vol_windows:
        s = rolling_std(r, w)
        sig[w] = s
        with np.errstate(divide="ignore"):
            out[f"vol_{w}"] = np.log(s + eps)
    ws = sorted(vol_windows)
    out["vol_ratio_short"] = sig[ws[0]] / (sig[ws[1]] + eps)
    out["vol_ratio_long"] = sig[ws[1]] / (sig[ws[2]] + eps)
    # Section 8: EWMA variance -> log -> fractional transform of order 0.40.
    v = ewma_variance(r, ewma_lambda, ewma_window)
    with np.errstate(invalid="ignore"):
        q = np.log(v + eps)
    out["fractional_volatility"] = engine.transform(q, volatility_order)
    out["ewma_variance"] = v
    # Section 15: range features.
    tr = (high - low) / close
    out["range_rel"] = tr
    out["range_z"] = robust_zscore(tr, z_window, eps)
    out["close_location"] = (2.0 * close - high - low) / (high - low + eps)
    out["_sigma"] = {w: sig[w] for w in vol_windows}  # type: ignore[assignment]
    out["_returns"] = r  # type: ignore[assignment]
    return out


__all__ = ["volatility_features"]
