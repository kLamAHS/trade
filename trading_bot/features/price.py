"""Fractional price-state and conventional return features (spec sections 4-5, 9-13)."""

from __future__ import annotations

import numpy as np

from ..fractional.engine import FractionalEngine
from .rolling import lag, robust_zscore
from .schema import FRACTIONAL_CHANNELS


def fractional_price_features(log_close: np.ndarray, engine: FractionalEngine, adaptive_d: float,
                              fixed_orders=(0.25, 0.50, 0.75), z_window: int = 250, eps: float = 1e-12,
                              slope_lags=(1, 4)) -> dict[str, np.ndarray]:
    orders = {"adaptive": adaptive_d, "025": fixed_orders[0], "050": fixed_orders[1], "075": fixed_orders[2]}
    out: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    z: dict[str, np.ndarray] = {}
    for name, d in orders.items():
        f = engine.transform(log_close, d)
        raw[name] = f
        z[name] = robust_zscore(f, z_window, eps)
        out[f"fd_{name}"] = f
        out[f"fd_{name}_z"] = z[name]
    # Fractional slope (section 10) and curvature (section 11) on the raw series.
    for k in slope_lags:
        for name in FRACTIONAL_CHANNELS:
            out[f"fd_slope_{k}_{name}"] = raw[name] - lag(raw[name], k)
    for name in FRACTIONAL_CHANNELS:
        out[f"fd_curvature_{name}"] = raw[name] - 2.0 * lag(raw[name], 1) + lag(raw[name], 2)
    # Cross-scale features (section 12) on the z-scored series.
    out["fd_cross_sm"] = z["025"] - z["050"]
    out["fd_cross_mf"] = z["050"] - z["075"]
    out["fd_cross_sf"] = z["025"] - z["075"]
    return out


def conventional_return_features(log_close: np.ndarray, sigma: np.ndarray, return_lags=(1, 2, 4, 8, 16),
                                 eps: float = 1e-12) -> dict[str, np.ndarray]:
    """NR_{k,t} = (p_t - p_{t-k}) / (sigma_{t,50} sqrt(k) + eps)  (section 13)."""
    out: dict[str, np.ndarray] = {}
    for k in return_lags:
        r = log_close - lag(log_close, k)
        out[f"return_{k}"] = r / (sigma * np.sqrt(k) + eps)
    return out


__all__ = ["fractional_price_features", "conventional_return_features"]
