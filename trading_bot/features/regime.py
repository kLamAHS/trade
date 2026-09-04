"""Continuous market-state (regime) variables (spec section 18).

These are never converted into discrete labels; the models consume them directly.
"""

from __future__ import annotations

import numpy as np

from .rolling import rolling_corr_with_index, rolling_std


def regime_features(log_close: np.ndarray, returns: np.ndarray, fd_slow_z: np.ndarray, fd_fast_z: np.ndarray,
                    trend_window: int = 50, vol_short: int = 20, vol_long: int = 200,
                    frac_floor: float = 0.1, eps: float = 1e-12) -> dict[str, np.ndarray]:
    trend = rolling_corr_with_index(log_close, trend_window)
    vol_regime = rolling_std(returns, vol_short) / (rolling_std(returns, vol_long) + eps)
    frac_regime = np.abs(fd_slow_z) / (np.abs(fd_fast_z) + frac_floor)
    return {"trend_state": trend, "volatility_state": vol_regime, "fractional_state": frac_regime}


__all__ = ["regime_features"]
