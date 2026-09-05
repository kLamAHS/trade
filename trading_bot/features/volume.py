"""Volume features (spec section 16).  Disabled globally via config -- never silently filled."""

from __future__ import annotations

import numpy as np

from .rolling import lag, robust_zscore


def volume_features(volume: np.ndarray, z_window: int = 50, eps: float = 1e-12) -> dict[str, np.ndarray]:
    lv = np.log1p(np.asarray(volume, dtype=float))
    return {"volume_z": robust_zscore(lv, z_window, eps), "volume_change": lv - lag(lv, 1)}


__all__ = ["volume_features"]
