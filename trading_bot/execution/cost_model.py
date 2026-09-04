"""Transaction-cost model (spec section 24).

Round-trip cost (relative to notional):

    Cost_t = Commission_t + Spread_t + Slippage_t

* Commission_t : two sides of ``commission_per_side``.
* Spread_t     : one full relative spread (entry and exit each cross half).
* Slippage_t   : round-trip slippage, two sides of alpha * (H_t - L_t) / C_t,
                 matching the fill model of section 43 which applies slippage
                 on every fill.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import CostEstimate


class CostModel:
    def __init__(self, commission_per_side: float = 0.0, default_spread: float = 1e-4,
                 slippage_range_fraction: float = 0.05):
        self.commission_per_side = float(commission_per_side)
        self.default_spread = float(default_spread)
        self.alpha = float(slippage_range_fraction)

    @classmethod
    def from_config(cls, cfg) -> "CostModel":
        e = cfg.execution
        return cls(e.commission_per_side, e.default_spread, e.slippage_range_fraction)

    def spread(self, spread_rel: float | None) -> float:
        if spread_rel is None or not math.isfinite(spread_rel) or spread_rel < 0:
            return self.default_spread
        return float(spread_rel)

    def slippage_per_side(self, range_rel: float) -> float:
        if not math.isfinite(range_rel) or range_rel < 0:
            return 0.0
        return self.alpha * float(range_rel)

    def estimate(self, range_rel: float, spread_rel: float | None = None) -> CostEstimate:
        return CostEstimate(commission=2.0 * self.commission_per_side, spread=self.spread(spread_rel),
                            slippage=2.0 * self.slippage_per_side(range_rel))

    def estimate_array(self, range_rel: np.ndarray, spread_rel: np.ndarray | None = None) -> np.ndarray:
        """Vectorised round-trip cost for the validation simulator."""
        rr = np.asarray(range_rel, dtype=float)
        slip = np.where(np.isfinite(rr) & (rr >= 0), self.alpha * rr, 0.0)
        if spread_rel is None:
            spr = np.full(len(rr), self.default_spread)
        else:
            s = np.asarray(spread_rel, dtype=float)
            spr = np.where(np.isfinite(s) & (s >= 0), s, self.default_spread)
        return 2.0 * self.commission_per_side + spr + 2.0 * slip

    def per_side_array(self, range_rel: np.ndarray, spread_rel: np.ndarray | None = None) -> np.ndarray:
        return 0.5 * self.estimate_array(range_rel, spread_rel)


__all__ = ["CostModel"]
