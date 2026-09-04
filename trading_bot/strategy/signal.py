"""SignalEngine: prediction + market state + cost -> Signal (spec sections 25-29).

The engine is stateless apart from a rolling record of sigma_{t,50} used to
compute the reference volatility (median over the trailing ``vol_reference_days``
sessions, section 28).
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

from ..types import CostEstimate, FeatureVector, Prediction, Signal
from .sizing import (confidence_from_edge, direction_from_edge, net_edge, raw_exposure, volatility_multiplier)


class SignalEngine:
    def __init__(self, cost_multiplier: float = 3.0, confidence_cost_multiplier: float = 6.0,
                 horizon: int = 4, vol_reference_bars: int = 390, vol_multiplier_min: float = 0.25,
                 vol_multiplier_max: float = 1.5, max_abs_exposure: float = 1.0, eps: float = 1e-12):
        self.cost_multiplier = float(cost_multiplier)
        self.confidence_cost_multiplier = float(confidence_cost_multiplier)
        self.horizon = int(horizon)
        self.vol_reference_bars = int(vol_reference_bars)
        self.vm_min = float(vol_multiplier_min)
        self.vm_max = float(vol_multiplier_max)
        self.max_abs = float(max_abs_exposure)
        self.eps = float(eps)
        self._sigma_history: deque[float] = deque(maxlen=self.vol_reference_bars)

    @classmethod
    def from_config(cls, cfg) -> "SignalEngine":
        s = cfg.signal
        return cls(s.cost_multiplier, s.confidence_cost_multiplier, cfg.prediction.horizon_bars,
                   int(s.vol_reference_days) * int(cfg.market.bars_per_day), s.vol_multiplier_min,
                   s.vol_multiplier_max, cfg.risk.max_absolute_exposure, cfg.features.epsilon)

    # ------------------------------------------------------- reference vol
    def observe_sigma(self, sigma: float) -> None:
        if math.isfinite(sigma) and sigma > 0:
            self._sigma_history.append(float(sigma))

    def seed_sigma_history(self, sigmas) -> None:
        self._sigma_history.clear()
        for s in sigmas:
            self.observe_sigma(float(s))

    @property
    def has_history(self) -> bool:
        return len(self._sigma_history) > 0

    def reference_volatility(self) -> float:
        if not self._sigma_history:
            return math.nan
        return float(np.median(np.asarray(self._sigma_history)))

    # --------------------------------------------------------------- create
    def expected_raw_return(self, expected_normalized: float, sigma: float) -> float:
        """ER_t = E_t * sigma_{t,50} * sqrt(H) (section 25)."""
        if not (math.isfinite(expected_normalized) and math.isfinite(sigma)):
            return math.nan
        return expected_normalized * sigma * math.sqrt(self.horizon)

    def create(self, prediction: Prediction, market_state: FeatureVector, estimated_cost: CostEstimate,
               sigma_ref: Optional[float] = None) -> Signal:
        sigma = market_state.get("sigma_h")
        cost = estimated_cost.total
        er = prediction.expected_raw_return
        if not math.isfinite(er):
            er = self.expected_raw_return(prediction.expected_normalized_return, sigma)
        ref = self.reference_volatility() if sigma_ref is None else sigma_ref
        return self.build(prediction.timestamp, er, cost, sigma, ref)

    def build(self, timestamp: datetime, expected_return: float, cost: float, sigma: float, sigma_ref: float) -> Signal:
        direction = direction_from_edge(expected_return, cost, self.cost_multiplier)
        ne = net_edge(expected_return, cost, self.cost_multiplier)
        conf = confidence_from_edge(expected_return, cost, self.confidence_cost_multiplier, self.eps) if direction else 0.0
        vm = volatility_multiplier(sigma_ref, sigma, self.vm_min, self.vm_max)
        q = raw_exposure(direction, conf, vm, self.max_abs) if direction else 0.0
        return Signal(timestamp=timestamp, direction=direction, expected_return=float(expected_return),
                      estimated_cost=float(cost), expected_net_edge=float(ne), confidence=float(conf),
                      target_exposure=float(q), volatility_multiplier=float(vm),
                      reference_volatility=float(sigma_ref) if math.isfinite(sigma_ref) else math.nan,
                      current_volatility=float(sigma) if math.isfinite(sigma) else math.nan)


__all__ = ["SignalEngine"]
