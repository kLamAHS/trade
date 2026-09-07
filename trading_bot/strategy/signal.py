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

from ..types import CostEstimate, ExecutionEstimate, FeatureVector, Prediction, Signal
from .policy import DecisionPolicy
from .sizing import raw_exposure, volatility_multiplier


class SignalEngine:
    def __init__(self, cost_multiplier: float = 3.0, confidence_cost_multiplier: float = 6.0,
                 horizon: int = 4, vol_reference_bars: int = 390, vol_multiplier_min: float = 0.25,
                 vol_multiplier_max: float = 1.5, max_abs_exposure: float = 1.0, eps: float = 1e-12,
                 policy: DecisionPolicy | None = None):
        self.cost_multiplier = float(cost_multiplier)
        self.confidence_cost_multiplier = float(confidence_cost_multiplier)
        self.policy = policy or DecisionPolicy(mode="legacy", cost_multiplier=float(cost_multiplier),
                                               confidence_cost_multiplier=float(confidence_cost_multiplier), eps=float(eps))
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
                   s.vol_multiplier_max, cfg.risk.max_absolute_exposure, cfg.features.epsilon,
                   policy=DecisionPolicy.from_config(cfg))

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

    def create(self, prediction: Prediction, market_state: FeatureVector, estimated_cost,
               sigma_ref: Optional[float] = None) -> Signal:
        """``estimated_cost`` is a CostEstimate (spread / slippage / commission) or an ExecutionEstimate
        (adds adverse selection and uncertainty, market-state spec section 3)."""
        sigma = market_state.get("sigma_h")
        er = prediction.expected_raw_return
        if not math.isfinite(er):
            er = self.expected_raw_return(prediction.expected_normalized_return, sigma)
        ref = self.reference_volatility() if sigma_ref is None else sigma_ref
        stress = market_state.get("regime_stress_p", math.nan)
        if isinstance(estimated_cost, ExecutionEstimate):
            est = estimated_cost
            return self.build(prediction.timestamp, er, (est.spread_bps + est.slippage_bps + est.fee_bps) / 1e4, sigma, ref,
                              adverse=est.adverse_selection_bps / 1e4, model_uncertainty=est.model_uncertainty_bps / 1e4,
                              stress_p=stress, estimate=est)
        return self.build(prediction.timestamp, er, estimated_cost.total, sigma, ref, stress_p=stress)

    def build(self, timestamp: datetime, expected_return: float, cost: float, sigma: float, sigma_ref: float,
              adverse: float = 0.0, model_uncertainty: float = 0.0, stress_p: float = math.nan,
              estimate: ExecutionEstimate | None = None) -> Signal:
        """``cost`` is the round-trip spread + slippage + commission (fraction)."""
        dec = self.policy.decide(expected_return, cost, 0.0, adverse, model_uncertainty, stress_p, fees=self.policy.extra_fee)
        direction = dec.direction
        conf = dec.confidence if direction else 0.0
        vm = volatility_multiplier(sigma_ref, sigma, self.vm_min, self.vm_max)
        q = raw_exposure(direction, conf, vm, self.max_abs * dec.exposure_cap_multiplier) if direction else 0.0
        policy_rec = {"mode": self.policy.mode, "required_edge_bps": dec.required_edge * 1e4, "net_edge_bps": dec.net_edge * 1e4,
                      "trade": direction != 0, "abstain_reason": dec.abstain_reason,
                      "exposure_multiplier": dec.exposure_cap_multiplier, **{k: v for k, v in dec.components.items()}}
        return Signal(timestamp=timestamp, direction=direction, expected_return=float(expected_return),
                      estimated_cost=float(cost), expected_net_edge=float(dec.net_edge), confidence=float(conf),
                      target_exposure=float(q), volatility_multiplier=float(vm),
                      reference_volatility=float(sigma_ref) if math.isfinite(sigma_ref) else math.nan,
                      current_volatility=float(sigma) if math.isfinite(sigma) else math.nan,
                      execution=estimate.to_dict() if estimate is not None else None, policy=policy_rec)


__all__ = ["SignalEngine"]
