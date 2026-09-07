"""Decision policy: from expected gross return and expected execution costs to a trade / abstain
decision (market-state spec sections 3, 11, 60).  One implementation serves the live SignalEngine,
the trainer's validation simulator and the research simulator.

    net_edge = |expected_return| - spread - slippage - adverse_selection - fees - uncertainty_buffer
    trade only when net_edge > minimum_net_edge (scaled up in stress regimes)

``legacy`` keeps the original rule of spec section 26 (|ER| > k * round-trip cost) so that the two
policies can be compared like for like.  ABSTAIN is a first-class outcome: the policy answers
"is there enough forecastable movement to overcome execution with adequate confidence?".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RegimeAdjustment:
    enabled: bool = False
    stress_probability_threshold: float = 0.70
    exposure_multiplier: float = 0.40
    edge_multiplier: float = 1.5
    confidence_multiplier: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> "RegimeAdjustment":
        r = (cfg.risk.get("regime_adjustment", {}) or {}) if hasattr(cfg, "risk") else {}
        if isinstance(r, bool):
            return cls(enabled=bool(r))
        return cls(enabled=bool(r.get("enabled", False)), stress_probability_threshold=float(r.get("stress_probability_threshold", 0.70)),
                   exposure_multiplier=float(r.get("exposure_multiplier", 0.40)), edge_multiplier=float(r.get("edge_multiplier", 1.5)),
                   confidence_multiplier=float(r.get("confidence_multiplier", 1.0)))

    def factors(self, stress_p: float) -> tuple[float, float]:
        """(exposure multiplier, required-edge multiplier) for the stress probability."""
        if not self.enabled or not math.isfinite(stress_p) or stress_p <= self.stress_probability_threshold:
            return 1.0, 1.0
        return self.exposure_multiplier, self.edge_multiplier


@dataclass(frozen=True)
class Decision:
    direction: int
    confidence: float
    net_edge: float                 # in return units (fraction)
    exposure_cap_multiplier: float
    required_edge: float
    components: dict[str, float]
    abstain_reason: str = ""

    @property
    def trade(self) -> bool:
        return self.direction != 0


@dataclass(frozen=True)
class DecisionPolicy:
    mode: str = "net_edge"                  # net_edge | legacy
    cost_multiplier: float = 3.0            # legacy: trade when |ER| > k * cost
    confidence_cost_multiplier: float = 6.0
    minimum_net_edge: float = 8e-4          # fraction (8 bps)
    uncertainty_buffer: float = 3e-4        # fraction (3 bps), added to the model-based uncertainty
    fee_roundtrip: float = 0.0              # fraction: 2 x commission + extra fees (live path, where costs arrive decomposed)
    regime: RegimeAdjustment = RegimeAdjustment()
    eps: float = 1e-12
    extra_fee: float = 0.0                  # fraction: fees *not* in the cost model (simulators pass this; commission is in cost_rt)

    @classmethod
    def from_config(cls, cfg) -> "DecisionPolicy":
        e = cfg.execution
        s = cfg.signal
        mode = str(e.get("decision_policy", "net_edge"))
        if mode not in ("net_edge", "legacy"):
            raise ValueError("execution.decision_policy must be 'net_edge' or 'legacy'")
        return cls(mode=mode, cost_multiplier=float(s.cost_multiplier), confidence_cost_multiplier=float(s.confidence_cost_multiplier),
                   minimum_net_edge=float(e.get("minimum_net_edge_bps", 8.0)) / 1e4,
                   uncertainty_buffer=float(e.get("uncertainty_buffer_bps", 3.0)) / 1e4,
                   fee_roundtrip=2.0 * float(e.get("commission_per_side", 0.0)) + float(e.get("fee_bps", 0.0)) / 1e4,
                   regime=RegimeAdjustment.from_config(cfg), eps=float(cfg.features.epsilon),
                   extra_fee=float(e.get("fee_bps", 0.0)) / 1e4)

    def decide(self, expected_return: float, spread: float, slippage: float, adverse: float = 0.0,
               model_uncertainty: float = 0.0, stress_p: float = math.nan, fees: float | None = None) -> Decision:
        """All inputs are fractions of notional over the round trip (``spread`` = one full spread,
        ``slippage`` = two sides); ``adverse`` = expected post-fill adverse move beyond the forecast."""
        fees = self.fee_roundtrip if fees is None else fees
        exp_mult, edge_mult = self.regime.factors(stress_p)
        if not math.isfinite(expected_return):
            return Decision(0, 0.0, -math.inf, exp_mult, self.minimum_net_edge * edge_mult, {}, "non-finite forecast")
        spread = spread if math.isfinite(spread) else 0.0
        slippage = slippage if math.isfinite(slippage) else 0.0
        adverse = adverse if math.isfinite(adverse) else 0.0
        model_unc = max(0.0, model_uncertainty) if math.isfinite(model_uncertainty) else 0.0
        gross = abs(expected_return)
        cost_rt = spread + slippage + fees
        comps = {"gross": gross, "spread": spread, "slippage": slippage, "adverse_selection": adverse, "fees": fees,
                 "uncertainty": self.uncertainty_buffer + model_unc, "stress_p": stress_p if math.isfinite(stress_p) else 0.0,
                 "exposure_multiplier": exp_mult, "edge_multiplier": edge_mult}
        if self.mode == "legacy":
            required = self.cost_multiplier * cost_rt * edge_mult
            net = gross - required
            comps["required"] = required
            if net <= 0:
                return Decision(0, 0.0, net, exp_mult, required, comps, "|ER| <= k * cost")
            conf = min(1.0, gross / (self.confidence_cost_multiplier * cost_rt * edge_mult + self.eps))
        else:
            net = gross - cost_rt - adverse - self.uncertainty_buffer - model_unc
            required = self.minimum_net_edge * edge_mult
            comps["required"] = required
            comps["net_edge"] = net
            if net <= required:
                return Decision(0, 0.0, net, exp_mult, required, comps, "net edge below minimum")
            conf = min(1.0, net / (self.confidence_cost_multiplier * max(cost_rt, self.minimum_net_edge) * edge_mult + self.eps))
        conf *= self.regime.confidence_multiplier if exp_mult < 1.0 else 1.0
        direction = 1 if expected_return > 0 else -1
        return Decision(direction, float(min(1.0, conf)), float(net), exp_mult, required, comps)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "cost_multiplier": self.cost_multiplier, "confidence_cost_multiplier": self.confidence_cost_multiplier,
                "minimum_net_edge_bps": self.minimum_net_edge * 1e4, "uncertainty_buffer_bps": self.uncertainty_buffer * 1e4,
                "fee_bps": self.fee_roundtrip * 1e4, "regime_adjustment": self.regime.__dict__}


__all__ = ["DecisionPolicy", "Decision", "RegimeAdjustment"]
