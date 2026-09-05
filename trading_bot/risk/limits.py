"""Pure position-level risk rules shared by the live RiskEngine and the
walk-forward validation simulator (spec sections 30-32)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..strategy.sizing import direction_sign, turnover_suppressed


@dataclass(frozen=True)
class PositionRuleResult:
    exposure: float
    new_entry: bool
    reason: str
    max_holding_status: str
    stop_status: str


def stop_distance(sigma_entry: float, horizon: int, multiple: float = 4.0) -> float:
    """StopDistance = 4 sigma_{t,50} sqrt(H) in log-return units (section 32)."""
    if not math.isfinite(sigma_entry):
        return math.inf
    return multiple * sigma_entry * math.sqrt(horizon)


def stop_triggered(position_return: float, sigma_entry: float, horizon: int, multiple: float = 4.0) -> bool:
    if not math.isfinite(position_return):
        return False
    return position_return < -stop_distance(sigma_entry, horizon, multiple)


def apply_position_rules(q_raw: float, q_current: float, holding_bars: int, max_holding_bars: int,
                         stop_hit: bool, rebalance_threshold: float, reevaluate_every: int = 1) -> PositionRuleResult:
    """Combine the emergency stop, maximum holding time and turnover suppression.

    * Stop hit            -> flat (section 32).
    * Holding >= maximum  -> flat unless the *fresh* signal independently opens a
                             position, in which case it is a new entry (section 31).
    * Between re-evaluation points (``reevaluate_every`` > 1: hold-to-horizon mode) an
                             open position is maintained unchanged.
    * Otherwise           -> turnover suppression (section 30); a change of
                             direction (including from flat) counts as a new entry.
    """
    in_position = q_current != 0.0
    if in_position and stop_hit:
        return PositionRuleResult(0.0, False, "STOP_LOSS", "OK", "TRIGGERED")
    if in_position and holding_bars >= max_holding_bars:
        if q_raw != 0.0:
            return PositionRuleResult(float(q_raw), True, "MAX_HOLDING_REENTRY", "EXPIRED", "OK")
        return PositionRuleResult(0.0, False, "MAX_HOLDING_EXIT", "EXPIRED", "OK")
    if in_position and reevaluate_every > 1 and holding_bars % reevaluate_every != 0:
        return PositionRuleResult(float(q_current), False, "HOLD_TO_HORIZON", "OK", "OK")
    q = turnover_suppressed(q_raw, q_current, rebalance_threshold)
    new_entry = q != 0.0 and direction_sign(q) != direction_sign(q_current)
    reason = "OK" if q == q_raw else "TURNOVER_SUPPRESSED"
    return PositionRuleResult(float(q), new_entry, reason, "OK", "OK")


__all__ = ["PositionRuleResult", "apply_position_rules", "stop_distance", "stop_triggered"]
