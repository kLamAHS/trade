"""Pure position-sizing arithmetic (spec sections 25-30).

These functions are shared by the live signal/risk engines and by the
walk-forward validation simulator so that both paths size positions identically.
"""

from __future__ import annotations

import math


def direction_sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def direction_from_edge(expected_return: float, cost: float, cost_multiplier: float = 3.0) -> int:
    """+1 when ER > k*Cost, -1 when ER < -k*Cost, else 0 (section 26)."""
    if not math.isfinite(expected_return) or not math.isfinite(cost):
        return 0
    threshold = cost_multiplier * cost
    if expected_return > threshold:
        return 1
    if expected_return < -threshold:
        return -1
    return 0


def net_edge(expected_return: float, cost: float, cost_multiplier: float = 3.0) -> float:
    """NE_t = |ER_t| - k * Cost_t (section 25)."""
    if not math.isfinite(expected_return):
        return -math.inf
    return abs(expected_return) - cost_multiplier * cost


def confidence_from_edge(expected_return: float, cost: float, confidence_multiplier: float = 6.0,
                         eps: float = 1e-12) -> float:
    """Confidence_t = min(1, |ER_t| / (6 Cost_t + eps)) (section 27)."""
    if not math.isfinite(expected_return):
        return 0.0
    return min(1.0, abs(expected_return) / (confidence_multiplier * cost + eps))


def volatility_multiplier(sigma_ref: float, sigma_now: float, lo: float = 0.25, hi: float = 1.5) -> float:
    """VM_t = clip(sigma_ref / sigma_{t,50}, lo, hi) (section 28)."""
    if not (math.isfinite(sigma_ref) and math.isfinite(sigma_now)) or sigma_now <= 0 or sigma_ref <= 0:
        return lo
    return float(min(hi, max(lo, sigma_ref / sigma_now)))


def raw_exposure(direction: int, confidence: float, vol_mult: float, max_abs: float = 1.0) -> float:
    """Q_t = Direction * Confidence * VM clipped to [-max_abs, max_abs] (section 29)."""
    q = direction * confidence * vol_mult
    return float(max(-max_abs, min(max_abs, q)))


def turnover_suppressed(target: float, current: float, threshold: float = 0.15) -> float:
    """Section 30: rebalance only when |Q_t - Q_current| >= threshold or the direction changes."""
    if direction_sign(target) != direction_sign(current):
        return target
    if abs(target - current) >= threshold:
        return target
    return current


__all__ = ["direction_sign", "direction_from_edge", "net_edge", "confidence_from_edge", "volatility_multiplier",
           "raw_exposure", "turnover_suppressed"]
