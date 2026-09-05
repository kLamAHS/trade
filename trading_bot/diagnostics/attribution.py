"""Performance attribution groupings (spec section 52): by volatility regime, time of day, fractional d."""

from __future__ import annotations

import math
from typing import Any


def volatility_bucket(vol_state: float, edges=(0.8, 1.2)) -> str:
    if not math.isfinite(vol_state):
        return "unknown"
    if vol_state < edges[0]:
        return "low"
    if vol_state < edges[1]:
        return "normal"
    return "high"


def time_bucket(minutes_since_open: int, session_minutes: int) -> str:
    frac = minutes_since_open / max(session_minutes, 1)
    if frac < 1 / 3:
        return "open"
    if frac < 2 / 3:
        return "mid"
    return "close"


def attribution_groups(trade_contexts: list[dict[str, Any]], vol_edges=(0.8, 1.2)) -> dict[str, list[tuple[str, float]]]:
    """``trade_contexts`` items: {pnl, vol_state, minutes_since_open, session_minutes, fractional_d}."""
    groups: dict[str, list[tuple[str, float]]] = {"volatility_regime": [], "time_of_day": [], "fractional_d": []}
    for t in trade_contexts:
        pnl = float(t.get("pnl", 0.0))
        groups["volatility_regime"].append((volatility_bucket(t.get("vol_state", math.nan), vol_edges), pnl))
        groups["time_of_day"].append((time_bucket(int(t.get("minutes_since_open", 0)), int(t.get("session_minutes", 390))), pnl))
        d = t.get("fractional_d", math.nan)
        groups["fractional_d"].append((f"{d:.2f}" if math.isfinite(d) else "unknown", pnl))
    return groups


__all__ = ["attribution_groups", "volatility_bucket", "time_bucket"]
