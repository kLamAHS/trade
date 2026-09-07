"""Market representation (market-state spec sections 6-7): which bars the strategy sees.

    market_representation:
      bar_type: time | volume | dollar | tick
      target_shares / target_dollar_volume / target_ticks / target_minutes
      candidates: [time_30m, volume_500k, dollar_250m]      # research mode (selection inside training only)

``apply_representation`` turns the raw time bars of a feed into the configured representation and
returns the configuration whose ``market.bars_per_day`` reflects the empirical bars per session
(the annualisation basis for event bars).
"""

from __future__ import annotations

import re
from typing import Any

from ..types import Bar
from .bars import BAR_KINDS, build_event_bars, empirical_bars_per_session
from .calendar import SessionCalendar

_TARGET_KEYS = {"volume": "target_shares", "dollar": "target_dollar_volume", "tick": "target_ticks", "time": "target_minutes"}


def parse_candidate(name: str) -> dict[str, Any]:
    """'time_30m' -> time bars of 30 minutes; 'volume_500k', 'dollar_250m', 'tick_20'."""
    m = re.fullmatch(r"(time|volume|dollar|tick)_([0-9.]+)([kmbKMB]?)(m?)", str(name).strip())
    if not m:
        raise ValueError(f"cannot parse representation candidate {name!r} (e.g. time_30m, volume_500k, dollar_250m, tick_20)")
    kind, num, suffix, _ = m.groups()
    mult = {"": 1.0, "k": 1e3, "m": 1e6, "b": 1e9}[suffix.lower()] if kind != "time" else 1.0
    return {"bar_type": kind, "target": float(num) * mult, "name": name}


def representation_spec(cfg) -> dict[str, Any]:
    rep = cfg.get("market_representation", {}) or {}
    kind = str(rep.get("bar_type", "time")).lower()
    if kind not in BAR_KINDS:
        raise ValueError(f"market_representation.bar_type must be one of {BAR_KINDS}")
    target = rep.get(_TARGET_KEYS[kind])
    if kind == "time":
        target = float(target or cfg.market.bar_minutes)
    elif target is None:
        raise ValueError(f"market_representation.{_TARGET_KEYS[kind]} is required for {kind} bars")
    return {"bar_type": kind, "target": float(target), "name": f"{kind}_{target:g}" if kind != "time" else f"time_{int(target)}m",
            "candidates": [parse_candidate(c) for c in (rep.get("candidates") or [])]}


def build_representation(bars: list[Bar], spec: dict[str, Any], calendar: SessionCalendar) -> list[Bar]:
    if spec["bar_type"] == "time":
        return list(bars)
    return build_event_bars(bars, spec["bar_type"], spec["target"], calendar)


def apply_representation(bars: list[Bar], cfg, calendar: SessionCalendar | None = None):
    """(bars in the configured representation, effective config).  Time bars pass through untouched."""
    cal = calendar or SessionCalendar.from_config(cfg)
    spec = representation_spec(cfg)
    if spec["bar_type"] == "time":
        return list(bars), cfg
    out = build_representation(bars, spec, cal)
    bpd = max(1, int(round(empirical_bars_per_session(out, cal)))) if out else int(cfg.market.bars_per_day)
    return out, cfg.with_overrides({"market": {"bars_per_day": bpd}})


__all__ = ["representation_spec", "parse_candidate", "build_representation", "apply_representation"]
