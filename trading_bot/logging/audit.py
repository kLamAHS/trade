"""AuditLogger: one JSON line per bar decision, per fill and per event.

The design objective is that any trade can be completely reconstructed months
later: every record carries the bar, all feature values, the fractional order
and kernel size, model outputs, cost estimate, signal, risk decision, orders,
fills, ledger state and the bot state (spec section 51).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ..types import Bar, FeatureVector, Fill, Order, Prediction, RiskDecision, Signal


def _clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _clean(obj.to_dict())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


class AuditLogger:
    def __init__(self, directory: str | Path, run_id: str, echo=None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._bars = open(self.dir / f"{run_id}_bars.jsonl", "a", encoding="utf-8")
        self._fills = open(self.dir / f"{run_id}_fills.jsonl", "a", encoding="utf-8")
        self._events = open(self.dir / f"{run_id}_events.jsonl", "a", encoding="utf-8")
        self.echo = echo
        self.n_records = 0

    def _write(self, handle, record: dict) -> None:
        handle.write(json.dumps(_clean(record), default=str) + "\n")
        handle.flush()

    def record(self, bar: Bar, state: str, features: Optional[FeatureVector] = None,
               prediction: Optional[Prediction] = None, signal: Optional[Signal] = None,
               risk: Optional[RiskDecision] = None, order: Optional[Order] = None, cost=None,
               portfolio=None, model_version: str = "none", validation=None, extra: Optional[dict] = None) -> None:
        rec = {
            "type": "bar", "run_id": self.run_id, "timestamp": bar.timestamp.isoformat(),
            "close_time": bar.close_time.isoformat(), "state": state, "model_version": model_version,
            "bar": bar.to_dict(), "validation": validation,
            "fractional_d": features.fractional_d if features else None,
            "fractional_kernel_size": features.fractional_kernel_size if features else None,
            "feature_timestamp": features.timestamp.isoformat() if features else None,
            "latest_source_timestamp": features.latest_source_timestamp.isoformat() if features else None,
            "features": dict(features.values) if features else None,
            "prediction": prediction, "cost_estimate": cost, "signal": signal, "risk": risk, "order": order,
            "portfolio": portfolio, "extra": extra or {},
        }
        self._write(self._bars, rec)
        self.n_records += 1

    def record_fill(self, fill: Fill, portfolio=None) -> None:
        self._write(self._fills, {"type": "fill", "run_id": self.run_id, "fill": fill, "portfolio": portfolio})

    def event(self, kind: str, **info: Any) -> None:
        self._write(self._events, {"type": "event", "run_id": self.run_id, "event": kind, **info})
        if self.echo:
            self.echo(f"[{kind}] " + ", ".join(f"{k}={v}" for k, v in info.items() if k != "details"))

    def close(self) -> None:
        for h in (self._bars, self._fills, self._events):
            try:
                h.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["AuditLogger"]
