"""Expected-versus-realised execution cost bookkeeping (market-state spec sections 18-19, 45).

Per fill (all in basis points of notional, round-trip equivalents where the estimate is round-trip):

    spread        2 x the half-spread actually paid on this fill
    slippage      2 x the slippage actually charged
    decision_to_fill   direction x (fill price - decision price) / decision price   (price moved before the fill)
    adverse_selection  |ER| / H - direction x (close of the fill bar - fill price) / fill price
                       (post-fill move beyond the forecast's per-bar share, h = 1 bar; same definition as the
                        execution model's target, so predicted and realised are comparable)
    total         spread + slippage + adverse_selection + fees

The summary reports mean / median / P90 / P95 / worst of the prediction error, the share of fills
whose realised total lay within the modelled P95 band, and a calibration table (predicted bucket ->
realised mean).  It runs identically in backtests (simulated fills) and Alpaca paper trading.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class ExecutionCalibration:
    def __init__(self, horizon: int = 4):
        self.horizon = int(horizon)
        self.records: list[dict[str, Any]] = []

    def record(self, expected: dict[str, Any], order, fill, bar) -> dict[str, Any]:
        notional = fill.units * fill.fill_price if fill.units else 0.0
        direction = 1 if fill.side == "buy" else -1
        spread = 2.0 * fill.spread_cost / notional * 1e4 if notional else 0.0
        slip = 2.0 * fill.slippage_cost / notional * 1e4 if notional else 0.0
        fees = fill.commission / notional * 1e4 if notional else 0.0
        dp = expected.get("decision_price") or fill.reference_price
        drift = direction * (fill.fill_price - dp) / dp * 1e4 if dp else 0.0
        abs_er = float(expected.get("abs_er") or 0.0)
        post = direction * (bar.close - fill.fill_price) / fill.fill_price if fill.fill_price else 0.0
        adverse = (abs_er / max(1, self.horizon) - post) * 1e4
        realised_total = spread + slip + adverse + fees
        predicted_total = float(expected.get("total_expected_bps", 0.0))
        unc = float(expected.get("uncertainty_bps", 0.0))
        model_unc = float(expected.get("model_uncertainty_bps", 0.0))
        band = predicted_total + (1.645 * model_unc if model_unc > 0 else unc)
        rec = {"order_id": fill.order_id, "fill_timestamp": fill.fill_timestamp.isoformat(), "side": fill.side,
               "price_source": fill.price_source, "new_entry": bool(fill.new_entry),
               "expected": {"spread_bps": float(expected.get("spread_bps", 0.0)), "slippage_bps": float(expected.get("slippage_bps", 0.0)),
                            "adverse_selection_bps": float(expected.get("adverse_selection_bps", 0.0)), "fee_bps": float(expected.get("fee_bps", 0.0)),
                            "total_bps": predicted_total, "uncertainty_bps": unc, "p95_bps": band},
               "realised": {"spread_bps": spread, "slippage_bps": slip, "adverse_selection_bps": adverse, "fee_bps": fees,
                            "decision_to_fill_bps": drift, "total_bps": realised_total},
               "error_bps": realised_total - predicted_total, "within_p95": bool(realised_total <= band)}
        self.records.append(rec)
        return rec

    @staticmethod
    def _dist(x: list[float]) -> dict[str, float]:
        if not x:
            return {"n": 0}
        a = np.asarray(x, dtype=float)
        return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
                "p95": float(np.quantile(a, 0.95)), "worst": float(a.max()), "best": float(a.min())}

    def summary(self) -> dict[str, Any]:
        r = self.records
        if not r:
            return {"fills": 0}
        comp = ("spread_bps", "slippage_bps", "adverse_selection_bps", "total_bps")
        out: dict[str, Any] = {"fills": len(r)}
        for c in comp:
            exp = [x["expected"][c] for x in r]
            rea = [x["realised"][c] for x in r]
            out[c] = {"expected": self._dist(exp), "realised": self._dist(rea),
                      "error": self._dist([b - a for a, b in zip(exp, rea)])}
        out["decision_to_fill_bps"] = self._dist([x["realised"]["decision_to_fill_bps"] for x in r])
        out["within_p95_fraction"] = float(np.mean([x["within_p95"] for x in r]))
        bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, math.inf)]
        table = []
        for lo, hi in bins:
            sel = [x for x in r if lo <= x["expected"]["total_bps"] < hi]
            if sel:
                table.append({"predicted_bucket_bps": f"{lo}-{hi if math.isfinite(hi) else '+'}", "n": len(sel),
                              "predicted_mean_bps": float(np.mean([x["expected"]["total_bps"] for x in sel])),
                              "realised_mean_bps": float(np.mean([x["realised"]["total_bps"] for x in sel]))})
        out["calibration"] = table
        out["by_price_source"] = {}
        for src in sorted({x["price_source"] for x in r}):
            sel = [x for x in r if x["price_source"] == src]
            out["by_price_source"][src] = {"n": len(sel), "realised_total_mean_bps": float(np.mean([x["realised"]["total_bps"] for x in sel])),
                                           "expected_total_mean_bps": float(np.mean([x["expected"]["total_bps"] for x in sel]))}
        return out

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        return self.records[-n:]


__all__ = ["ExecutionCalibration"]
