"""Full performance metric set for research runs (spec sections 14-15)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class StrategyMetrics:
    values: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, k):
        return self.values[k]

    def get(self, k, default=None):
        return self.values.get(k, default)

    def __contains__(self, k) -> bool:
        return k in self.values

    def keys(self):
        return self.values.keys()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def drawdown_stats(equity: np.ndarray) -> dict[str, float]:
    if len(equity) < 2:
        return {"max_drawdown": 0.0, "max_drawdown_duration": 0, "recovery_duration": 0, "current_drawdown": 0.0,
                "ulcer_index": 0.0}
    dd = drawdown_series(equity)
    max_dd = float(-dd.min())
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(equity[: trough + 1]))
    # recovery: first index after the trough where equity regains the prior peak
    recovered = np.flatnonzero(equity[trough:] >= equity[peak_idx])
    recovery = int(recovered[0]) if len(recovered) else -1
    # longest stretch under water
    under = dd < 0
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {"max_drawdown": max_dd, "max_drawdown_duration": int(longest), "recovery_duration": recovery,
            "current_drawdown": float(dd[-1]), "ulcer_index": float(math.sqrt(np.mean(dd ** 2)))}


def _annualise(rets: np.ndarray, bars_per_year: int) -> dict[str, float]:
    n = len(rets)
    if n == 0:
        return {"volatility": 0.0, "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0}
    mean, sd = float(rets.mean()), float(rets.std())
    downside = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))
    total = float(np.prod(1.0 + rets))
    years = n / bars_per_year
    cagr = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else 0.0
    return {"volatility": sd * math.sqrt(bars_per_year), "sharpe": (mean / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0,
            "sortino": (mean / downside * math.sqrt(bars_per_year)) if downside > 0 else 0.0, "cagr": cagr}


def rolling_sharpe(rets: np.ndarray, window: int, bars_per_year: int) -> np.ndarray:
    out = np.full(len(rets), np.nan)
    if len(rets) >= window:
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(rets, window)
        sd = w.std(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window - 1:] = np.where(sd > 0, w.mean(axis=1) / sd * math.sqrt(bars_per_year), 0.0)
    return out


def compute_strategy_metrics(bar_pnl: np.ndarray, equity: np.ndarray, exposures: np.ndarray, trades: list[dict],
                             cost_bar: np.ndarray, bars_per_day: int = 13, trading_days_per_year: int = 252,
                             session_ids: Optional[np.ndarray] = None, timestamps: Optional[list] = None,
                             y_norm: Optional[np.ndarray] = None, E: Optional[np.ndarray] = None,
                             capital: float = 1.0, rolling_window_days: int = 20) -> StrategyMetrics:
    rets = np.asarray(bar_pnl, dtype=float)
    eq = np.asarray(equity, dtype=float)
    ex = np.asarray(exposures, dtype=float)
    bpy = bars_per_day * trading_days_per_year
    n = len(rets)
    v: dict[str, Any] = {"bars": int(n), "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
                         "terminal_equity": float(eq[-1] * capital) if len(eq) else capital}
    v.update(_annualise(rets, bpy))
    dd = drawdown_stats(eq)
    v.update(dd)
    v["calmar"] = (v["cagr"] / dd["max_drawdown"]) if dd["max_drawdown"] > 0 else 0.0
    # distribution
    if n > 3 and rets.std() > 0:
        z = (rets - rets.mean()) / rets.std()
        v["skew"] = float(np.mean(z ** 3))
        v["kurtosis"] = float(np.mean(z ** 4) - 3.0)
    else:
        v["skew"] = v["kurtosis"] = 0.0
    if n:
        q = float(np.quantile(rets, 0.05))
        v["var_95"] = -q
        tail = rets[rets <= q]
        v["es_95"] = float(-tail.mean()) if len(tail) else -q
    else:
        v["var_95"] = v["es_95"] = 0.0
    # sessions
    if session_ids is not None and n:
        sid = np.asarray(session_ids)
        daily = {}
        for s, r in zip(sid, rets):
            daily[s] = daily.get(s, 1.0) * (1.0 + r)
        d_rets = np.array([x - 1.0 for x in daily.values()])
        v["worst_daily_loss"] = float(d_rets.min()) if len(d_rets) else 0.0
        v["sessions"] = int(len(d_rets))
        if len(d_rets) >= 5:
            cum = np.cumprod(1.0 + d_rets)
            five = cum[5:] / cum[:-5] - 1.0
            v["worst_5_day_loss"] = float(min(five.min(), (cum[4] - 1.0)))
        else:
            v["worst_5_day_loss"] = float(np.prod(1.0 + d_rets) - 1.0) if len(d_rets) else 0.0
    # execution / exposure
    dq = np.abs(np.diff(np.concatenate(([0.0], ex)))) if n else np.array([])
    v["turnover_per_year"] = float(dq.sum() / n * bpy) if n else 0.0
    v["turnover_total"] = float(dq.sum())
    v["total_cost"] = float(np.sum(cost_bar))
    v["total_cost_currency"] = float(np.sum(cost_bar) * capital)
    v["mean_gross_exposure"] = float(np.mean(np.abs(ex))) if n else 0.0
    v["mean_net_exposure"] = float(np.mean(ex)) if n else 0.0
    v["time_invested"] = float(np.mean(ex != 0)) if n else 0.0
    # trades
    tp = np.asarray([t["net_pnl"] for t in trades], dtype=float)
    wins, losses = tp[tp > 0], tp[tp < 0]
    v["trade_count"] = int(len(tp))
    v["win_rate"] = float(np.mean(tp > 0)) if len(tp) else 0.0
    v["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else 0.0)
    v["expectancy"] = float(tp.mean()) if len(tp) else 0.0
    v["average_win"] = float(wins.mean()) if len(wins) else 0.0
    v["average_loss"] = float(losses.mean()) if len(losses) else 0.0
    v["win_loss_ratio"] = float(v["average_win"] / -v["average_loss"]) if v["average_loss"] < 0 else 0.0
    v["median_trade"] = float(np.median(tp)) if len(tp) else 0.0
    v["trade_p95"] = float(np.quantile(tp, 0.95)) if len(tp) else 0.0
    v["trade_p05"] = float(np.quantile(tp, 0.05)) if len(tp) else 0.0
    v["long_pnl"] = float(sum(t["net_pnl"] for t in trades if t["direction"] > 0))
    v["short_pnl"] = float(sum(t["net_pnl"] for t in trades if t["direction"] < 0))
    v["long_trades"] = int(sum(1 for t in trades if t["direction"] > 0))
    v["short_trades"] = int(sum(1 for t in trades if t["direction"] < 0))
    v["mean_bars_held"] = float(np.mean([t["bars_held"] for t in trades])) if trades else 0.0
    by_hold: dict[str, list[float]] = {}
    for t in trades:
        b = t["bars_held"]
        key = "1" if b <= 1 else "2-4" if b <= 4 else "5-8" if b <= 8 else "9-12" if b <= 12 else "13+"
        by_hold.setdefault(key, []).append(t["net_pnl"])
    v["pnl_by_holding_period"] = {k: {"count": len(x), "total": float(np.sum(x)), "mean": float(np.mean(x))} for k, x in sorted(by_hold.items())}
    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t.get("exit_reason", "?")] = exit_reasons.get(t.get("exit_reason", "?"), 0) + 1
    v["exits_by_reason"] = exit_reasons
    # calendar attribution
    if timestamps is not None and n:
        by_year: dict[str, float] = {}
        by_month: dict[str, float] = {}
        for ts, r in zip(timestamps, rets):
            if ts is None:
                continue
            y, m = str(ts)[:4], str(ts)[:7]
            by_year[y] = by_year.get(y, 1.0) * (1.0 + r)
            by_month[m] = by_month.get(m, 1.0) * (1.0 + r)
        v["pnl_by_year"] = {k: float(x - 1.0) for k, x in sorted(by_year.items())}
        v["pnl_by_month"] = {k: float(x - 1.0) for k, x in sorted(by_month.items())}
    # forecast quality
    if y_norm is not None and E is not None and n:
        E = np.asarray(E, dtype=float)
        y = np.asarray(y_norm, dtype=float)
        called = E != 0
        v["directional_accuracy"] = float(np.mean(np.sign(E[called]) == np.sign(y[called]))) if called.any() else 0.0
        v["forecast_correlation"] = float(np.corrcoef(E, y)[0, 1]) if E.std() > 0 and y.std() > 0 else 0.0
        v["calibration_error"] = float(np.mean(np.abs(E - y)))
        v["prediction_loss"] = float(np.mean((E - y) ** 2))
        v["signal_rate"] = float(np.mean(called))
    # stability
    rw = rolling_window_days * bars_per_day
    rs = rolling_sharpe(rets, rw, bpy)
    finite = rs[np.isfinite(rs)]
    v["rolling_sharpe_window_bars"] = int(rw)
    v["rolling_sharpe_min"] = float(finite.min()) if len(finite) else 0.0
    v["rolling_sharpe_median"] = float(np.median(finite)) if len(finite) else 0.0
    v["rolling_sharpe_positive_fraction"] = float(np.mean(finite > 0)) if len(finite) else 0.0
    return StrategyMetrics(v)


def summarize_distribution(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {f"{prefix}n": 0}
    return {f"{prefix}n": int(len(x)), f"{prefix}mean": float(x.mean()), f"{prefix}median": float(np.median(x)),
            f"{prefix}std": float(x.std()), f"{prefix}p05": float(np.quantile(x, 0.05)),
            f"{prefix}p25": float(np.quantile(x, 0.25)), f"{prefix}p75": float(np.quantile(x, 0.75)),
            f"{prefix}p95": float(np.quantile(x, 0.95)), f"{prefix}min": float(x.min()), f"{prefix}max": float(x.max()),
            f"{prefix}positive_fraction": float(np.mean(x > 0))}


__all__ = ["StrategyMetrics", "compute_strategy_metrics", "drawdown_series", "drawdown_stats", "rolling_sharpe",
           "summarize_distribution"]
