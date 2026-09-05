"""Performance database metrics (spec section 52)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .ledger import PortfolioLedger, TradeRecord


@dataclass(frozen=True)
class PerformanceMetrics:
    gross_return: float
    net_return: float
    transaction_costs: float
    turnover: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    average_trade: float
    median_trade: float
    win_percentage: float
    average_winning_trade: float
    average_losing_trade: float
    trade_count: int
    average_exposure: float
    long_performance: dict[str, float]
    short_performance: dict[str, float]
    by_volatility_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    by_time_of_day: dict[str, dict[str, float]] = field(default_factory=dict)
    by_fractional_d: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _group_stats(pnls: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(pnls, dtype=float)
    if len(arr) == 0:
        return {"count": 0, "total_pnl": 0.0, "mean_pnl": 0.0, "win_rate": 0.0}
    return {"count": int(len(arr)), "total_pnl": float(arr.sum()), "mean_pnl": float(arr.mean()),
            "win_rate": float(np.mean(arr > 0))}


def compute_metrics(ledger: PortfolioLedger, bars_per_day: int = 13, trading_days_per_year: int = 252,
                    groups: dict[str, list[tuple[str, float]]] | None = None) -> PerformanceMetrics:
    eq = np.asarray([e for _, e, _ in ledger.equity_history], dtype=float)
    ex = np.asarray([x for _, _, x in ledger.equity_history], dtype=float)
    init = ledger.initial_capital
    net = float(eq[-1] / init - 1.0) if len(eq) else 0.0
    gross = float((eq[-1] + ledger.total_costs) / init - 1.0) if len(eq) else 0.0
    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    bpy = bars_per_day * trading_days_per_year
    sd = float(rets.std()) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0
    dsd = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2))) if len(rets) else 0.0   # target-downside deviation
    sortino = float(rets.mean() / dsd * math.sqrt(bpy)) if dsd > 0 else 0.0
    if len(eq):
        peak = np.maximum.accumulate(eq)
        mdd = float(-np.min((eq - peak) / peak))
    else:
        mdd = 0.0
    trades: list[TradeRecord] = list(ledger.trades)
    tp = np.asarray([t.net_pnl for t in trades], dtype=float)   # fill prices already carry spread/slippage
    wins = tp[tp > 0]
    losses = tp[tp < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else 0.0)
    longs = [t.net_pnl for t in trades if t.direction > 0]
    shorts = [t.net_pnl for t in trades if t.direction < 0]
    turnover = float(ledger.turnover_notional / init) if init > 0 else 0.0
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    if groups:
        for gname, items in groups.items():
            buckets: dict[str, list[float]] = {}
            for key, pnl in items:
                buckets.setdefault(key, []).append(pnl)
            grouped[gname] = {k: _group_stats(v) for k, v in sorted(buckets.items())}
    return PerformanceMetrics(
        gross_return=gross, net_return=net, transaction_costs=float(ledger.total_costs), turnover=turnover,
        sharpe=sharpe, sortino=sortino, max_drawdown=mdd, profit_factor=pf,
        average_trade=float(tp.mean()) if len(tp) else 0.0, median_trade=float(np.median(tp)) if len(tp) else 0.0,
        win_percentage=float(np.mean(tp > 0)) if len(tp) else 0.0,
        average_winning_trade=float(wins.mean()) if len(wins) else 0.0,
        average_losing_trade=float(losses.mean()) if len(losses) else 0.0, trade_count=int(len(tp)),
        average_exposure=float(np.mean(np.abs(ex))) if len(ex) else 0.0,
        long_performance=_group_stats(longs), short_performance=_group_stats(shorts),
        by_volatility_regime=grouped.get("volatility_regime", {}), by_time_of_day=grouped.get("time_of_day", {}),
        by_fractional_d=grouped.get("fractional_d", {}))


__all__ = ["PerformanceMetrics", "compute_metrics"]
