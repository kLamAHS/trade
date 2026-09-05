"""Walk-forward validation simulator, metrics and acceptance tests (spec sections 39-41).

``simulate_validation`` replays the *signal -> sizing -> position rules ->
next-open execution with costs* chain on a block of out-of-sample predictions
using the same functions as the live engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..risk.limits import apply_position_rules, stop_triggered
from ..strategy.sizing import (confidence_from_edge, direction_from_edge, raw_exposure, volatility_multiplier)


@dataclass(frozen=True)
class ValidationMetrics:
    n: int
    accuracy: float
    correlation: float
    correlation_raw: float
    net_pnl: float
    gross_pnl: float
    total_cost: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    turnover_penalty: float
    drawdown_penalty: float
    score: float
    n_trades: int
    n_signals: int
    avg_exposure: float
    equity_curve: tuple[float, ...] = field(default_factory=tuple, repr=False)
    exposures: tuple[float, ...] = field(default_factory=tuple, repr=False)
    trade_pnls: tuple[float, ...] = field(default_factory=tuple, repr=False)
    bar_pnls: tuple[float, ...] = field(default_factory=tuple, repr=False)

    def to_dict(self, include_curves: bool = False) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__
             if k not in ("equity_curve", "exposures", "trade_pnls", "bar_pnls")}
        if include_curves:
            d["equity_curve"] = list(self.equity_curve)
            d["exposures"] = list(self.exposures)
        return d


@dataclass(frozen=True)
class SimulationParams:
    horizon: int = 4
    cost_multiplier: float = 3.0
    confidence_cost_multiplier: float = 6.0
    vol_multiplier_min: float = 0.25
    vol_multiplier_max: float = 1.5
    max_abs_exposure: float = 1.0
    rebalance_threshold: float = 0.15
    max_holding_bars: int = 12
    stop_sigma_multiple: float = 4.0
    bars_per_day: int = 13
    trading_days_per_year: int = 252
    turnover_weight: float = 0.25
    drawdown_weight: float = 0.25
    drawdown_reference: float = 0.10
    eps: float = 1e-12

    @classmethod
    def from_config(cls, cfg) -> "SimulationParams":
        return cls(horizon=int(cfg.prediction.horizon_bars), cost_multiplier=float(cfg.signal.cost_multiplier),
                   confidence_cost_multiplier=float(cfg.signal.confidence_cost_multiplier),
                   vol_multiplier_min=float(cfg.signal.vol_multiplier_min),
                   vol_multiplier_max=float(cfg.signal.vol_multiplier_max),
                   max_abs_exposure=float(cfg.risk.max_absolute_exposure),
                   rebalance_threshold=float(cfg.signal.rebalance_threshold),
                   max_holding_bars=int(cfg.risk.maximum_holding_bars),
                   stop_sigma_multiple=float(cfg.risk.stop_sigma_multiple), bars_per_day=int(cfg.market.bars_per_day),
                   trading_days_per_year=int(cfg.market.trading_days_per_year),
                   turnover_weight=float(cfg.training.score.turnover_weight),
                   drawdown_weight=float(cfg.training.score.drawdown_weight),
                   drawdown_reference=float(cfg.training.score.drawdown_reference), eps=float(cfg.features.epsilon))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(-dd.min()) if len(dd) else 0.0


def simulate_validation(E: np.ndarray, y_norm: np.ndarray, sigma: np.ndarray, sigma_ref: np.ndarray,
                        cost_roundtrip: np.ndarray, cost_side_exec: np.ndarray, log_close: np.ndarray,
                        open_next: np.ndarray, open_next2: np.ndarray, params: SimulationParams,
                        M: np.ndarray | None = None) -> ValidationMetrics:
    n = len(E)
    H = params.horizon
    q_cur = 0.0
    holding = 0
    entry_price = math.nan
    entry_sigma = math.nan
    entry_dir = 0
    equity = 1.0
    curve = np.empty(n + 1)
    curve[0] = equity
    exposures = np.empty(n)
    bar_pnl = np.empty(n)
    gross = 0.0
    costs = 0.0
    trade_pnls: list[float] = []
    trade_acc = 0.0
    in_trade = False
    n_signals = 0
    for i in range(n):
        # Stop check on the signal bar close relative to the entry fill.
        stop_hit = False
        if q_cur != 0.0 and math.isfinite(entry_price):
            pos_ret = entry_dir * (log_close[i] - math.log(entry_price))
            stop_hit = stop_triggered(pos_ret, entry_sigma, H, params.stop_sigma_multiple)
        er = E[i] * sigma[i] * math.sqrt(H)
        direction = direction_from_edge(er, cost_roundtrip[i], params.cost_multiplier)
        if direction:
            n_signals += 1
            conf = confidence_from_edge(er, cost_roundtrip[i], params.confidence_cost_multiplier, params.eps)
            vm = volatility_multiplier(sigma_ref[i], sigma[i], params.vol_multiplier_min, params.vol_multiplier_max)
            q_raw = raw_exposure(direction, conf, vm, params.max_abs_exposure)
        else:
            q_raw = 0.0
        rule = apply_position_rules(q_raw, q_cur, holding, params.max_holding_bars, stop_hit, params.rebalance_threshold)
        q_new = rule.exposure
        # Execution at open_{t+1}; exposure held until open_{t+2}.
        ret = open_next2[i] / open_next[i] - 1.0
        turnover = abs(q_new - q_cur)
        cost = turnover * cost_side_exec[i]
        pnl = q_new * ret - cost
        gross += q_new * ret
        costs += cost
        equity *= (1.0 + pnl)
        curve[i + 1] = equity
        exposures[i] = q_new
        bar_pnl[i] = pnl
        # Round-trip bookkeeping.  Every unit of bar P&L (including exit costs) belongs to exactly one trade.
        flipped = q_cur != 0.0 and q_new != 0.0 and (q_cur > 0) != (q_new > 0)
        closing = q_cur != 0.0 and (q_new == 0.0 or flipped or rule.new_entry)
        remaining = pnl
        if closing:
            if q_new == 0.0:
                close_cost = cost                               # the whole turnover is the exit
            elif flipped:
                close_cost = abs(q_cur) * cost_side_exec[i]     # closing leg of the flip
            else:
                close_cost = 0.0                                # same-direction re-entry: resize cost is the new trade's
            trade_pnls.append(trade_acc - close_cost)
            trade_acc = 0.0
            in_trade = False
            remaining = pnl + close_cost
        if q_new != 0.0:
            in_trade = True
            trade_acc += remaining
            if rule.new_entry or q_cur == 0.0:
                entry_price = open_next[i]
                entry_sigma = sigma[i]
                entry_dir = 1 if q_new > 0 else -1
                holding = 0          # the position is entered at the next open; the live ledger counts marks after it
            else:
                holding += 1
        else:
            holding = 0
            entry_price = math.nan
            entry_dir = 0
        q_cur = q_new
    if in_trade:
        trade_pnls.append(trade_acc)
    tp = np.asarray(trade_pnls)
    wins = tp[tp > 0].sum() if len(tp) else 0.0
    losses = -tp[tp < 0].sum() if len(tp) else 0.0
    profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    made_call = E != 0
    if made_call.any():
        accuracy = float(np.mean(np.sign(E[made_call]) == np.sign(y_norm[made_call])))
    else:
        accuracy = 0.0
    corr = _safe_corr(E, y_norm)
    corr_raw = _safe_corr(M, y_norm) if M is not None else math.nan
    bars_per_year = params.bars_per_day * params.trading_days_per_year
    sd = float(np.std(bar_pnl)) if n > 1 else 0.0
    sharpe = float(np.mean(bar_pnl) / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    dq = np.abs(np.diff(np.concatenate(([0.0], exposures))))
    turnover_penalty = float(dq.mean() * params.bars_per_day) if n else 0.0
    mdd = max_drawdown(curve)
    dd_penalty = mdd / params.drawdown_reference
    score = sharpe - params.turnover_weight * turnover_penalty - params.drawdown_weight * dd_penalty
    return ValidationMetrics(n=n, accuracy=accuracy, correlation=corr, correlation_raw=corr_raw,
                             net_pnl=float(equity - 1.0), gross_pnl=float(gross), total_cost=float(costs),
                             profit_factor=profit_factor, max_drawdown=mdd, sharpe=sharpe,
                             turnover_penalty=turnover_penalty, drawdown_penalty=float(dd_penalty), score=float(score),
                             n_trades=int(len(tp)), n_signals=n_signals, avg_exposure=float(np.mean(np.abs(exposures))) if n else 0.0,
                             equity_curve=tuple(curve.tolist()), exposures=tuple(exposures.tolist()),
                             trade_pnls=tuple(tp.tolist()), bar_pnls=tuple(bar_pnl.tolist()))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    checks: dict[str, bool]
    values: dict[str, float]
    folds_beating_baseline: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "checks": dict(self.checks), "values": dict(self.values),
                "folds_beating_baseline": self.folds_beating_baseline, "reasons": list(self.reasons)}


class ModelValidator:
    """Model acceptance criteria (spec section 39)."""

    def __init__(self, min_accuracy: float = 0.51, min_correlation: float = 0.0, min_net_pnl: float = 0.0,
                 min_profit_factor: float = 1.05, max_drawdown: float = 0.15, min_folds_beating_baseline: int = 3):
        self.min_accuracy = min_accuracy
        self.min_correlation = min_correlation
        self.min_net_pnl = min_net_pnl
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown
        self.min_folds_beating_baseline = min_folds_beating_baseline

    @classmethod
    def from_config(cls, cfg) -> "ModelValidator":
        a = cfg.training.acceptance
        return cls(a.min_accuracy, a.min_correlation, a.min_net_pnl, a.min_profit_factor, a.max_drawdown,
                   a.min_folds_beating_baseline)

    def evaluate(self, aggregate: ValidationMetrics, fold_scores: list[float],
                 baseline_fold_scores: list[float]) -> AcceptanceResult:
        beating = sum(1 for f, b in zip(fold_scores, baseline_fold_scores) if f > b)
        checks = {
            "accuracy": aggregate.accuracy > self.min_accuracy,
            "correlation": aggregate.correlation > self.min_correlation,
            "net_pnl": aggregate.net_pnl > self.min_net_pnl,
            "profit_factor": aggregate.profit_factor > self.min_profit_factor,
            "max_drawdown": aggregate.max_drawdown < self.max_drawdown,
            "beats_baseline": beating >= self.min_folds_beating_baseline,
        }
        values = {"accuracy": aggregate.accuracy, "correlation": aggregate.correlation, "net_pnl": aggregate.net_pnl,
                  "profit_factor": aggregate.profit_factor, "max_drawdown": aggregate.max_drawdown,
                  "beats_baseline": float(beating), "folds_beating_baseline": float(beating)}
        reasons = tuple(f"{k} failed ({values[k]:.4f})" for k, ok in checks.items() if not ok)
        return AcceptanceResult(all(checks.values()), checks, values, beating, reasons)


__all__ = ["ValidationMetrics", "SimulationParams", "simulate_validation", "max_drawdown", "ModelValidator",
           "AcceptanceResult"]
