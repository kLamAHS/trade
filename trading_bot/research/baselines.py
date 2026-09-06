"""Baseline strategies for the OOS comparison (research spec section 10).

Every baseline is traded on the *same* OOS rows, execution timing (next open to the open after)
and per-side cost as the strategy, so the comparison isolates the signal:

    cash            flat
    buy_and_hold    constant +1 exposure
    vol_scaled      +1 scaled to a constant annualised volatility target (capped at 1)
    momentum        sign/strength of the trailing return, turnover-suppressed
    random_signal   the strategy's own forecasts randomly permuted (same trade frequency and
                    forecast distribution; several seeds -> a distribution, not a point)
    no_fractional   the ablation baseline model (conventional features only)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .metrics import compute_strategy_metrics, summarize_distribution
from .simulate import SimResult, simulate_strategy


def simulate_exposure(exposure: np.ndarray, open_next: np.ndarray, open_next2: np.ndarray, cost_side: np.ndarray,
                      capital: float = 1.0, timestamps=None, exec_timestamps=None) -> SimResult:
    """Trade a prescribed exposure series with the strategy's execution convention."""
    q = np.asarray(exposure, dtype=float)
    n = len(q)
    ret = np.asarray(open_next2, dtype=float) / np.asarray(open_next, dtype=float) - 1.0
    prev = np.concatenate(([0.0], q[:-1]))
    turnover = np.abs(q - prev)
    cost = turnover * np.asarray(cost_side, dtype=float)
    gross = q * ret
    pnl = gross - cost
    equity = np.concatenate(([1.0], np.cumprod(1.0 + pnl)))
    trades: list[dict[str, Any]] = []
    open_trade = None
    for i in range(n):
        flipped = prev[i] != 0 and q[i] != 0 and (prev[i] > 0) != (q[i] > 0)
        if open_trade is not None and (q[i] == 0 or flipped):
            open_trade.update({"exit_row": i, "exit_price": float(open_next[i]), "bars_held": i - open_trade["entry_row"],
                               "exit_reason": "flip" if flipped else "flat",
                               "cost": open_trade["cost"] + (abs(prev[i]) * cost_side[i]),
                               "exit_timestamp": _iso(exec_timestamps, i)})
            open_trade["net_pnl"] = open_trade["gross_pnl"] - open_trade["cost"]
            trades.append(open_trade)
            open_trade = None
        if q[i] != 0:
            if open_trade is None:
                open_trade = {"trade_id": len(trades) + 1, "entry_row": i, "decision_timestamp": _iso(timestamps, i),
                              "execution_timestamp": _iso(exec_timestamps, i), "direction": 1 if q[i] > 0 else -1,
                              "entry_price": float(open_next[i]), "approved_exposure": float(q[i]), "max_exposure": abs(q[i]),
                              "gross_pnl": float(gross[i]), "cost": float(abs(q[i]) * cost_side[i]), "equity_before": float(equity[i])}
            else:
                open_trade["gross_pnl"] += float(gross[i])
                open_trade["cost"] += float(cost[i])
                open_trade["max_exposure"] = max(open_trade["max_exposure"], abs(q[i]))
    if open_trade is not None:
        open_trade.update({"exit_row": n - 1, "exit_price": float(open_next2[-1]), "bars_held": n - 1 - open_trade["entry_row"],
                           "exit_reason": "end", "net_pnl": open_trade["gross_pnl"] - open_trade["cost"],
                           "exit_timestamp": _iso(exec_timestamps, n - 1)})
        trades.append(open_trade)
    for t in trades:
        for k in ("net_pnl", "gross_pnl", "cost"):
            t[k + "_currency"] = t[k] * capital * t["equity_before"]
    return SimResult(pnl, gross, cost, equity, q, q.copy(), np.zeros(n, dtype=bool), trades,
                     int(np.sum(q != 0)), int(np.sum(turnover > 0)), 0, 1.0, None, capital)


def _iso(seq, i):
    if seq is None or i >= len(seq):
        return None
    t = seq[i]
    return t.isoformat() if hasattr(t, "isoformat") else t


def momentum_exposure(log_close: np.ndarray, sigma: np.ndarray, lookback: int, rebalance_threshold: float,
                      max_abs: float = 1.0) -> np.ndarray:
    n = len(log_close)
    q = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if i >= lookback:
            z = (log_close[i] - log_close[i - lookback]) / (sigma[i] * math.sqrt(lookback) + 1e-12)
            target = float(np.clip(z, -max_abs, max_abs))
        else:
            target = 0.0
        if abs(target - cur) >= rebalance_threshold or (target == 0.0 and cur != 0.0):
            cur = target
        q[i] = cur
    return q


def vol_scaled_exposure(sigma: np.ndarray, bars_per_year: int, target_annual_vol: float, max_abs: float = 1.0) -> np.ndarray:
    ann = np.asarray(sigma, dtype=float) * math.sqrt(bars_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(ann > 0, target_annual_vol / ann, 0.0)
    return np.clip(q, 0.0, max_abs)


def run_baselines(runner, oos, sims: dict[str, SimResult], n_random: int = 20, seed: int = 0,
                  vol_target: float = 0.10, momentum_bars: int = 65) -> dict[str, Any]:
    cfg = runner.cfg
    bpd, tdy = int(cfg.market.bars_per_day), int(cfg.market.trading_days_per_year)
    bpy = bpd * tdy
    capital = runner.capital
    cost_side = oos.cost_side_exec
    common = dict(bars_per_day=bpd, trading_days_per_year=tdy, session_ids=oos.session_ids, timestamps=oos.decision_at,
                  capital=capital)
    out: dict[str, Any] = {}

    def add(name: str, sim: SimResult, note: str = "") -> None:
        m = compute_strategy_metrics(sim.bar_pnl, sim.equity, sim.exposures, sim.trades, sim.cost_bar, **common)
        out[name] = {"metrics": m.to_dict(), "equity": [float(x) for x in sim.equity], "note": note}

    n = len(oos)
    exposures = {
        "cash": np.zeros(n),
        "buy_and_hold": np.ones(n),
        "vol_scaled": vol_scaled_exposure(oos.sigma, bpy, vol_target, float(cfg.risk.max_absolute_exposure)),
        "momentum": momentum_exposure(oos.log_close, oos.sigma, momentum_bars, float(cfg.signal.rebalance_threshold),
                                      float(cfg.risk.max_absolute_exposure)),
    }
    notes = {"cash": "flat", "buy_and_hold": "+1 exposure, same execution timing and costs",
             "vol_scaled": f"long only, scaled to {vol_target:.0%} annualised volatility",
             "momentum": f"trailing {momentum_bars}-bar return / sigma, clipped to ±1, rebalance threshold applied"}
    for name, q in exposures.items():
        add(name, simulate_exposure(q, oos.open_next, oos.open_next2, cost_side, capital, oos.decision_at, oos.execution_at), notes[name])
    add("no_fractional", sims["baseline"], "ablation baseline model: conventional features only, identical protocol")
    # random forecasts: the strategy's own E permuted (same frequency / magnitude distribution)
    rng = np.random.default_rng(seed)
    E = oos.forecasts["full"]["E"]
    random_sims: list[SimResult] = []
    sharpes, rets, dds = [], [], []
    for _ in range(n_random):
        sim = runner.simulate(oos, "full", E=E[rng.permutation(n)])
        m = compute_strategy_metrics(sim.bar_pnl, sim.equity, sim.exposures, sim.trades, sim.cost_bar, **common)
        random_sims.append(sim)
        sharpes.append(m["sharpe"]); rets.append(m["total_return"]); dds.append(m["max_drawdown"])
    if random_sims:
        med = int(np.argsort(sharpes)[len(sharpes) // 2])          # the median-Sharpe permutation is the representative
        add("random_signal", random_sims[med], f"median of {n_random} random permutations of the strategy's own forecasts")
        out["random_signal"]["distribution"] = {**summarize_distribution(np.array(sharpes), "sharpe_"),
                                                **summarize_distribution(np.array(rets), "return_"),
                                                **summarize_distribution(np.array(dds), "max_drawdown_")}
    strategy = compute_strategy_metrics(sims["full"].bar_pnl, sims["full"].equity, sims["full"].exposures, sims["full"].trades,
                                        sims["full"].cost_bar, **common)
    out["_comparison"] = {name: {"sharpe": v["metrics"]["sharpe"], "total_return": v["metrics"]["total_return"],
                                 "max_drawdown": v["metrics"]["max_drawdown"], "strategy_beats": bool(strategy["sharpe"] > v["metrics"]["sharpe"])}
                          for name, v in out.items() if not name.startswith("_")}
    out["_comparison"]["strategy"] = {"sharpe": strategy["sharpe"], "total_return": strategy["total_return"],
                                      "max_drawdown": strategy["max_drawdown"]}
    if sharpes:
        out["_comparison"]["strategy"]["random_signal_percentile"] = float(np.mean(np.array(sharpes) < strategy["sharpe"]))
    return out


__all__ = ["simulate_exposure", "momentum_exposure", "vol_scaled_exposure", "run_baselines"]
