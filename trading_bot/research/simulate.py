"""Strategy simulator for research runs (spec sections 5, 11-14, 22, 28).

One implementation of the trading rules serves model selection (``simulate_validation``),
the walk-forward OOS evaluation, the cost / timing stress tests and the sanity tests:

* decisions at the close of row ``i`` execute at ``open_next[i + delay]`` and are held to
  ``open_next2[i + delay]`` (``delay`` = extra bars of execution latency, section 22),
* per-side cost = ``cost_side_exec`` (scaled by ``cost_scale``) or a flat ``cost_bps`` override,
* position rules (sizing, turnover suppression, stop, max holding, re-evaluation cadence)
  are the shared ``apply_position_rules`` used by the live RiskEngine,
* optional portfolio-level circuit breakers: daily loss halt (rest of session) and drawdown
  halt (until the next walk-forward window, the research proxy for "until an accepted retrain"),
* every round trip is recorded with the fields of the trade audit trail (section 28), sized
  from information available before execution only (section 13).

All quantities are in return units on a unit of capital; ``capital`` scales the reported
currency fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..risk.limits import apply_position_rules, stop_triggered
from ..strategy.sizing import confidence_from_edge, direction_from_edge, raw_exposure, volatility_multiplier


@dataclass
class SimInputs:
    E: np.ndarray                 # calibrated expected normalised return per row
    y_norm: np.ndarray            # realised normalised label (for accuracy / correlation only, never for trading)
    sigma: np.ndarray
    sigma_ref: np.ndarray
    cost_roundtrip: np.ndarray
    cost_side_exec: np.ndarray
    log_close: np.ndarray
    open_next: np.ndarray
    open_next2: np.ndarray
    M: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    session_ids: Optional[np.ndarray] = None    # int per row: session ordinal (daily loss halt)
    window_ids: Optional[np.ndarray] = None     # int per row: walk-forward window (drawdown halt release)
    timestamps: Optional[list] = None           # decision timestamps (bar close) per row
    exec_timestamps: Optional[list] = None      # execution timestamps (next bar open) per row
    model_ids: Optional[list] = None            # model id per row

    @classmethod
    def from_dataset(cls, ds, rows: np.ndarray, E: np.ndarray, M: np.ndarray | None = None,
                     P: np.ndarray | None = None, **extra) -> "SimInputs":
        return cls(E=np.asarray(E, dtype=float), y_norm=ds.y_norm[rows], sigma=ds.sigma[rows], sigma_ref=ds.sigma_ref[rows],
                   cost_roundtrip=ds.cost_roundtrip[rows], cost_side_exec=ds.cost_side_exec[rows],
                   log_close=ds.log_close[rows], open_next=ds.open_next[rows], open_next2=ds.open_next2[rows],
                   M=None if M is None else np.asarray(M, dtype=float), P=None if P is None else np.asarray(P, dtype=float),
                   timestamps=[ds.close_times[i] for i in rows], **extra)

    def __len__(self) -> int:
        return len(self.E)


@dataclass
class SimResult:
    bar_pnl: np.ndarray           # net return per row (on total equity)
    gross_bar_pnl: np.ndarray
    cost_bar: np.ndarray
    equity: np.ndarray            # length n + 1, starts at 1.0
    exposures: np.ndarray         # exposure in effect after row i's decision
    targets: np.ndarray           # raw target exposure Q_t before position rules
    halted: np.ndarray            # bool per row: a portfolio circuit breaker forced flat
    trades: list[dict[str, Any]] = field(default_factory=list)
    n_signals: int = 0
    n_fills: int = 0
    delay: int = 0
    cost_scale: float = 1.0
    cost_bps: Optional[float] = None
    capital: float = 1.0

    @property
    def net_pnl(self) -> float:
        return float(self.equity[-1] - 1.0)

    @property
    def total_cost(self) -> float:
        return float(self.cost_bar.sum())

    @property
    def gross_pnl(self) -> float:
        return float(self.gross_bar_pnl.sum())

    def trade_pnls(self) -> np.ndarray:
        return np.asarray([t["net_pnl"] for t in self.trades], dtype=float)


def simulate_strategy(inp: SimInputs, params, cost_scale: float = 1.0, cost_bps: float | None = None, delay: int = 0,
                      daily_loss_limit: float | None = None, drawdown_halt: float | None = None,
                      capital: float = 1.0, model_ids: list | None = None) -> SimResult:
    n = len(inp)
    H = params.horizon
    # cost arrays under the scenario
    if cost_bps is not None:
        cost_rt = np.full(n, cost_bps / 1e4)
        cost_side = np.full(n, cost_bps / 2e4)
    else:
        cost_rt = np.asarray(inp.cost_roundtrip, dtype=float) * cost_scale
        cost_side = np.asarray(inp.cost_side_exec, dtype=float) * cost_scale
    q_cur = 0.0
    holding = 0
    entry_price = math.nan
    entry_sigma = math.nan
    entry_dir = 0
    equity = 1.0
    peak = 1.0
    curve = np.empty(n + 1)
    curve[0] = equity
    exposures = np.zeros(n)
    targets = np.zeros(n)
    bar_pnl = np.zeros(n)
    gross_bar = np.zeros(n)
    cost_bar = np.zeros(n)
    halted = np.zeros(n, dtype=bool)
    trades: list[dict[str, Any]] = []
    open_trade: Optional[dict[str, Any]] = None
    n_signals = 0
    n_fills = 0
    session_start_equity = 1.0
    current_session = None
    daily_halt_session = None
    dd_halt_window = None
    current_window = None
    usable = n - delay if delay > 0 else n
    for i in range(usable):
        j = i + delay                                 # execution row
        # session / window bookkeeping
        sid = int(inp.session_ids[i]) if inp.session_ids is not None else 0
        wid = int(inp.window_ids[i]) if inp.window_ids is not None else 0
        if sid != current_session:
            current_session = sid
            session_start_equity = equity
        if wid != current_window:
            current_window = wid
            if dd_halt_window is not None and dd_halt_window != wid:
                dd_halt_window = None             # a new window = a new accepted model lifts the drawdown halt
                peak = equity                     # ... and re-bases the drawdown reference (RiskEngine.after_retrain)
        if q_cur != 0.0:
            holding += 1
        stop_hit = False
        if q_cur != 0.0 and math.isfinite(entry_price):
            pos_ret = entry_dir * (inp.log_close[i] - math.log(entry_price))
            stop_hit = stop_triggered(pos_ret, entry_sigma, H, params.stop_sigma_multiple)
        er = inp.E[i] * inp.sigma[i] * math.sqrt(H)
        direction = direction_from_edge(er, cost_rt[i], params.cost_multiplier)
        conf = 0.0
        if direction:
            n_signals += 1
            conf = confidence_from_edge(er, cost_rt[i], params.confidence_cost_multiplier, params.eps)
            vm = volatility_multiplier(inp.sigma_ref[i], inp.sigma[i], params.vol_multiplier_min, params.vol_multiplier_max)
            q_raw = raw_exposure(direction, conf, vm, params.max_abs_exposure)
        else:
            q_raw = 0.0
        targets[i] = q_raw
        # portfolio circuit breakers (evaluated on mark-to-market equity known at the decision)
        force_flat = False
        if daily_loss_limit is not None:
            if daily_halt_session == sid or (equity / session_start_equity - 1.0 < -daily_loss_limit):
                daily_halt_session = sid
                force_flat = True
        if drawdown_halt is not None:
            if dd_halt_window == wid or (equity / peak - 1.0 < -drawdown_halt):
                dd_halt_window = wid
                force_flat = True
        if force_flat:
            halted[i] = True
            q_raw = 0.0
        rule = apply_position_rules(q_raw, q_cur, holding, params.max_holding_bars, stop_hit,
                                    params.rebalance_threshold, params.reevaluate_every)
        q_new = 0.0 if force_flat else rule.exposure
        reason = "HALT" if force_flat and q_cur != 0.0 else rule.reason
        # execution at open_next[j]; exposure held until open_next2[j]
        ret = inp.open_next2[j] / inp.open_next[j] - 1.0
        turnover = abs(q_new - q_cur)
        cost = turnover * cost_side[j]
        if turnover > 0:
            n_fills += 1
        pnl = q_new * ret - cost
        gross_bar[i] = q_new * ret
        cost_bar[i] = cost
        equity_before = equity
        equity *= (1.0 + pnl)
        peak = max(peak, equity)
        curve[i + 1] = equity
        exposures[i] = q_new
        bar_pnl[i] = pnl
        flipped = q_cur != 0.0 and q_new != 0.0 and (q_cur > 0) != (q_new > 0)
        closing = q_cur != 0.0 and (q_new == 0.0 or flipped or rule.new_entry)
        remaining = pnl
        if closing and open_trade is not None:
            if q_new == 0.0:
                close_cost = cost
            elif flipped:
                close_cost = abs(q_cur) * cost_side[j]
            else:
                close_cost = 0.0
            exit_reason = ("stop" if rule.stop_status == "TRIGGERED" else "halt" if force_flat else
                           "flip" if flipped else "reentry" if rule.new_entry else
                           "max_holding" if rule.max_holding_status == "EXPIRED" else "flat")
            open_trade.update({
                "exit_row": int(i), "exit_price": float(inp.open_next[j]), "bars_held": int(holding),
                "exit_reason": exit_reason, "cost": open_trade["cost"] + close_cost,
                "gross_pnl": open_trade["gross_pnl"],
                "net_pnl": open_trade["gross_pnl"] - open_trade["cost"] - close_cost,
                "exit_timestamp": _ts(inp.exec_timestamps, j),
            })
            open_trade["net_pnl_currency"] = open_trade["net_pnl"] * capital * open_trade["equity_before"]
            open_trade["gross_pnl_currency"] = open_trade["gross_pnl"] * capital * open_trade["equity_before"]
            open_trade["cost_currency"] = open_trade["cost"] * capital * open_trade["equity_before"]
            trades.append(open_trade)
            open_trade = None
            remaining = pnl + close_cost
        if q_new != 0.0:
            if open_trade is None:
                open_trade = {
                    "trade_id": len(trades) + 1, "entry_row": int(i), "decision_timestamp": _ts(inp.timestamps, i),
                    "execution_timestamp": _ts(inp.exec_timestamps, j), "model_id": (model_ids[i] if model_ids else None),
                    "forecast": float(inp.E[i]), "expected_return": float(er),
                    "probability_up": float(inp.P[i]) if inp.P is not None else None,
                    "confidence": float(conf), "target_exposure": float(q_raw), "approved_exposure": float(q_new),
                    "direction": 1 if q_new > 0 else -1, "entry_price": float(inp.open_next[j]),
                    "equity_before": float(equity_before), "max_exposure": abs(q_new), "gross_pnl": 0.0, "cost": 0.0,
                    "delay": delay,
                }
                # cost of the opening leg (a flip attributes only its opening part here)
                open_cost = cost - (abs(q_cur) * cost_side[j] if flipped else 0.0) if (flipped or q_cur == 0.0) else cost
                open_trade["cost"] += open_cost
                open_trade["gross_pnl"] += q_new * ret
            else:
                open_trade["gross_pnl"] += q_new * ret
                open_trade["cost"] += cost
                open_trade["max_exposure"] = max(open_trade["max_exposure"], abs(q_new))
            if rule.new_entry or q_cur == 0.0:
                entry_price = inp.open_next[j]
                entry_sigma = inp.sigma[i]
                entry_dir = 1 if q_new > 0 else -1
                holding = 0
        else:
            holding = 0
            entry_price = math.nan
            entry_dir = 0
        q_cur = q_new
    if open_trade is not None:
        last = usable - 1
        open_trade.update({"exit_row": int(last), "exit_price": float(inp.open_next2[last + delay]) if usable else math.nan,
                           "bars_held": int(holding), "exit_reason": "end", "net_pnl": open_trade["gross_pnl"] - open_trade["cost"],
                           "exit_timestamp": _ts(inp.exec_timestamps, min(last + delay + 1, n - 1)) if inp.exec_timestamps else None})
        open_trade["net_pnl_currency"] = open_trade["net_pnl"] * capital * open_trade["equity_before"]
        open_trade["gross_pnl_currency"] = open_trade["gross_pnl"] * capital * open_trade["equity_before"]
        open_trade["cost_currency"] = open_trade["cost"] * capital * open_trade["equity_before"]
        trades.append(open_trade)
    if usable < n:
        curve[usable + 1:] = curve[usable]
    return SimResult(bar_pnl, gross_bar, cost_bar, curve, exposures, targets, halted, trades, n_signals, n_fills,
                     delay, cost_scale, cost_bps, capital)


def _ts(seq, i):
    if seq is None or i < 0 or i >= len(seq):
        return None
    t = seq[i]
    return t.isoformat() if hasattr(t, "isoformat") else t


__all__ = ["SimInputs", "SimResult", "simulate_strategy"]
