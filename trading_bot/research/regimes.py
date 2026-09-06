"""Ex-ante regime tagging and P&L attribution (research spec section 18).

Regimes are tagged from information available at the decision bar only: trailing 20-bar
realised volatility against the trailing 200-bar level, and the close against its trailing
200-bar mean.  Nothing about the regime uses later bars, so the attribution can be reproduced
live bar by bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import _annualise, drawdown_stats


def tag_regimes(log_close_full: np.ndarray, bar_index: np.ndarray, short: int = 20, long: int = 200) -> dict[str, np.ndarray]:
    lc = np.asarray(log_close_full, dtype=float)
    r = np.diff(lc, prepend=lc[0])
    vol_tag = np.empty(len(bar_index), dtype=object)
    trend_tag = np.empty(len(bar_index), dtype=object)
    for k, i in enumerate(bar_index):
        i = int(i)
        if i < long:
            vol_tag[k], trend_tag[k] = "warmup", "warmup"
            continue
        v_short = r[i - short + 1: i + 1].std()
        v_long = r[i - long + 1: i + 1].std()
        vol_tag[k] = "high_vol" if v_short > v_long else "low_vol"
        trend_tag[k] = "uptrend" if lc[i] > lc[i - long + 1: i + 1].mean() else "downtrend"
    return {"volatility": vol_tag, "trend": trend_tag}


def regime_attribution(bar_pnl: np.ndarray, exposures: np.ndarray, trades: list[dict], tags: dict[str, np.ndarray],
                       bars_per_year: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    r = np.asarray(bar_pnl, dtype=float)
    for kind, tag in tags.items():
        table = {}
        for name in sorted(set(tag.tolist())):
            rows = np.flatnonzero(tag == name)
            rr = r[rows]
            eq = np.concatenate(([1.0], np.cumprod(1.0 + rr)))
            tp = [t["net_pnl"] for t in trades if tag[t["entry_row"]] == name]
            table[name] = {"bars": int(len(rows)), "fraction": float(len(rows) / len(r)) if len(r) else 0.0,
                           "net_return": float(eq[-1] - 1.0), "sharpe": _annualise(rr, bars_per_year)["sharpe"],
                           "max_drawdown": drawdown_stats(eq)["max_drawdown"], "trades": int(len(tp)),
                           "trade_pnl": float(np.sum(tp)) if tp else 0.0,
                           "win_rate": float(np.mean(np.array(tp) > 0)) if tp else 0.0,
                           "time_invested": float(np.mean(np.asarray(exposures)[rows] != 0)) if len(rows) else 0.0}
        out[kind] = table
    return out


__all__ = ["tag_regimes", "regime_attribution"]
