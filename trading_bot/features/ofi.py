"""Order-flow imbalance from bar-close NBBO snapshots (market-state spec sections 13-15).

Tier B data (best bid / ask *with displayed sizes* at each bar close) is required.  Between two
consecutive snapshots the contribution is the Cont-Kukanov-Stoikov form, which handles best-price
changes instead of the naive size difference:

    e_n =  1{P_b,n >= P_b,n-1} q_b,n  - 1{P_b,n <= P_b,n-1} q_b,n-1
         - 1{P_a,n <= P_a,n-1} q_a,n  + 1{P_a,n >= P_a,n-1} q_a,n-1

Without sizes the family is *unavailable* and reported as such: nothing is approximated from
Tier A data.  Every value at bar n uses snapshots n-1 and n only (causal); the quote timestamp
of snapshot n is the feature's newest source time.
"""

from __future__ import annotations

import numpy as np

from .rolling import lag, robust_zscore


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.full(len(x), np.nan)
    s = np.nan
    for i, v in enumerate(x):
        if np.isfinite(v):
            s = v if not np.isfinite(s) else (1 - a) * s + a * v
        out[i] = s
    return out


def ofi_events(bid: np.ndarray, ask: np.ndarray, bid_size: np.ndarray, ask_size: np.ndarray) -> np.ndarray:
    n = len(bid)
    e = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isfinite(bid[i]) and np.isfinite(bid[i - 1]) and np.isfinite(ask[i]) and np.isfinite(ask[i - 1])
                and np.isfinite(bid_size[i]) and np.isfinite(bid_size[i - 1]) and np.isfinite(ask_size[i]) and np.isfinite(ask_size[i - 1])):
            continue
        eb = (bid_size[i] if bid[i] >= bid[i - 1] else 0.0) - (bid_size[i - 1] if bid[i] <= bid[i - 1] else 0.0)
        ea = (ask_size[i] if ask[i] <= ask[i - 1] else 0.0) - (ask_size[i - 1] if ask[i] >= ask[i - 1] else 0.0)
        e[i] = eb - ea
    return e


def ofi_feature_arrays(bid: np.ndarray, ask: np.ndarray, bid_size: np.ndarray, ask_size: np.ndarray, close: np.ndarray,
                       return_1: np.ndarray, ema_short: int = 5, ema_long: int = 20, z_window: int = 100,
                       eps: float = 1e-12) -> dict[str, np.ndarray]:
    raw = ofi_events(bid, ask, bid_size, ask_size)
    depth = bid_size + ask_size
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = raw / (depth + eps)
        qi = (bid_size - ask_size) / (depth + eps)
        mid = 0.5 * (bid + ask)
        spread = np.where(np.isfinite(mid) & (mid > 0), (ask - bid) / mid, np.nan)
    es, el = _ema(norm, ema_short), _ema(norm, ema_long)
    z = robust_zscore(norm, z_window, eps)
    return {
        "ofi_raw": raw, "ofi_normalized": norm, "ofi_ema_short": es, "ofi_ema_long": el, "ofi_zscore": z,
        "ofi_acceleration": es - lag(es, 1),
        "ofi_price_divergence": z - robust_zscore(return_1, z_window, eps),
        "quote_imbalance": qi, "ofi_spread": spread, "ofi_spread_zscore": robust_zscore(spread, z_window, eps),
    }


OFI_NAMES = ("ofi_raw", "ofi_normalized", "ofi_ema_short", "ofi_ema_long", "ofi_zscore", "ofi_acceleration",
             "ofi_price_divergence", "quote_imbalance", "ofi_spread", "ofi_spread_zscore")

__all__ = ["ofi_events", "ofi_feature_arrays", "OFI_NAMES"]
