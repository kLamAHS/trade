"""Cross-asset context with a strict causal as-of join (market-state spec sections 21-24).

For a primary decision at bar close t, the context instrument's value is the latest one whose
*close time* is <= t.  Never the nearest neighbour (which may lie in the future).  Every aligned
value carries its age; values older than ``max_staleness_seconds`` are invalid (NaN) rather than
stale-but-used.  Features are relative to the primary so the model sees context, not a second
alpha model of the other instrument.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime

import numpy as np

from .rolling import lag, robust_zscore, rolling_mean, rolling_std


def causal_asof_join(target_times: list[datetime], source_times: list[datetime], source_values: np.ndarray,
                     max_staleness_seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each target time the latest source value with source_time <= target time.
    Returns (values, age_seconds); NaN where nothing precedes the target or the value is too stale."""
    src = [t.timestamp() for t in source_times]
    vals = np.asarray(source_values, dtype=float)
    out = np.full(len(target_times), np.nan)
    age = np.full(len(target_times), np.nan)
    for i, t in enumerate(target_times):
        k = bisect.bisect_right(src, t.timestamp()) - 1
        if k < 0:
            continue
        a = t.timestamp() - src[k]
        if max_staleness_seconds is not None and a > max_staleness_seconds:
            continue
        out[i] = vals[k]
        age[i] = a
    return out, age


def _rolling_corr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    wa, wb = sliding_window_view(a, n), sliding_window_view(b, n)
    ma, mb = wa.mean(axis=1, keepdims=True), wb.mean(axis=1, keepdims=True)
    cov = ((wa - ma) * (wb - mb)).mean(axis=1)
    sa, sb = wa.std(axis=1), wb.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where((sa > 0) & (sb > 0), cov / (sa * sb), np.nan)
    out[n - 1:] = c
    return out


def cross_asset_feature_arrays(primary_log_close: np.ndarray, primary_volume: np.ndarray, primary_sigma: np.ndarray,
                               close_times: list[datetime], context: dict[str, tuple[list[datetime], np.ndarray, np.ndarray]],
                               max_staleness_seconds: float, corr_short: int = 20, corr_long: int = 200,
                               divergence_window: int = 50, eps: float = 1e-12) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``context`` maps symbol -> (close_times, log_close, volume).  Returns (features, ages)."""
    n = len(primary_log_close)
    r1_p = primary_log_close - lag(primary_log_close, 1)
    r4_p = primary_log_close - lag(primary_log_close, 4)
    s = np.where(np.isfinite(primary_sigma) & (primary_sigma > 0), primary_sigma, np.nan)
    feats: dict[str, np.ndarray] = {}
    ages: dict[str, np.ndarray] = {}
    rel1_all = []
    r1_all = []
    for sym, (times, lc, vol) in context.items():
        aligned, age = causal_asof_join(close_times, times, lc, max_staleness_seconds)
        vol_al, _ = causal_asof_join(close_times, times, vol, max_staleness_seconds)
        ages[sym] = age
        r1 = aligned - lag(aligned, 1)
        r4 = aligned - lag(aligned, 4)
        rel1 = (r1 - r1_p) / (s + eps)
        rel4 = (r4 - r4_p) / (2.0 * s + eps)
        key = sym.lower()
        feats[f"xa_{key}_rel_return_1"] = rel1
        feats[f"xa_{key}_rel_return_4"] = rel4
        feats[f"xa_{key}_divergence"] = robust_zscore(aligned - primary_log_close, divergence_window, eps)
        c_s = _rolling_corr(np.nan_to_num(r1, nan=0.0), np.nan_to_num(r1_p, nan=0.0), corr_short)
        c_l = _rolling_corr(np.nan_to_num(r1, nan=0.0), np.nan_to_num(r1_p, nan=0.0), corr_long)
        feats[f"xa_{key}_corr_short"] = c_s
        feats[f"xa_{key}_corr_breakdown"] = c_s - c_l
        with np.errstate(divide="ignore", invalid="ignore"):
            feats[f"xa_{key}_rel_volume"] = robust_zscore(np.log1p(vol_al) - np.log1p(primary_volume), divergence_window, eps)
            feats[f"xa_{key}_vol_divergence"] = np.log((rolling_std(r1, corr_short) + eps) / (rolling_std(r1_p, corr_short) + eps))
        feats[f"xa_{key}_age_s"] = age
        rel1_all.append(rel1)
        r1_all.append(r1)
    if r1_all:
        R = np.column_stack(r1_all)
        with np.errstate(invalid="ignore"):
            feats["xa_breadth"] = np.nanmean(R > 0, axis=1) if R.shape[1] else np.full(n, np.nan)
            feats["xa_dispersion"] = np.nanstd(R, axis=1) / (s + eps)
    return feats, ages


def cross_asset_names(symbols) -> tuple[str, ...]:
    out = []
    for sym in symbols:
        k = sym.lower()
        out += [f"xa_{k}_rel_return_1", f"xa_{k}_rel_return_4", f"xa_{k}_divergence", f"xa_{k}_corr_short",
                f"xa_{k}_corr_breakdown", f"xa_{k}_rel_volume", f"xa_{k}_vol_divergence"]
    if symbols:
        out += ["xa_breadth", "xa_dispersion"]
    return tuple(out)


__all__ = ["causal_asof_join", "cross_asset_feature_arrays", "cross_asset_names"]
