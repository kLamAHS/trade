"""Statistical robustness: block bootstrap, ablation bootstrap, trade Monte Carlo and multiple-testing
bookkeeping (research spec sections 9, 16, 17)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from .metrics import _annualise, drawdown_stats, summarize_distribution


def _sharpe(r: np.ndarray, bpy: int) -> float:
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(bpy)) if sd > 0 else 0.0


def block_bootstrap(bar_pnl: np.ndarray, bars_per_year: int, block: int = 65, n_boot: int = 1000, seed: int = 0,
                    confidence: float = 0.95) -> dict[str, Any]:
    """Circular block bootstrap of the bar returns: confidence intervals for Sharpe, CAGR and the
    distribution of the maximum drawdown a strategy with these return characteristics produces."""
    r = np.asarray(bar_pnl, dtype=float)
    n = len(r)
    if n < 2 * block:
        block = max(1, n // 4)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    sharpes = np.empty(n_boot)
    cagrs = np.empty(n_boot)
    dds = np.empty(n_boot)
    rets = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        take = np.concatenate([(idx[s: s + block] if s + block <= n else np.concatenate((idx[s:], idx[: s + block - n]))) for s in starts])[:n]
        rb = r[take]
        ann = _annualise(rb, bars_per_year)
        sharpes[b] = ann["sharpe"]
        cagrs[b] = ann["cagr"]
        eq = np.concatenate(([1.0], np.cumprod(1.0 + rb)))
        dds[b] = drawdown_stats(eq)["max_drawdown"]
        rets[b] = eq[-1] - 1.0
    a = (1.0 - confidence) / 2.0
    point = _annualise(r, bars_per_year)
    return {"n_bars": int(n), "block_bars": int(block), "n_boot": int(n_boot), "confidence": confidence,
            "sharpe": {"point": point["sharpe"], "ci_low": float(np.quantile(sharpes, a)), "ci_high": float(np.quantile(sharpes, 1 - a)),
                       "p_negative": float(np.mean(sharpes <= 0)), **summarize_distribution(sharpes, "boot_")},
            "cagr": {"point": point["cagr"], "ci_low": float(np.quantile(cagrs, a)), "ci_high": float(np.quantile(cagrs, 1 - a))},
            "total_return": {"ci_low": float(np.quantile(rets, a)), "ci_high": float(np.quantile(rets, 1 - a)), "p_loss": float(np.mean(rets < 0))},
            "max_drawdown": {"point": drawdown_stats(np.concatenate(([1.0], np.cumprod(1.0 + r))))["max_drawdown"],
                             "median": float(np.median(dds)), "p95": float(np.quantile(dds, 0.95)), "p_over_20pct": float(np.mean(dds > 0.20)),
                             "p_over_10pct": float(np.mean(dds > 0.10))}}


def ablation_bootstrap(delta_sharpe: np.ndarray, n_boot: int = 5000, seed: int = 0, confidence: float = 0.95) -> dict[str, Any]:
    """Resample walk-forward windows: CI of the mean and median ΔSharpe and P(mean ΔSharpe > 0)."""
    d = np.asarray([x for x in delta_sharpe if x is not None and np.isfinite(x)], dtype=float)
    if len(d) == 0:
        return {"n_windows": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    medians = np.empty(n_boot)
    for b in range(n_boot):
        s = d[rng.integers(0, len(d), size=len(d))]
        means[b] = s.mean()
        medians[b] = np.median(s)
    a = (1.0 - confidence) / 2.0
    sign_test_p = float(stats.binomtest(int(np.sum(d > 0)), int(np.sum(d != 0)), 0.5, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    return {"n_windows": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "std": float(d.std()),
            "positive_fraction": float(np.mean(d > 0)),
            "mean_ci": [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))],
            "median_ci": [float(np.quantile(medians, a)), float(np.quantile(medians, 1 - a))],
            "p_mean_positive": float(np.mean(means > 0)), "sign_test_p_value": sign_test_p,
            "t_stat": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 1 and d.std(ddof=1) > 0 else 0.0}


def monte_carlo_trades(trade_pnls: np.ndarray, n_paths: int = 2000, seed: int = 0, dd_levels=(0.10, 0.20)) -> dict[str, Any]:
    """Resample the closed-trade P&L sequence with replacement: distribution of terminal wealth and of
    the maximum drawdown over a path of the same number of trades (section 17)."""
    tp = np.asarray(trade_pnls, dtype=float)
    n = len(tp)
    if n == 0:
        return {"n_trades": 0}
    rng = np.random.default_rng(seed)
    terminal = np.empty(n_paths)
    max_dd = np.empty(n_paths)
    for p in range(n_paths):
        path = tp[rng.integers(0, n, size=n)]
        eq = np.concatenate(([1.0], np.cumprod(1.0 + path)))
        terminal[p] = eq[-1] - 1.0
        max_dd[p] = drawdown_stats(eq)["max_drawdown"]
    actual = np.concatenate(([1.0], np.cumprod(1.0 + tp)))
    out = {"n_trades": int(n), "n_paths": int(n_paths),
           "terminal_return": {**summarize_distribution(terminal, ""), "p_loss": float(np.mean(terminal < 0)),
                               "actual": float(actual[-1] - 1.0)},
           "max_drawdown": {**summarize_distribution(max_dd, ""), "actual": drawdown_stats(actual)["max_drawdown"]}}
    for lvl in dd_levels:
        out["max_drawdown"][f"p_over_{int(round(lvl * 100))}pct"] = float(np.mean(max_dd > lvl))
    return out


def multiple_testing(sharpe_annual: float, n_bars: int, bars_per_year: int, n_tests: int) -> dict[str, Any]:
    """Naive p-value of the observed Sharpe against zero, and the Bonferroni-adjusted p-value for the
    number of configurations that were evaluated on the way to it (section 16).  Trials are not
    independent, so the adjusted value is conservative; it is reported, never used to inflate."""
    if n_bars < 2:
        return {"n_tests": int(n_tests)}
    sr_bar = sharpe_annual / math.sqrt(bars_per_year)      # per-bar Sharpe
    t = sr_bar * math.sqrt(n_bars)
    p = float(1.0 - stats.norm.cdf(t))
    return {"n_tests": int(n_tests), "t_stat": float(t), "p_value": p,
            "bonferroni_p_value": float(min(1.0, p * max(1, n_tests))),
            "sharpe_required_for_5pct_after_correction": float(stats.norm.ppf(1.0 - 0.05 / max(1, n_tests)) / math.sqrt(n_bars) * math.sqrt(bars_per_year))}


__all__ = ["block_bootstrap", "ablation_bootstrap", "monte_carlo_trades", "multiple_testing"]
