"""Adaptive fractional-order estimation (spec section 7).

For each candidate d in {adaptive_min, ..., adaptive_max} (step adaptive_step):
  1. fractionally transform the training-window log-price series,
  2. Augmented Dickey-Fuller test,
  3. KPSS test,
  4. correlation with the original log-price series.

Acceptable when p_ADF < adf_pvalue_max and p_KPSS > kpss_pvalue_min; choose the
minimum acceptable d.  If none is acceptable, choose the candidate with the most
negative ADF statistic (subject to d <= adaptive_max).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .transform import fractional_transform


@dataclass(frozen=True)
class CandidateResult:
    d: float
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    correlation: float
    n_obs: int
    kernel_size: int

    @property
    def acceptable(self) -> bool:
        return False  # replaced by estimator with thresholds; kept for schema completeness

    def to_dict(self) -> dict:
        return {
            "d": self.d, "adf_stat": self.adf_stat, "adf_pvalue": self.adf_pvalue,
            "kpss_stat": self.kpss_stat, "kpss_pvalue": self.kpss_pvalue,
            "correlation": self.correlation, "n_obs": self.n_obs, "kernel_size": self.kernel_size,
        }


@dataclass(frozen=True)
class StationarityResult:
    d_star: float
    selected_by: str                     # "min_acceptable" | "strongest_adf"
    candidates: tuple[CandidateResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {"d_star": self.d_star, "selected_by": self.selected_by,
                "candidates": [c.to_dict() for c in self.candidates]}


def _adf(x: np.ndarray, maxlag: int | None) -> tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = adfuller(x, maxlag=maxlag, regression="c", autolag="AIC", result_object=False)
    return float(res[0]), float(res[1])


def _kpss(x: np.ndarray) -> tuple[float, float]:
    from statsmodels.tsa.stattools import kpss

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kpss(x, regression="c", nlags="auto")
    return float(res[0]), float(res[1])


class StationaryOrderEstimator:
    def __init__(self, adaptive_min: float = 0.05, adaptive_max: float = 0.95, adaptive_step: float = 0.05,
                 adf_pvalue_max: float = 0.05, kpss_pvalue_min: float = 0.05,
                 weight_threshold: float = 1e-5, weight_threshold_run: int = 10, max_lags: int = 500,
                 adf_maxlag: int | None = 20, candidates: Sequence[float] | None = None):
        if candidates is None:
            n_steps = int(round((adaptive_max - adaptive_min) / adaptive_step)) + 1
            candidates = [round(adaptive_min + i * adaptive_step, 10) for i in range(n_steps)]
        self.candidates = tuple(float(c) for c in candidates if c <= adaptive_max + 1e-12)
        self.adf_pvalue_max = adf_pvalue_max
        self.kpss_pvalue_min = kpss_pvalue_min
        self.weight_threshold = weight_threshold
        self.weight_threshold_run = weight_threshold_run
        self.max_lags = max_lags
        self.adf_maxlag = adf_maxlag

    def evaluate(self, log_price: np.ndarray) -> list[CandidateResult]:
        p = np.asarray(log_price, dtype=float)
        if p.ndim != 1:
            raise ValueError("log_price must be one-dimensional")
        results: list[CandidateResult] = []
        for d in self.candidates:
            fd = fractional_transform(p, d, self.weight_threshold, self.weight_threshold_run, self.max_lags)
            valid = ~np.isnan(fd)
            x = fd[valid]
            orig = p[valid]
            kernel = int(np.argmax(valid)) if valid.any() else len(p)
            if len(x) < 50 or not np.isfinite(x).all() or np.std(x) == 0:
                results.append(CandidateResult(d, math.nan, math.nan, math.nan, math.nan, math.nan, len(x), kernel))
                continue
            try:
                adf_stat, adf_p = _adf(x, self.adf_maxlag)
                kpss_stat, kpss_p = _kpss(x)
            except Exception:
                # A failed test never guesses: the candidate is recorded as unusable (NaN statistics).
                results.append(CandidateResult(d, math.nan, math.nan, math.nan, math.nan, math.nan, len(x), kernel))
                continue
            corr = float(np.corrcoef(x, orig)[0, 1]) if np.std(orig) > 0 else math.nan
            results.append(CandidateResult(d, adf_stat, adf_p, kpss_stat, kpss_p, corr, len(x), kernel))
        return results

    def is_acceptable(self, c: CandidateResult) -> bool:
        return (math.isfinite(c.adf_pvalue) and math.isfinite(c.kpss_pvalue)
                and c.adf_pvalue < self.adf_pvalue_max and c.kpss_pvalue > self.kpss_pvalue_min)

    def estimate(self, log_price: np.ndarray) -> StationarityResult:
        results = self.evaluate(log_price)
        acceptable = [c for c in results if self.is_acceptable(c)]
        if acceptable:
            best = min(acceptable, key=lambda c: c.d)
            return StationarityResult(best.d, "min_acceptable", tuple(results))
        finite = [c for c in results if math.isfinite(c.adf_stat)]
        if not finite:
            raise ValueError("stationarity estimation failed: no candidate produced a finite ADF statistic")
        best = min(finite, key=lambda c: (c.adf_stat, c.d))
        return StationarityResult(best.d, "strongest_adf", tuple(results))


__all__ = ["CandidateResult", "StationarityResult", "StationaryOrderEstimator"]
