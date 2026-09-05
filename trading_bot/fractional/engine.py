"""FractionalEngine: the public fractional-operator API (spec section 46)."""

from __future__ import annotations

import numpy as np

from .stationarity import StationarityResult, StationaryOrderEstimator
from .transform import fractional_latest, fractional_transform
from .weights import build_weights


class FractionalEngine:
    def __init__(self, threshold: float = 1e-5, threshold_run: int = 10, max_lags: int = 500,
                 estimator: StationaryOrderEstimator | None = None):
        self.threshold = float(threshold)
        self.threshold_run = int(threshold_run)
        self.max_lags = int(max_lags)
        self.estimator = estimator or StationaryOrderEstimator(
            weight_threshold=threshold, weight_threshold_run=threshold_run, max_lags=max_lags)
        self._weights: dict[float, np.ndarray] = {}

    @classmethod
    def from_config(cls, cfg) -> "FractionalEngine":
        f = cfg.fractional
        estimator = StationaryOrderEstimator(
            adaptive_min=f.adaptive_min, adaptive_max=f.adaptive_max, adaptive_step=f.adaptive_step,
            adf_pvalue_max=f.adf_pvalue_max, kpss_pvalue_min=f.kpss_pvalue_min,
            weight_threshold=f.weight_threshold, weight_threshold_run=f.weight_threshold_run,
            max_lags=f.max_lags, adf_maxlag=f.adf_maxlag, min_observations=int(f.get("min_observations", 50)))
        return cls(f.weight_threshold, f.weight_threshold_run, f.max_lags, estimator)

    def build_weights(self, d: float, threshold: float | None = None, max_lags: int | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        ml = self.max_lags if max_lags is None else max_lags
        key = (float(d), thr, ml)
        if key not in self._weights:
            self._weights[key] = build_weights(d, thr, self.threshold_run, ml)
        return self._weights[key]

    def kernel_size(self, d: float) -> int:
        return len(self.build_weights(d)) - 1

    def transform(self, series: np.ndarray, d: float) -> np.ndarray:
        return fractional_transform(series, d, weights=self.build_weights(d))

    def latest(self, series: np.ndarray, d: float) -> float:
        return fractional_latest(series, d, weights=self.build_weights(d))

    def estimate_stationary_d(self, series: np.ndarray) -> float:
        return self.estimate_stationarity(series).d_star

    def estimate_stationarity(self, series: np.ndarray) -> StationarityResult:
        return self.estimator.estimate(np.asarray(series, dtype=float))


__all__ = ["FractionalEngine"]
