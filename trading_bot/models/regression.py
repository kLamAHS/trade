"""Gradient-boosted regression of the normalized forward return (spec section 20).

Backends: LightGBM (default) or scikit-learn HistGradientBoosting (fallback).
The same backend is used for backtesting and live simulation because the model
object is serialised into the artifact and reloaded unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionParams:
    backend: str = "lightgbm"
    max_depth: int = 3
    learning_rate: float = 0.03
    n_estimators: int = 300
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    num_threads: int = 2
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, cfg, n_estimators: int, min_child_samples: int, seed: int) -> "RegressionParams":
        r = cfg.models.regression
        return cls(backend=r.backend, max_depth=int(r.max_depth), learning_rate=float(r.learning_rate),
                   n_estimators=int(n_estimators), min_child_samples=int(min_child_samples),
                   subsample=float(r.subsample), colsample_bytree=float(r.colsample_bytree),
                   reg_alpha=float(r.reg_alpha), reg_lambda=float(r.reg_lambda), num_threads=int(r.num_threads),
                   seed=int(seed))


class BoostedRegressor:
    def __init__(self, params: RegressionParams):
        self.params = params
        self._model = None
        self.feature_names: tuple[str, ...] = tuple()

    def _build(self):
        p = self.params
        if p.backend == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                objective="regression", max_depth=p.max_depth, num_leaves=2 ** p.max_depth,
                learning_rate=p.learning_rate, n_estimators=p.n_estimators, min_child_samples=p.min_child_samples,
                subsample=p.subsample, subsample_freq=1, colsample_bytree=p.colsample_bytree,
                reg_alpha=p.reg_alpha, reg_lambda=p.reg_lambda, random_state=p.seed, n_jobs=p.num_threads,
                deterministic=True, force_row_wise=True, verbose=-1)
        if p.backend == "sklearn":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                loss="squared_error", max_depth=p.max_depth, learning_rate=p.learning_rate, max_iter=p.n_estimators,
                min_samples_leaf=p.min_child_samples, l2_regularization=p.reg_lambda, random_state=p.seed)
        raise ValueError(f"unknown regression backend {p.backend!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> "BoostedRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self._model = self._build()
        self._model.fit(X, y)
        self.feature_names = tuple(feature_names) if feature_names is not None else tuple(f"f{i}" for i in range(X.shape[1]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fitted")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=float)

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        imp = getattr(self._model, "feature_importances_", None)
        if imp is None:
            return {}
        return {n: float(v) for n, v in zip(self.feature_names, imp)}


__all__ = ["BoostedRegressor", "RegressionParams"]
