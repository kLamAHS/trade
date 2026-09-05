"""Secondary direction model: L2 logistic regression on the same features (spec section 21).

Feature scaling statistics are fitted on the training rows only (leakage test:
scaling must never include validation observations).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DirectionModel:
    def __init__(self, C: float = 0.5, max_iter: int = 1000, seed: int = 42):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self._pipe: Pipeline | None = None
        self.constant_class: int | None = None

    def fit(self, X: np.ndarray, y_raw: np.ndarray) -> "DirectionModel":
        X = np.asarray(X, dtype=float)
        labels = (np.asarray(y_raw, dtype=float) > 0).astype(int)
        if labels.min() == labels.max():
            self.constant_class = int(labels[0])
            self._pipe = None
            return self
        self.constant_class = None
        self._pipe = Pipeline([
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)),
        ])
        self._pipe.fit(X, labels)
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._pipe is None:
            if self.constant_class is None:
                raise RuntimeError("model not fitted")
            return np.full(len(X), float(self.constant_class))
        return self._pipe.predict_proba(X)[:, 1]

    @property
    def scaler_mean(self) -> np.ndarray | None:
        if self._pipe is None:
            return None
        return self._pipe.named_steps["scale"].mean_

    @property
    def scaler_scale(self) -> np.ndarray | None:
        if self._pipe is None:
            return None
        return self._pipe.named_steps["scale"].scale_


__all__ = ["DirectionModel"]
