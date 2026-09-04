"""Monotonic forecast calibration g(A) -> E[Y~ | A] (spec section 23).

Fitted on out-of-sample validation predictions only.  ``A = 0`` (the two models
disagree) always maps to ``E = 0`` so that no risk is taken on disagreement,
consistent with section 22.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self, method: str = "isotonic", bins: int = 20):
        if method not in ("isotonic", "binned"):
            raise ValueError("calibration method must be 'isotonic' or 'binned'")
        self.method = method
        self.bins = int(bins)
        self._iso: IsotonicRegression | None = None
        self._edges: np.ndarray | None = None
        self._levels: np.ndarray | None = None
        self.n_fit = 0

    def fit(self, A: np.ndarray, y: np.ndarray) -> "Calibrator":
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(A) & np.isfinite(y)
        A, y = A[m], y[m]
        self.n_fit = int(len(A))
        if len(A) < 10 or np.ptp(A) == 0:
            # Degenerate: fall back to identity scaled by the slope of a least-squares fit through the origin.
            denom = float(np.sum(A * A)) if len(A) else 0.0
            slope = float(np.sum(A * y) / denom) if denom > 0 else 0.0
            self._iso = None
            self._edges = np.array([-np.inf, np.inf])
            self._levels = np.array([0.0])
            self._identity_slope = slope
            return self
        self._identity_slope = None
        if self.method == "isotonic":
            self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self._iso.fit(A, y)
        else:
            qs = np.linspace(0, 1, self.bins + 1)
            edges = np.unique(np.quantile(A, qs))
            idx = np.clip(np.searchsorted(edges, A, side="right") - 1, 0, len(edges) - 2)
            levels = np.array([y[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])
            # forward-fill NaNs then enforce monotonicity via cumulative max
            for i in range(len(levels)):
                if np.isnan(levels[i]):
                    levels[i] = levels[i - 1] if i > 0 else 0.0
            levels = np.maximum.accumulate(levels)
            self._edges, self._levels = edges, levels
        return self

    def predict(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        out = np.zeros(len(A))
        finite = np.isfinite(A)
        nonzero = finite & (A != 0)
        if not nonzero.any():
            return out
        if getattr(self, "_identity_slope", None) is not None:
            out[nonzero] = self._identity_slope * A[nonzero]
            return out
        if self.method == "isotonic" and self._iso is not None:
            out[nonzero] = self._iso.predict(A[nonzero])
        elif self._edges is not None:
            idx = np.clip(np.searchsorted(self._edges, A[nonzero], side="right") - 1, 0, len(self._edges) - 2)
            out[nonzero] = self._levels[idx]
        return out

    def curve(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        if self.method == "isotonic" and self._iso is not None:
            xs = np.linspace(self._iso.X_min_, self._iso.X_max_, n)
            return xs, self._iso.predict(xs)
        if self._edges is not None and len(self._edges) > 2:
            xs = 0.5 * (self._edges[:-1] + self._edges[1:])
            return xs, self._levels
        return np.array([]), np.array([])


__all__ = ["Calibrator"]
