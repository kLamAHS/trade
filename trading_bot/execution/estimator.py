"""Execution intelligence (market-state spec sections 16-20).

``ExecutionModel`` predicts the *expected post-fill adverse move beyond what the alpha forecast
already expects*, at bar resolution (h = one bar):

    target_t = |ER_t| / H  -  direction_t * r_fill_t,      r_fill_t = (C_{t+1} - O_{t+1}) / O_{t+1}

i.e. how much the first bar after the fill falls short of the forecast's per-bar share.  A well
calibrated model has zero mean; the *conditional* structure (spread, range, volatility, volume,
time of day, regime, quote imbalance, OFI, forecast size and confidence) is what the ridge model
learns.  It is fitted on out-of-fold forecasts only (the same discipline as the calibrator) so it
never sees in-sample optimism.  Its residual standard deviation is the model part of the
uncertainty buffer.  ``ExecutionEstimator`` combines it with the spread / slippage / fee model
into an :class:`ExecutionEstimate` per decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..types import ExecutionEstimate
from .cost_model import CostModel

EXEC_INPUT_NAMES = ("spread_rel", "range_rel", "sigma_h", "volume_z", "time_sin", "time_cos", "regime_stress_p",
                    "quote_imbalance", "ofi_zscore", "volatility_state")


def exec_inputs_from_columns(get, n: int, default_spread: float) -> np.ndarray:
    """Build the (n, len(EXEC_INPUT_NAMES)) execution-input matrix from a column accessor
    (``get(name) -> array or None``).  Unavailable families contribute zeros (never an estimate)."""
    cols = []
    for name in EXEC_INPUT_NAMES:
        c = get(name)
        if c is None:
            c = np.zeros(n)
        c = np.asarray(c, dtype=float)
        if name == "spread_rel":
            c = np.where(np.isfinite(c) & (c >= 0), c, default_spread)
        c = np.where(np.isfinite(c), c, 0.0)
        cols.append(c)
    return np.column_stack(cols) if n else np.empty((0, len(EXEC_INPUT_NAMES)))


def _design(X_exec: np.ndarray, abs_er: np.ndarray, confidence: np.ndarray, direction: np.ndarray) -> np.ndarray:
    qi = X_exec[:, EXEC_INPUT_NAMES.index("quote_imbalance")]
    ofi = X_exec[:, EXEC_INPUT_NAMES.index("ofi_zscore")]
    return np.column_stack([X_exec, abs_er, confidence, direction * qi, direction * ofi])


DESIGN_NAMES = EXEC_INPUT_NAMES + ("abs_er", "confidence", "dir_x_quote_imbalance", "dir_x_ofi")


@dataclass
class ExecutionModel:
    alpha: float = 1.0
    min_rows: int = 50
    clip_sigma: float = 4.0
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    residual_std: float = 0.0
    target_mean: float = 0.0
    n_fit: int = 0
    r2: float = float("nan")

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    @staticmethod
    def target(abs_er: np.ndarray, direction: np.ndarray, r_fill: np.ndarray, horizon: int) -> np.ndarray:
        return np.asarray(abs_er, dtype=float) / max(1, horizon) - np.asarray(direction, dtype=float) * np.asarray(r_fill, dtype=float)

    def fit(self, X_exec: np.ndarray, abs_er: np.ndarray, confidence: np.ndarray, direction: np.ndarray,
            r_fill: np.ndarray, horizon: int) -> "ExecutionModel":
        D = _design(X_exec, abs_er, confidence, direction)
        y = self.target(abs_er, direction, r_fill, horizon)
        ok = np.isfinite(D).all(axis=1) & np.isfinite(y) & (np.asarray(direction) != 0)
        if ok.sum() < self.min_rows:
            self.coef = None
            self.n_fit = int(ok.sum())
            return self
        D, y = D[ok], y[ok]
        # winsorise the target: a single jump must not define the execution model
        s = y.std() if y.std() > 0 else 1.0
        y = np.clip(y, y.mean() - self.clip_sigma * s, y.mean() + self.clip_sigma * s)
        self.mean = D.mean(axis=0)
        self.scale = np.where(D.std(axis=0) > 1e-12, D.std(axis=0), 1.0)
        Z = (D - self.mean) / self.scale
        self.target_mean = float(y.mean())
        yc = y - self.target_mean
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ yc)
        self.intercept = self.target_mean
        pred = Z @ self.coef + self.intercept
        resid = y - pred
        self.residual_std = float(resid.std())
        tss = float(((y - y.mean()) ** 2).sum())
        self.r2 = float(1.0 - (resid ** 2).sum() / tss) if tss > 0 else float("nan")
        self.n_fit = int(ok.sum())
        return self

    def predict(self, X_exec: np.ndarray, abs_er: np.ndarray, confidence: np.ndarray, direction: np.ndarray) -> np.ndarray:
        n = len(abs_er)
        if not self.fitted:
            return np.zeros(n)
        D = _design(X_exec, abs_er, confidence, direction)
        D = np.where(np.isfinite(D), D, self.mean)
        return ((D - self.mean) / self.scale) @ self.coef + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {"fitted": self.fitted, "n_fit": self.n_fit, "residual_std": self.residual_std, "r2": self.r2,
                "target_mean": self.target_mean, "alpha": self.alpha,
                "coefficients": (dict(zip(DESIGN_NAMES, [float(c) for c in self.coef])) if self.fitted else None)}


@dataclass
class ExecutionEstimator:
    """spread / slippage (cost model) + adverse selection (execution model) + fees + uncertainty."""

    cost_model: CostModel
    model: Optional[ExecutionModel] = None
    horizon: int = 4
    uncertainty_buffer: float = 3e-4
    uncertainty_z: float = 0.0            # multiplier on the execution model's residual std
    fee_roundtrip: float = 0.0
    use_model: bool = True
    adverse_floor: float = 0.0            # fraction: predictions below this are clipped (0 = the model can only add cost)

    @classmethod
    def from_config(cls, cfg, cost_model: CostModel, model: ExecutionModel | None = None) -> "ExecutionEstimator":
        e = cfg.execution
        return cls(cost_model, model, int(cfg.prediction.horizon_bars), float(e.get("uncertainty_buffer_bps", 3.0)) / 1e4,
                   float(e.get("uncertainty_z", 0.0)), 2.0 * float(e.get("commission_per_side", 0.0)) + float(e.get("fee_bps", 0.0)) / 1e4,
                   bool(e.get("adverse_selection_model", True)), float(e.get("adverse_selection_floor_bps", 0.0)) / 1e4)

    def adverse_arrays(self, X_exec: np.ndarray, abs_er: np.ndarray, confidence: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(adverse selection, model uncertainty) per row, fractions."""
        n = len(abs_er)
        if not (self.use_model and self.model is not None and self.model.fitted):
            return np.zeros(n), np.zeros(n)
        adv = np.maximum(self.model.predict(X_exec, abs_er, confidence, direction), self.adverse_floor)
        unc = np.full(n, self.uncertainty_z * self.model.residual_std)
        return adv, unc

    def estimate(self, range_rel: float, spread_rel: float | None, x_exec: np.ndarray, expected_return: float,
                 confidence: float) -> ExecutionEstimate:
        cost = self.cost_model.estimate(range_rel, spread_rel)
        direction = 1 if expected_return > 0 else (-1 if expected_return < 0 else 0)
        adv, unc = self.adverse_arrays(x_exec.reshape(1, -1), np.array([abs(expected_return)]), np.array([confidence]),
                                       np.array([direction], dtype=float))
        model_unc = float(unc[0]) if self.model is not None and self.model.fitted else 0.0
        return ExecutionEstimate(spread_bps=cost.spread * 1e4, slippage_bps=cost.slippage * 1e4,
                                 adverse_selection_bps=float(adv[0]) * 1e4, fee_bps=(cost.commission + self.fee_roundtrip - 2 * self.cost_model.commission_per_side) * 1e4,
                                 uncertainty_bps=(self.uncertainty_buffer + model_unc) * 1e4,
                                 gross_bps=expected_return * 1e4 if math.isfinite(expected_return) else math.nan,
                                 model_uncertainty_bps=model_unc * 1e4,
                                 adverse_selection_source="model" if (self.model is not None and self.model.fitted and self.use_model) else "none")


__all__ = ["ExecutionModel", "ExecutionEstimator", "EXEC_INPUT_NAMES", "DESIGN_NAMES", "exec_inputs_from_columns"]
