"""Causal Kalman latent-price features (market-state spec sections 25-27).

Local linear trend model on the log price:

    level_t    = level_{t-1} + velocity_{t-1} + w_l,   w_l ~ N(0, q_l)
    velocity_t = velocity_{t-1} + w_v,                 w_v ~ N(0, q_v)
    y_t        = level_t + e_t,                        e_t ~ N(0, r)

Only the *filter* is used for features (FILTERED = causal).  ``smooth`` (RTS) exists for
diagnostics and is never called by the feature engine.  The noise parameters are fitted on the
training block by maximising the innovation likelihood on a small grid and frozen until the next
retrain; the latent state itself updates every bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rolling import lag


@dataclass
class KalmanParams:
    q_level: float
    q_velocity: float
    r: float
    log_likelihood: float = float("nan")
    n_fit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"q_level": self.q_level, "q_velocity": self.q_velocity, "r": self.r,
                "log_likelihood": self.log_likelihood, "n_fit": self.n_fit}


def kalman_filter(y: np.ndarray, params: KalmanParams, x0: np.ndarray | None = None,
                  P0: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Forward filter.  Row t uses y_0..y_t only.  Returns level, velocity, innovation, innovation
    variance, level variance and the one-step prediction."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([params.q_level, params.q_velocity])
    H = np.array([1.0, 0.0])
    x = np.array([y[0] if np.isfinite(y[0]) else 0.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float).copy()
    P = np.diag([params.r * 10.0, params.r]) if P0 is None else np.asarray(P0, dtype=float).copy()
    level = np.full(n, np.nan)
    velocity = np.full(n, np.nan)
    innovation = np.full(n, np.nan)
    innovation_var = np.full(n, np.nan)
    level_var = np.full(n, np.nan)
    predicted = np.full(n, np.nan)
    ll = 0.0
    for t in range(n):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        yp = H @ x
        S = H @ P @ H + params.r
        predicted[t] = yp
        if np.isfinite(y[t]):
            v = y[t] - yp
            Kg = (P @ H) / S
            x = x + Kg * v
            P = P - np.outer(Kg, H @ P)
            innovation[t] = v
            innovation_var[t] = S
            if t > 0:
                ll += -0.5 * (math.log(2 * math.pi * S) + v * v / S)
        level[t], velocity[t] = x[0], x[1]
        level_var[t] = P[0, 0]
    return {"level": level, "velocity": velocity, "innovation": innovation, "innovation_var": innovation_var,
            "level_var": level_var, "predicted": predicted, "log_likelihood": ll}


def fit_kalman(y: np.ndarray, grid_q_velocity=(1e-4, 1e-3, 1e-2, 1e-1), grid_r=(0.25, 0.5, 1.0, 2.0, 4.0)) -> KalmanParams:
    """Maximum innovation likelihood on a grid.  Scales are relative to the sample return variance so
    the grid is meaningful for any price level; ``q_level`` is held at that variance."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 50:
        raise ValueError("too few observations to fit the Kalman parameters")
    base = float(np.nanvar(np.diff(y[ok])))
    base = base if base > 0 else 1e-8
    best = None
    for qv in grid_q_velocity:
        for r in grid_r:
            p = KalmanParams(base, base * qv, base * r)
            ll = kalman_filter(y, p)["log_likelihood"]
            if best is None or ll > best.log_likelihood:
                best = KalmanParams(float(p.q_level), float(p.q_velocity), float(p.r), float(ll), int(ok.sum()))
    return best


def kalman_feature_arrays(log_close: np.ndarray, sigma: np.ndarray, params: KalmanParams, eps: float = 1e-12) -> dict[str, np.ndarray]:
    f = kalman_filter(log_close, params)
    s = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    vel = f["velocity"]
    return {
        "kalman_price": (f["level"] - log_close) / (s + eps),               # latent level minus observed, in sigma units
        "kalman_velocity": vel / (s + eps),
        "kalman_acceleration": (vel - lag(vel, 1)) / (s + eps),
        "kalman_innovation": f["innovation"] / (s + eps),
        "kalman_innovation_z": f["innovation"] / np.sqrt(np.maximum(f["innovation_var"], eps)),
        "kalman_state_uncertainty": np.sqrt(np.maximum(f["level_var"], 0.0)) / (s + eps),
        "kalman_observed_minus_filtered": (log_close - f["level"]) / (s + eps),
    }


KALMAN_SUBSETS = {"latent_price": ("kalman_price", "kalman_observed_minus_filtered"),
                  "innovation": ("kalman_innovation", "kalman_innovation_z"),
                  "full": ("kalman_price", "kalman_velocity", "kalman_acceleration", "kalman_innovation", "kalman_innovation_z",
                           "kalman_state_uncertainty", "kalman_observed_minus_filtered")}


def rts_smooth(log_close: np.ndarray, params: KalmanParams) -> np.ndarray:  # pragma: no cover - diagnostics only
    """Rauch-Tung-Striebel smoother: DIAGNOSTICS ONLY (uses the whole series)."""
    y = np.asarray(log_close, dtype=float)
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([params.q_level, params.q_velocity])
    H = np.array([1.0, 0.0])
    xs, Ps, xp, Pp = [], [], [], []
    x = np.array([y[0], 0.0])
    P = np.diag([params.r * 10.0, params.r])
    for t in range(n):
        if t > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xp.append(x.copy()); Pp.append(P.copy())
        S = H @ P @ H + params.r
        Kg = (P @ H) / S
        x = x + Kg * (y[t] - H @ x)
        P = P - np.outer(Kg, H @ P)
        xs.append(x.copy()); Ps.append(P.copy())
    out = np.empty(n)
    xn = xs[-1]
    out[-1] = xn[0]
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.pinv(Pp[t + 1])
        xn = xs[t] + C @ (xn - xp[t + 1])
        out[t] = xn[0]
    return out


__all__ = ["KalmanParams", "kalman_filter", "fit_kalman", "kalman_feature_arrays", "rts_smooth", "KALMAN_SUBSETS"]
