"""Causal Gaussian hidden Markov model for market regimes (market-state spec sections 8-12).

Fitting uses the full training block (EM with forward-backward: in-sample, allowed).  *Features*
use the forward filter only:

    P(S_t | X_0 .. X_t)          filtered  -- causal, the only thing exposed to the models
    P(S_t | X_0 .. X_T), T > t   smoothed  -- diagnostics only (``smooth``); never a feature

States are re-indexed by their volatility rank after every fit so that ``regime_p_0`` is always
the calmest state and the highest rank the stress state, whatever order EM converged to (state-index
switching, section 12).  Semantic metadata per state (volatility, mean return, spread, volume,
persistence, occupancy) is stored with the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .rolling import lag, robust_zscore, rolling_corr_with_index, rolling_std

HMM_OBS_NAMES = ("return", "abs_return", "log_vol", "volume_intensity", "spread", "trend")


def hmm_observations(log_close: np.ndarray, volume: np.ndarray | None, spread_rel: np.ndarray | None,
                     vol_window: int = 20, trend_window: int = 50, eps: float = 1e-12) -> np.ndarray:
    """Observation matrix (T, 6) built from bars only; NaN during warm-up.  Every column is causal."""
    n = len(log_close)
    r = np.full(n, np.nan)
    r[1:] = np.diff(log_close)
    vol = rolling_std(r, vol_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.log(vol + eps)
    obs = np.column_stack([
        r, np.abs(r), log_vol,
        (np.log1p(np.where(np.isfinite(volume), volume, np.nan)) if volume is not None else np.zeros(n)),
        (np.where(np.isfinite(spread_rel), spread_rel, np.nan) if spread_rel is not None else np.zeros(n)),
        rolling_corr_with_index(log_close, trend_window),
    ])
    return obs


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class GaussianHMM:
    n_states: int = 3
    n_iter: int = 30
    tol: float = 1e-4
    seed: int = 0
    min_var: float = 1e-6
    means: Optional[np.ndarray] = None       # (K, D) in standardised observation space
    variances: Optional[np.ndarray] = None   # (K, D)
    transition: Optional[np.ndarray] = None  # (K, K)
    initial: Optional[np.ndarray] = None     # (K,)
    obs_mean: Optional[np.ndarray] = None    # standardisation fitted on the training block
    obs_std: Optional[np.ndarray] = None
    state_meta: list[dict[str, Any]] = field(default_factory=list)
    log_likelihood: float = float("nan")
    n_fit: int = 0
    converged: bool = False

    # ----------------------------------------------------------- helpers
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.obs_mean) / self.obs_std

    def _log_emission(self, Z: np.ndarray) -> np.ndarray:
        """log N(z | mu_k, diag(var_k)) per state; rows with any NaN get a flat likelihood (no information)."""
        T, D = Z.shape
        K = self.n_states
        out = np.zeros((T, K))
        ok = np.isfinite(Z).all(axis=1)
        if ok.any():
            z = Z[ok]
            for k in range(K):
                var = self.variances[k]
                ll = -0.5 * (np.sum(np.log(2 * math.pi * var)) + np.sum((z - self.means[k]) ** 2 / var, axis=1))
                out[ok, k] = ll
        return out

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        ok = np.isfinite(X).all(axis=1)
        if ok.sum() < 10 * self.n_states:
            raise ValueError("too few finite observations to fit the regime model")
        self.obs_mean = X[ok].mean(axis=0)
        self.obs_std = X[ok].std(axis=0)
        self.obs_std = np.where(self.obs_std > 1e-12, self.obs_std, 1.0)
        Z = self._standardise(X)
        K, D = self.n_states, X.shape[1]
        # deterministic initialisation: split by the volatility observation (column 2) quantiles
        z_ok = Z[ok]
        order = np.argsort(z_ok[:, 2])
        chunks = np.array_split(order, K)
        self.means = np.array([z_ok[c].mean(axis=0) for c in chunks])
        self.variances = np.array([np.maximum(z_ok[c].var(axis=0), self.min_var) for c in chunks])
        self.transition = np.full((K, K), 0.05 / max(1, K - 1))
        np.fill_diagonal(self.transition, 0.95)
        self.initial = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        self.converged = False
        for it in range(self.n_iter):
            logB = self._log_emission(Z)
            alpha, beta, ll, B = self._forward_backward_scaled(logB)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
            # xi summed over time (scaled quantities: alpha_t A B_{t+1} beta_{t+1} sums to 1 per t)
            xi = alpha[:-1, :, None] * self.transition[None, :, :] * (B[1:] * beta[1:])[:, None, :]
            xi /= np.maximum(xi.sum(axis=(1, 2), keepdims=True), 1e-300)
            xi_sum = xi.sum(axis=0)
            # M-step
            self.initial = gamma[0] / gamma[0].sum()
            self.transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
            self.transition = np.maximum(self.transition, 1e-6)
            self.transition /= self.transition.sum(axis=1, keepdims=True)
            g = gamma[ok]
            w = g.sum(axis=0)[:, None] + 1e-12
            self.means = (g.T @ z_ok) / w
            self.variances = np.maximum((g.T @ (z_ok ** 2)) / w - self.means ** 2, self.min_var)
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(ll)):
                self.converged = True
                self.log_likelihood = float(ll)
                break
            prev_ll = ll
            self.log_likelihood = float(ll)
        self.n_fit = int(ok.sum())
        self._relabel_by_volatility()
        self._build_meta(X, ok)
        return self

    def _forward_backward_scaled(self, logB: np.ndarray):
        """Scaled forward / backward passes (Rabiner): O(T K^2) with tiny per-step numpy work."""
        T, K = logB.shape
        B = np.exp(logB - logB.max(axis=1, keepdims=True))      # per-row rescaled emissions (scale cancels)
        A = self.transition
        alpha = np.empty((T, K))
        scales = np.empty(T)
        a = self.initial * B[0]
        s = a.sum()
        alpha[0] = a / s if s > 0 else np.full(K, 1.0 / K)
        scales[0] = s if s > 0 else 1.0
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            s = a.sum()
            if s <= 0:
                a = np.full(K, 1.0 / K)
                s = 1.0
            alpha[t] = a / s
            scales[t] = s
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            b = A @ (B[t + 1] * beta[t + 1])
            beta[t] = b / scales[t + 1]
        ll = float(np.sum(np.log(scales)) + np.sum(logB.max(axis=1)))
        return alpha, beta, ll, B

    def _forward_backward(self, logB: np.ndarray):
        """Log-space quantities for ``smooth`` (diagnostics)."""
        alpha, beta, ll, _ = self._forward_backward_scaled(logB)
        return np.log(alpha + 1e-300), np.log(beta + 1e-300), ll

    def _relabel_by_volatility(self) -> None:
        """Permute states so that index 0 is the calmest (lowest |return| / log-vol) and K-1 the most volatile."""
        score = self.means[:, 2] + self.means[:, 1]
        order = np.argsort(score)
        self.means = self.means[order]
        self.variances = self.variances[order]
        self.transition = self.transition[np.ix_(order, order)]
        self.initial = self.initial[order]

    def _build_meta(self, X: np.ndarray, ok: np.ndarray) -> None:
        probs = self.filter(X)
        most = np.argmax(probs, axis=1)
        raw_mean = self.means * self.obs_std + self.obs_mean
        self.state_meta = []
        for k in range(self.n_states):
            occ = float(np.mean(most[ok] == k)) if ok.any() else 0.0
            self.state_meta.append({
                "state": k, "name": ("calm" if k == 0 else "stress" if k == self.n_states - 1 else f"mid_{k}"),
                "volatility": float(math.exp(raw_mean[k, 2])), "mean_return": float(raw_mean[k, 0]),
                "abs_return": float(raw_mean[k, 1]), "spread": float(raw_mean[k, 4]), "volume": float(raw_mean[k, 3]),
                "trend": float(raw_mean[k, 5]), "persistence": float(self.transition[k, k]),
                "expected_duration_bars": float(1.0 / max(1e-9, 1.0 - self.transition[k, k])), "occupancy": occ})

    # ------------------------------------------------------------- filter
    def filter(self, X: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
        """Filtered state probabilities P(S_t | X_0..X_t): a forward pass only.  Row t depends on
        rows <= t exclusively (prefix invariance is unit-tested)."""
        Z = self._standardise(np.asarray(X, dtype=float))
        logB = self._log_emission(Z)
        T, K = logB.shape
        out = np.empty((T, K))
        p = (self.initial if initial is None else np.asarray(initial, dtype=float)).copy()
        for t in range(T):
            if t > 0:
                p = p @ self.transition
            like = np.exp(logB[t] - logB[t].max())
            p = p * like
            s = p.sum()
            p = p / s if s > 0 else np.full(K, 1.0 / K)
            out[t] = p
        return out

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """Smoothed probabilities P(S_t | X_0..X_T).  DIAGNOSTICS ONLY: uses the future."""
        Z = self._standardise(np.asarray(X, dtype=float))
        alpha, beta, _, _ = self._forward_backward_scaled(self._log_emission(Z))
        g = alpha * beta
        return g / np.maximum(g.sum(axis=1, keepdims=True), 1e-300)

    @property
    def stress_state(self) -> int:
        return self.n_states - 1

    def to_dict(self) -> dict[str, Any]:
        return {"n_states": self.n_states, "means": self.means.tolist(), "variances": self.variances.tolist(),
                "transition": self.transition.tolist(), "initial": self.initial.tolist(),
                "obs_mean": self.obs_mean.tolist(), "obs_std": self.obs_std.tolist(), "state_meta": self.state_meta,
                "log_likelihood": self.log_likelihood, "n_fit": self.n_fit, "converged": self.converged,
                "obs_names": list(HMM_OBS_NAMES)}


def regime_feature_arrays(probs: np.ndarray, transition: np.ndarray, prefix: str = "regime") -> dict[str, np.ndarray]:
    """Feature contract (section 10) from filtered probabilities: per-state probabilities, entropy,
    most-likely state (volatility rank), age of the current most-likely state, probability of leaving it."""
    T, K = probs.shape
    out: dict[str, np.ndarray] = {}
    for k in range(K):
        out[f"{prefix}_p_{k}"] = probs[:, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1) / math.log(K) if K > 1 else np.zeros(T)
    out[f"{prefix}_entropy"] = ent
    most = np.argmax(probs, axis=1)
    out[f"{prefix}_most_likely"] = most.astype(float)
    age = np.zeros(T)
    for t in range(1, T):
        age[t] = age[t - 1] + 1 if most[t] == most[t - 1] else 0.0
    out[f"{prefix}_age"] = age
    out[f"{prefix}_transition_prob"] = 1.0 - transition[most, most]
    out[f"{prefix}_stress_p"] = probs[:, K - 1]
    return out


__all__ = ["GaussianHMM", "hmm_observations", "regime_feature_arrays", "HMM_OBS_NAMES"]
