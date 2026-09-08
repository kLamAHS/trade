"""Pooled training rows from secondary instruments (the training panel).

The bot trades **one** instrument.  The panel does not change that: it only enlarges the sample the
models are *fitted* on, by adding rows from other symbols that carry the same features and the same
scale-free label.  Every fold, the outer holdout, the acceptance criteria and the simulated P&L stay
on the primary instrument's own rows, so what is accepted is still a statement about the instrument
that will be traded.

Three properties make pooling legitimate here:

* **The label is already scale free.**  ``Y = (log P_{t+1+H} - log P_{t+1}) / (sigma_{t,50} sqrt(H))``
  is measured in that symbol's own volatility units, so a quiet index and a noisy small cap
  contribute comparable targets.
* **Rows are pooled causally.**  A panel row may only join a fit when the newest bar entering its
  label is not after the newest bar entering the primary training block's label.  Nothing from the
  future of the primary's training cutoff can be fitted on, whatever the panel symbol's calendar.
* **Level-dependent features are aligned.**  Most features are already ratios or rolling robust
  z-scores, but a few (log volatility, the fractional volatility channel) sit at a level that
  identifies the symbol.  Left alone, a tree can split on that level and silently un-pool the data.
  Each panel symbol is therefore mapped onto the primary's training distribution feature by feature,
  using statistics from eligible (past-only) rows.  The primary's own rows are never rescaled, so the
  deployed model still reads the live feature vector exactly as it is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

from .dataset import TrainingDataset

EPS = 1e-12


@dataclass(frozen=True)
class PanelRows:
    """One secondary instrument's contribution, in the primary's feature order."""

    symbol: str
    X: np.ndarray                 # (n_rows, n_features)
    y_norm: np.ndarray
    y_raw: np.ndarray
    label_end: np.ndarray         # POSIX seconds of the newest bar entering each row's label
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.y_norm)


def label_end_times(ds: TrainingDataset, window, horizon: int) -> np.ndarray:
    """POSIX seconds of the newest bar whose price enters each row's label.

    The label of row *i* spans bars ``bar_index[i] + 1 .. bar_index[i] + 1 + H``; a row is only usable
    once that last bar has closed, which is what makes it comparable with the primary's cutoff.
    """
    last = np.asarray(ds.bar_index, dtype=int) + int(horizon) + 1
    n = len(window)
    out = np.empty(len(last), dtype=float)
    for i, idx in enumerate(last):
        bar = window[int(min(idx, n - 1))]
        out[i] = bar.close_time.timestamp()
    return out


def build_panel_rows(symbol: str, ds: TrainingDataset, window, horizon: int, names: tuple[str, ...]) -> Optional[PanelRows]:
    """A secondary instrument's dataset reduced to the primary's model inputs.

    A symbol whose feature set differs from the primary's (a family its data cannot support, or a
    cross-asset context the primary does not share) is skipped rather than padded: a pooled row must
    mean the same thing as a primary row, column for column.
    """
    have = set(ds.feature_names)
    if not set(names) <= have or len(ds) == 0:
        return None
    X = ds.columns(names)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(ds.y_norm) & np.isfinite(ds.y_raw)
    if not ok.any():
        return None
    return PanelRows(symbol, X[ok], ds.y_norm[ok], ds.y_raw[ok],
                     label_end_times(ds, window, horizon)[ok], tuple(names))


class TrainingPanel:
    """Pooled rows from every secondary instrument, filtered and aligned per fit."""

    def __init__(self, rows: list[PanelRows], align: bool = True, max_rows_per_fit: int = 0):
        self.rows = [r for r in rows if r is not None and len(r)]
        self.align = bool(align)
        self.max_rows_per_fit = int(max_rows_per_fit)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(r.symbol for r in self.rows)

    def __len__(self) -> int:
        return sum(len(r) for r in self.rows)

    def eligible(self, cutoff: datetime, primary_X: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Rows whose labels are complete by ``cutoff``, aligned to ``primary_X``'s distribution.

        ``primary_X`` are the primary rows the same fit uses, so the alignment statistics come from
        exactly the data the model is allowed to see.
        """
        t = cutoff.timestamp()
        parts_X, parts_yn, parts_yr = [], [], []
        mu_p, sd_p = _moments(primary_X)
        for r in self.rows:
            m = r.label_end <= t
            if not m.any():
                continue
            X = r.X[m]
            if self.align:
                X = _align(X, mu_p, sd_p)
            parts_X.append(X)
            parts_yn.append(r.y_norm[m])
            parts_yr.append(r.y_raw[m])
        if not parts_X:
            return None
        X = np.concatenate(parts_X)
        yn = np.concatenate(parts_yn)
        yr = np.concatenate(parts_yr)
        if 0 < self.max_rows_per_fit < len(yn):
            keep = np.linspace(0, len(yn) - 1, self.max_rows_per_fit).astype(int)   # deterministic thinning
            X, yn, yr = X[keep], yn[keep], yr[keep]
        return X, yn, yr

    def out_of_sample(self, start: datetime, end: datetime) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Per symbol, the rows whose labels complete inside (start, end]: the panel view of a holdout."""
        lo, hi = start.timestamp(), end.timestamp()
        out = []
        for r in self.rows:
            m = (r.label_end > lo) & (r.label_end <= hi)
            if m.any():
                out.append((r.symbol, r.X[m], r.y_norm[m]))
        return out


def _moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > EPS), sd, 1.0)
    return mu, sd


def _align(X: np.ndarray, mu_p: np.ndarray, sd_p: np.ndarray) -> np.ndarray:
    """Map this symbol's feature distribution onto the primary's: standardise, then rescale.

    A feature that is constant in either sample is left alone; there is nothing to align and the
    rescaling would only inject noise.
    """
    mu_s, sd_s = _moments(X)
    return (X - mu_s) / sd_s * sd_p + mu_p


def panel_forecast_quality(model, panel_rows: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """How well does the fitted model forecast the *other* instruments over the same span?

    One instrument gives one experiment.  The panel asks the harder question: does the same fitted
    model carry directional information across names it will not trade?  A pooled correlation near
    zero with a healthy primary correlation is the signature of a result that will not repeat.
    """
    from .validation import _safe_corr

    def _score(E: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        made = E != 0
        if not made.any():
            # A collapsed calibrator forecasts the same number everywhere.  Reporting a correlation of
            # zero here would read as a measured result; there is no forecast to measure.
            return {"rows": int(len(y)), "correlation": float("nan"), "accuracy": float("nan"),
                    "signal_rows": 0, "forecast_constant": True}
        return {"rows": int(len(y)), "correlation": _safe_corr(E, y),
                "accuracy": float(np.mean(np.sign(E[made]) == np.sign(y[made]))),
                "signal_rows": int(made.sum()), "forecast_constant": False}

    per: dict[str, Any] = {}
    E_all, Y_all = [], []
    for sym, X, y in panel_rows:
        E = model.predict_arrays(X)["E"]
        per[sym] = _score(E, y)
        E_all.append(E)
        Y_all.append(y)
    if not E_all:
        return {"symbols": {}, "rows": 0, "correlation": float("nan"), "accuracy": float("nan"),
                "signal_rows": 0, "forecast_constant": True}
    out = _score(np.concatenate(E_all), np.concatenate(Y_all))
    out["symbols"] = per
    return out


__all__ = ["PanelRows", "TrainingPanel", "build_panel_rows", "label_end_times", "panel_forecast_quality"]
