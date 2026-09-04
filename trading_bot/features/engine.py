"""FeatureEngine: turns validated bar history into immutable FeatureVectors.

Two entry points share one implementation so that training (batch) and live
(streaming) features are identical:

* ``compute_matrix``  -- every bar of a history window (training dataset builder)
* ``compute_latest``  -- the newest bar only (main loop), computed on the trailing
  ``required_history`` bars, which is enough for every lag/kernel/window.

All computations are causal.  A FeatureVector's ``latest_source_timestamp`` is
the close of the newest bar used; the record type itself rejects any look-ahead
(spec section 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ..data.calendar import SessionCalendar
from ..data.store import BarStore
from ..fractional.engine import FractionalEngine
from ..types import FeatureVector
from .price import conventional_return_features, fractional_price_features
from .regime import regime_features
from .rolling import rolling_std
from .schema import FeatureSchema, build_schema
from .volatility import volatility_features
from .volume import volume_features


@dataclass(frozen=True)
class FeatureMatrix:
    names: tuple[str, ...]
    values: np.ndarray                   # shape (n_bars, len(names))
    close_times: tuple[datetime, ...]    # bar close times
    adaptive_d: float
    kernel_size: int
    source_times: tuple[datetime, ...] = ()   # newest information time per bar (close, or a later quote)

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def valid_mask(self, names) -> np.ndarray:
        idx = [self.names.index(n) for n in names]
        return np.isfinite(self.values[:, idx]).all(axis=1)

    def row(self, i: int) -> dict[str, float]:
        return {n: float(self.values[i, j]) for j, n in enumerate(self.names)}


class FeatureEngine:
    def __init__(self, cfg, fractional_engine: FractionalEngine, calendar: SessionCalendar,
                 adaptive_d: Optional[float] = None):
        self.cfg = cfg
        self.fe = fractional_engine
        self.calendar = calendar
        f = cfg.features
        self.z_window = int(f.robust_z_window)
        self.volume_z_window = int(f.volume_z_window)
        self.ewma_lambda = float(f.ewma_lambda)
        self.ewma_window = int(f.ewma_window)
        self.eps = float(f.epsilon)
        self.vol_windows = tuple(int(w) for w in f.vol_windows)
        self.return_lags = tuple(int(k) for k in f.return_lags)
        self.slope_lags = tuple(int(k) for k in f.slope_lags)
        self.trend_window = int(f.trend_window)
        self.vol_regime_short = int(f.vol_regime_short)
        self.vol_regime_long = int(f.vol_regime_long)
        self.frac_floor = float(f.frac_regime_floor)
        self.fixed_orders = tuple(float(d) for d in cfg.fractional.fixed_orders)
        self.volatility_order = float(cfg.fractional.volatility_order)
        self.sigma_window = int(cfg.prediction.volatility_window)
        self.volume_enabled = bool(cfg.data.volume_enabled)
        self.schema: FeatureSchema = build_schema(self.volume_enabled, bool(f.use_raw_fractional_levels),
                                                 self.slope_lags, self.return_lags, self.vol_windows)
        self._adaptive_d = float(adaptive_d) if adaptive_d is not None else float(self.fixed_orders[1])

    # ---------------------------------------------------------------- state
    @property
    def adaptive_d(self) -> float:
        return self._adaptive_d

    def set_adaptive_d(self, d: float) -> None:
        self._adaptive_d = float(d)

    @property
    def max_kernel(self) -> int:
        ds = (*self.fixed_orders, self.volatility_order, self._adaptive_d)
        return max(self.fe.kernel_size(d) for d in ds)

    @property
    def required_history(self) -> int:
        """Bars needed before every feature is finite."""
        k_price = self.fe.kernel_size(self._adaptive_d)
        k_fixed = max(self.fe.kernel_size(d) for d in self.fixed_orders)
        k_vol = self.fe.kernel_size(self.volatility_order)
        need_price = max(k_price, k_fixed) + self.z_window + max(self.slope_lags + (2,))
        need_vol = self.ewma_window + k_vol + 1              # returns start at bar 1
        need_windows = max(self.vol_windows + (self.trend_window, self.vol_regime_long, self.z_window)) + 1
        return max(need_price, need_vol, need_windows) + 1

    def ready(self, store: BarStore) -> bool:
        return len(store) >= self.required_history

    # -------------------------------------------------------------- compute
    def compute_matrix(self, store: BarStore, start: int = 0, stop: int | None = None) -> FeatureMatrix:
        arrays = store.arrays(start, stop)
        close = arrays["close"]
        n = len(close)
        log_close = np.log(close)
        cols: dict[str, np.ndarray] = {}

        vol = volatility_features(log_close, arrays["high"], arrays["low"], close, self.fe, self.vol_windows,
                                  self.ewma_lambda, self.ewma_window, self.volatility_order, self.z_window, self.eps)
        vol.pop("_sigma")
        returns = vol.pop("_returns")
        # sigma_{t,50} for labels / edge is computed on its own window so it never depends on vol_windows.
        sigma_h = rolling_std(returns, self.sigma_window)
        cols.update(vol)

        cols.update(fractional_price_features(log_close, self.fe, self._adaptive_d, self.fixed_orders, self.z_window,
                                              self.eps, self.slope_lags))
        cols.update(conventional_return_features(log_close, sigma_h, self.return_lags, self.eps))
        if self.volume_enabled:
            cols.update(volume_features(arrays["volume"], self.volume_z_window, self.eps))
        cols.update(regime_features(log_close, returns, cols["fd_025_z"], cols["fd_075_z"], self.trend_window,
                                    self.vol_regime_short, self.vol_regime_long, self.frac_floor, self.eps))

        # Section 17: cyclic time-of-day, M = total minutes of the bar's own session (early closes included).
        minutes = np.array([self.calendar.minutes_since_open(ts) for ts in arrays["timestamp"]], dtype=float)
        session_len = np.array([self.calendar.session_minutes_for(self.calendar.session_date(ts))
                                for ts in arrays["timestamp"]], dtype=float)
        phase = 2.0 * math.pi * minutes / session_len
        cols["time_sin"] = np.sin(phase)
        cols["time_cos"] = np.cos(phase)

        # Auxiliary market state for the signal/risk engines.
        cols["sigma_h"] = sigma_h
        cols["close"] = close
        cols["log_close"] = log_close
        bid, ask = arrays["bid"], arrays["ask"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mid = 0.5 * (bid + ask)
            spread = np.where(np.isfinite(mid) & (mid > 0), (ask - bid) / mid, np.nan)
        cols["spread_rel"] = spread

        names = self.schema.all_names
        matrix = np.column_stack([cols[nm] for nm in names]) if n else np.empty((0, len(names)))
        close_times = tuple(ts + _minutes(store.bar_minutes) for ts in arrays["timestamp"])
        source_times = tuple(b.latest_source_time for b in store.bars[start:stop])
        return FeatureMatrix(names, matrix, close_times, self._adaptive_d, self.fe.kernel_size(self._adaptive_d),
                             source_times)

    def compute_latest(self, store: BarStore) -> FeatureVector:
        if not self.ready(store):
            raise RuntimeError(f"feature engine not ready: {len(store)} < {self.required_history} bars")
        n = len(store)
        start = max(0, n - self.required_history)
        fm = self.compute_matrix(store, start, n)
        return self.vector_from_matrix(fm, len(fm.close_times) - 1, store.instrument, bar_index=n - 1)

    def vector_from_matrix(self, fm: FeatureMatrix, i: int, instrument: str, bar_index: int) -> FeatureVector:
        """Feature timestamp = the bar close, or later if a source (e.g. a live quote) arrived after it.
        ``latest_source_timestamp`` is the newest information time among all bars used, so the
        section-3 guard is checked against real source times."""
        close = fm.close_times[i]
        latest_source = max(fm.source_times[: i + 1]) if fm.source_times else close
        feature_ts = max(close, latest_source)
        return FeatureVector(instrument=instrument, timestamp=feature_ts, latest_source_timestamp=latest_source,
                             bar_index=bar_index, fractional_d=fm.adaptive_d, fractional_kernel_size=fm.kernel_size,
                             values=fm.row(i), bar_close_time=close)


def _minutes(m: int):
    from datetime import timedelta
    return timedelta(minutes=m)


__all__ = ["FeatureEngine", "FeatureMatrix"]
