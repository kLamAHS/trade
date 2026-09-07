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
from .components import FeatureFamilyConfig, FittedComponents
from .cross_asset import cross_asset_feature_arrays
from .hmm import hmm_observations, regime_feature_arrays
from .kalman import kalman_feature_arrays
from .ofi import ofi_feature_arrays
from .price import conventional_return_features, fractional_price_features
from .regime import regime_features
from .rolling import lag, robust_zscore, rolling_std
from .schema import FAMILY_ORDER, FeatureSchema, build_schema
from .toxicity import vpin_feature_arrays
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
    families_available: tuple[str, ...] = ()   # families whose values were actually computable from this data
    components_digest: str = ""

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def valid_mask(self, names) -> np.ndarray:
        idx = [self.names.index(n) for n in names]
        return np.isfinite(self.values[:, idx]).all(axis=1)

    def row(self, i: int) -> dict[str, float]:
        return {n: float(self.values[i, j]) for j, n in enumerate(self.names)}


class FeatureEngine:
    def __init__(self, cfg, fractional_engine: FractionalEngine, calendar: SessionCalendar,
                 adaptive_d: Optional[float] = None, components: Optional[FittedComponents] = None):
        self.cfg = cfg
        self.fe = fractional_engine
        self.calendar = calendar
        f = cfg.features
        self.families = FeatureFamilyConfig.from_config(cfg)
        self.components: Optional[FittedComponents] = components
        rep = cfg.get("market_representation", {}) or {}
        self.event_bars = str(rep.get("bar_type", "time")).lower() != "time"
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
        fam = self.families
        self.schema: FeatureSchema = build_schema(
            self.volume_enabled, bool(f.use_raw_fractional_levels), self.slope_lags, self.return_lags, self.vol_windows,
            fractional=fam.fractional, regime_states=fam.regime_states if fam.regime else 0, ofi=fam.ofi,
            kalman_subset=fam.kalman_subset if fam.kalman else None,
            cross_symbols=fam.cross_asset_symbols if fam.cross_asset else (), toxicity=fam.vpin, event_bars=self.event_bars)
        self._adaptive_d = float(adaptive_d) if adaptive_d is not None else float(self.fixed_orders[1])

    # ------------------------------------------------------------ components
    def set_components(self, components: Optional[FittedComponents]) -> None:
        """Install fitted HMM / Kalman parameters (frozen until the next promotion, like adaptive_d)."""
        self.components = components

    def fit_components(self, store: BarStore) -> FittedComponents:
        return FittedComponents.fit(store, self.families, self.sigma_window, int(self.cfg.seed))

    def available_families(self, store: BarStore, context: Optional[dict[str, BarStore]] = None) -> tuple[str, ...]:
        """Enabled families whose inputs the data actually provides (nothing is ever approximated):
        OFI needs displayed quote sizes (Tier B), CROSS_ASSET every configured context instrument,
        REGIME / KALMAN fitted components, TOXICITY a confident volume classification."""
        out = []
        for fam in self.schema.enabled_families:
            if fam == "OFI" and not store.has_quote_sizes():
                continue
            if fam == "CROSS_ASSET" and not (context and all(s in context and len(context[s]) for s in self.families.cross_asset_symbols)):
                continue
            if fam == "REGIME" and (self.components is None or self.components.hmm is None):
                continue
            if fam == "KALMAN" and (self.components is None or self.components.kalman is None):
                continue
            if fam == "TOXICITY" and (self.components is None or not self.components.vpin_enabled):
                continue
            out.append(fam)
        return tuple(out)

    def data_tier(self, store: BarStore) -> str:
        return store.data_tier()

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
    def compute_matrix(self, store: BarStore, start: int = 0, stop: int | None = None,
                       context: Optional[dict[str, BarStore]] = None) -> FeatureMatrix:
        arrays = store.arrays(start, stop)
        close = arrays["close"]
        n = len(close)
        log_close = np.log(close)
        cols: dict[str, np.ndarray] = {}
        available = list(self.available_families(store, context))

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
        bars_slice = store.bars[start:stop]
        close_times = tuple(b.close_time for b in bars_slice)
        duration = arrays["duration"]
        cols["duration_seconds"] = duration
        if self.event_bars:
            # Event bars have variable elapsed time (section 6.3): expose it, and per-unit-time versions.
            with np.errstate(divide="ignore", invalid="ignore"):
                ld = np.log(np.maximum(duration, 1.0))
                hours = np.maximum(duration, 1.0) / 3600.0
                cols["duration_z"] = robust_zscore(ld, self.z_window, self.eps)
                cols["return_per_time"] = (log_close - lag(log_close, 1)) / (sigma_h * np.sqrt(hours) + self.eps)
                cols["volume_intensity"] = robust_zscore(np.log1p(arrays["volume"] / hours), self.volume_z_window, self.eps)

        # ---- optional families (market-state spec).  Missing inputs -> NaN, never an approximation.
        comp = self.components
        fam = self.families
        if "REGIME" in self.schema.enabled_families:
            if "REGIME" in available:
                spread_col = spread if np.isfinite(spread).mean() > 0.5 else None
                probs = comp.hmm.filter(hmm_observations(log_close, arrays["volume"], spread_col))
                cols.update(regime_feature_arrays(probs, comp.hmm.transition))
            else:
                for nm in self.schema.families["REGIME"] + ("regime_stress_p",):
                    cols[nm] = np.full(n, np.nan)
        else:
            cols["regime_stress_p"] = np.full(n, np.nan)
        if "KALMAN" in self.schema.enabled_families:
            if "KALMAN" in available:
                cols.update(kalman_feature_arrays(log_close, sigma_h, comp.kalman, self.eps))
            else:
                from .kalman import KALMAN_SUBSETS
                for nm in KALMAN_SUBSETS["full"]:
                    cols[nm] = np.full(n, np.nan)
        if "OFI" in self.schema.enabled_families:
            if "OFI" in available:
                r1 = log_close - lag(log_close, 1)
                cols.update(ofi_feature_arrays(bid, ask, arrays["bid_size"], arrays["ask_size"], close, r1, eps=self.eps))
            else:
                from .ofi import OFI_NAMES
                for nm in OFI_NAMES:
                    cols[nm] = np.full(n, np.nan)
        else:
            cols["ofi_raw"] = np.full(n, np.nan)
            cols["ofi_spread"] = np.full(n, np.nan)
        if "CROSS_ASSET" in self.schema.enabled_families:
            xa_names = self.schema.families["CROSS_ASSET"] + tuple(f"xa_{s.lower()}_age_s" for s in fam.cross_asset_symbols)
            if "CROSS_ASSET" in available:
                ctx = {}
                for sym in fam.cross_asset_symbols:
                    cs = context[sym]
                    ca = cs.arrays()
                    ctx[sym] = ([b.close_time for b in cs.bars], np.log(ca["close"]), ca["volume"])
                feats, _ = cross_asset_feature_arrays(log_close, arrays["volume"], sigma_h, list(close_times), ctx,
                                                      fam.cross_asset_max_staleness_seconds, eps=self.eps)
                cols.update(feats)
            else:
                for nm in xa_names:
                    cols[nm] = np.full(n, np.nan)
        if "TOXICITY" in self.schema.enabled_families:
            if "TOXICITY" in available:
                feats, _ = vpin_feature_arrays(returns, sigma_h, arrays["volume"], fam.vpin_bucket_bars, fam.vpin_n_buckets,
                                               self.z_window, self.eps)
                cols.update(feats)
            else:
                from .toxicity import VPIN_NAMES
                for nm in VPIN_NAMES:
                    cols[nm] = np.full(n, np.nan)

        names = self.schema.all_names
        matrix = np.column_stack([cols[nm] for nm in names]) if n else np.empty((0, len(names)))
        source_times = tuple(b.latest_source_time for b in bars_slice)
        return FeatureMatrix(names, matrix, close_times, self._adaptive_d, self.fe.kernel_size(self._adaptive_d),
                             source_times, tuple(f for f in FAMILY_ORDER if f in available),
                             comp.digest() if comp is not None else "")

    def compute_latest(self, store: BarStore, context: Optional[dict[str, BarStore]] = None) -> FeatureVector:
        if not self.ready(store):
            raise RuntimeError(f"feature engine not ready: {len(store)} < {self.required_history} bars")
        n = len(store)
        start = max(0, n - self.required_history)
        fm = self.compute_matrix(store, start, n, context=context)
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
