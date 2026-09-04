"""TrainingDatasetBuilder: history window -> features, labels and simulation inputs.

Label (spec section 19):  Y_t = p_{t+H+1} - p_{t+1},   Y~_t = Y_t / (sigma_{t,50} sqrt(H) + eps)

Label columns live in a separate object from the feature engine; the feature
engine never sees them (label isolation, section 54).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..data.store import BarStore
from ..execution.cost_model import CostModel
from ..features.engine import FeatureEngine, FeatureMatrix
from ..features.rolling import rolling_median


@dataclass(frozen=True)
class TrainingDataset:
    feature_names: tuple[str, ...]
    X: np.ndarray                 # (n_rows, n_features) model inputs (full feature set)
    y_norm: np.ndarray            # normalised label Y~
    y_raw: np.ndarray             # raw label Y (log return)
    sigma: np.ndarray             # sigma_{t,50} at the signal bar
    sigma_ref: np.ndarray         # reference volatility (trailing median of sigma)
    cost_roundtrip: np.ndarray    # estimated round-trip cost at the signal bar
    cost_side_exec: np.ndarray    # per-side cost applied at the execution bar
    log_close: np.ndarray         # p_t at the signal bar
    open_next: np.ndarray         # O_{t+1}
    open_next2: np.ndarray        # O_{t+2}
    close_next: np.ndarray        # C_{t+1}
    close_times: tuple[datetime, ...]
    bar_index: np.ndarray         # index of the signal bar inside the window
    adaptive_d: float
    kernel_size: int
    window_checksum: str
    window_start: datetime
    window_end: datetime

    def __len__(self) -> int:
        return len(self.y_norm)

    def columns(self, names) -> np.ndarray:
        idx = [self.feature_names.index(n) for n in names]
        return self.X[:, idx]


def reference_sigma(sigma: np.ndarray, window: int) -> np.ndarray:
    """Trailing median of sigma over ``window`` bars; expanding median before the window fills."""
    s = np.asarray(sigma, dtype=float)
    out = rolling_median(np.where(np.isfinite(s), s, np.nan), window)
    # expanding fallback for the first bars (causal)
    for i in range(min(window - 1, len(s))):
        hist = s[: i + 1]
        hist = hist[np.isfinite(hist)]
        out[i] = np.median(hist) if len(hist) else np.nan
    # NaN-robust: if any window had NaNs, np.median returns NaN -> use nanmedian over that window
    bad = ~np.isfinite(out)
    if bad.any():
        for i in np.flatnonzero(bad):
            lo = max(0, i - window + 1)
            hist = s[lo: i + 1]
            hist = hist[np.isfinite(hist)]
            out[i] = np.median(hist) if len(hist) else np.nan
    return out


class TrainingDatasetBuilder:
    def __init__(self, feature_engine: FeatureEngine, cost_model: CostModel, horizon: int,
                 vol_reference_bars: int, eps: float = 1e-12, slippage_reference: str = "execution_bar"):
        self.fe = feature_engine
        self.cost_model = cost_model
        self.horizon = int(horizon)
        self.vol_reference_bars = int(vol_reference_bars)
        self.eps = float(eps)
        self.slippage_reference = slippage_reference

    @classmethod
    def from_config(cls, cfg, feature_engine: FeatureEngine, cost_model: CostModel) -> "TrainingDatasetBuilder":
        return cls(feature_engine, cost_model, cfg.prediction.horizon_bars,
                   int(cfg.signal.vol_reference_days) * int(cfg.market.bars_per_day), cfg.features.epsilon,
                   cfg.execution.slippage_reference)

    def build(self, window: BarStore, adaptive_d: float) -> TrainingDataset:
        self.fe.set_adaptive_d(adaptive_d)
        fm: FeatureMatrix = self.fe.compute_matrix(window)
        n = len(window)
        arrays = window.arrays()
        log_close = np.log(arrays["close"])
        H = self.horizon
        sigma = fm.column("sigma_h")
        range_rel = fm.column("range_rel")
        spread_rel = fm.column("spread_rel")

        # Labels: need bars t+1 .. t+H+1.
        y_raw = np.full(n, np.nan)
        if n > H + 1:
            y_raw[: n - (H + 1)] = log_close[H + 1:] - log_close[1: n - H]
        y_norm = y_raw / (sigma * np.sqrt(H) + self.eps)

        cost_rt = self.cost_model.estimate_array(range_rel, spread_rel)
        side = self.cost_model.per_side_array(range_rel, spread_rel)
        cost_side_exec = np.full(n, np.nan)
        if self.slippage_reference == "execution_bar":
            cost_side_exec[:-1] = side[1:]
        else:
            cost_side_exec[:] = side
        open_next = np.full(n, np.nan)
        open_next2 = np.full(n, np.nan)
        close_next = np.full(n, np.nan)
        open_next[:-1] = arrays["open"][1:]
        open_next2[:-2] = arrays["open"][2:]
        close_next[:-1] = arrays["close"][1:]
        sigma_ref = reference_sigma(sigma, self.vol_reference_bars)

        model_names = self.fe.schema.model_names
        valid = fm.valid_mask(model_names) & np.isfinite(y_norm) & np.isfinite(open_next2) & np.isfinite(cost_side_exec)
        rows = np.flatnonzero(valid)
        X = fm.values[:, [fm.names.index(nm) for nm in model_names]][rows]
        ts = tuple(fm.close_times[i] for i in rows)
        return TrainingDataset(
            feature_names=tuple(model_names), X=X, y_norm=y_norm[rows], y_raw=y_raw[rows], sigma=sigma[rows],
            sigma_ref=sigma_ref[rows], cost_roundtrip=cost_rt[rows], cost_side_exec=cost_side_exec[rows],
            log_close=log_close[rows], open_next=open_next[rows], open_next2=open_next2[rows],
            close_next=close_next[rows], close_times=ts, bar_index=rows, adaptive_d=fm.adaptive_d,
            kernel_size=fm.kernel_size, window_checksum=window.checksum(),
            window_start=window[0].timestamp, window_end=window[-1].timestamp)


__all__ = ["TrainingDataset", "TrainingDatasetBuilder", "reference_sigma"]
