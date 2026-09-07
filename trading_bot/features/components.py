"""Fitted feature components that are *parameters* (frozen between retrains) as opposed to state
that updates every bar (market-state spec sections 38-40):

    HMM       transition / emission parameters   frozen   -> filtered probabilities update every bar
    Kalman    Q / R noise parameters             frozen   -> latent state updates every bar
    VPIN      classification-confidence gate     frozen

They are fitted on a training block (never on rows they are later evaluated on) and travel with the
model artifact so that live features use exactly the parameters the model was trained with.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ..data.store import BarStore
from .hmm import GaussianHMM, hmm_observations
from .kalman import KalmanParams, fit_kalman
from .rolling import rolling_std
from .toxicity import vpin_feature_arrays


@dataclass
class FeatureFamilyConfig:
    """Which optional families are enabled and how (from ``features.*`` of the strategy config)."""

    regime: bool = False
    regime_states: int = 3
    regime_causal_filter_only: bool = True
    ofi: bool = False
    ofi_normalization: str = "displayed_depth"
    kalman: bool = False
    kalman_subset: str = "full"
    cross_asset: bool = False
    cross_asset_symbols: tuple[str, ...] = ()
    cross_asset_max_staleness_seconds: float = 1800.0
    vpin: bool = False
    vpin_bucket_bars: int = 5
    vpin_n_buckets: int = 50
    vpin_min_confidence: float = 0.15
    fractional: bool = True

    @classmethod
    def from_config(cls, cfg) -> "FeatureFamilyConfig":
        f = cfg.features
        g = lambda name: (f.get(name, {}) or {})  # noqa: E731
        reg, ofi, kal, xa, vp, fr = g("regime"), g("ofi"), g("kalman"), g("cross_asset"), g("vpin"), g("fractional")
        return cls(regime=bool(reg.get("enabled", False)), regime_states=int(reg.get("states", 3)),
                   regime_causal_filter_only=bool(reg.get("causal_filter_only", True)),
                   ofi=bool(ofi.get("enabled", False)), ofi_normalization=str(ofi.get("normalization", "displayed_depth")),
                   kalman=bool(kal.get("enabled", False)), kalman_subset=str(kal.get("subset", "full")),
                   cross_asset=bool(xa.get("enabled", False)), cross_asset_symbols=tuple(str(s) for s in (xa.get("symbols") or ())),
                   cross_asset_max_staleness_seconds=float(xa.get("max_staleness_seconds", 1800)),
                   vpin=bool(vp.get("enabled", False)), vpin_bucket_bars=int(vp.get("bucket_bars", 5)),
                   vpin_n_buckets=int(vp.get("n_buckets", 50)), vpin_min_confidence=float(vp.get("min_classification_confidence", 0.15)),
                   fractional=bool(fr.get("enabled", True)) if fr else True)

    @property
    def enabled_families(self) -> tuple[str, ...]:
        out = ["PRICE", "VOLATILITY", "VOLUME"]
        if self.fractional:
            out.append("FRACTIONAL")
        if self.regime:
            out.append("REGIME")
        if self.ofi:
            out.append("OFI")
        if self.kalman:
            out.append("KALMAN")
        if self.cross_asset and self.cross_asset_symbols:
            out.append("CROSS_ASSET")
        if self.vpin:
            out.append("TOXICITY")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__, cross_asset_symbols=list(self.cross_asset_symbols))


@dataclass
class FittedComponents:
    hmm: Optional[GaussianHMM] = None
    kalman: Optional[KalmanParams] = None
    vpin_confidence: float = float("nan")
    vpin_enabled: bool = False
    fitted_start: Optional[datetime] = None
    fitted_end: Optional[datetime] = None
    n_bars: int = 0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, store: BarStore, families: FeatureFamilyConfig, sigma_window: int = 50, seed: int = 0,
            hmm_iter: int = 30) -> "FittedComponents":
        """Fit every enabled parametric component on ``store`` (the training block only)."""
        out = cls(fitted_start=store[0].timestamp if len(store) else None, fitted_end=store[-1].timestamp if len(store) else None,
                  n_bars=len(store))
        arrays = store.arrays()
        lc = np.log(arrays["close"])
        if families.regime:
            bid, ask = arrays["bid"], arrays["ask"]
            with np.errstate(invalid="ignore", divide="ignore"):
                mid = 0.5 * (bid + ask)
                spread = np.where(np.isfinite(mid) & (mid > 0), (ask - bid) / mid, np.nan)
            spread_col = spread if np.isfinite(spread).mean() > 0.5 else None
            obs = hmm_observations(lc, arrays["volume"], spread_col)
            try:
                out.hmm = GaussianHMM(n_states=int(families.regime_states), n_iter=int(hmm_iter), seed=seed).fit(obs)
            except ValueError as exc:
                out.notes.append(f"regime model not fitted: {exc}")
        if families.kalman:
            try:
                out.kalman = fit_kalman(lc)
            except ValueError as exc:
                out.notes.append(f"kalman parameters not fitted: {exc}")
        if families.vpin:
            r = np.full(len(lc), np.nan)
            r[1:] = np.diff(lc)
            sig = rolling_std(r, sigma_window)
            _, conf = vpin_feature_arrays(r, sig, arrays["volume"], families.vpin_bucket_bars, families.vpin_n_buckets)
            out.vpin_confidence = conf
            out.vpin_enabled = bool(conf >= families.vpin_min_confidence)
            if not out.vpin_enabled:
                out.notes.append(f"VPIN disabled: classification confidence {conf:.3f} < {families.vpin_min_confidence}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"hmm": self.hmm.to_dict() if self.hmm is not None else None,
                "kalman": self.kalman.to_dict() if self.kalman is not None else None,
                "vpin_confidence": self.vpin_confidence, "vpin_enabled": self.vpin_enabled,
                "fitted_start": self.fitted_start.isoformat() if self.fitted_start else None,
                "fitted_end": self.fitted_end.isoformat() if self.fitted_end else None, "n_bars": self.n_bars,
                "notes": list(self.notes)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    @property
    def key(self) -> str:
        return self.digest()


__all__ = ["FeatureFamilyConfig", "FittedComponents"]
