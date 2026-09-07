"""Feature schema: ordered names, feature *families* and the model-input subset.

Families (market-state spec section 28) are the unit of ablation: PRICE, VOLATILITY, VOLUME,
FRACTIONAL, REGIME (causal HMM), OFI, KALMAN, CROSS_ASSET, TOXICITY.  The schema version
participates in the model artifact metadata (spec section 56).  The FRACTIONAL family is exactly
the set removed for the classic ablation baseline (spec section 40).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cross_asset import cross_asset_names
from .kalman import KALMAN_SUBSETS
from .toxicity import VPIN_NAMES

FEATURE_SCHEMA_VERSION = "2.0.0"

FRACTIONAL_CHANNELS = ("adaptive", "025", "050", "075")
FAMILY_ORDER = ("PRICE", "VOLATILITY", "VOLUME", "FRACTIONAL", "REGIME", "OFI", "KALMAN", "CROSS_ASSET", "TOXICITY")
OFI_MODEL_NAMES = ("ofi_normalized", "ofi_ema_short", "ofi_ema_long", "ofi_zscore", "ofi_acceleration",
                   "ofi_price_divergence", "quote_imbalance", "ofi_spread_zscore")


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    all_names: tuple[str, ...]           # every value stored in FeatureVector.values
    model_names: tuple[str, ...]         # inputs to the full model (every enabled family)
    fractional_names: tuple[str, ...]    # subset of model_names removed for the ablation baseline
    aux_names: tuple[str, ...]           # market-state values used by signal/risk, never by models
    families: dict[str, tuple[str, ...]] = field(default_factory=dict)   # family -> model feature names
    enabled_families: tuple[str, ...] = ()

    @property
    def baseline_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.model_names if n not in set(self.fractional_names))

    def names_for(self, families) -> tuple[str, ...]:
        """Model inputs restricted to ``families`` (schema order preserved)."""
        keep = set(families)
        return tuple(n for n in self.model_names if self.family_of(n) in keep)

    def without(self, family: str) -> tuple[str, ...]:
        return tuple(n for n in self.model_names if self.family_of(n) != family)

    def family_of(self, name: str) -> str:
        for fam, names in self.families.items():
            if name in names:
                return fam
        return "AUX"

    def to_dict(self) -> dict:
        return {"version": self.version, "all_names": list(self.all_names), "model_names": list(self.model_names),
                "fractional_names": list(self.fractional_names), "aux_names": list(self.aux_names),
                "families": {k: list(v) for k, v in self.families.items()}, "enabled_families": list(self.enabled_families)}


def regime_names(n_states: int) -> tuple[str, ...]:
    return tuple(f"regime_p_{k}" for k in range(n_states)) + ("regime_entropy", "regime_age", "regime_transition_prob",
                                                               "regime_most_likely")


def build_schema(volume_enabled: bool = True, use_raw_fractional_levels: bool = False,
                 slope_lags=(1, 4), return_lags=(1, 2, 4, 8, 16), vol_windows=(10, 50, 200),
                 fractional: bool = True, regime_states: int = 0, ofi: bool = False, kalman_subset: str | None = None,
                 cross_symbols=(), toxicity: bool = False, event_bars: bool = False) -> FeatureSchema:
    raw_levels = tuple(f"fd_{c}" for c in FRACTIONAL_CHANNELS)
    z_levels = tuple(f"fd_{c}_z" for c in FRACTIONAL_CHANNELS)
    slopes = tuple(f"fd_slope_{k}_{c}" for k in slope_lags for c in FRACTIONAL_CHANNELS)
    curvature = tuple(f"fd_curvature_{c}" for c in FRACTIONAL_CHANNELS)
    cross = ("fd_cross_sm", "fd_cross_mf", "fd_cross_sf")
    returns = tuple(f"return_{k}" for k in return_lags)
    vols = tuple(f"vol_{w}" for w in vol_windows) + ("vol_ratio_short", "vol_ratio_long")
    frac_vol = ("fractional_volatility",)
    rng = ("range_z", "close_location")
    volume = ("volume_z", "volume_change") if volume_enabled else tuple()
    regime = ("trend_state", "volatility_state", "fractional_state")
    tod = ("time_sin", "time_cos")
    event = ("duration_z", "return_per_time", "volume_intensity") if event_bars else tuple()
    aux = ("sigma_h", "range_rel", "spread_rel", "close", "log_close", "ewma_variance", "regime_stress_p", "ofi_raw",
           "ofi_spread", "duration_seconds")

    fractional_model = (z_levels + slopes + curvature + cross + frac_vol + ("fractional_state",))
    if use_raw_fractional_levels:
        fractional_model = raw_levels + fractional_model
    families: dict[str, tuple[str, ...]] = {
        "PRICE": returns + rng + tod + event,
        "VOLATILITY": vols + ("trend_state", "volatility_state"),
        "VOLUME": volume,
        "FRACTIONAL": fractional_model if fractional else tuple(),
        "REGIME": regime_names(regime_states) if regime_states > 0 else tuple(),
        "OFI": OFI_MODEL_NAMES if ofi else tuple(),
        "KALMAN": tuple(KALMAN_SUBSETS[kalman_subset]) if kalman_subset else tuple(),
        "CROSS_ASSET": cross_asset_names(cross_symbols),
        "TOXICITY": VPIN_NAMES if toxicity else tuple(),
    }
    enabled = tuple(f for f in FAMILY_ORDER if families[f])
    model_names = tuple(n for f in FAMILY_ORDER for n in families[f])
    xa_aux = tuple(f"xa_{s.lower()}_age_s" for s in cross_symbols)
    all_names = (raw_levels + z_levels + slopes + curvature + cross + returns + vols + frac_vol + rng + volume
                 + regime + tod + event + families["REGIME"] + (tuple(KALMAN_SUBSETS["full"]) if kalman_subset else tuple())
                 + (("ofi_normalized", "ofi_ema_short", "ofi_ema_long", "ofi_zscore", "ofi_acceleration",
                     "ofi_price_divergence", "quote_imbalance", "ofi_spread_zscore") if ofi else tuple())
                 + families["CROSS_ASSET"] + xa_aux + families["TOXICITY"] + aux)
    # de-duplicate while preserving order
    seen: set[str] = set()
    all_names = tuple(n for n in all_names if not (n in seen or seen.add(n)))
    return FeatureSchema(FEATURE_SCHEMA_VERSION, all_names, model_names, families["FRACTIONAL"], aux + xa_aux, families, enabled)


__all__ = ["FeatureSchema", "build_schema", "FEATURE_SCHEMA_VERSION", "FRACTIONAL_CHANNELS", "FAMILY_ORDER", "regime_names",
           "OFI_MODEL_NAMES"]
