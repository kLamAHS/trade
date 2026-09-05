"""Feature schema: ordered names, groups, and the model-input subset.

The schema version participates in the model artifact metadata (spec section
56).  The fractional group is exactly the set removed for the ablation baseline
(spec section 40).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FEATURE_SCHEMA_VERSION = "1.0.0"

FRACTIONAL_CHANNELS = ("adaptive", "025", "050", "075")


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    all_names: tuple[str, ...]           # every value stored in FeatureVector.values
    model_names: tuple[str, ...]         # inputs to the full model
    fractional_names: tuple[str, ...]    # subset of model_names removed for the ablation baseline
    aux_names: tuple[str, ...]           # market-state values used by signal/risk, never by models

    @property
    def baseline_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.model_names if n not in set(self.fractional_names))

    def to_dict(self) -> dict:
        return {"version": self.version, "all_names": list(self.all_names), "model_names": list(self.model_names),
                "fractional_names": list(self.fractional_names), "aux_names": list(self.aux_names)}


def build_schema(volume_enabled: bool = True, use_raw_fractional_levels: bool = False,
                 slope_lags=(1, 4), return_lags=(1, 2, 4, 8, 16), vol_windows=(10, 50, 200)) -> FeatureSchema:
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
    aux = ("sigma_h", "range_rel", "spread_rel", "close", "log_close", "ewma_variance")

    fractional_model = (z_levels + slopes + curvature + cross + frac_vol + ("fractional_state",))
    if use_raw_fractional_levels:
        fractional_model = raw_levels + fractional_model
    conventional_model = returns + vols + rng + volume + ("trend_state", "volatility_state") + tod
    model_names = fractional_model + conventional_model
    all_names = (raw_levels + z_levels + slopes + curvature + cross + returns + vols + frac_vol + rng + volume
                 + regime + tod + aux)
    return FeatureSchema(FEATURE_SCHEMA_VERSION, all_names, model_names, fractional_model, aux)


__all__ = ["FeatureSchema", "build_schema", "FEATURE_SCHEMA_VERSION", "FRACTIONAL_CHANNELS"]
