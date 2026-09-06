"""CombinedModel: boosted magnitude x logistic direction -> calibrated forecast (spec section 22-23)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..types import FeatureVector, Prediction
from .calibration import Calibrator
from .direction import DirectionModel
from .regression import BoostedRegressor


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to reproduce a model exactly (spec section 56)."""

    model_id: str
    training_start: str
    training_end: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    source_data_checksum: str
    fractional_d: float
    fractional_kernel_size: int
    normalization: dict[str, Any]
    model_params: dict[str, Any]
    random_seed: int
    validation_metrics: dict[str, Any]
    software_version: str
    config_digest: str
    created_at: str
    n_training_rows: int
    is_baseline: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)   # python / platform / package versions / git commit

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["feature_names"] = list(self.feature_names)
        return d


def combine(M: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A_t = M_t * |D_t| where D_t = 2P_t - 1; A_t = 0 when sign(M_t) != sign(D_t)."""
    M = np.asarray(M, dtype=float)
    D = 2.0 * np.asarray(P, dtype=float) - 1.0
    agree = np.sign(M) == np.sign(D)
    A = np.where(agree, M * np.abs(D), 0.0)
    return A, D


class CombinedModel:
    def __init__(self, regression: BoostedRegressor, direction: DirectionModel, calibration: Calibrator,
                 feature_names, d_star: float, horizon: int, metadata: ModelMetadata | None = None):
        self.regression = regression
        self.direction = direction
        self.calibration = calibration
        self.feature_names = tuple(feature_names)
        self.d_star = float(d_star)
        self.horizon = int(horizon)
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata.model_id if self.metadata else "unversioned"

    def predict_arrays(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        M = self.regression.predict(X)
        P = self.direction.predict_proba_up(X)
        A, D = combine(M, P)
        E = self.calibration.predict(A)
        return {"M": M, "P": P, "D": D, "A": A, "E": E}

    def predict(self, features: FeatureVector) -> Prediction:
        x = np.array([[features.values[n] for n in self.feature_names]], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("non-finite feature input")
        out = self.predict_arrays(x)
        sigma = features.get("sigma_h")
        E = float(out["E"][0])
        er = E * sigma * math.sqrt(self.horizon) if math.isfinite(sigma) else math.nan
        conf = float(abs(out["D"][0])) if out["A"][0] != 0 else 0.0
        return Prediction(timestamp=features.timestamp, expected_normalized_return=E, expected_raw_return=er,
                          probability_up=float(out["P"][0]), model_confidence=conf, model_version=self.version,
                          regression_output=float(out["M"][0]), combined_output=float(out["A"][0]))


__all__ = ["CombinedModel", "ModelMetadata", "combine"]
