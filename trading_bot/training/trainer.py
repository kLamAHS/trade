"""ModelTrainer: the retraining process of spec sections 37-41 and 58.

    history -> d* -> dataset -> walk-forward folds -> (grid) boosted + logistic
    -> leave-one-fold-out calibration -> validation simulation -> best candidate
    -> ablation baseline (no fractional features) -> acceptance -> final refit

Randomness is controlled by the configured seed; every artifact records the
information required to reproduce it (spec section 56).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .. import __version__
from ..data.store import BarStore
from ..execution.cost_model import CostModel
from ..features.engine import FeatureEngine
from ..fractional.engine import FractionalEngine
from ..fractional.stationarity import StationarityResult
from ..models.calibration import Calibrator
from ..models.combined import CombinedModel, ModelMetadata, combine
from ..models.direction import DirectionModel
from ..models.regression import BoostedRegressor, RegressionParams
from .dataset import TrainingDataset, TrainingDatasetBuilder
from .validation import (AcceptanceResult, ModelValidator, SimulationParams, ValidationMetrics, simulate_validation)
from .walkforward import Fold, walk_forward_folds


@dataclass
class FoldPrediction:
    fold: int
    rows: np.ndarray
    M: np.ndarray
    P: np.ndarray
    A: np.ndarray


@dataclass
class CandidateEvaluation:
    params: dict[str, Any]
    fold_predictions: list[FoldPrediction]
    fold_metrics: list[ValidationMetrics]
    aggregate: ValidationMetrics
    calibrator_fold: list[Calibrator]
    calibration_rows: list[np.ndarray]     # rows whose labels calibrated each fold (all strictly before its validation)

    @property
    def fold_scores(self) -> list[float]:
        return [m.score for m in self.fold_metrics]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_metrics else float("-inf")


@dataclass
class TrainingReport:
    model: Optional[CombinedModel]
    accepted: bool
    acceptance: Optional[AcceptanceResult]
    stationarity: Optional[StationarityResult]
    best_params: dict[str, Any]
    grid_results: list[dict[str, Any]]
    full_fold_metrics: list[ValidationMetrics]
    baseline_fold_metrics: list[ValidationMetrics]
    aggregate_metrics: Optional[ValidationMetrics]
    baseline_aggregate_metrics: Optional[ValidationMetrics]
    delta_score: float
    n_rows: int
    window_start: Optional[datetime]
    window_end: Optional[datetime]
    elapsed_seconds: float
    error: Optional[str] = None
    feature_importance: dict[str, float] = field(default_factory=dict)

    @property
    def full_score(self) -> float:
        return float(np.mean([m.score for m in self.full_fold_metrics])) if self.full_fold_metrics else float("nan")

    @property
    def baseline_score(self) -> float:
        return float(np.mean([m.score for m in self.baseline_fold_metrics])) if self.baseline_fold_metrics else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model.version if self.model is not None else None,
            "accepted": self.accepted,
            "acceptance": self.acceptance.to_dict() if self.acceptance else None,
            "d_star": self.stationarity.d_star if self.stationarity else None,
            "d_star_selected_by": self.stationarity.selected_by if self.stationarity else None,
            "best_params": self.best_params,
            "grid_results": self.grid_results,
            "full_fold_scores": [m.score for m in self.full_fold_metrics],
            "baseline_fold_scores": [m.score for m in self.baseline_fold_metrics],
            "full_score": self.full_score,
            "baseline_score": self.baseline_score,
            "delta_score": self.delta_score,
            "aggregate_metrics": self.aggregate_metrics.to_dict() if self.aggregate_metrics else None,
            "baseline_aggregate_metrics": self.baseline_aggregate_metrics.to_dict() if self.baseline_aggregate_metrics else None,
            "n_rows": self.n_rows,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "feature_importance": self.feature_importance,
        }


def software_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        git = sha.stdout.strip() if sha.returncode == 0 else "nogit"
    except Exception:  # pragma: no cover
        git = "nogit"
    return f"{__version__}+{git}"


class ModelTrainer:
    def __init__(self, cfg, feature_engine: FeatureEngine, fractional_engine: FractionalEngine,
                 cost_model: CostModel, validator: ModelValidator | None = None):
        self.cfg = cfg
        self.fe = feature_engine
        self.fractional = fractional_engine
        self.cost_model = cost_model
        self.builder = TrainingDatasetBuilder.from_config(cfg, feature_engine, cost_model)
        self.validator = validator or ModelValidator.from_config(cfg)
        self.sim_params = SimulationParams.from_config(cfg)
        t = cfg.training
        self.window_bars = int(t.window_bars)
        self.minimum_bars = int(t.minimum_bars)
        self.n_folds = int(t.walk_forward_folds)
        self.first_train_fraction = float(t.first_train_fraction)
        self.fold_validation_fraction = float(t.fold_validation_fraction)
        self.purge = int(t.purge_bars)
        self.embargo = int(t.embargo_bars)
        self.seed = int(cfg.seed)
        grid = t.hyperparameter_grid
        keys = list(grid.keys())
        self.grid = [dict(zip(keys, combo)) for combo in itertools.product(*[list(grid[k]) for k in keys])]
        self.inner_calibration_fraction = float(t.get("inner_calibration_fraction", 0.25))
        self.horizon = int(cfg.prediction.horizon_bars)
        self.software_version = software_version()

    # ------------------------------------------------------------- helpers
    def _params(self, combo: dict[str, Any]) -> RegressionParams:
        return RegressionParams.from_config(self.cfg, combo.get("n_estimators", 300),
                                            combo.get("min_child_samples", 100), self.seed)

    def _fit_pair(self, X: np.ndarray, y_norm: np.ndarray, y_raw: np.ndarray, combo: dict[str, Any],
                  names) -> tuple[BoostedRegressor, DirectionModel]:
        reg = BoostedRegressor(self._params(combo)).fit(X, y_norm, names)
        d = self.cfg.models.direction
        direction = DirectionModel(C=float(d.C), max_iter=int(d.max_iter), seed=self.seed).fit(X, y_raw)
        return reg, direction

    def _calibrator(self) -> Calibrator:
        c = self.cfg.models.calibration
        return Calibrator(method=str(c.method), bins=int(c.bins))

    def _simulate(self, ds: TrainingDataset, rows: np.ndarray, E: np.ndarray, M: np.ndarray | None = None) -> ValidationMetrics:
        return simulate_validation(E, ds.y_norm[rows], ds.sigma[rows], ds.sigma_ref[rows], ds.cost_roundtrip[rows],
                                   ds.cost_side_exec[rows], ds.log_close[rows], ds.open_next[rows],
                                   ds.open_next2[rows], self.sim_params, M=M)

    def _inner_calibration_set(self, ds: TrainingDataset, Xall: np.ndarray, train_rows: np.ndarray,
                               combo: dict[str, Any], names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronological inner split of a training block: models fitted on the earlier part predict the
        later part (purged + embargoed), giving out-of-sample (A, Y) pairs that lie entirely before the
        fold's validation window."""
        n = len(train_rows)
        split = int(n * (1.0 - self.inner_calibration_fraction))
        inner_train = train_rows[: max(0, split - self.purge)]
        inner_cal = train_rows[min(n, split + self.embargo):]
        if len(inner_train) < 50 or len(inner_cal) < 20:
            raise ValueError("training block too small for an inner calibration split")
        reg, direction = self._fit_pair(Xall[inner_train], ds.y_norm[inner_train], ds.y_raw[inner_train], combo, names)
        A, _ = combine(reg.predict(Xall[inner_cal]), direction.predict_proba_up(Xall[inner_cal]))
        return A, ds.y_norm[inner_cal], inner_cal

    def evaluate_candidate(self, ds: TrainingDataset, folds: list[Fold], combo: dict[str, Any],
                           feature_names) -> CandidateEvaluation:
        names = tuple(feature_names)
        Xall = ds.columns(names)
        preds: list[FoldPrediction] = []
        for fold in folds:
            reg, direction = self._fit_pair(Xall[fold.train], ds.y_norm[fold.train], ds.y_raw[fold.train], combo, names)
            M = reg.predict(Xall[fold.validate])
            P = direction.predict_proba_up(Xall[fold.validate])
            A, _ = combine(M, P)
            preds.append(FoldPrediction(fold.index, fold.validate, M, P, A))
        # Chronological calibration: fold i's g(A) is fitted only on out-of-sample predictions that lie
        # strictly before its validation window -- the earlier folds' validation predictions, and for the
        # first fold an inner chronological split of its own training block.  No fold metric, acceptance
        # statistic or grid score ever sees labels from after the fold's validation window.
        fold_metrics: list[ValidationMetrics] = []
        calibrators: list[Calibrator] = []
        calibration_rows: list[np.ndarray] = []
        E_all: list[np.ndarray] = []
        M_all: list[np.ndarray] = []
        rows_all: list[np.ndarray] = []
        inner_A, inner_Y, inner_rows = self._inner_calibration_set(ds, Xall, folds[0].train, combo, names)
        for i, fp in enumerate(preds):
            earlier = [q for q in preds[:i] if q.rows[-1] < fp.rows[0]]
            cal_A = np.concatenate([inner_A] + [q.A for q in earlier])
            cal_Y = np.concatenate([inner_Y] + [ds.y_norm[q.rows] for q in earlier])
            cal_rows = np.concatenate([inner_rows] + [q.rows for q in earlier])
            cal = self._calibrator().fit(cal_A, cal_Y)
            E = cal.predict(fp.A)
            fold_metrics.append(self._simulate(ds, fp.rows, E, fp.M))
            calibrators.append(cal)
            calibration_rows.append(cal_rows)
            E_all.append(E)
            M_all.append(fp.M)
            rows_all.append(fp.rows)
        rows_cat = np.concatenate(rows_all)
        aggregate = self._simulate(ds, rows_cat, np.concatenate(E_all), np.concatenate(M_all))
        return CandidateEvaluation(dict(combo), preds, fold_metrics, aggregate, calibrators, calibration_rows)

    def _metadata(self, ds: TrainingDataset, params: RegressionParams, names, validation: dict[str, Any],
                  direction: DirectionModel, is_baseline: bool, extra: dict[str, Any],
                  effective_params: dict[str, Any] | None = None) -> ModelMetadata:
        payload = json.dumps({
            "start": ds.window_start.isoformat(), "end": ds.window_end.isoformat(), "checksum": ds.window_checksum,
            "schema": self.fe.schema.version, "d": ds.adaptive_d, "seed": self.seed, "cfg": self.cfg.digest(),
            "params": params.to_dict(), "baseline": is_baseline, "names": list(names),
        }, sort_keys=True).encode()
        model_id = ("baseline_" if is_baseline else "model_") + hashlib.sha256(payload).hexdigest()[:12]
        norm = {"scaler_mean": None if direction.scaler_mean is None else [float(v) for v in direction.scaler_mean],
                "scaler_scale": None if direction.scaler_scale is None else [float(v) for v in direction.scaler_scale]}
        return ModelMetadata(
            model_id=model_id, training_start=ds.window_start.isoformat(), training_end=ds.window_end.isoformat(),
            feature_schema_version=self.fe.schema.version, feature_names=tuple(names),
            source_data_checksum=ds.window_checksum, fractional_d=ds.adaptive_d, fractional_kernel_size=ds.kernel_size,
            normalization=norm, model_params={"regression": effective_params or params.to_dict(),
                                              "direction": {"C": float(self.cfg.models.direction.C)},
                                              "calibration": dict(self.cfg.models.calibration.to_dict())},
            random_seed=self.seed, validation_metrics=validation, software_version=self.software_version,
            config_digest=self.cfg.digest(), created_at=datetime.now(timezone.utc).isoformat(),
            n_training_rows=len(ds), is_baseline=is_baseline, extra=extra)

    # -------------------------------------------------------------- retrain
    def retrain(self, history: BarStore, log=None) -> TrainingReport:
        t0 = time.time()
        _log = log or (lambda *a, **k: None)
        window = history.last(self.window_bars)
        if len(window) < self.minimum_bars:
            return TrainingReport(None, False, None, None, {}, [], [], [], None, None, float("nan"), 0, None, None,
                                  time.time() - t0, error=f"insufficient history: {len(window)} < {self.minimum_bars}")
        previous_d = self.fe.adaptive_d
        stationarity = None
        try:
            stationarity = self.fractional.estimate_stationarity(window.log_close())
            d_star = stationarity.d_star
            _log(f"d* = {d_star:.2f} ({stationarity.selected_by})")
            ds = self.builder.build(window, d_star)
            if len(ds) < 200:
                raise ValueError(f"too few valid training rows: {len(ds)}")
            folds = walk_forward_folds(len(ds), self.n_folds, self.first_train_fraction,
                                       self.fold_validation_fraction, self.purge, self.embargo)
            full_names = ds.feature_names
            base_names = self.fe.schema.baseline_names

            # Fixed, small hyperparameter search (section 41) on the full feature set.
            evaluations: list[CandidateEvaluation] = []
            for combo in self.grid:
                ev = self.evaluate_candidate(ds, folds, combo, full_names)
                evaluations.append(ev)
                _log(f"grid {combo}: score={ev.mean_score:.3f} sharpe={ev.aggregate.sharpe:.2f} "
                     f"acc={ev.aggregate.accuracy:.3f} pnl={ev.aggregate.net_pnl:.4f}")
            best = max(evaluations, key=lambda e: e.mean_score)  # first max wins ties (deterministic)
            grid_results = [{"params": e.params, "score": e.mean_score, "fold_scores": e.fold_scores,
                             "aggregate": e.aggregate.to_dict()} for e in evaluations]

            # Ablation baseline: identical procedure without fractional features (section 40).
            baseline = self.evaluate_candidate(ds, folds, best.params, base_names)
            delta = best.mean_score - baseline.mean_score
            acceptance = self.validator.evaluate(best.aggregate, best.fold_scores, baseline.fold_scores)
            _log(f"baseline score={baseline.mean_score:.3f} delta={delta:.3f} accepted={acceptance.accepted} "
                 f"{'; '.join(acceptance.reasons)}")

            # Final refit on the whole window; calibration from pooled out-of-fold predictions.
            Xall = ds.columns(full_names)
            reg, direction = self._fit_pair(Xall, ds.y_norm, ds.y_raw, best.params, full_names)
            pooled_A = np.concatenate([fp.A for fp in best.fold_predictions])
            pooled_Y = np.concatenate([ds.y_norm[fp.rows] for fp in best.fold_predictions])
            calibration = self._calibrator().fit(pooled_A, pooled_Y)
            params = self._params(best.params)
            validation_summary = {
                "aggregate": best.aggregate.to_dict(), "fold_scores": best.fold_scores,
                "baseline_fold_scores": baseline.fold_scores, "baseline_aggregate": baseline.aggregate.to_dict(),
                "delta_score": delta, "acceptance": acceptance.to_dict(),
                "grid": grid_results, "n_folds": len(folds), "purge": self.purge, "embargo": self.embargo,
            }
            extra = {"stationarity": stationarity.to_dict(), "baseline_feature_names": list(base_names),
                     "calibration_points": calibration.n_fit, "previous_adaptive_d": previous_d}
            meta = self._metadata(ds, params, full_names, validation_summary, direction, False, extra,
                                  reg.effective_params())
            model = CombinedModel(reg, direction, calibration, full_names, d_star, self.horizon, meta)
            report = TrainingReport(model, acceptance.accepted, acceptance, stationarity, dict(best.params),
                                    grid_results, best.fold_metrics, baseline.fold_metrics, best.aggregate,
                                    baseline.aggregate, float(delta), len(ds), ds.window_start, ds.window_end,
                                    time.time() - t0, feature_importance=reg.feature_importance())
            return report
        except Exception as exc:
            # A failed retraining cycle never reaches the main loop: the previous accepted model keeps
            # trading (spec section 37) and the failure is reported as RETRAIN_FAILED.
            return TrainingReport(None, False, None, stationarity, {}, [], [], [], None, None, float("nan"), 0,
                                  window[0].timestamp, window[-1].timestamp, time.time() - t0,
                                  error=f"{type(exc).__name__}: {exc}")
        finally:
            # The live feature engine keeps its previous d until a model is promoted (atomic swap by the bot).
            self.fe.set_adaptive_d(previous_d)


__all__ = ["ModelTrainer", "TrainingReport", "CandidateEvaluation", "FoldPrediction", "software_version"]
