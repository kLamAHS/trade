"""ModelTrainer: the retraining process of spec sections 37-41 and 58, hardened for research validity.

    window -> reference dataset (rows) -> [outer holdout split] -> inner walk-forward folds
    -> per fold: d* on the fold's own training block, features rebuilt with that d,
       boosted + logistic fitted on the training block, chronological calibration
    -> grid selection on the inner folds (full and baseline feature sets, best-of-grid each)
    -> untouched outer holdout: models refit on the inner block (d* from the inner block only),
       calibrated on inner out-of-fold predictions, scored once -> acceptance + ablation ΔS
    -> final production refit on the whole window (d* on the whole window: all of it is history)

Nothing that is selected (d, hyperparameters, calibration, thresholds) ever sees the rows it is
evaluated on.  Randomness is controlled by the configured seed and every artifact records the
data, configuration and software environment needed to reproduce it (spec section 56).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import platform
import subprocess
import sys
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
class FoldSet:
    """A walk-forward fold together with the dataset built from its own fold-local d*."""

    fold: Fold
    dataset: TrainingDataset
    d_star: float
    stationarity: Optional[StationarityResult] = None
    fixed: bool = False                       # diagnostics: d fixed by the caller, no fold-local estimation
    window: Optional[BarStore] = None         # the price history the fold was carved from


@dataclass
class FoldPrediction:
    fold: int
    rows: np.ndarray
    M: np.ndarray
    P: np.ndarray
    A: np.ndarray
    regressor: Optional[BoostedRegressor] = None
    direction_model: Optional[DirectionModel] = None


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
class HoldoutEvaluation:
    rows: np.ndarray
    d_star: float
    metrics: ValidationMetrics
    A: np.ndarray
    E: np.ndarray
    train_rows: np.ndarray


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
    baseline_params: dict[str, Any] = field(default_factory=dict)
    oos_by_d: list[dict[str, Any]] = field(default_factory=list)
    fold_d_stars: list[float] = field(default_factory=list)
    holdout_d_star: float = float("nan")
    holdout_metrics: Optional[ValidationMetrics] = None
    baseline_holdout_metrics: Optional[ValidationMetrics] = None
    holdout_rows: int = 0
    holdout_span: Optional[tuple[datetime, datetime]] = None   # first/last bar timestamps of the holdout rows
    d_full: float = float("nan")                                # whole-window d* (diagnostics only)

    @property
    def fold_full_score(self) -> float:
        return float(np.mean([m.score for m in self.full_fold_metrics])) if self.full_fold_metrics else float("nan")

    @property
    def fold_baseline_score(self) -> float:
        return float(np.mean([m.score for m in self.baseline_fold_metrics])) if self.baseline_fold_metrics else float("nan")

    @property
    def full_score(self) -> float:
        """S_F: the untouched holdout score when a holdout is configured, else the mean inner-fold score."""
        return self.holdout_metrics.score if self.holdout_metrics is not None else self.fold_full_score

    @property
    def baseline_score(self) -> float:
        return self.baseline_holdout_metrics.score if self.baseline_holdout_metrics is not None else self.fold_baseline_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model.version if self.model is not None else None,
            "accepted": self.accepted,
            "acceptance": self.acceptance.to_dict() if self.acceptance else None,
            "d_star": self.stationarity.d_star if self.stationarity else None,
            "d_star_selected_by": self.stationarity.selected_by if self.stationarity else None,
            "fold_d_stars": self.fold_d_stars,
            "holdout_d_star": self.holdout_d_star,
            "holdout_rows": self.holdout_rows,
            "holdout_span": [t.isoformat() for t in self.holdout_span] if self.holdout_span else None,
            "d_full": self.d_full,
            "best_params": self.best_params,
            "baseline_params": self.baseline_params,
            "grid_results": self.grid_results,
            "oos_by_d": self.oos_by_d,
            "full_fold_scores": [m.score for m in self.full_fold_metrics],
            "baseline_fold_scores": [m.score for m in self.baseline_fold_metrics],
            "fold_full_score": self.fold_full_score,
            "fold_baseline_score": self.fold_baseline_score,
            "full_score": self.full_score,
            "baseline_score": self.baseline_score,
            "delta_score": self.delta_score,
            "aggregate_metrics": self.aggregate_metrics.to_dict() if self.aggregate_metrics else None,
            "baseline_aggregate_metrics": self.baseline_aggregate_metrics.to_dict() if self.baseline_aggregate_metrics else None,
            "holdout_metrics": self.holdout_metrics.to_dict() if self.holdout_metrics else None,
            "baseline_holdout_metrics": self.baseline_holdout_metrics.to_dict() if self.baseline_holdout_metrics else None,
            "n_rows": self.n_rows,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "feature_importance": self.feature_importance,
        }


def git_commit() -> str:
    """Commit of the repository that contains *this package* (not the process cwd), with a ``-dirty``
    suffix when the package's working tree has uncommitted changes; ``nogit`` for wheel installs."""
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1]
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=pkg, capture_output=True, text=True, timeout=5)
        if top.returncode != 0 or not str(pkg).startswith(top.stdout.strip()):
            return "nogit"
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=pkg, capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return "nogit"
        status = subprocess.run(["git", "status", "--porcelain", "--", str(pkg)], cwd=pkg, capture_output=True,
                                text=True, timeout=5)
        dirty = status.returncode == 0 and status.stdout.strip() != ""
        return sha.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "nogit"


def software_version() -> str:
    return f"{__version__}+{git_commit()[:8]}"


def environment_info() -> dict[str, Any]:
    """Interpreter, platform and the numerical stack that produced a model (spec section 56)."""
    import importlib.metadata as md

    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "statsmodels", "PyYAML", "joblib"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover
            packages[name] = None
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": packages,
            "git_commit": git_commit(), "software_version": software_version()}


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
        self.outer_holdout_fraction = float(t.get("outer_holdout_fraction", 0.15))
        self.fold_local_d = bool(t.get("fold_local_d", True))
        self.seed = int(cfg.seed)
        grid = t.hyperparameter_grid
        keys = list(grid.keys())
        valid_keys = {f.name for f in dataclasses.fields(RegressionParams)} - {"backend", "seed", "num_threads"}
        unknown = [k for k in keys if k not in valid_keys]
        if unknown:
            raise ValueError(f"hyperparameter_grid keys not applicable to the regression model: {unknown}")
        self.grid = [dict(zip(keys, combo)) for combo in itertools.product(*[list(grid[k]) for k in keys])]
        self.inner_calibration_fraction = float(t.get("inner_calibration_fraction", 0.25))
        inner_min = list(t.get("inner_calibration_min_rows", [50, 20]))
        self.inner_min_fit, self.inner_min_cal = int(inner_min[0]), int(inner_min[1])
        self.min_dataset_rows = int(t.get("min_dataset_rows", 200))
        d = cfg.get("diagnostics", {}) or {}
        self.oos_by_d_every_retrain = bool(d.get("oos_by_d_every_retrain", False))
        self.oos_by_d_step = int(d.get("oos_by_d_step", 3))
        self.horizon = int(cfg.prediction.horizon_bars)
        self.software_version = software_version()
        self.environment = environment_info()

    # ------------------------------------------------------------- helpers
    def _params(self, combo: dict[str, Any]) -> RegressionParams:
        """Every key of the grid combination is applied to the regression parameters."""
        base = RegressionParams.from_config(self.cfg, combo.get("n_estimators", 300),
                                            combo.get("min_child_samples", 100), self.seed)
        extra = {k: v for k, v in combo.items() if k not in ("n_estimators", "min_child_samples")}
        return dataclasses.replace(base, **extra) if extra else base

    def _fit_pair(self, X: np.ndarray, y_norm: np.ndarray, y_raw: np.ndarray, combo: dict[str, Any],
                  names) -> tuple[BoostedRegressor, DirectionModel]:
        reg = BoostedRegressor(self._params(combo)).fit(X, y_norm, names)
        d = self.cfg.models.direction
        direction = DirectionModel(C=float(d.C), max_iter=int(d.max_iter), seed=self.seed).fit(X, y_raw)
        return reg, direction

    def _calibrator(self) -> Calibrator:
        c = self.cfg.models.calibration
        return Calibrator(method=str(c.method), bins=int(c.bins), min_points=int(c.get("min_points", 10)))

    def _simulate(self, ds: TrainingDataset, rows: np.ndarray, E: np.ndarray, M: np.ndarray | None = None) -> ValidationMetrics:
        return simulate_validation(E, ds.y_norm[rows], ds.sigma[rows], ds.sigma_ref[rows], ds.cost_roundtrip[rows],
                                   ds.cost_side_exec[rows], ds.log_close[rows], ds.open_next[rows],
                                   ds.open_next2[rows], self.sim_params, M=M)

    def _estimate_d(self, window: BarStore, upto_bar: int) -> StationarityResult:
        """d* from the window's log prices up to and including bar ``upto_bar`` (nothing later)."""
        return self.fractional.estimate_stationarity(window.log_close()[: upto_bar + 1])

    def _last_label_bar(self, ds: TrainingDataset, last_row: int) -> int:
        """Index of the newest bar whose price enters the label of ``last_row``."""
        return int(ds.bar_index[last_row]) + self.horizon + 1

    def build_fold_sets(self, window: BarStore, ds_ref: TrainingDataset, folds: list[Fold],
                        fixed_d: float | None = None) -> list[FoldSet]:
        """One dataset per fold, built with d* estimated on that fold's training block only
        (or with ``fixed_d`` for diagnostics).  Rows must coincide with the reference dataset."""
        sets: list[FoldSet] = []
        cache: dict[float, TrainingDataset] = {}
        for fold in folds:
            if fixed_d is not None:
                d, st = float(fixed_d), None
            elif self.fold_local_d:
                st = self._estimate_d(window, self._last_label_bar(ds_ref, int(fold.train[-1])))
                d = st.d_star
            else:
                d, st = ds_ref.adaptive_d, None
            sets.append(FoldSet(fold, self._dataset_for(window, ds_ref, d, cache), d, st, fixed_d is not None, window))
        return sets

    def _dataset_for(self, window: BarStore, ds_ref: TrainingDataset, d: float, cache: dict) -> TrainingDataset:
        if d not in cache:
            ds_i = ds_ref if d == ds_ref.adaptive_d else self.builder.build(window, d)
            if not np.array_equal(ds_i.bar_index, ds_ref.bar_index):
                raise ValueError("dataset rows differ from the reference dataset (kernel/warm-up mismatch)")
            cache[d] = ds_i
        return cache[d]

    def evaluate_candidate(self, fold_sets: list[FoldSet], combo: dict[str, Any], feature_names) -> CandidateEvaluation:
        names = tuple(feature_names)
        preds: list[FoldPrediction] = []
        for fs in fold_sets:
            fold, ds = fs.fold, fs.dataset
            X = ds.columns(names)
            reg, direction = self._fit_pair(X[fold.train], ds.y_norm[fold.train], ds.y_raw[fold.train], combo, names)
            M = reg.predict(X[fold.validate])
            P = direction.predict_proba_up(X[fold.validate])
            A, _ = combine(M, P)
            preds.append(FoldPrediction(fold.index, fold.validate, M, P, A, reg, direction))
        # Chronological calibration: fold i's g(A) is fitted only on out-of-sample predictions that lie
        # strictly before its validation window -- the earlier folds' validation predictions, and for the
        # first fold an inner chronological split of its own training block.
        fold_metrics: list[ValidationMetrics] = []
        calibrators: list[Calibrator] = []
        calibration_rows: list[np.ndarray] = []
        E_all: list[np.ndarray] = []
        M_all: list[np.ndarray] = []
        rows_all: list[np.ndarray] = []
        first = fold_sets[0]
        inner_A, inner_Y, inner_rows = self._inner_calibration_set(first, combo, names)
        ref_ds = fold_sets[0].dataset          # labels / simulation inputs are identical across d
        for i, fp in enumerate(preds):
            earlier = [q for q in preds[:i] if q.rows[-1] < fp.rows[0]]
            cal_A = np.concatenate([inner_A] + [q.A for q in earlier])
            cal_Y = np.concatenate([inner_Y] + [ref_ds.y_norm[q.rows] for q in earlier])
            cal_rows = np.concatenate([inner_rows] + [q.rows for q in earlier])
            cal = self._calibrator().fit(cal_A, cal_Y)
            E = cal.predict(fp.A)
            fold_metrics.append(self._simulate(ref_ds, fp.rows, E, fp.M))
            calibrators.append(cal)
            calibration_rows.append(cal_rows)
            E_all.append(E)
            M_all.append(fp.M)
            rows_all.append(fp.rows)
        rows_cat = np.concatenate(rows_all)
        aggregate = self._simulate(ref_ds, rows_cat, np.concatenate(E_all), np.concatenate(M_all))
        return CandidateEvaluation(dict(combo), preds, fold_metrics, aggregate, calibrators, calibration_rows)

    def _inner_calibration_set(self, first: FoldSet, combo: dict[str, Any], names) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronological inner split of the first fold's training block.  The models (and, with
        fold-local d, the adaptive order itself) are fitted on the earlier part only, so the (A, Y)
        pairs from the later part are out of sample in every respect."""
        ds, train_rows = first.dataset, first.fold.train
        n = len(train_rows)
        split = int(n * (1.0 - self.inner_calibration_fraction))
        inner_train = train_rows[: max(0, split - self.purge)]
        inner_cal = train_rows[min(n, split + self.embargo):]
        if len(inner_train) < self.inner_min_fit or len(inner_cal) < self.inner_min_cal:
            raise ValueError("training block too small for an inner calibration split")
        if self.fold_local_d and not first.fixed and first.window is not None:
            st = self._estimate_d(first.window, self._last_label_bar(ds, int(inner_train[-1])))
            ds = self._dataset_for(first.window, ds, st.d_star, {})
        Xall = ds.columns(tuple(names))
        reg, direction = self._fit_pair(Xall[inner_train], ds.y_norm[inner_train], ds.y_raw[inner_train], combo, names)
        A, _ = combine(reg.predict(Xall[inner_cal]), direction.predict_proba_up(Xall[inner_cal]))
        return A, ds.y_norm[inner_cal], inner_cal

    def evaluate_holdout(self, window: BarStore, ds_ref: TrainingDataset, inner_rows: np.ndarray,
                         holdout_rows: np.ndarray, candidate: CandidateEvaluation, names) -> HoldoutEvaluation:
        """Score a candidate once on the untouched outer holdout.

        d* comes from the inner block only, the models are refit on the inner block, the calibrator is
        fitted on the candidate's inner out-of-fold predictions, and the holdout rows are predicted once.
        """
        # The holdout's d* always comes from the inner block, whatever fold_local_d says: the holdout
        # must never influence the deployed feature definition it is scoring.
        st = self._estimate_d(window, self._last_label_bar(ds_ref, int(inner_rows[-1])))
        d_h = st.d_star
        ds_h = self._dataset_for(window, ds_ref, d_h, {})
        X = ds_h.columns(tuple(names))
        reg, direction = self._fit_pair(X[inner_rows], ds_h.y_norm[inner_rows], ds_h.y_raw[inner_rows],
                                        candidate.params, names)
        pooled_A = np.concatenate([fp.A for fp in candidate.fold_predictions])
        pooled_Y = np.concatenate([ds_ref.y_norm[fp.rows] for fp in candidate.fold_predictions])
        cal = self._calibrator().fit(pooled_A, pooled_Y)
        M = reg.predict(X[holdout_rows])
        P = direction.predict_proba_up(X[holdout_rows])
        A, _ = combine(M, P)
        E = cal.predict(A)
        metrics = self._simulate(ds_ref, holdout_rows, E, M)
        return HoldoutEvaluation(holdout_rows, d_h, metrics, A, E, inner_rows)

    def _metadata(self, ds: TrainingDataset, params: RegressionParams, names, validation: dict[str, Any],
                  direction: DirectionModel, is_baseline: bool, extra: dict[str, Any],
                  effective_params: dict[str, Any] | None = None) -> ModelMetadata:
        payload = json.dumps({
            "start": ds.window_start.isoformat(), "end": ds.window_end.isoformat(), "checksum": ds.window_checksum,
            "schema": self.fe.schema.version, "d": ds.adaptive_d, "seed": self.seed, "cfg": self.cfg.digest(),
            "params": params.to_dict(), "baseline": is_baseline, "names": list(names),
            "software": self.software_version, "environment": self.environment,
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
            n_training_rows=len(ds), is_baseline=is_baseline, extra=extra, environment=self.environment)

    def _layout(self, n_rows: int) -> tuple[np.ndarray, np.ndarray, list[Fold]]:
        """Rows -> (inner rows, holdout rows, inner walk-forward folds)."""
        if self.outer_holdout_fraction > 0:
            holdout_start = int(np.floor(n_rows * (1.0 - self.outer_holdout_fraction)))
            inner_end = holdout_start - self.purge
            holdout_rows = np.arange(min(n_rows, holdout_start + self.embargo), n_rows)
            if inner_end <= 0 or len(holdout_rows) < 20:
                raise ValueError(f"not enough rows ({n_rows}) for an outer holdout of {self.outer_holdout_fraction:.0%}")
        else:
            inner_end, holdout_rows = n_rows, np.arange(0)
        inner_rows = np.arange(0, inner_end)
        folds = walk_forward_folds(inner_end, self.n_folds, self.first_train_fraction, self.fold_validation_fraction,
                                   self.purge, self.embargo)
        return inner_rows, holdout_rows, folds

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
            # Reference dataset: d* on the whole window is what the *production* model uses (the whole
            # window is history at deployment time).  Folds and the holdout never use it for evaluation.
            stationarity = self.fractional.estimate_stationarity(window.log_close())
            d_full = stationarity.d_star
            _log(f"d* (whole window) = {d_full:.2f} ({stationarity.selected_by})")
            ds = self.builder.build(window, d_full)
            if len(ds) < self.min_dataset_rows:
                raise ValueError(f"too few valid training rows: {len(ds)} < {self.min_dataset_rows}")
            inner_rows, holdout_rows, folds = self._layout(len(ds))
            fold_sets = self.build_fold_sets(window, ds, folds)
            fold_ds = [round(fs.d_star, 4) for fs in fold_sets]
            _log(f"fold-local d*: {fold_ds}; holdout rows: {len(holdout_rows)}")
            full_names = ds.feature_names
            base_names = self.fe.schema.baseline_names

            # Fixed, small hyperparameter search (section 41) on the inner folds, full feature set.
            evaluations = [self.evaluate_candidate(fold_sets, combo, full_names) for combo in self.grid]
            for ev in evaluations:
                _log(f"grid {ev.params}: score={ev.mean_score:.3f} sharpe={ev.aggregate.sharpe:.2f} "
                     f"acc={ev.aggregate.accuracy:.3f} pnl={ev.aggregate.net_pnl:.4f}")
            best = max(evaluations, key=lambda e: e.mean_score)  # first max wins ties (deterministic)
            grid_results = [{"params": e.params, "score": e.mean_score, "fold_scores": e.fold_scores,
                             "aggregate": e.aggregate.to_dict()} for e in evaluations]
            # Ablation baseline: identical procedure without fractional features (section 40).
            base_evals = [self.evaluate_candidate(fold_sets, combo, base_names) for combo in self.grid]
            baseline = max(base_evals, key=lambda e: e.mean_score)

            # Untouched outer holdout: one evaluation each for the selected full and baseline candidates.
            holdout_full: Optional[HoldoutEvaluation] = None
            holdout_base: Optional[HoldoutEvaluation] = None
            if len(holdout_rows):
                holdout_full = self.evaluate_holdout(window, ds, inner_rows, holdout_rows, best, full_names)
                holdout_base = self.evaluate_holdout(window, ds, inner_rows, holdout_rows, baseline, base_names)
                acceptance_sample = holdout_full.metrics
                delta = holdout_full.metrics.score - holdout_base.metrics.score
                _log(f"holdout (d*={holdout_full.d_star:.2f}): full score={holdout_full.metrics.score:.3f} "
                     f"baseline score={holdout_base.metrics.score:.3f} delta={delta:.3f}")
            else:
                acceptance_sample = best.aggregate
                delta = best.mean_score - baseline.mean_score
            acceptance = self.validator.evaluate(acceptance_sample, best.fold_scores, baseline.fold_scores,
                                                 holdout_delta=delta if len(holdout_rows) else None)
            _log(f"baseline {baseline.params}: folds delta={best.mean_score - baseline.mean_score:.3f}; "
                 f"accepted={acceptance.accepted} {'; '.join(acceptance.reasons)}")

            oos_rows: list[dict[str, Any]] = []
            if self.oos_by_d_every_retrain:
                from ..diagnostics.fractional_analysis import FractionalDiagnostics

                candidates = [c.d for c in stationarity.candidates][:: max(1, self.oos_by_d_step)]
                oos_rows = FractionalDiagnostics.oos_score_by_d(self, history, candidates, combo=best.params, log=_log)

            # Final production refit on the whole window with the *validated* feature definition: the
            # holdout's d* (estimated on the inner block) when a holdout exists, so the deployed adaptive
            # channel is the one whose out-of-sample score was accepted.  Calibration pools every
            # out-of-fold prediction.  The whole-window d* is kept for diagnostics only.
            d_prod = holdout_full.d_star if holdout_full is not None else d_full
            ds_prod = self._dataset_for(window, ds, d_prod, {})
            Xall = ds_prod.columns(full_names)
            reg, direction = self._fit_pair(Xall, ds_prod.y_norm, ds_prod.y_raw, best.params, full_names)
            pooled_A = [fp.A for fp in best.fold_predictions]
            pooled_Y = [ds.y_norm[fp.rows] for fp in best.fold_predictions]
            if holdout_full is not None:
                pooled_A.append(holdout_full.A)
                pooled_Y.append(ds.y_norm[holdout_full.rows])
            calibration = self._calibrator().fit(np.concatenate(pooled_A), np.concatenate(pooled_Y))
            params = self._params(best.params)
            validation_summary = {
                "aggregate": best.aggregate.to_dict(), "fold_scores": best.fold_scores,
                "baseline_fold_scores": baseline.fold_scores, "baseline_aggregate": baseline.aggregate.to_dict(),
                "baseline_params": baseline.params, "delta_score": delta,
                "fold_delta_score": best.mean_score - baseline.mean_score, "acceptance": acceptance.to_dict(),
                "holdout": holdout_full.metrics.to_dict() if holdout_full else None,
                "baseline_holdout": holdout_base.metrics.to_dict() if holdout_base else None,
                "holdout_rows": int(len(holdout_rows)), "fold_d_stars": fold_ds,
                "holdout_d_star": holdout_full.d_star if holdout_full else None,
                "grid": grid_results, "baseline_grid": [{"params": e.params, "score": e.mean_score} for e in base_evals],
                "n_folds": len(folds), "purge": self.purge, "embargo": self.embargo,
            }
            extra = {"stationarity": stationarity.to_dict(), "baseline_feature_names": list(base_names),
                     "calibration_points": calibration.n_fit, "previous_adaptive_d": previous_d,
                     "config": self.cfg.to_dict(), "label_price": self.builder.label_price,
                     "d_full": d_full, "d_production": d_prod}
            meta = self._metadata(ds_prod, params, full_names, validation_summary, direction, False, extra,
                                  reg.effective_params())
            model = CombinedModel(reg, direction, calibration, full_names, d_prod, self.horizon, meta)
            holdout_span = None
            if holdout_full is not None:
                holdout_span = (window[int(ds.bar_index[holdout_full.rows[0]])].timestamp,
                                window[int(ds.bar_index[holdout_full.rows[-1]])].timestamp)
            return TrainingReport(model, acceptance.accepted, acceptance, stationarity, dict(best.params),
                                  grid_results, best.fold_metrics, baseline.fold_metrics, best.aggregate,
                                  baseline.aggregate, float(delta), len(ds), ds.window_start, ds.window_end,
                                  time.time() - t0, feature_importance=reg.feature_importance(),
                                  baseline_params=dict(baseline.params), oos_by_d=oos_rows, fold_d_stars=fold_ds,
                                  holdout_d_star=holdout_full.d_star if holdout_full else float("nan"),
                                  holdout_metrics=holdout_full.metrics if holdout_full else None,
                                  baseline_holdout_metrics=holdout_base.metrics if holdout_base else None,
                                  holdout_rows=int(len(holdout_rows)), holdout_span=holdout_span, d_full=d_full)
        except Exception as exc:
            return TrainingReport(None, False, None, stationarity, {}, [], [], [], None, None, float("nan"), 0,
                                  window[0].timestamp, window[-1].timestamp, time.time() - t0,
                                  error=f"{type(exc).__name__}: {exc}")
        finally:
            # The live feature engine keeps its previous d until a model is promoted (atomic swap by the bot).
            self.fe.set_adaptive_d(previous_d)


__all__ = ["ModelTrainer", "TrainingReport", "CandidateEvaluation", "FoldPrediction", "FoldSet", "HoldoutEvaluation",
           "software_version", "environment_info", "git_commit"]
