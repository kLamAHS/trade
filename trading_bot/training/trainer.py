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
import math
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
from ..execution.estimator import ExecutionModel
from ..features.components import FittedComponents
from ..features.engine import FeatureEngine
from ..fractional.engine import FractionalEngine
from ..fractional.stationarity import StationarityResult
from ..models.calibration import Calibrator
from ..models.combined import CombinedModel, ModelMetadata, combine
from ..models.direction import DirectionModel
from .panel import TrainingPanel, build_panel_rows, panel_forecast_quality
from ..models.regression import BoostedRegressor, RegressionParams
from .dataset import TrainingDataset, TrainingDatasetBuilder
from .validation import (AcceptanceResult, ModelValidator, SimulationParams, ValidationMetrics, _safe_corr,
                         simulate_validation)
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
    components: Optional[FittedComponents] = None   # HMM / Kalman parameters fitted on this fold's training block


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
    exec_models: list[ExecutionModel] = field(default_factory=list)   # adverse-selection model per fold (same rows as the calibrator)
    fold_E: list[np.ndarray] = field(default_factory=list)
    fold_adverse: list[np.ndarray] = field(default_factory=list)

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
    components: Optional[FittedComponents] = None
    exec_model: Optional[ExecutionModel] = None
    P: Optional[np.ndarray] = None
    adverse: Optional[np.ndarray] = None
    diagnosis: dict[str, Any] = field(default_factory=dict)   # why this block traded, or did not
    panel: dict[str, Any] = field(default_factory=dict)       # the same model's forecasts on the other instruments


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
    holdout_diagnosis: dict[str, Any] = field(default_factory=dict)   # why the acceptance sample traded, or did not
    panel_diagnosis: dict[str, Any] = field(default_factory=dict)     # pooled training rows and their out-of-sample forecast quality
    holdout_rows: int = 0
    holdout_span: Optional[tuple[datetime, datetime]] = None   # first/last bar timestamps of the holdout rows
    d_full: float = float("nan")                                # whole-window d* (diagnostics only)
    baseline_model: Optional[CombinedModel] = None              # final refit of the ablation baseline (research runs)

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


def fitted_model_hash(model: CombinedModel) -> str:
    """Digest of the *fitted* parameters (trees, logistic weights, calibration map) -- independent of
    the configuration and metadata hashes, so an identical manifest that yields a different fitted
    model is detected (research section 26)."""
    h = hashlib.sha256()
    reg = model.regression._model
    if reg is not None:
        booster = getattr(reg, "booster_", None)
        if booster is not None:
            h.update(booster.model_to_string().encode())
        else:  # sklearn fallback: hash the predictions on a fixed probe grid
            h.update(np.ascontiguousarray(reg.predict(np.zeros((1, len(model.feature_names))))).tobytes())
            h.update(str(sorted((k, str(v)) for k, v in reg.get_params().items())).encode())
    d = model.direction
    if d._pipe is not None:
        logit = d._pipe.named_steps["logit"]
        h.update(np.ascontiguousarray(logit.coef_, dtype=float).tobytes())
        h.update(np.ascontiguousarray(logit.intercept_, dtype=float).tobytes())
        h.update(np.ascontiguousarray(d.scaler_mean, dtype=float).tobytes())
        h.update(np.ascontiguousarray(d.scaler_scale, dtype=float).tobytes())
    else:
        h.update(f"constant:{d.constant_class}".encode())
    cal = model.calibration
    probe = np.linspace(-3.0, 3.0, 241)
    h.update(np.ascontiguousarray(cal.predict(probe), dtype=float).tobytes())
    h.update(f"d={model.d_star!r};H={model.horizon};names={list(model.feature_names)}".encode())
    comp = getattr(model, "components", None)
    if comp is not None:
        h.update(comp.digest().encode())
    ex = getattr(model, "execution_model", None)
    if ex is not None and ex.fitted:
        h.update(np.ascontiguousarray(ex.coef, dtype=float).tobytes())
        h.update(f"{ex.intercept!r}|{ex.residual_std!r}".encode())
    return h.hexdigest()[:16]


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
                 cost_model: CostModel, validator: ModelValidator | None = None, fit_final_baseline: bool = False):
        self.cfg = cfg
        self.fit_final_baseline = bool(fit_final_baseline)   # research: also refit the ablation baseline as a model
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
        e = cfg.execution
        self.exec_enabled = bool(e.get("adverse_selection_model", True))
        self.uncertainty_z = float(e.get("uncertainty_z", 0.0))
        self.exec_alpha = float(e.get("adverse_selection_ridge_alpha", 1.0))
        self.adverse_floor = float(e.get("adverse_selection_floor_bps", 0.0)) / 1e4
        self.fold_local_components = bool(t.get("fold_local_components", True))
        self._context: Optional[dict[str, BarStore]] = None      # cross-asset stores for the current retrain
        p = t.get("panel", {}) or {}                             # pooled training rows from other instruments
        self.panel_enabled = bool(p.get("enabled", False))
        self.panel_symbols = tuple(str(s).upper() for s in (p.get("symbols") or []))
        self.panel_align = bool(p.get("align_features", True))
        self.panel_max_symbols = int(p.get("max_symbols", 20))
        self.panel_max_rows = int(p.get("max_rows_per_fit", 0))
        self._panel_stores: dict[str, BarStore] = {}
        self._panel_cache: dict = {}

    # ------------------------------------------------------------- helpers
    def _params(self, combo: dict[str, Any]) -> RegressionParams:
        """Every key of the grid combination is applied to the regression parameters."""
        base = RegressionParams.from_config(self.cfg, combo.get("n_estimators", 300),
                                            combo.get("min_child_samples", 100), self.seed)
        extra = {k: v for k, v in combo.items() if k not in ("n_estimators", "min_child_samples")}
        return dataclasses.replace(base, **extra) if extra else base

    def _fit_pair(self, X: np.ndarray, y_norm: np.ndarray, y_raw: np.ndarray, combo: dict[str, Any],
                  names, extra: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                  ) -> tuple[BoostedRegressor, DirectionModel]:
        """``extra`` are pooled rows from the training panel (other instruments).  They enlarge the fit
        only: the primary's rows are passed through unchanged and every evaluation downstream still
        reads the primary alone."""
        if extra is not None and len(extra[0]):
            Xf = np.concatenate([X, extra[0]])
            yn = np.concatenate([y_norm, extra[1]])
            yr = np.concatenate([y_raw, extra[2]])
        else:
            Xf, yn, yr = X, y_norm, y_raw
        reg = BoostedRegressor(self._params(combo)).fit(Xf, yn, names)
        d = self.cfg.models.direction
        direction = DirectionModel(C=float(d.C), max_iter=int(d.max_iter), seed=self.seed).fit(Xf, yr)
        return reg, direction

    # ------------------------------------------------------ training panel
    def set_panel(self, stores: Optional[dict[str, BarStore]]) -> None:
        """Secondary instruments whose rows join every fit.  Cleared between retrains."""
        self._panel_stores = dict(stores or {})
        self._panel_cache = {}

    def _panel_for(self, window: BarStore, d: float, names: tuple[str, ...],
                   cutoff: Optional[datetime] = None) -> Optional[TrainingPanel]:
        """The panel built with the primary's feature definition ``d`` for this fit.

        Each secondary instrument gets its own fitted components (its HMM / Kalman parameters describe
        its own history) but the *same* fractional order as the primary, so a pooled column means the
        same transformation everywhere.

        Feature columns are causal within each symbol, so rows may be filtered by label completion
        afterwards.  Fitted components are not: parameters estimated over a symbol's whole history would
        carry its future into every pooled row.  When any parametric family is enabled each symbol's
        history is therefore truncated at ``cutoff`` before it is built, and the panel is cached per
        cutoff; with no parametric families the build is cutoff-independent and cached once.
        """
        if not getattr(self, "_panel_stores", None) or not self.panel_enabled:
            return None
        parametric = self._parametric_families
        key = (round(float(d), 10), names, cutoff.isoformat() if (parametric and cutoff is not None) else "")
        cached = self._panel_cache.get(key)
        if cached is not None:
            return cached
        rows = []
        for sym, store in list(self._panel_stores.items())[: self.panel_max_symbols]:
            src = self._truncate(store, cutoff) if (parametric and cutoff is not None) else store
            if len(src) < self.min_dataset_rows:
                continue
            try:
                comp = self._components_for(src, len(src) - 1, {}) if parametric else None
                ds_s = self.builder.build(src, float(d), components=comp, context=self._context_for(src))
                pr = build_panel_rows(sym, ds_s, src, self.horizon, names)
            except Exception:
                pr = None                      # a symbol that cannot produce the primary's columns is skipped
            if pr is not None:
                rows.append(pr)
        panel = TrainingPanel(rows, align=self.panel_align, max_rows_per_fit=self.panel_max_rows)
        self._panel_cache[key] = panel
        return panel

    @staticmethod
    def _truncate(store: BarStore, cutoff: datetime) -> BarStore:
        """The prefix of ``store`` that had closed by ``cutoff``."""
        n = len(store)
        lo, hi = 0, n
        while lo < hi:                                   # bars are chronological: binary search the cutoff
            mid = (lo + hi) // 2
            if store[mid].close_time <= cutoff:
                lo = mid + 1
            else:
                hi = mid
        return store if lo == n else store.slice(0, lo)

    def _panel_out_of_sample(self, window: BarStore, ds: TrainingDataset, inner_rows: np.ndarray,
                             holdout_rows: np.ndarray, d: float, model) -> dict[str, Any]:
        """The holdout question asked of the instruments the bot will *not* trade.

        The model was fitted on the inner block only, so its forecasts over the holdout span are out of
        sample for every symbol.  One instrument gives one experiment; a pooled correlation near zero
        beside a healthy primary correlation is the signature of a result that will not repeat.
        """
        last = len(window) - 1
        start = window[min(self._last_label_bar(ds, int(inner_rows[-1])), last)].close_time
        end = window[min(self._last_label_bar(ds, int(holdout_rows[-1])), last)].close_time
        panel = self._panel_for(window, d, tuple(model.feature_names), start)   # fitted only on the inner block
        if panel is None or not len(panel):
            return {}
        rows = panel.out_of_sample(start, end)
        if not rows:
            return {}
        out = panel_forecast_quality(model, rows)
        out["span"] = [start.isoformat(), end.isoformat()]
        return out

    def _pooled(self, window: BarStore, ds: TrainingDataset, rows: np.ndarray, d: float,
                names: tuple[str, ...], primary_X: np.ndarray):
        """Pooled rows eligible for a fit whose primary rows end at ``rows[-1]``.

        The cutoff is the close of the newest bar entering that row's label: a panel row may join only
        once its own label is complete by then, so no fit ever sees a label the primary could not.
        """
        if not len(rows):
            return None
        idx = min(self._last_label_bar(ds, int(rows[-1])), len(window) - 1)
        cutoff = window[idx].close_time
        panel = self._panel_for(window, d, names, cutoff)
        if panel is None or not len(panel):
            return None
        return panel.eligible(cutoff, primary_X)

    def _calibrator(self) -> Calibrator:
        c = self.cfg.models.calibration
        return Calibrator(method=str(c.method), bins=int(c.bins), min_points=int(c.get("min_points", 10)))

    def forecast_diagnosis(self, ds: TrainingDataset, rows: np.ndarray, E: np.ndarray, metrics: ValidationMetrics,
                           adverse: np.ndarray | None = None, uncertainty: np.ndarray | None = None,
                           calibration_corr: float = float("nan"), calibration_rows: int = 0) -> dict[str, Any]:
        """Why did this block of rows trade, or not?  A rejected retrain otherwise reports a row of zeros
        (no trades -> no P&L, no profit factor, no ablation delta), which says nothing about the cause.
        The two causes that matter are distinguishable: a calibrator that collapsed to a constant because
        the out-of-fold forecasts carried no monotone relation to the label, and forecasts that are real
        but too small to clear the trade threshold."""
        pol = self.sim_params.policy
        er = np.abs(np.asarray(E, dtype=float)) * ds.sigma[rows] * math.sqrt(self.horizon)
        cost = ds.cost_roundtrip[rows]
        adv = np.zeros(len(rows)) if adverse is None else np.asarray(adverse, dtype=float)
        unc = np.zeros(len(rows)) if uncertainty is None else np.asarray(uncertainty, dtype=float)
        if pol.mode == "legacy":
            required = pol.cost_multiplier * cost
        else:
            required = cost + adv + unc + pol.uncertainty_buffer + pol.minimum_net_edge
        ok = np.isfinite(er) & np.isfinite(required)
        bps = lambda v: float(v * 1e4)  # noqa: E731
        out: dict[str, Any] = {
            "rows": int(len(rows)), "n_signals": int(metrics.n_signals), "n_trades": int(metrics.n_trades),
            "policy": pol.mode, "forecast_distinct_values": int(len(np.unique(np.round(np.asarray(E, dtype=float), 12)))),
            "forecast_std": float(np.std(E)) if len(E) else 0.0,
            "calibration_correlation": float(calibration_corr), "calibration_rows": int(calibration_rows),
            "median_abs_edge_bps": bps(np.median(er[ok])) if ok.any() else 0.0,
            "p90_abs_edge_bps": bps(np.percentile(er[ok], 90)) if ok.any() else 0.0,
            "max_abs_edge_bps": bps(np.max(er[ok])) if ok.any() else 0.0,
            "median_required_bps": bps(np.median(required[ok])) if ok.any() else 0.0,
            "share_clearing_threshold": float(np.mean(er[ok] > required[ok])) if ok.any() else 0.0,
        }
        out["reason"] = self._diagnosis_text(out)
        return out

    @staticmethod
    def _diagnosis_text(d: dict[str, Any]) -> str:
        if d["n_signals"] > 0:
            return (f"{d['n_signals']} signals and {d['n_trades']} trades from {d['rows']} rows "
                    f"(median |ER| {d['median_abs_edge_bps']:.1f} bps vs required {d['median_required_bps']:.1f} bps)")
        if d["forecast_distinct_values"] <= 2 or d["forecast_std"] < 1e-12:
            corr = d["calibration_correlation"]
            corr_txt = f"{corr:+.4f}" if math.isfinite(corr) else "n/a"
            return ("the calibrator collapsed to a constant forecast, so no bar can produce a signal: the "
                    f"out-of-fold predictions had no monotone relation to the label (corr {corr_txt} over "
                    f"{d['calibration_rows']} rows). The model found no edge; this is a verdict, not a fault")
        return ("forecasts are real but never clear the trade threshold: |ER| median "
                f"{d['median_abs_edge_bps']:.1f} bps, p90 {d['p90_abs_edge_bps']:.1f} bps, max "
                f"{d['max_abs_edge_bps']:.1f} bps against a required {d['median_required_bps']:.1f} bps "
                f"({d['policy']} policy)")

    def _simulate(self, ds: TrainingDataset, rows: np.ndarray, E: np.ndarray, M: np.ndarray | None = None,
                  adverse: np.ndarray | None = None, uncertainty: np.ndarray | None = None) -> ValidationMetrics:
        stress = ds.stress_p[rows] if ds.stress_p is not None else None
        return simulate_validation(E, ds.y_norm[rows], ds.sigma[rows], ds.sigma_ref[rows], ds.cost_roundtrip[rows],
                                   ds.cost_side_exec[rows], ds.log_close[rows], ds.open_next[rows],
                                   ds.open_next2[rows], self.sim_params, M=M, adverse=adverse, uncertainty=uncertainty,
                                   stress_p=stress)

    # ------------------------------------------------------ components
    @property
    def _parametric_families(self) -> bool:
        f = self.fe.families
        return bool(f.regime or f.kalman or f.vpin)

    def _components_for(self, window: BarStore, upto_bar: int, cache: dict) -> Optional[FittedComponents]:
        """HMM / Kalman / VPIN parameters fitted on the window's bars up to and including ``upto_bar``."""
        if not self._parametric_families:
            return None
        key = int(min(upto_bar, len(window) - 1))
        if key not in cache:
            cache[key] = self.fe.fit_components(window.slice(0, key + 1))
        return cache[key]

    def _context_for(self, window: BarStore) -> Optional[dict[str, BarStore]]:
        """Cross-asset stores restricted to bars that closed no later than the window's last close."""
        if not self._context:
            return None
        end = window[-1].close_time
        out = {}
        for sym, cs in self._context.items():
            bars = [b for b in cs.bars if b.close_time <= end]
            out[sym] = BarStore(sym, cs.bar_minutes, bars)
        return out

    # ------------------------------------------------ execution model
    def _fit_exec_model(self, ds: TrainingDataset, rows: np.ndarray, E: np.ndarray, P: np.ndarray) -> Optional[ExecutionModel]:
        """Adverse-selection model on out-of-fold forecasts of ``rows`` (fractions)."""
        if not self.exec_enabled or ds.x_exec is None:
            return None
        abs_er = np.abs(E) * ds.sigma[rows] * math.sqrt(self.horizon)
        conf = np.abs(2.0 * P - 1.0)
        direction = np.sign(E)
        return ExecutionModel(alpha=self.exec_alpha).fit(ds.x_exec[rows], abs_er, conf, direction, ds.r_fill[rows], self.horizon)

    def _adverse_for(self, model: Optional[ExecutionModel], ds: TrainingDataset, rows: np.ndarray, E: np.ndarray,
                     P: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if model is None or not model.fitted:
            return None, None
        abs_er = np.abs(E) * ds.sigma[rows] * math.sqrt(self.horizon)
        adv = np.maximum(model.predict(ds.x_exec[rows], abs_er, np.abs(2.0 * P - 1.0), np.sign(E)), self.adverse_floor)
        return adv, np.full(len(rows), self.uncertainty_z * model.residual_std)

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
        cache: dict = {}
        comp_cache: dict = {}
        for fold in folds:
            if fixed_d is not None:
                d, st = float(fixed_d), None
            elif self.fold_local_d:
                st = self._estimate_d(window, self._last_label_bar(ds_ref, int(fold.train[-1])))
                d = st.d_star
            else:
                d, st = ds_ref.adaptive_d, None
            if self.fold_local_components and fixed_d is None:
                comp = self._components_for(window, self._last_label_bar(ds_ref, int(fold.train[-1])), comp_cache)
            else:
                comp = self._ref_components
            sets.append(FoldSet(fold, self._dataset_for(window, ds_ref, d, cache, comp), d, st, fixed_d is not None, window, comp))
        return sets

    _ref_components: Optional[FittedComponents] = None   # components of the reference dataset (whole window)

    def _dataset_for(self, window: BarStore, ds_ref: TrainingDataset, d: float, cache: dict,
                     components: Optional[FittedComponents] = "ref") -> TrainingDataset:
        comp = self._ref_components if components == "ref" else components
        key = (round(float(d), 10), comp.key if comp is not None else "")
        if key not in cache:
            same = d == ds_ref.adaptive_d and (comp.key if comp is not None else "") == ds_ref.components_digest
            ds_i = ds_ref if same else self.builder.build(window, d, components=comp, context=self._context_for(window))
            if not np.array_equal(ds_i.bar_index, ds_ref.bar_index):
                raise ValueError("dataset rows differ from the reference dataset (kernel/warm-up mismatch)")
            cache[key] = ds_i
        return cache[key]

    def evaluate_candidate(self, fold_sets: list[FoldSet], combo: dict[str, Any], feature_names) -> CandidateEvaluation:
        names = tuple(feature_names)
        preds: list[FoldPrediction] = []
        for fs in fold_sets:
            fold, ds = fs.fold, fs.dataset
            X = ds.columns(names)
            pooled = self._pooled(fs.window, ds, fold.train, fs.d_star, names, X[fold.train]) if fs.window is not None else None
            reg, direction = self._fit_pair(X[fold.train], ds.y_norm[fold.train], ds.y_raw[fold.train], combo, names,
                                            extra=pooled)
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
        exec_models: list[ExecutionModel] = []
        E_all: list[np.ndarray] = []
        M_all: list[np.ndarray] = []
        adv_all: list[np.ndarray] = []
        unc_all: list[np.ndarray] = []
        rows_all: list[np.ndarray] = []
        first = fold_sets[0]
        inner_A, inner_Y, inner_rows, inner_P = self._inner_calibration_set(first, combo, names)
        ref_ds = fold_sets[0].dataset          # labels / simulation inputs are identical across d
        for i, fp in enumerate(preds):
            earlier = [q for q in preds[:i] if q.rows[-1] < fp.rows[0]]
            cal_A = np.concatenate([inner_A] + [q.A for q in earlier])
            cal_Y = np.concatenate([inner_Y] + [ref_ds.y_norm[q.rows] for q in earlier])
            cal_rows = np.concatenate([inner_rows] + [q.rows for q in earlier])
            cal_P = np.concatenate([inner_P] + [q.P for q in earlier])
            cal = self._calibrator().fit(cal_A, cal_Y)
            E = cal.predict(fp.A)
            # Execution model: same chronological rows as the calibrator (out of sample for this fold).
            ex = self._fit_exec_model(ref_ds, cal_rows, cal.predict(cal_A), cal_P)
            adv, unc = self._adverse_for(ex, ref_ds, fp.rows, E, fp.P)
            fold_metrics.append(self._simulate(ref_ds, fp.rows, E, fp.M, adv, unc))
            calibrators.append(cal)
            calibration_rows.append(cal_rows)
            exec_models.append(ex)
            E_all.append(E)
            M_all.append(fp.M)
            adv_all.append(adv if adv is not None else np.zeros(len(fp.rows)))
            unc_all.append(unc if unc is not None else np.zeros(len(fp.rows)))
            rows_all.append(fp.rows)
        rows_cat = np.concatenate(rows_all)
        use_adv = any(x is not None and x.fitted for x in exec_models)
        aggregate = self._simulate(ref_ds, rows_cat, np.concatenate(E_all), np.concatenate(M_all),
                                   np.concatenate(adv_all) if use_adv else None, np.concatenate(unc_all) if use_adv else None)
        return CandidateEvaluation(dict(combo), preds, fold_metrics, aggregate, calibrators, calibration_rows, exec_models,
                                   E_all, adv_all)

    def _inner_calibration_set(self, first: FoldSet, combo: dict[str, Any], names) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            comp = first.components
            if self.fold_local_components and self._parametric_families:
                comp = self._components_for(first.window, self._last_label_bar(ds, int(inner_train[-1])), {})
            ds = self._dataset_for(first.window, ds, st.d_star, {}, comp)
        Xall = ds.columns(tuple(names))
        pooled = (self._pooled(first.window, ds, inner_train, ds.adaptive_d, tuple(names), Xall[inner_train])
                  if first.window is not None else None)
        reg, direction = self._fit_pair(Xall[inner_train], ds.y_norm[inner_train], ds.y_raw[inner_train], combo, names,
                                        extra=pooled)
        P = direction.predict_proba_up(Xall[inner_cal])
        A, _ = combine(reg.predict(Xall[inner_cal]), P)
        return A, ds.y_norm[inner_cal], inner_cal, P

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
        comp_h = self._components_for(window, self._last_label_bar(ds_ref, int(inner_rows[-1])), {})
        ds_h = self._dataset_for(window, ds_ref, d_h, {}, comp_h)
        X = ds_h.columns(tuple(names))
        pooled = self._pooled(window, ds_h, inner_rows, d_h, tuple(names), X[inner_rows])
        reg, direction = self._fit_pair(X[inner_rows], ds_h.y_norm[inner_rows], ds_h.y_raw[inner_rows],
                                        candidate.params, names, extra=pooled)
        pooled_A = np.concatenate([fp.A for fp in candidate.fold_predictions])
        pooled_Y = np.concatenate([ds_ref.y_norm[fp.rows] for fp in candidate.fold_predictions])
        pooled_P = np.concatenate([fp.P for fp in candidate.fold_predictions])
        pooled_rows = np.concatenate([fp.rows for fp in candidate.fold_predictions])
        cal = self._calibrator().fit(pooled_A, pooled_Y)
        ex = self._fit_exec_model(ds_ref, pooled_rows, cal.predict(pooled_A), pooled_P)
        M = reg.predict(X[holdout_rows])
        P = direction.predict_proba_up(X[holdout_rows])
        A, _ = combine(M, P)
        E = cal.predict(A)
        adv, unc = self._adverse_for(ex, ds_ref, holdout_rows, E, P)
        metrics = self._simulate(ds_ref, holdout_rows, E, M, adv, unc)
        cal_corr = _safe_corr(pooled_A, pooled_Y)      # the relation isotonic calibration had to work with
        diag = self.forecast_diagnosis(ds_ref, holdout_rows, E, metrics, adv, unc, cal_corr, len(pooled_A))
        panel = self._panel_out_of_sample(window, ds_ref, inner_rows, holdout_rows, d_h,
                                          CombinedModel(reg, direction, cal, tuple(names), float(d_h), self.horizon,
                                                        None, components=comp_h, families=ds_h.families))
        return HoldoutEvaluation(holdout_rows, d_h, metrics, A, E, inner_rows, comp_h, ex, P, adv, diag, panel)

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

    # ---------------------------------------------------------- light refit
    def light_refit(self, history: BarStore, d: float, params: dict[str, Any], names, calibration_fraction: float = 0.25,
                    shuffle_labels: bool = False, shuffle_features: bool = False, label_offset_bars: int = 0,
                    seed: int | None = None) -> CombinedModel:
        """Cheap refit used by the research stress / sanity / leakage tests (sections 20-24).

        No d* search, no grid, no holdout: the models are fitted on the earliest
        ``1 - calibration_fraction`` of the window's rows (purged) and calibrated chronologically on the
        rest.  ``shuffle_labels`` permutes the training labels, ``shuffle_features`` permutes every
        feature column independently (both destroy any real structure: a strategy that still
        "works" afterwards is broken), ``label_offset_bars`` shifts the label into the future.
        The reference for every perturbation is the same light refit with the perturbation off.
        """
        window = history.last(self.window_bars)
        previous_d = self.fe.adaptive_d
        try:
            comp = self._components_for(window, len(window) - 1, {})
            ds = self.builder.build(window, float(d), label_offset_bars=int(label_offset_bars), components=comp,
                                    context=self._context_for(window))
            names = tuple(n for n in names if n in ds.feature_names)
            X = ds.columns(names)
            y_norm, y_raw = ds.y_norm.copy(), ds.y_raw.copy()
            rng = np.random.default_rng(self.seed if seed is None else int(seed))
            if shuffle_labels:
                perm = rng.permutation(len(ds))
                y_norm, y_raw = y_norm[perm], y_raw[perm]
            if shuffle_features:
                X = X.copy()
                for j in range(X.shape[1]):
                    X[:, j] = X[rng.permutation(len(ds)), j]
            n = len(ds)
            split = int(n * (1.0 - calibration_fraction))
            fit_rows = np.arange(0, max(0, split - self.purge))
            cal_rows = np.arange(min(n, split + self.embargo), n)
            if len(fit_rows) < self.inner_min_fit or len(cal_rows) < self.inner_min_cal:
                raise ValueError("window too small for a light refit")
            # A sabotaged refit must stay sabotaged: real pooled rows would undo the shuffle.
            pooled = (None if (shuffle_labels or shuffle_features)
                      else self._pooled(window, ds, fit_rows, float(d), names, X[fit_rows]))
            reg, direction = self._fit_pair(X[fit_rows], y_norm[fit_rows], y_raw[fit_rows], params, names, extra=pooled)
            P = direction.predict_proba_up(X[cal_rows])
            A, _ = combine(reg.predict(X[cal_rows]), P)
            cal = self._calibrator().fit(A, y_norm[cal_rows])
            ex = self._fit_exec_model(ds, cal_rows, cal.predict(A), P) if not (shuffle_labels or shuffle_features) else None
            return CombinedModel(reg, direction, cal, names, float(d), self.horizon, None, components=comp,
                                 execution_model=ex, families=ds.families)
        finally:
            self.fe.set_adaptive_d(previous_d)

    # -------------------------------------------------------------- retrain
    def retrain(self, history: BarStore, log=None, context: Optional[dict[str, BarStore]] = None,
                panel: Optional[dict[str, BarStore]] = None) -> TrainingReport:
        """``panel`` are secondary instruments whose rows join every model fit (pooled training).  They
        never enter a fold's validation block, the holdout, the acceptance metrics or the simulated
        P&L: what is accepted remains a statement about the instrument the bot trades."""
        t0 = time.time()
        _log = log or (lambda *a, **k: None)
        self._context = context
        self.set_panel(panel)
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
            comp_full = self._components_for(window, len(window) - 1, {})
            self._ref_components = comp_full
            ds = self.builder.build(window, d_full, components=comp_full, context=self._context_for(window))
            if comp_full is not None and comp_full.notes:
                _log("components: " + "; ".join(comp_full.notes))
            _log(f"feature families: {list(ds.families)} ({len(ds.feature_names)} inputs)")
            if len(ds) < self.min_dataset_rows:
                raise ValueError(f"too few valid training rows: {len(ds)} < {self.min_dataset_rows}")
            inner_rows, holdout_rows, folds = self._layout(len(ds))
            fold_sets = self.build_fold_sets(window, ds, folds)
            fold_ds = [round(fs.d_star, 4) for fs in fold_sets]
            _log(f"fold-local d*: {fold_ds}; holdout rows: {len(holdout_rows)}")
            full_names = ds.feature_names
            frac = set(self.fe.schema.fractional_names)
            base_names = tuple(n for n in full_names if n not in frac)

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
                _log(f"holdout (d*={holdout_full.d_star:.2f}, {len(holdout_rows)} rows): "
                     f"full score={holdout_full.metrics.score:.3f} baseline score={holdout_base.metrics.score:.3f} "
                     f"delta={delta:.3f}")
                _log(f"holdout verdict: {holdout_full.diagnosis.get('reason', '')}")
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
            comp_prod = holdout_full.components if holdout_full is not None else comp_full
            ds_prod = self._dataset_for(window, ds, d_prod, {}, comp_prod)
            Xall = ds_prod.columns(full_names)
            all_rows = np.arange(len(ds_prod))
            pooled_prod = self._pooled(window, ds_prod, all_rows, d_prod, full_names, Xall)
            reg, direction = self._fit_pair(Xall, ds_prod.y_norm, ds_prod.y_raw, best.params, full_names,
                                            extra=pooled_prod)
            pooled_A = [fp.A for fp in best.fold_predictions]
            pooled_Y = [ds.y_norm[fp.rows] for fp in best.fold_predictions]
            pooled_P = [fp.P for fp in best.fold_predictions]
            pooled_rows = [fp.rows for fp in best.fold_predictions]
            if holdout_full is not None:
                pooled_A.append(holdout_full.A)
                pooled_Y.append(ds.y_norm[holdout_full.rows])
                pooled_P.append(holdout_full.P)
                pooled_rows.append(holdout_full.rows)
            calibration = self._calibrator().fit(np.concatenate(pooled_A), np.concatenate(pooled_Y))
            exec_final = self._fit_exec_model(ds, np.concatenate(pooled_rows), calibration.predict(np.concatenate(pooled_A)),
                                              np.concatenate(pooled_P))
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
                     "d_full": d_full, "d_production": d_prod, "feature_families": list(ds.families),
                     "components": comp_prod.to_dict() if comp_prod is not None else None,
                     "components_digest": comp_prod.digest() if comp_prod is not None else None,
                     "execution_model": exec_final.to_dict() if exec_final is not None else None,
                     "data_tier": window.data_tier()}
            meta = self._metadata(ds_prod, params, full_names, validation_summary, direction, False, extra,
                                  reg.effective_params())
            model = CombinedModel(reg, direction, calibration, full_names, d_prod, self.horizon, meta,
                                  components=comp_prod, execution_model=exec_final, families=ds.families)
            baseline_model = None
            if self.fit_final_baseline:
                # Same protocol for the conventional-feature baseline (research ablation, section 9).
                Xb = ds_prod.columns(base_names)
                reg_b, dir_b = self._fit_pair(Xb, ds_prod.y_norm, ds_prod.y_raw, baseline.params, base_names,
                                              extra=self._pooled(window, ds_prod, all_rows, d_prod, base_names, Xb))
                pooled_Ab = [fp.A for fp in baseline.fold_predictions]
                pooled_Yb = [ds.y_norm[fp.rows] for fp in baseline.fold_predictions]
                if holdout_base is not None:
                    pooled_Ab.append(holdout_base.A)
                    pooled_Yb.append(ds.y_norm[holdout_base.rows])
                cal_b = self._calibrator().fit(np.concatenate(pooled_Ab), np.concatenate(pooled_Yb))
                meta_b = self._metadata(ds_prod, self._params(baseline.params), base_names,
                                        {"aggregate": baseline.aggregate.to_dict(), "fold_scores": baseline.fold_scores,
                                         "holdout": holdout_base.metrics.to_dict() if holdout_base else None},
                                        dir_b, True, {"d_production": d_prod}, reg_b.effective_params())
                pooled_Pb = [fp.P for fp in baseline.fold_predictions]
                pooled_rb = [fp.rows for fp in baseline.fold_predictions]
                if holdout_base is not None:
                    pooled_Pb.append(holdout_base.P)
                    pooled_rb.append(holdout_base.rows)
                exec_b = self._fit_exec_model(ds, np.concatenate(pooled_rb), cal_b.predict(np.concatenate(pooled_Ab)),
                                              np.concatenate(pooled_Pb))
                baseline_model = CombinedModel(reg_b, dir_b, cal_b, base_names, d_prod, self.horizon, meta_b,
                                               components=comp_prod, execution_model=exec_b,
                                               families=tuple(f for f in ds.families if f != "FRACTIONAL"))
            holdout_span = None
            prod_cutoff = window[min(self._last_label_bar(ds_prod, len(ds_prod) - 1), len(window) - 1)].close_time
            panel_ref = self._panel_for(window, d_prod, full_names, prod_cutoff)
            panel_diag: dict[str, Any] = {
                "enabled": bool(self.panel_enabled), "symbols": list(panel_ref.symbols) if panel_ref else [],
                "pooled_rows": len(panel_ref) if panel_ref else 0,
                "rows_in_final_fit": int(len(pooled_prod[1])) if pooled_prod is not None else 0,
                "primary_rows_in_final_fit": int(len(ds_prod)), "aligned": bool(self.panel_align),
            }
            if holdout_full is not None and holdout_full.panel:
                panel_diag["out_of_sample"] = holdout_full.panel
                p = holdout_full.panel
                head = f"panel out of sample: {p['rows']} rows across {len(p.get('symbols', {}))} instruments, "
                _log(head + ("no forecast (the calibrator is constant on them too)" if p.get("forecast_constant")
                             else f"correlation {p['correlation']:+.4f}, accuracy {p['accuracy']:.3f}"))
            if panel_diag["pooled_rows"]:
                _log(f"training panel: {panel_diag['rows_in_final_fit']} pooled rows from "
                     f"{len(panel_diag['symbols'])} instruments joined the final fit of "
                     f"{panel_diag['primary_rows_in_final_fit']} primary rows")
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
                                  holdout_rows=int(len(holdout_rows)), holdout_span=holdout_span, d_full=d_full,
                                  baseline_model=baseline_model,
                                  holdout_diagnosis=dict(holdout_full.diagnosis) if holdout_full else
                                  self.forecast_diagnosis(ds, np.concatenate([fp.rows for fp in best.fold_predictions]),
                                                          np.concatenate(best.fold_E), best.aggregate),
                                  panel_diagnosis=panel_diag)
        except Exception as exc:
            return TrainingReport(None, False, None, stationarity, {}, [], [], [], None, None, float("nan"), 0,
                                  window[0].timestamp, window[-1].timestamp, time.time() - t0,
                                  error=f"{type(exc).__name__}: {exc}")
        finally:
            # The live feature engine keeps its previous d until a model is promoted (atomic swap by the bot).
            self.fe.set_adaptive_d(previous_d)


__all__ = ["ModelTrainer", "TrainingReport", "CandidateEvaluation", "FoldPrediction", "FoldSet", "HoldoutEvaluation",
           "software_version", "environment_info", "git_commit", "fitted_model_hash"]
