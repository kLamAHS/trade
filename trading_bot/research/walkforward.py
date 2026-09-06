"""Rolling walk-forward OOS evaluation (research spec sections 4-9, 13, 28).

For every window the production trainer is run on the training block exactly as the live bot
would run it (nested d* / hyper-parameter / calibration selection inside the block, untouched
outer holdout for acceptance), then the *fitted* models -- the full model and the conventional
ablation baseline -- forecast the following unseen OOS block.  The OOS blocks are concatenated
into one continuous decision series that is traded by ``simulate_strategy`` with the live
position rules and portfolio circuit breakers, so the equity curve is the one a bot that
retrained on that schedule would have produced.

Timestamps carried per decision row (section 3):

    feature_available_at  newest information time of the bars used by the features
    decision_at           the bar close at which the forecast is made (== feature timestamp)
    execution_at          the next bar's open, where the resulting order fills

The bar close and the next bar's open share the boundary timestamp (a 30-minute bar closing at
18:30 is followed by the open print at 18:30:00 or later), so the no-lookahead invariant is
``feature_available_at <= decision_at <= execution_at`` with the decision using only bars that
closed at or before ``decision_at``; the runner refuses rows that violate it.  Timing latency
beyond the boundary is exercised separately by the +1/+2 bar delay stress test (section 22).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from ..config import FrozenConfig
from ..data.calendar import SessionCalendar
from ..data.store import BarStore
from ..execution.cost_model import CostModel
from ..features.engine import FeatureEngine
from ..fractional.engine import FractionalEngine
from ..models.combined import CombinedModel
from ..training.trainer import ModelTrainer, TrainingReport, fitted_model_hash
from ..training.validation import SimulationParams
from .metrics import StrategyMetrics, _annualise, compute_strategy_metrics, drawdown_stats
from .simulate import SimInputs, SimResult, simulate_strategy
from .windows import WalkForwardSchedule, Window, build_schedule

VARIANTS = ("full", "baseline", "production")


@dataclass
class WindowResult:
    window: Window
    model_id: Optional[str]
    baseline_model_id: Optional[str]
    deployed_model_id: Optional[str]
    accepted: bool
    error: Optional[str]
    d_star: float
    d_full: float
    fold_d_stars: list[float]
    best_params: dict[str, Any]
    baseline_params: dict[str, Any]
    holdout_delta: float
    holdout_score: float
    baseline_holdout_score: float
    elapsed_seconds: float
    n_training_rows: int
    n_oos_rows: int
    train_span: tuple[Optional[str], Optional[str]]
    oos_span: tuple[Optional[str], Optional[str]]
    fitted_model_hash: Optional[str]
    baseline_fitted_model_hash: Optional[str]
    carried: bool                       # OOS forecast by a model carried over from an earlier window
    report: Optional[dict[str, Any]] = None
    model: Optional[CombinedModel] = field(default=None, repr=False)
    baseline_model: Optional[CombinedModel] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("report", "model", "baseline_model")}
        d["window"] = self.window.to_dict()
        return d


@dataclass
class OOSSeries:
    """Concatenated OOS decision rows of a walk-forward pass."""

    bar_index: np.ndarray
    window_ids: np.ndarray
    session_ids: np.ndarray
    model_ids: list[str]
    decision_at: list[datetime]
    feature_available_at: list[datetime]
    execution_at: list[datetime]
    y_norm: np.ndarray
    y_raw: np.ndarray
    sigma: np.ndarray
    sigma_ref: np.ndarray
    cost_roundtrip: np.ndarray
    cost_side_exec: np.ndarray
    log_close: np.ndarray
    open_next: np.ndarray
    open_next2: np.ndarray
    forecasts: dict[str, dict[str, np.ndarray]]      # variant -> {"E", "M", "P"}

    def __len__(self) -> int:
        return len(self.bar_index)

    def sim_inputs(self, variant: str = "full", E: np.ndarray | None = None) -> SimInputs:
        f = self.forecasts[variant]
        return SimInputs(E=np.asarray(f["E"] if E is None else E, dtype=float), y_norm=self.y_norm, sigma=self.sigma,
                         sigma_ref=self.sigma_ref, cost_roundtrip=self.cost_roundtrip, cost_side_exec=self.cost_side_exec,
                         log_close=self.log_close, open_next=self.open_next, open_next2=self.open_next2,
                         M=f["M"], P=f["P"], session_ids=self.session_ids, window_ids=self.window_ids,
                         timestamps=list(self.decision_at), exec_timestamps=list(self.execution_at),
                         model_ids=list(self.model_ids))

    def subset(self, mask: np.ndarray) -> "OOSSeries":
        idx = np.flatnonzero(mask)
        pick = lambda seq: [seq[i] for i in idx]  # noqa: E731
        return OOSSeries(self.bar_index[idx], self.window_ids[idx], self.session_ids[idx], pick(self.model_ids),
                         pick(self.decision_at), pick(self.feature_available_at), pick(self.execution_at),
                         self.y_norm[idx], self.y_raw[idx], self.sigma[idx], self.sigma_ref[idx], self.cost_roundtrip[idx],
                         self.cost_side_exec[idx], self.log_close[idx], self.open_next[idx], self.open_next2[idx],
                         {v: {k: a[idx] for k, a in f.items()} for v, f in self.forecasts.items()})

    @staticmethod
    def concat(parts: list["OOSSeries"]) -> "OOSSeries":
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("no OOS rows")
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
        lst = lambda name: [x for p in parts for x in getattr(p, name)]  # noqa: E731
        variants = parts[0].forecasts.keys()
        return OOSSeries(cat("bar_index"), cat("window_ids"), cat("session_ids"), lst("model_ids"), lst("decision_at"),
                         lst("feature_available_at"), lst("execution_at"), cat("y_norm"), cat("y_raw"), cat("sigma"),
                         cat("sigma_ref"), cat("cost_roundtrip"), cat("cost_side_exec"), cat("log_close"), cat("open_next"),
                         cat("open_next2"),
                         {v: {k: np.concatenate([p.forecasts[v][k] for p in parts]) for k in ("E", "M", "P")} for v in variants})


@dataclass
class WalkForwardResult:
    label: str                                   # "development" | "holdout"
    windows: list[WindowResult]
    oos: OOSSeries
    sims: dict[str, SimResult]
    metrics: dict[str, StrategyMetrics]
    per_window: list[dict[str, Any]]
    timestamp_audit: dict[str, Any]
    elapsed_seconds: float
    configurations_tested: int

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def summary(self) -> dict[str, Any]:
        return {"label": self.label, "n_windows": self.n_windows, "n_rows": len(self.oos),
                "accepted_windows": int(sum(w.accepted for w in self.windows)),
                "failed_windows": int(sum(w.error is not None for w in self.windows)),
                "span": [self.oos.decision_at[0].isoformat(), self.oos.decision_at[-1].isoformat()] if len(self.oos) else None,
                "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
                "per_window": self.per_window, "windows": [w.to_dict() for w in self.windows],
                "timestamp_audit": self.timestamp_audit, "elapsed_seconds": self.elapsed_seconds,
                "configurations_tested": self.configurations_tested}


def window_statistics(sim: SimResult, window_ids: np.ndarray, bars_per_year: int) -> list[dict[str, Any]]:
    """Per-window return / Sharpe / drawdown / trade count from a continuous simulation."""
    out: list[dict[str, Any]] = []
    entries = {}
    for t in sim.trades:
        entries.setdefault(int(window_ids[t["entry_row"]]), []).append(t["net_pnl"])
    for wid in np.unique(window_ids):
        rows = np.flatnonzero(window_ids == wid)
        r = sim.bar_pnl[rows]
        eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
        ann = _annualise(r, bars_per_year)
        tp = entries.get(int(wid), [])
        out.append({"window": int(wid), "bars": int(len(rows)), "net_return": float(eq[-1] - 1.0), "sharpe": ann["sharpe"],
                    "sortino": ann["sortino"], "max_drawdown": drawdown_stats(eq)["max_drawdown"],
                    "trades": int(len(tp)), "trade_pnl": float(np.sum(tp)) if tp else 0.0,
                    "time_invested": float(np.mean(sim.exposures[rows] != 0)), "cost": float(sim.cost_bar[rows].sum())})
    return out


class WalkForwardRunner:
    def __init__(self, cfg: FrozenConfig, store: BarStore, trainer: ModelTrainer | None = None,
                 log: Callable[[str], None] | None = None, deploy_policy: str | None = None):
        self.cfg = cfg
        self.store = store
        self.log = log or (lambda *_: None)
        self.calendar = SessionCalendar.from_config(cfg)
        r = cfg.get("research", {}) or {}
        wf = r.get("walkforward", {}) or {}
        self.deploy_policy = deploy_policy or str(wf.get("deploy_policy", "always"))
        if self.deploy_policy not in ("always", "production"):
            raise ValueError("research.walkforward.deploy_policy must be 'always' or 'production'")
        if trainer is None:
            fractional = FractionalEngine.from_config(cfg)
            fe = FeatureEngine(cfg, fractional, self.calendar)
            trainer = ModelTrainer(cfg, fe, fractional, CostModel.from_config(cfg), fit_final_baseline=True)
        trainer.fit_final_baseline = True
        self.trainer = trainer
        self.sim_params = SimulationParams.from_config(cfg)
        self.horizon = int(cfg.prediction.horizon_bars)
        self.bars_per_year = int(cfg.market.bars_per_day) * int(cfg.market.trading_days_per_year)
        sim = r.get("simulation", {}) or {}
        self.apply_halts = bool(sim.get("portfolio_halts", True))
        self.capital = float(cfg.portfolio.initial_capital)
        self._train_bars = int(wf.get("train_bars") or cfg.training.window_bars)
        self._first_train_bars = int(wf.get("first_train_bars") or cfg.training.minimum_bars)
        self._oos_bars = int(wf.get("oos_bars") or cfg.training.retrain_every_bars)
        self._step_bars = int(wf.get("step_bars") or self._oos_bars)
        self._expanding = bool(wf.get("expanding", False))
        self._min_oos_bars = wf.get("min_oos_bars")
        hold = r.get("holdout", {}) or {}
        self.holdout_fraction = float(hold.get("fraction", 0.15))
        # newest information time of everything up to and including each bar (FeatureEngine.vector_from_matrix)
        src = [b.latest_source_time for b in store.bars]
        self._source_time_upto = list(src)
        for i in range(1, len(src)):
            if self._source_time_upto[i - 1] > self._source_time_upto[i]:
                self._source_time_upto[i] = self._source_time_upto[i - 1]

    # ------------------------------------------------------------ schedule
    def schedule(self) -> WalkForwardSchedule:
        return build_schedule(len(self.store), self._train_bars, self._oos_bars, self._step_bars,
                              holdout_fraction=self.holdout_fraction, expanding=self._expanding,
                              min_oos_bars=self._min_oos_bars, first_train_bars=self._first_train_bars)

    def holdout_schedule(self) -> WalkForwardSchedule:
        sched = self.schedule()
        if sched.holdout_start is None:
            raise ValueError("no holdout is configured (research.holdout.fraction = 0)")
        return build_schedule(len(self.store), self._train_bars, self._oos_bars, self._step_bars,
                              holdout_start=sched.holdout_start, expanding=self._expanding,
                              min_oos_bars=self._min_oos_bars, span=(sched.holdout_start, len(self.store)))

    # ------------------------------------------------------------- windows
    def _lead_bars(self, d: float) -> int:
        fe = self.trainer.fe
        prev = fe.adaptive_d
        try:
            fe.set_adaptive_d(float(d))
            lead = fe.required_history
        finally:
            fe.set_adaptive_d(prev)
        return int(lead + self.trainer.builder.vol_reference_bars + 5)

    def _oos_dataset(self, w: Window, d: float, cache: dict):
        """Dataset over the OOS block (plus warm-up history before it and the label bars after it)
        built with the feature definition ``d``; returns (dataset, row mask of the OOS block, offset)."""
        key = round(float(d), 10)
        if key in cache:
            return cache[key]
        start = max(0, w.train_end - self._lead_bars(d))
        end = min(len(self.store), w.oos_end + self.horizon + 1)
        ext = self.store.slice(start, end)
        ds = self.trainer.builder.build(ext, float(d))
        global_idx = ds.bar_index + start
        mask = (global_idx >= w.train_end) & (global_idx < w.oos_end)
        cache[key] = (ds, mask, start)
        return cache[key]

    def _forecast(self, model: CombinedModel, ds, mask: np.ndarray) -> dict[str, np.ndarray]:
        X = ds.columns(model.feature_names)[mask]
        out = model.predict_arrays(X)
        return {"E": out["E"], "M": out["M"], "P": out["P"]}

    def _series_for_window(self, w: Window, ds, mask: np.ndarray, offset: int, forecasts: dict[str, dict[str, np.ndarray]],
                           model_id: str) -> OOSSeries:
        rows = np.flatnonzero(mask)
        bar_idx = ds.bar_index[rows] + offset
        decision_at = [ds.close_times[i] for i in rows]
        feature_available_at = [self._source_time_upto[int(g)] for g in bar_idx]
        execution_at = [self.store[int(g) + 1].timestamp for g in bar_idx]
        for fa, da, ea in zip(feature_available_at, decision_at, execution_at):
            if fa > da:
                raise ValueError(f"look-ahead: feature source {fa} newer than decision {da}")
            if ea < da:
                raise ValueError(f"look-ahead: execution {ea} before decision {da}")
        return OOSSeries(bar_idx, np.full(len(rows), w.index, dtype=int), np.zeros(len(rows), dtype=int),
                         [model_id] * len(rows), decision_at, feature_available_at, execution_at,
                         ds.y_norm[rows], ds.y_raw[rows], ds.sigma[rows], ds.sigma_ref[rows], ds.cost_roundtrip[rows],
                         ds.cost_side_exec[rows], ds.log_close[rows], ds.open_next[rows], ds.open_next2[rows], forecasts)

    def _flat_forecast(self, n: int) -> dict[str, np.ndarray]:
        return {"E": np.zeros(n), "M": np.zeros(n), "P": np.full(n, 0.5)}

    def run_windows(self, windows: list[Window] | tuple[Window, ...], label: str = "development",
                    carry: tuple[Optional[CombinedModel], Optional[CombinedModel], Optional[CombinedModel]] | None = None,
                    on_window: Callable[[WindowResult], None] | None = None) -> WalkForwardResult:
        t0 = time.time()
        results: list[WindowResult] = []
        parts: list[OOSSeries] = []
        model_prev, base_prev, deployed = carry or (None, None, None)
        for w in windows:
            tw = time.time()
            history = self.store.slice(w.train_start, w.train_end)
            self.log(f"[{label}] window {w.index}: train bars {w.train_start}-{w.train_end}, OOS {w.train_end}-{w.oos_end}")
            report: TrainingReport = self.trainer.retrain(history, self.log)
            carried = report.model is None
            model = report.model if report.model is not None else model_prev
            base = report.baseline_model if report.baseline_model is not None else base_prev
            if self.deploy_policy == "always":
                deployed = model
            elif report.model is not None and report.accepted:
                deployed = report.model
            # OOS forecasts with the fitted models (features rebuilt with each model's own d)
            cache: dict = {}
            ref_d = model.d_star if model is not None else self.trainer.fe.adaptive_d
            ds, mask, offset = self._oos_dataset(w, ref_d, cache)
            n = int(mask.sum())
            forecasts = {"full": self._forecast(model, ds, mask) if model is not None else self._flat_forecast(n)}
            if base is not None:
                ds_b, mask_b, _ = self._oos_dataset(w, base.d_star, cache)
                forecasts["baseline"] = self._forecast(base, ds_b, mask_b)
            else:
                forecasts["baseline"] = self._flat_forecast(n)
            if deployed is None:
                forecasts["production"] = self._flat_forecast(n)
            elif deployed is model:
                forecasts["production"] = {k: v.copy() for k, v in forecasts["full"].items()}
            else:
                ds_p, mask_p, _ = self._oos_dataset(w, deployed.d_star, cache)
                forecasts["production"] = self._forecast(deployed, ds_p, mask_p)
            model_id = model.version if model is not None else "none"
            series = self._series_for_window(w, ds, mask, offset, forecasts, model_id)
            parts.append(series)
            rep = report.to_dict()
            rep.pop("grid_results", None)
            wr = WindowResult(
                window=w, model_id=report.model.version if report.model else None,
                baseline_model_id=report.baseline_model.version if report.baseline_model else None,
                deployed_model_id=deployed.version if deployed is not None else None,
                accepted=bool(report.accepted), error=report.error,
                d_star=float(model.d_star) if model is not None else float("nan"), d_full=float(report.d_full),
                fold_d_stars=list(report.fold_d_stars), best_params=dict(report.best_params),
                baseline_params=dict(report.baseline_params), holdout_delta=float(report.delta_score),
                holdout_score=float(report.full_score) if report.model else float("nan"),
                baseline_holdout_score=float(report.baseline_score) if report.model else float("nan"),
                elapsed_seconds=time.time() - tw, n_training_rows=int(report.n_rows), n_oos_rows=n,
                train_span=(self.store[w.train_start].timestamp.isoformat(), self.store[w.train_end - 1].timestamp.isoformat()),
                oos_span=(series.decision_at[0].isoformat() if n else None, series.decision_at[-1].isoformat() if n else None),
                fitted_model_hash=fitted_model_hash(report.model) if report.model else None,
                baseline_fitted_model_hash=fitted_model_hash(report.baseline_model) if report.baseline_model else None,
                carried=carried, report=rep, model=report.model, baseline_model=report.baseline_model)
            results.append(wr)
            self.log(f"[{label}] window {w.index}: model={wr.model_id} accepted={wr.accepted} d*={wr.d_star:.2f} "
                     f"holdout Δ={wr.holdout_delta:+.3f} oos rows={n} ({wr.elapsed_seconds:.1f}s)"
                     + (f" ERROR {wr.error}" if wr.error else ""))
            if on_window:
                on_window(wr)
            model_prev, base_prev = model, base
        oos = OOSSeries.concat(parts)
        oos.session_ids[:] = self._session_ids(oos.decision_at)
        sims = {v: self.simulate(oos, v) for v in VARIANTS}
        metrics = {v: self.metrics(oos, s, v) for v, s in sims.items()}
        per_window = self._per_window(sims, oos.window_ids, results)
        gaps = [(e - d).total_seconds() / 60.0 for d, e in zip(oos.decision_at, oos.execution_at)]
        audit = {"rows": len(oos), "violations": 0,
                 "checked": ["feature_available_at <= decision_at", "decision_at <= execution_at (next open print)",
                             "labels use bars after execution_at only"],
                 "min_decision_to_execution_minutes": float(min(gaps)), "max_decision_to_execution_minutes": float(max(gaps))}
        n_grid = len(self.trainer.grid)
        return WalkForwardResult(label, results, oos, sims, metrics, per_window, audit, time.time() - t0,
                                 configurations_tested=2 * n_grid * len(results))

    def run(self, on_window: Callable[[WindowResult], None] | None = None) -> WalkForwardResult:
        sched = self.schedule()
        if not sched.windows:
            raise ValueError(f"no walk-forward window fits: {len(self.store)} bars, first train {self._first_train_bars}, "
                             f"OOS {self._oos_bars}, holdout {self.holdout_fraction:.0%}")
        return self.run_windows(sched.windows, "development", on_window=on_window)

    def run_holdout(self, development: WalkForwardResult | None = None,
                    on_window: Callable[[WindowResult], None] | None = None) -> WalkForwardResult:
        sched = self.holdout_schedule()
        if not sched.windows:
            raise ValueError("the locked holdout is too short for one OOS window")
        carry = None
        if development is not None and development.windows:
            last = development.windows[-1]
            carry = (last.model, last.baseline_model, None)
        return self.run_windows(sched.windows, "holdout", carry=carry, on_window=on_window)

    # ---------------------------------------------------------- simulation
    def simulate(self, oos: OOSSeries, variant: str = "full", E: np.ndarray | None = None, **overrides) -> SimResult:
        kw: dict[str, Any] = {"capital": self.capital, "model_ids": list(oos.model_ids)}
        if self.apply_halts:
            kw["daily_loss_limit"] = float(self.cfg.risk.daily_loss_limit)
            kw["drawdown_halt"] = float(self.cfg.risk.drawdown_halt)
        kw.update(overrides)
        params = kw.pop("params", self.sim_params)
        return simulate_strategy(oos.sim_inputs(variant, E), params, **kw)

    def metrics(self, oos: OOSSeries, sim: SimResult, variant: str | None = None, E: np.ndarray | None = None) -> StrategyMetrics:
        """Full metric set; forecast-quality metrics use the variant's forecasts (or ``E`` when given)."""
        if E is None and variant in oos.forecasts:
            E = oos.forecasts[variant]["E"]
        return compute_strategy_metrics(sim.bar_pnl, sim.equity, sim.exposures, sim.trades, sim.cost_bar,
                                        int(self.cfg.market.bars_per_day), int(self.cfg.market.trading_days_per_year),
                                        session_ids=oos.session_ids, timestamps=oos.decision_at, y_norm=oos.y_norm,
                                        E=E, capital=self.capital)

    def _per_window(self, sims: dict[str, SimResult], window_ids: np.ndarray, results: list[WindowResult]) -> list[dict[str, Any]]:
        stats = {v: {s["window"]: s for s in window_statistics(sim, window_ids, self.bars_per_year)} for v, sim in sims.items()}
        out = []
        for wr in results:
            k = wr.window.index
            row: dict[str, Any] = {"window": k, "model_id": wr.model_id, "accepted": wr.accepted, "d_star": wr.d_star,
                                   "holdout_delta": wr.holdout_delta, "oos_span": list(wr.oos_span)}
            for v in sims:
                s = stats[v].get(k)
                for key in ("net_return", "sharpe", "sortino", "max_drawdown", "trades", "time_invested", "cost", "bars"):
                    row[f"{v}_{key}"] = s[key] if s else None
            if stats["full"].get(k) and stats["baseline"].get(k):
                row["delta_sharpe"] = stats["full"][k]["sharpe"] - stats["baseline"][k]["sharpe"]
                row["delta_return"] = stats["full"][k]["net_return"] - stats["baseline"][k]["net_return"]
            out.append(row)
        return out

    def _session_ids(self, timestamps: list[datetime]) -> np.ndarray:
        ids = np.zeros(len(timestamps), dtype=int)
        last = None
        k = -1
        for i, ts in enumerate(timestamps):
            d = self.calendar.session_date(ts)
            if d != last:
                k += 1
                last = d
            ids[i] = k
        return ids

    # ----------------------------------------------------- reproducibility
    def reproduce_window(self, wr: WindowResult) -> dict[str, Any]:
        """Retrain one window from scratch and compare the fitted model and its OOS forecasts bit for bit."""
        w = wr.window
        report = self.trainer.retrain(self.store.slice(w.train_start, w.train_end))
        if report.model is None or wr.model is None:
            return {"window": w.index, "identical": report.model is None and wr.model is None,
                    "reason": "no model" if report.model is None else "reference had no model"}
        ds, mask, _ = self._oos_dataset(w, report.model.d_star, {})
        E_new = self._forecast(report.model, ds, mask)["E"]
        ds0, mask0, _ = self._oos_dataset(w, wr.model.d_star, {})
        E_old = self._forecast(wr.model, ds0, mask0)["E"]
        same_hash = fitted_model_hash(report.model) == wr.fitted_model_hash
        same_E = len(E_new) == len(E_old) and bool(np.array_equal(E_new, E_old))
        return {"window": w.index, "identical": bool(same_hash and same_E), "fitted_model_hash_match": bool(same_hash),
                "forecasts_match": same_E, "model_id_match": report.model.version == wr.model_id,
                "max_abs_forecast_difference": float(np.max(np.abs(E_new - E_old))) if len(E_new) == len(E_old) else None}


__all__ = ["WalkForwardRunner", "WalkForwardResult", "WindowResult", "OOSSeries", "window_statistics", "VARIANTS"]
