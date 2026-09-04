"""TradingBot: the production event loop and state machine (spec sections 44, 57-58).

    on_bar(bar):
        fills for orders queued at the previous bar (at this bar's open)
        validate -> store -> (retrain schedule) -> features -> prediction
        -> signal -> risk -> order queued for the next bar -> audit record
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import FrozenConfig
from .data.calendar import SessionCalendar
from .data.store import BarStore
from .data.validator import DataValidator, ValidationResult
from .diagnostics.attribution import attribution_groups
from .diagnostics.fractional_analysis import FractionalDiagnostics
from .execution.simulator import AlpacaPaperBroker, ExecutionEngine
from .features.engine import FeatureEngine
from .fractional.engine import FractionalEngine
from .logging.audit import AuditLogger
from .models.registry import ModelRegistry
from .portfolio.ledger import PortfolioLedger
from .portfolio.metrics import compute_metrics
from .risk.manager import RiskEngine
from .strategy.signal import SignalEngine
from .training.trainer import ModelTrainer, TrainingReport
from .types import Bar, BotState, FeatureVector, Order


class TradingBot:
    def __init__(self, cfg: FrozenConfig, run_id: str | None = None, artifacts_dir: str | Path | None = None,
                 broker: AlpacaPaperBroker | None = None, log: Callable[[str], None] | None = print,
                 async_retrain: bool = False):
        self.cfg = cfg
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        root = Path(artifacts_dir) if artifacts_dir is not None else Path(cfg.paths.artifacts_dir)
        self.artifacts_dir = root
        self.log = log or (lambda *_: None)
        self.instrument = cfg.market.instrument
        self.calendar = SessionCalendar.from_config(cfg)
        self.validator = DataValidator.from_config(cfg, self.calendar)
        self.store = BarStore(self.instrument, int(cfg.market.bar_minutes))
        self.fractional = FractionalEngine.from_config(cfg)
        self.feature_engine = FeatureEngine(cfg, self.fractional, self.calendar)
        self.registry = ModelRegistry(root / "models")
        self.execution = ExecutionEngine.from_config(cfg, broker)
        # The trainer works on its own feature engine so the live one is never mutated mid-bar.
        self.trainer = ModelTrainer(cfg, FeatureEngine(cfg, self.fractional, self.calendar), self.fractional,
                                    self.execution.cost_model)
        self.signal_engine = SignalEngine.from_config(cfg)
        self.risk = RiskEngine.from_config(cfg)
        self.ledger = PortfolioLedger(float(cfg.portfolio.initial_capital), self.instrument)
        self.audit = AuditLogger(root / "audit", self.run_id, echo=self.log)
        self.diagnostics = FractionalDiagnostics(root / "diagnostics")
        self.state = BotState.INITIALIZING
        self.bars_since_retrain = 0
        self.clean_bars_since_halt = 0
        self.retrain_count = 0
        self.last_report: Optional[TrainingReport] = None
        self.reports: list[dict] = []
        self.trade_contexts: list[dict] = []
        self._entry_context: Optional[dict] = None
        self._pending_signal_bar: Optional[Bar] = None
        self.async_retrain = async_retrain
        self._executor = ThreadPoolExecutor(max_workers=1) if async_retrain else None
        self._retrain_future: Optional[Future] = None
        self._lock = threading.Lock()
        self.minimum_bars = int(cfg.training.minimum_bars)
        self.retrain_every = int(cfg.training.retrain_every_bars)
        self.halt_recovery_bars = int(cfg.data.halt_recovery_bars)
        self.processed = 0
        self.rejected_bars = 0      # corrupt bars discarded
        self.halted_bars = 0        # stored bars that triggered a data halt (gap / extreme jump)

    # ------------------------------------------------------------ helpers
    def _set_state(self, new_state: BotState, reason: str = "") -> None:
        if new_state != self.state:
            self.audit.event("STATE", **{"from": self.state.value, "to": new_state.value, "reason": reason,
                                         "timestamp": self.store.last().timestamp.isoformat() if len(self.store) else None})
            self.state = new_state

    def _refresh_state(self) -> None:
        if self.state == BotState.INITIALIZING:
            return
        self._set_state(self.risk.state_for(self.ledger.exposure), "refresh")

    @property
    def model(self):
        return self.registry.current

    # ---------------------------------------------------------------- fills
    def on_next_bar_open(self, bar: Bar) -> None:
        """Execute orders queued at the previous bar at this bar's open (spec section 57)."""
        for order in self.execution.pending_orders():
            fill = self.execution.simulate_fill(order, bar, self._pending_signal_bar)
            prev_trades = len(self.ledger.trades)
            self.ledger.apply(fill)
            self.audit.record_fill(fill, self.ledger.state())
            if fill.new_entry and self._entry_context is not None:
                self._entry_context["entry_time"] = fill.fill_timestamp.isoformat()
            if len(self.ledger.trades) > prev_trades:
                trade = self.ledger.trades[-1]
                ctx = dict(self._entry_context or {})
                ctx["pnl"] = trade.realized_pnl - trade.costs
                self.trade_contexts.append(ctx)
        self._pending_signal_bar = None

    # ------------------------------------------------------------- retrain
    def _maybe_apply_retrain(self) -> None:
        if self._retrain_future is not None and self._retrain_future.done():
            report = self._retrain_future.result()
            self._retrain_future = None
            self._apply_report(report)

    def retrain(self, blocking: bool = True) -> Optional[TrainingReport]:
        self.bars_since_retrain = 0
        if self.async_retrain and not blocking:
            if self._retrain_future is None:
                snapshot = self.store.slice(0, len(self.store))
                self._retrain_future = self._executor.submit(self.trainer.retrain, snapshot, self.log)
                self.audit.event("RETRAIN_STARTED", bars=len(self.store), mode="async")
            return None
        self.audit.event("RETRAIN_STARTED", bars=len(self.store), mode="sync")
        report = self.trainer.retrain(self.store, self.log)
        self._apply_report(report)
        return report

    def _apply_report(self, report: TrainingReport) -> None:
        self.retrain_count += 1
        self.last_report = report
        ts = self.store.last().timestamp if len(self.store) else datetime.now(timezone.utc)
        summary = report.to_dict()
        summary["retrain_index"] = self.retrain_count
        summary["timestamp"] = ts.isoformat()
        self.reports.append(summary)
        if report.error:
            self.audit.event("RETRAIN_FAILED", error=report.error, elapsed=round(report.elapsed_seconds, 1))
            return
        if report.stationarity is not None:
            self.diagnostics.record_stationarity(report.stationarity, ts)
        self.diagnostics.record_contribution(ts, report.full_score, report.baseline_score, report.delta_score,
                                             report.accepted, report.stationarity.d_star if report.stationarity else math.nan)
        self.risk.record_retrain(report.accepted, report.delta_score)
        for ev in self.risk.events:
            self.audit.event(ev["event"], **{k: v for k, v in ev.items() if k != "event"})
        self.risk.events.clear()
        if report.accepted and report.model is not None:
            with self._lock:
                self.registry.promote(report.model)
                self.feature_engine.set_adaptive_d(report.model.d_star)
            self.audit.event("MODEL_PROMOTED", model_id=report.model.version, d_star=report.model.d_star,
                             delta_score=round(report.delta_score, 4), full_score=round(report.full_score, 4),
                             baseline_score=round(report.baseline_score, 4), elapsed=round(report.elapsed_seconds, 1))
        else:
            self.audit.event("MODEL_REJECTED", reasons=list(report.acceptance.reasons) if report.acceptance else [],
                             delta_score=round(report.delta_score, 4) if math.isfinite(report.delta_score) else None,
                             elapsed=round(report.elapsed_seconds, 1))
        with open(self.artifacts_dir / "audit" / f"{self.run_id}_retrains.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, default=str) + "\n")
        self._refresh_state()

    # ------------------------------------------------------------- on_bar
    def on_bar(self, bar: Bar) -> None:
        self.processed += 1
        result: ValidationResult = self.validator.validate(bar)
        if not result.ok:
            self._handle_bad_bar(bar, result)
            if not result.storable:
                return
        else:
            if self.risk.data_halted:
                self.clean_bars_since_halt += 1
                if self.clean_bars_since_halt >= self.halt_recovery_bars:
                    self.risk.set_data_halt(False)
                    self.audit.event("DATA_HALT_CLEARED", clean_bars=self.clean_bars_since_halt)
        # 1. fills for orders queued at the previous bar (at this bar's open)
        if self.execution.has_pending():
            self.on_next_bar_open(bar)
        # 2. store + mark
        self.store.append(bar)
        self.ledger.mark(bar.close_time, bar.close, len(self.store) - 1, self.calendar.session_date(bar.timestamp))
        self.bars_since_retrain += 1
        self._maybe_apply_retrain()
        # 3. initial training / retraining schedule
        if self.model is None:
            if len(self.store) >= self.minimum_bars and (self.bars_since_retrain >= self.retrain_every or self.retrain_count == 0):
                self.retrain(blocking=not self.async_retrain)
            if self.model is None:
                self._record(bar, validation=result, note="no accepted model")
                return
            self._set_state(self.risk.state_for(self.ledger.exposure), "initial model accepted")
        elif self.bars_since_retrain >= self.retrain_every:
            self.retrain(blocking=not self.async_retrain)
        if self.state == BotState.INITIALIZING:
            self._set_state(self.risk.state_for(self.ledger.exposure), "model available")
        # 4. features
        if not self.feature_engine.ready(self.store):
            self._record(bar, validation=result, note="feature engine warming up")
            return
        with self._lock:
            features = self.feature_engine.compute_latest(self.store)
            model = self.model
        needed = list(model.feature_names) + ["sigma_h", "range_rel", "close"]
        if not features.is_finite(needed):
            bad = [n for n in needed if not math.isfinite(features.get(n))]
            self.risk.set_data_halt(True, f"non-finite features: {bad[:5]}")
            self.clean_bars_since_halt = 0
            self.audit.event("DATA_HALT", reason="feature NaN/inf", features=bad[:10])
            self._flatten_if_positioned(bar, features, "DATA_HALT_FLATTEN")
            self._refresh_state()
            self._record(bar, features=features, validation=result, note="non-finite features")
            return
        self.signal_engine.observe_sigma(features.get("sigma_h"))
        # 5. prediction -> signal -> risk
        prediction = model.predict(features)
        cost = self.execution.estimate_cost(features)
        signal = self.signal_engine.create(prediction, features, cost)
        decision = self.risk.evaluate(signal, self.ledger.state(), features, self.calendar.session_date(bar.timestamp))
        for ev in self.risk.events:
            self.audit.event(ev["event"], **{k: v for k, v in ev.items() if k != "event"})
        self.risk.events.clear()
        self._refresh_state()
        # 6. order
        order = self.execution.build_order(
            self.instrument, bar.close_time, self.ledger.units, self.ledger.exposure, decision.approved_exposure,
            self.ledger.equity, bar.close, self.state, reason=decision.reason, new_entry=decision.new_entry,
            entry_sigma=features.get("sigma_h"))
        if order is not None:
            self.execution.queue_for_next_bar(order)
            self._pending_signal_bar = bar
            if decision.new_entry:
                self._entry_context = {"vol_state": features.get("volatility_state"),
                                       "minutes_since_open": self.calendar.minutes_since_open(bar.timestamp),
                                       "session_minutes": self.calendar.session_minutes,
                                       "fractional_d": features.fractional_d, "direction": signal.direction,
                                       "signal_time": bar.close_time.isoformat()}
        self._record(bar, features, prediction, signal, decision, order, cost, validation=result)

    def _flatten_if_positioned(self, bar: Bar, features: Optional[FeatureVector], reason: str) -> None:
        if self.ledger.units == 0:
            return
        order = self.execution.build_order(self.instrument, bar.close_time, self.ledger.units, self.ledger.exposure, 0.0,
                                           self.ledger.equity, bar.close, self.risk.state_for(self.ledger.exposure),
                                           reason=reason)
        if order is not None:
            self.execution.queue_for_next_bar(order)
            self._pending_signal_bar = bar

    def _handle_bad_bar(self, bar: Bar, result: ValidationResult) -> None:
        if result.storable:
            self.halted_bars += 1
        else:
            self.rejected_bars += 1
        self.clean_bars_since_halt = 0
        if not self.risk.data_halted:
            self.risk.set_data_halt(True, ";".join(result.reasons))
            self.audit.event("DATA_HALT", reasons=list(result.reasons), timestamp=bar.timestamp.isoformat(),
                             stored=result.storable)
        if not result.storable:
            # Corrupt input: cancel anything queued (bad input produces no trade) and flatten at the next clean bar.
            cancelled = self.execution.pending_orders()
            if cancelled:
                self.audit.event("ORDERS_CANCELLED", count=len(cancelled), reason="rejected bar")
            self._refresh_state()
            self._record(bar, validation=result, note="rejected bar")
        self._refresh_state()
        if self.ledger.units != 0 and not self.execution.has_pending():
            self._flatten_if_positioned(bar, None, "DATA_HALT_FLATTEN")

    def _record(self, bar: Bar, features=None, prediction=None, signal=None, decision=None, order=None, cost=None,
                validation=None, note: str = "") -> None:
        self.audit.record(bar, self.state.value, features, prediction, signal, decision, order, cost,
                          self.ledger.state(), self.registry.current_version,
                          validation.to_dict() if validation else None,
                          {"note": note, "bars_since_retrain": self.bars_since_retrain,
                           "risk_halt": self.risk.halt_reason(self.calendar.session_date(bar.timestamp))})

    # ---------------------------------------------------------------- run
    def run(self, feed: Iterable[Bar], max_bars: int | None = None) -> dict:
        t0 = time.time()
        for i, bar in enumerate(feed):
            if max_bars is not None and i >= max_bars:
                break
            self.on_bar(bar)
        return self.finalize(elapsed=time.time() - t0)

    def finalize(self, elapsed: float = math.nan) -> dict:
        if self._retrain_future is not None:
            self._apply_report(self._retrain_future.result())
            self._retrain_future = None
        groups = attribution_groups(self.trade_contexts)
        metrics = compute_metrics(self.ledger, int(self.cfg.market.bars_per_day),
                                  int(self.cfg.market.trading_days_per_year), groups)
        summary = {
            "run_id": self.run_id, "instrument": self.instrument, "bars_processed": self.processed,
            "bars_stored": len(self.store), "rejected_bars": self.rejected_bars, "halted_bars": self.halted_bars,
            "final_state": self.state.value,
            "retrains": self.retrain_count, "model_version": self.registry.current_version,
            "model_history": self.registry.history, "elapsed_seconds": elapsed,
            "ledger": self.ledger.summary(), "metrics": metrics.to_dict(),
            "risk": {"drawdown_halted": self.risk.drawdown_halted, "ablation_halted": self.risk.ablation_halted,
                     "ablation_failures": self.risk.ablation_failures, "data_halted": self.risk.data_halted},
            "fractional_contribution": self.diagnostics.contribution,
        }
        (self.artifacts_dir / "audit").mkdir(parents=True, exist_ok=True)
        with open(self.artifacts_dir / "audit" / f"{self.run_id}_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(self.artifacts_dir / "audit" / f"{self.run_id}_trades.jsonl", "w", encoding="utf-8") as fh:
            for t in self.ledger.trades:
                fh.write(json.dumps(t.to_dict(), default=str) + "\n")
        self.diagnostics.save_json()
        self.diagnostics.plot()
        self.audit.event("RUN_COMPLETE", equity=round(self.ledger.equity, 2), trades=len(self.ledger.trades),
                         net_return=round(metrics.net_return, 5), sharpe=round(metrics.sharpe, 3))
        self.audit.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        return summary


__all__ = ["TradingBot"]
