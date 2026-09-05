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

import numpy as np

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
from .strategy.sizing import direction_sign
from .training.trainer import ModelTrainer, TrainingReport
from .types import Bar, BotState, FeatureVector


class TradingBot:
    def __init__(self, cfg: FrozenConfig, run_id: str | None = None, artifacts_dir: str | Path | None = None,
                 broker: AlpacaPaperBroker | None = None, log: Callable[[str], None] | None = print,
                 async_retrain: bool = False, on_event: Callable[[dict], None] | None = None):
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
        self.execution = ExecutionEngine.from_config(cfg, broker, id_prefix=self.run_id)
        # The trainer works on its own feature engine so the live one is never mutated mid-bar.
        self.trainer = ModelTrainer(cfg, FeatureEngine(cfg, self.fractional, self.calendar), self.fractional,
                                    self.execution.cost_model)
        self.signal_engine = SignalEngine.from_config(cfg)
        self.risk = RiskEngine.from_config(cfg)
        self.ledger = PortfolioLedger(float(cfg.portfolio.initial_capital), self.instrument)
        self.audit = AuditLogger(root / "audit", self.run_id, echo=self.log, on_event=on_event)
        self.last_record: Optional[dict] = None
        self.diagnostics = FractionalDiagnostics(root / "diagnostics")
        self.vol_edges = tuple(float(x) for x in (cfg.get("diagnostics", {}) or {}).get("vol_regime_edges", (0.8, 1.2)))
        self.state = BotState.INITIALIZING
        self.bars_since_retrain = 0
        self.retrain_count = 0
        self.last_report: Optional[TrainingReport] = None
        self.reports: list[dict] = []
        self.trade_contexts: list[dict] = []
        self._pending_entry_context: Optional[dict] = None    # context of the entry decision awaiting its fill
        self._open_trade_context: Optional[dict] = None       # context of the trade currently open in the ledger
        self._pending_signal_bar: Optional[Bar] = None
        self.async_retrain = async_retrain
        self._executor = ThreadPoolExecutor(max_workers=1) if async_retrain else None
        self._retrain_future: Optional[Future] = None
        self._retrain_pending = False
        self._lock = threading.Lock()
        self.minimum_bars = int(cfg.training.minimum_bars)
        self.retrain_every = int(cfg.training.retrain_every_bars)
        self.vol_reference_bars = int(cfg.signal.vol_reference_days) * int(cfg.market.bars_per_day)
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

    def _flush_events(self) -> None:
        for ev in self.risk.events:
            self.audit.event(ev["event"], **{k: v for k, v in ev.items() if k != "event"})
        self.risk.events.clear()
        for ev in self.execution.events:
            self.audit.event(ev["event"], **{k: v for k, v in ev.items() if k != "event"})
        self.execution.events.clear()

    @property
    def model(self):
        return self.registry.current

    def _seed_sigma_history(self) -> None:
        """Seed the signal engine's sigma history from the store so live sigma_ref uses the same
        trailing window (section 28) as the validation simulator did."""
        if self.signal_engine.has_history or not self.feature_engine.ready(self.store):
            return
        n = len(self.store)
        start = max(0, n - (self.vol_reference_bars + self.feature_engine.required_history))
        with self._lock:
            fm = self.feature_engine.compute_matrix(self.store, start, n)
        sigma = fm.column("sigma_h")
        sigma = sigma[np.isfinite(sigma)][-self.vol_reference_bars:]
        self.signal_engine.seed_sigma_history(sigma.tolist())

    # ---------------------------------------------------------------- fills
    def on_next_bar_open(self, bar: Bar) -> None:
        """Execute orders queued at the previous bar at this bar's open (spec section 57).

        An order whose decision time is later than this bar's close (a quote that arrived after a
        delivery delay) is deferred to the first bar that was still open at decision time.
        """
        orders = self.execution.pending_orders()
        for k, order in enumerate(orders):
            if bar.close_time <= order.signal_timestamp:
                self.execution.queue.push(order)
                self.audit.event("ORDER_DEFERRED", order_id=order.order_id, signal_timestamp=order.signal_timestamp.isoformat(),
                                 bar_close=bar.close_time.isoformat())
                continue
            try:
                fill = self.execution.simulate_fill(order, bar, self._pending_signal_bar)
            except RuntimeError as exc:
                # Execution simulator unavailable (section 35): halt, keep the orders for the next bar.
                self.risk.set_data_halt(True, str(exc))
                self.audit.event("DATA_HALT", reason=str(exc), requeued=len(orders) - k)
                for o in orders[k:]:
                    self.execution.queue.push(o)
                self._refresh_state()
                return
            except ValueError as exc:
                self.execution.queue.push(order)
                self.audit.event("ORDER_DEFERRED", order_id=order.order_id, reason=str(exc))
                continue
            prev_trades = len(self.ledger.trades)
            prev_ctx = self._open_trade_context
            self.ledger.apply(fill)
            self.audit.record_fill(fill, self.ledger.state())
            if len(self.ledger.trades) > prev_trades:
                trade = self.ledger.trades[-1]
                ctx = dict(prev_ctx or {})
                ctx["pnl"] = trade.net_pnl
                ctx["exit_reason"] = trade.exit_reason
                self.trade_contexts.append(ctx)
            if fill.new_entry or (self.ledger.units != 0 and self._open_trade_context is None):
                self._open_trade_context = self._pending_entry_context
        if not self.execution.has_pending():
            self._pending_signal_bar = None
        self.execution.poll_mirrors()
        self._flush_events()

    # ------------------------------------------------------------- retrain
    def _maybe_apply_retrain(self) -> None:
        if self._retrain_future is not None and self._retrain_future.done():
            report = self._retrain_future.result()
            self._retrain_future = None
            self._apply_report(report)
            if self._retrain_pending:
                self._retrain_pending = False
                self.retrain(blocking=False)

    def retrain(self, blocking: bool = True) -> Optional[TrainingReport]:
        if self.async_retrain and not blocking:
            if self._retrain_future is not None:
                if not self._retrain_pending:
                    self._retrain_pending = True
                    self.audit.event("RETRAIN_DEFERRED", reason="previous cycle still running", bars=len(self.store))
                return None
            self.bars_since_retrain = 0
            snapshot = self.store.slice(0, len(self.store))
            self._retrain_future = self._executor.submit(self.trainer.retrain, snapshot, self.log)
            self.audit.event("RETRAIN_STARTED", bars=len(self.store), mode="async")
            return None
        self.bars_since_retrain = 0
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
        if report.oos_by_d:
            self.diagnostics.record_oos_by_d(report.oos_by_d, ts)
        cleared_drawdown = self.risk.record_retrain(report.accepted, report.delta_score)
        if cleared_drawdown:
            # The halt is lifted by an accepted retrain (section 34); drawdown is measured afresh from here,
            # otherwise the still-depressed equity would re-arm the halt on the very next bar.
            self.ledger.rebase_peak()
            self.audit.event("DRAWDOWN_REBASED", equity=round(self.ledger.equity, 2))
        self._flush_events()
        if report.accepted and report.model is not None:
            with self._lock:
                self.registry.promote(report.model)
                self.feature_engine.set_adaptive_d(report.model.d_star)
            self._seed_sigma_history()
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

    # ---------------------------------------------------------- bootstrap
    def bootstrap(self, bars: Iterable[Bar]) -> int:
        """Load history without simulating trades (paper-mode start-up): validate and store every bar
        (with a minimal audit record each), mark the ledger at the end, then train once."""
        stored = 0
        for bar in bars:
            self.processed += 1
            result = self.validator.validate(bar)
            if not result.storable:
                self.rejected_bars += 1
                self.audit.record(bar, self.state.value, validation=result.to_dict(), extra={"note": "bootstrap: rejected"})
                continue
            if not result.ok:
                self.halted_bars += 1
                self.risk.set_data_halt(True, ";".join(result.reasons))
            elif self.risk.data_halted:
                self.risk.note_clean_bar()
            self.store.append(bar)
            stored += 1
            self.audit.record(bar, self.state.value, validation=result.to_dict(), extra={"note": "bootstrap"})
        if len(self.store):
            last = self.store.last()
            self.ledger.mark(last.close_time, last.close, len(self.store) - 1, self.calendar.session_date(last.timestamp))
            self._save_store()
        self.audit.event("BOOTSTRAP", bars=stored, rejected=self.rejected_bars, gaps=self.halted_bars,
                         data_halted=self.risk.data_halted)
        self._flush_events()
        if len(self.store) >= self.minimum_bars:
            self.retrain(blocking=True)
        if self.model is not None:
            self._set_state(self.risk.state_for(self.ledger.exposure), "bootstrap complete")
        return stored

    def _save_store(self) -> Path:
        path = self.artifacts_dir / "data" / f"{self.run_id}_{self.instrument}_{self.store.bar_minutes}m.csv"
        self.store.save(path)
        return path

    # ------------------------------------------------------------- on_bar
    def on_bar(self, bar: Bar, allow_orders: bool = True) -> None:
        """Process one completed bar.  ``allow_orders=False`` marks a catch-up bar delivered late in a
        batch: it is validated, stored, evaluated and audited, but no order can be queued on it because
        the next bar's open is no longer a tradable price (spec section 42)."""
        self.processed += 1
        result: ValidationResult = self.validator.validate(bar)
        if not result.storable:
            self._handle_rejected_bar(bar, result)
            return
        # 1. fills for orders queued at the previous bar, at this bar's open (before anything else
        #    can react to this bar: a decision made on bar t never fills at bar t's open).
        if self.execution.has_pending():
            self.on_next_bar_open(bar)
        if not result.ok:
            # Gap / extreme jump: the bar is real data and is stored, but no new orders may be
            # generated until halt_recovery_bars clean bars have arrived (section 35).
            self.halted_bars += 1
            self.risk.set_data_halt(True, ";".join(result.reasons))
            self.audit.event("DATA_HALT", reasons=list(result.reasons), timestamp=bar.timestamp.isoformat(), stored=True)
        elif self.risk.data_halted and self.risk.note_clean_bar():
            self.audit.event("DATA_HALT_CLEARED", clean_bars=self.risk.halt_recovery_bars)
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
            self.audit.event("DATA_HALT", reason="feature NaN/inf", features=bad[:10])
            self._flatten_if_positioned(bar, "DATA_HALT_FLATTEN")
            self._refresh_state()
            self._record(bar, features=features, validation=result, note="non-finite features")
            return
        self.signal_engine.observe_sigma(features.get("sigma_h"))
        # 5. prediction -> signal -> risk
        prediction = model.predict(features)
        cost = self.execution.estimate_cost(features)
        signal = self.signal_engine.create(prediction, features, cost)
        decision = self.risk.evaluate(signal, self.ledger.state(), features, self.calendar.session_date(bar.timestamp))
        self._flush_events()
        self._refresh_state()
        # 6. order (an unchanged, turnover-suppressed position is maintained: no order at all; nothing is
        #    queued while an earlier order is still pending, nor on a late catch-up bar)
        order = None
        blocked = None
        if self.execution.has_pending():
            blocked = "order pending"
        elif not allow_orders:
            blocked = "catch-up bar: no orders"
        if blocked is None and decision.reason != "TURNOVER_SUPPRESSED":
            if (decision.reason == "MAX_HOLDING_REENTRY" and self.ledger.units != 0
                    and direction_sign(decision.approved_exposure) == direction_sign(self.ledger.units)):
                # Same-direction re-entry (section 31): restart the holding clock and stop reference now.
                self.ledger.reenter(features.timestamp, bar.close, features.get("sigma_h"), len(self.store) - 1)
                if self._open_trade_context is not None:
                    trade = self.ledger.trades[-1]
                    ctx = dict(self._open_trade_context)
                    ctx.update({"pnl": trade.net_pnl, "exit_reason": trade.exit_reason})
                    self.trade_contexts.append(ctx)
                self._open_trade_context = self._entry_context(features, bar, signal)
                self.audit.event("MAX_HOLDING_REENTRY", exposure=round(self.ledger.exposure, 4))
            order = self.execution.build_order(
                self.instrument, features.timestamp, self.ledger.units, self.ledger.exposure, decision.approved_exposure,
                self.ledger.equity, bar.close, self.state, reason=decision.reason, new_entry=decision.new_entry,
                entry_sigma=features.get("sigma_h"))
        if order is not None:
            self.execution.queue_for_next_bar(order)
            self._pending_signal_bar = bar
            if decision.new_entry:
                self._pending_entry_context = self._entry_context(features, bar, signal)
            self._flush_events()
        self._record(bar, features, prediction, signal, decision, order, cost, validation=result, note=blocked or "")

    def _entry_context(self, features: FeatureVector, bar: Bar, signal) -> dict:
        day = self.calendar.session_date(bar.timestamp)
        return {"vol_state": features.get("volatility_state"),
                "minutes_since_open": self.calendar.minutes_since_open(bar.timestamp),
                "session_minutes": self.calendar.session_minutes_for(day),
                "fractional_d": features.fractional_d, "direction": signal.direction,
                "signal_time": features.timestamp.isoformat()}

    def _flatten_if_positioned(self, bar: Bar, reason: str) -> None:
        """Queue a flattening order.  ``bar`` is the last *stored* bar: the decision is stamped with its
        close and priced at its close, so the fill can only happen at a later bar's open."""
        if self.ledger.units == 0 or self.execution.has_pending():
            return
        order = self.execution.build_order(self.instrument, bar.close_time, self.ledger.units, self.ledger.exposure, 0.0,
                                           self.ledger.equity, bar.close, self.risk.state_for(self.ledger.exposure),
                                           reason=reason)
        if order is not None:
            self.execution.queue_for_next_bar(order)
            self._pending_signal_bar = bar
            self._flush_events()

    def _handle_rejected_bar(self, bar: Bar, result: ValidationResult) -> None:
        """Corrupt input (never stored): cancel queued orders, halt, and flatten at the next clean bar."""
        self.rejected_bars += 1
        self.risk.set_data_halt(True, ";".join(result.reasons))
        self.audit.event("DATA_HALT", reasons=list(result.reasons), timestamp=bar.timestamp.isoformat(), stored=False)
        cancelled = self.execution.cancel_pending()
        if cancelled:
            self.audit.event("ORDERS_CANCELLED", count=len(cancelled), reason="rejected bar",
                             order_ids=[o.order_id for o in cancelled])
        self._refresh_state()
        if len(self.store):
            self._flatten_if_positioned(self.store.last(), "DATA_HALT_FLATTEN")
        self._flush_events()
        self._record(bar, validation=result, note="rejected bar")

    def _record(self, bar: Bar, features=None, prediction=None, signal=None, decision=None, order=None, cost=None,
                validation=None, note: str = "") -> None:
        portfolio = self.ledger.state() if self.ledger.mark_time is not None else None
        self.last_record = {
            "timestamp": bar.timestamp.isoformat(), "close": bar.close, "state": self.state.value, "note": note,
            "prediction": prediction.to_dict() if prediction else None, "signal": signal.to_dict() if signal else None,
            "risk": decision.to_dict() if decision else None, "order": order.to_dict() if order else None,
            "cost": cost.to_dict() if cost else None,
            "fractional_d": features.fractional_d if features else None,
        }
        self.audit.record(bar, self.state.value, features, prediction, signal, decision, order, cost,
                          portfolio, self.registry.current_version,
                          validation.to_dict() if validation else None,
                          {"note": note, "bars_since_retrain": self.bars_since_retrain,
                           "risk_halt": self.risk.halt_reason(self.calendar.session_date(bar.timestamp)),
                           "clean_bars_since_halt": self.risk.clean_bars_since_halt})

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
        groups = attribution_groups(self.trade_contexts, self.vol_edges)
        metrics = compute_metrics(self.ledger, int(self.cfg.market.bars_per_day),
                                  int(self.cfg.market.trading_days_per_year), groups)
        mirror = self.execution.reconcile_mirror(self.instrument, self.ledger.units)
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
            "mirror_reconciliation": mirror,
            "config": self.cfg.to_dict(), "config_digest": self.cfg.digest(),
        }
        (self.artifacts_dir / "audit").mkdir(parents=True, exist_ok=True)
        with open(self.artifacts_dir / "audit" / f"{self.run_id}_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(self.artifacts_dir / "audit" / f"{self.run_id}_trades.jsonl", "w", encoding="utf-8") as fh:
            for t in self.ledger.trades:
                fh.write(json.dumps(t.to_dict(), default=str) + "\n")
        if len(self.store):
            summary["data_file"] = str(self._save_store())
            summary["data_checksum"] = self.store.checksum()
        self.diagnostics.save_json()
        self.diagnostics.plot()
        self.audit.event("RUN_COMPLETE", equity=round(self.ledger.equity, 2), trades=len(self.ledger.trades),
                         net_return=round(metrics.net_return, 5), sharpe=round(metrics.sharpe, 3))
        self.audit.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        return summary


__all__ = ["TradingBot"]
