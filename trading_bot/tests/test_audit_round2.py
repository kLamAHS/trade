"""Regression tests for the second adversarial audit round."""

import json
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.calendar import SessionCalendar, nyse_holidays
from trading_bot.data.feed import ReplayFeed
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.data.validator import DataValidator
from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.orders import OrderBuilder
from trading_bot.execution.simulator import AlpacaPaperBroker, ExecutionEngine, ExecutionSimulator
from trading_bot.features.engine import FeatureEngine
from trading_bot.fractional.engine import FractionalEngine
from trading_bot.portfolio.ledger import PortfolioLedger
from trading_bot.portfolio.metrics import compute_metrics
from trading_bot.risk.manager import RiskEngine
from trading_bot.strategy.signal import SignalEngine
from trading_bot.training.trainer import ModelTrainer
from trading_bot.training.validation import SimulationParams, simulate_validation
from trading_bot.types import Bar, Fill, Order, Prediction

NY = SessionCalendar()
UTC = timezone.utc
T0 = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)   # Monday 09:30 New York


def _fill(ts, side, units, price, spread=0.0, slip=0.0, comm=0.0, new_entry=False, sigma=0.01):
    return Fill("x", "SYN", ts - timedelta(minutes=30), ts, side, units, price, price, spread, slip, comm,
                new_entry=new_entry, entry_sigma=sigma)


# ------------------------------------------------------------ risk / ledger
def test_drawdown_halt_is_lifted_and_rebased_after_accepted_retrain():
    risk = RiskEngine(drawdown_halt=0.10, daily_loss_limit=1.0)     # isolate the drawdown rule
    eng = SignalEngine()
    led = PortfolioLedger(100000.0, "SYN")
    led.mark(T0, 100.0, 0, T0.date())
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 1000.0, 100.0, new_entry=True))
    led.mark(T0 + timedelta(minutes=30), 100.0, 1, T0.date())
    led.mark(T0 + timedelta(minutes=60), 88.0, 2, T0.date())          # -12 %
    d = risk.evaluate(eng.build(T0, 0.01, 0.001, 0.01, 0.01), led.state(), None, T0.date())
    assert d.approved_exposure == 0.0 and risk.drawdown_halted
    led.apply(_fill(T0 + timedelta(minutes=90), "sell", 1000.0, 88.0))
    led.mark(T0 + timedelta(minutes=90), 88.0, 3, T0.date())
    assert risk.record_retrain(accepted=True, delta_score=0.5) is True
    led.rebase_peak()
    assert led.drawdown == pytest.approx(0.0)
    d2 = risk.evaluate(eng.build(T0, 0.01, 0.001, 0.01, 0.01), led.state(), None, T0.date())
    assert d2.approved_exposure > 0 and not risk.drawdown_halted
    assert risk.record_retrain(accepted=True, delta_score=0.5) is False


def test_daily_return_includes_overnight_gap_and_opening_fills():
    led = PortfolioLedger(100000.0, "SYN")
    d1 = T0.date()
    led.mark(T0, 100.0, 0, d1)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 1000.0, 100.0, new_entry=True))
    led.mark(T0 + timedelta(minutes=30), 100.0, 1, d1)
    nxt = T0 + timedelta(days=1)
    led.mark(nxt, 96.0, 2, nxt.date())                                # -4 % overnight gap on a full position
    assert led.daily_return == pytest.approx(-0.04)
    assert led.session_start_equity == pytest.approx(100000.0)
    risk = RiskEngine(daily_loss_limit=0.025)
    d = risk.evaluate(SignalEngine().build(nxt, 0.01, 0.001, 0.01, 0.01), led.state(), None, nxt.date())
    assert d.daily_loss_status == "DAILY_RISK_HALT" and d.approved_exposure == 0.0


def test_ledger_reenter_restarts_holding_clock_and_closes_trade():
    led = PortfolioLedger(100000.0, "SYN")
    d = T0.date()
    led.mark(T0, 100.0, 0, d)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 100.0, 100.0, comm=1.0, new_entry=True, sigma=0.01))
    for i in range(1, 13):
        led.mark(T0 + timedelta(minutes=30 * i), 100.0 + i * 0.1, i, d)
    assert led.holding_bars == 12
    equity_before = led.equity
    led.reenter(led.mark_time, led.mark_price, 0.02, 12)
    assert led.holding_bars == 0 and led.entry_sigma == 0.02 and led.entry_price == led.mark_price
    assert led.units == 100.0 and led.equity == pytest.approx(equity_before)
    assert len(led.trades) == 1 and led.trades[0].exit_reason == "max_holding_reentry"
    assert led.trades[0].realized_pnl == pytest.approx(100 * 1.2) and led.trades[0].net_pnl == pytest.approx(120 - 1.0)
    assert led.realized_pnl == pytest.approx(120.0)
    led.mark(T0 + timedelta(minutes=30 * 13), 101.2, 13, d)
    assert led.holding_bars == 1 and led.unrealized_pnl == pytest.approx(0.0)


def test_trade_net_pnl_excludes_embedded_spread_and_slippage():
    led = PortfolioLedger(100000.0, "SYN")
    d = T0.date()
    led.mark(T0, 100.0, 0, d)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 100.0, 101.0, spread=5.0, slip=2.0, comm=1.0, new_entry=True))
    led.apply(_fill(T0 + timedelta(minutes=60), "sell", 100.0, 103.0, spread=5.0, slip=2.0, comm=1.0))
    led.mark(T0 + timedelta(minutes=60), 103.0, 2, d)
    t = led.trades[0]
    assert t.realized_pnl == pytest.approx(200.0) and t.commission == pytest.approx(2.0)
    assert t.net_pnl == pytest.approx(198.0) == pytest.approx(led.equity - 100000.0)
    m = compute_metrics(led)
    assert m.average_trade == pytest.approx(198.0) and m.long_performance["total_pnl"] == pytest.approx(198.0)


def test_flip_max_units_and_cost_split():
    led = PortfolioLedger(100000.0, "SYN")
    d = T0.date()
    led.mark(T0, 100.0, 0, d)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 100.0, 100.0, new_entry=True))
    led.apply(_fill(T0 + timedelta(minutes=60), "sell", 400.0, 110.0, comm=4.0, new_entry=True))
    assert led.trades[0].max_units == 100.0
    assert led.trades[0].commission == pytest.approx(1.0)      # closing quarter of the flip
    assert led._open_trade.commission == pytest.approx(3.0)


def test_sortino_uses_target_downside_deviation():
    led = PortfolioLedger(1000.0, "SYN")
    eq = [1000, 1010, 1000, 1020, 1030, 1020]
    for i, e in enumerate(eq):
        led.cash = float(e)
        led.mark(T0 + timedelta(minutes=30 * i), 1.0, i, T0.date())
    m = compute_metrics(led, 13, 252)
    rets = np.diff(np.array(eq, dtype=float)) / np.array(eq[:-1], dtype=float)
    dsd = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert m.sortino == pytest.approx(rets.mean() / dsd * math.sqrt(13 * 252))


def test_turnover_suppressed_keeps_drifted_exposure_unclipped():
    risk = RiskEngine(max_absolute_exposure=1.0)
    eng = SignalEngine()
    from trading_bot.types import PortfolioSnapshot
    snap = PortfolioSnapshot(T0, 0.0, 1000.0, 100.04, 100000.0, 1.0004, 100000.0, 0.0, 100000.0, 0.0, 0.0, 0.0, 0, 100.0,
                             0.01, 3, 0.0004)
    d = risk.evaluate(eng.build(T0, 0.05, 0.001, 0.01, 0.01), snap, None, T0.date())
    assert d.reason == "TURNOVER_SUPPRESSED" and d.approved_exposure == pytest.approx(1.0004)


# ------------------------------------------------------------ validation simulator
def test_validation_simulator_attributes_every_unit_of_pnl_to_a_trade():
    rng = np.random.RandomState(3)
    n = 400
    params = SimulationParams(horizon=4, bars_per_day=13, max_holding_bars=6)
    E = rng.choice([-3.0, 0.0, 0.0, 2.0, 4.0], size=n)
    sigma = np.full(n, 0.01); sigma_ref = np.full(n, 0.01)
    cost_rt = np.full(n, 0.001); cost_side = np.full(n, 0.0005)
    opens = 100.0 * np.exp(np.cumsum(rng.randn(n + 2) * 0.003))
    m = simulate_validation(E, np.sign(rng.randn(n)), sigma, sigma_ref, cost_rt, cost_side, np.log(opens[:n]),
                            opens[1:n + 1], opens[2:n + 2], params)
    assert m.n_trades > 5
    assert sum(m.trade_pnls) == pytest.approx(sum(m.bar_pnls), abs=1e-12)
    assert m.total_cost > 0 and m.gross_pnl - m.total_cost == pytest.approx(sum(m.bar_pnls), abs=1e-12)


def test_validation_simulator_exit_cost_charged_to_the_trade():
    n = 30
    params = SimulationParams(horizon=4, bars_per_day=13)
    E = np.zeros(n); E[5:8] = 2.0
    sigma = np.full(n, 0.01); sigma_ref = np.full(n, 0.01)
    cost_rt = np.full(n, 0.001); cost_side = np.full(n, 0.0005)
    opens = np.full(n + 2, 100.0)
    m = simulate_validation(E, np.ones(n), sigma, sigma_ref, cost_rt, cost_side, np.log(opens[:n]), opens[1:n + 1],
                            opens[2:n + 2], params)
    assert m.n_trades == 1
    assert m.trade_pnls[0] == pytest.approx(sum(m.bar_pnls), abs=1e-12) and m.trade_pnls[0] < 0
    assert m.profit_factor == 0.0


# ------------------------------------------------------------ validator / calendar
def test_missing_sessions_are_detected_and_holidays_are_not():
    v = DataValidator(NY, instrument="SYN")
    mon_last = (NY.session_open_datetime(date(2026, 9, 14)) + timedelta(minutes=360)).astimezone(UTC)
    v.accept(Bar("SYN", mon_last, 100, 101, 99, 100.5, 10, 30))
    wed = NY.session_open_datetime(date(2026, 9, 16)).astimezone(UTC)
    r = v.check(Bar("SYN", wed, 100, 101, 99, 100.5, 10, 30))
    assert not r.ok and r.storable and any("missing session" in x for x in r.reasons)
    tue = NY.session_open_datetime(date(2026, 9, 15)).astimezone(UTC)
    assert v.check(Bar("SYN", tue, 100, 101, 99, 100.5, 10, 30)).ok
    # Friday 2026-09-04 -> Tuesday 2026-09-08 (Labor Day Monday) is a legitimate gap
    fri_last = (NY.session_open_datetime(date(2026, 9, 4)) + timedelta(minutes=360)).astimezone(UTC)
    v.accept(Bar("SYN", fri_last, 100, 101, 99, 100.5, 10, 30))
    tue2 = NY.session_open_datetime(date(2026, 9, 8)).astimezone(UTC)
    assert v.check(Bar("SYN", tue2, 100, 101, 99, 100.5, 10, 30)).ok
    assert date(2026, 9, 7) in nyse_holidays(2026)
    assert NY.next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)
    assert date(2026, 4, 3) in nyse_holidays(2026)           # Good Friday 2026
    assert date(2027, 3, 26) in nyse_holidays(2027)          # Good Friday 2027
    assert date(2025, 6, 19) in nyse_holidays(2025) and date(2026, 7, 3) in nyse_holidays(2026)
    assert date(2022, 12, 26) in nyse_holidays(2022)         # Christmas observed
    cal2 = SessionCalendar(holidays=frozenset({date(2026, 9, 15)}))
    v2 = DataValidator(cal2, instrument="SYN")
    v2.accept(Bar("SYN", mon_last, 100, 101, 99, 100.5, 10, 30))
    assert v2.check(Bar("SYN", wed, 100, 101, 99, 100.5, 10, 30)).ok


# ------------------------------------------------------------ execution / mirror
def test_order_ids_are_unique_across_builders():
    a = OrderBuilder(id_prefix="runA").build("SYN", T0, 0, 0, 0.5, 1e5, 100.0)
    b = OrderBuilder(id_prefix="runB").build("SYN", T0, 0, 0, 0.5, 1e5, 100.0)
    assert a.order_id != b.order_id and a.order_id.startswith("runA-") and b.order_id.startswith("runB-")


def test_fill_timestamp_never_precedes_signal_and_rejects_closed_bars():
    sim = ExecutionSimulator(CostModel())
    exec_bar = Bar("SYN", T0 + timedelta(minutes=30), 100, 101, 99, 100.5, 10, 30)
    late = Order("SYN", T0 + timedelta(minutes=30, seconds=20), "buy", 1.0, 0.5, 0.0, 100.0)
    f = sim.simulate_fill(late, exec_bar)
    assert f.fill_timestamp == late.signal_timestamp and f.reference_price == exec_bar.open
    on_time = Order("SYN", T0 + timedelta(minutes=30), "buy", 1.0, 0.5, 0.0, 100.0)
    assert sim.simulate_fill(on_time, exec_bar).fill_timestamp == exec_bar.timestamp
    too_late = Order("SYN", T0 + timedelta(minutes=60), "buy", 1.0, 0.5, 0.0, 100.0)
    with pytest.raises(ValueError):
        sim.simulate_fill(too_late, exec_bar)


class _FakeTradingClient:
    def __init__(self):
        self.submitted = []
        self.status = "partially_filled"

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id=f"alp-{len(self.submitted)}", status="accepted", filled_avg_price=None, filled_qty="0")

    def get_order_by_id(self, oid):
        if self.status == "partially_filled":
            return SimpleNamespace(id=oid, status="partially_filled", filled_avg_price="100.5", filled_qty="40")
        return SimpleNamespace(id=oid, status="filled", filled_avg_price="100.6", filled_qty="100")

    def get_open_position(self, symbol):
        return SimpleNamespace(qty="100")


def test_alpaca_mirror_submits_at_queue_time_and_never_replaces_the_simulated_fill():
    client = _FakeTradingClient()
    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=client)
    eng = ExecutionEngine(CostModel(default_spread=0.0002), OrderBuilder(id_prefix="run1"), broker=broker)
    order = eng.build_order("SYN", T0 + timedelta(minutes=30), 0.0, 0.0, 0.1, 100000.0, 100.0, __import__("trading_bot.types", fromlist=["BotState"]).BotState.READY)
    assert order.units == pytest.approx(100.0)
    eng.queue_for_next_bar(order)
    assert len(client.submitted) == 1 and client.submitted[0].qty == 100          # whole shares, submitted immediately
    assert client.submitted[0].client_order_id == order.order_id[:48]
    assert eng.events[0]["event"] == "ORDER_MIRROR" and order.order_id in eng.pending_mirrors
    exec_bar = Bar("SYN", T0 + timedelta(minutes=30), 100.0, 101.0, 99.0, 100.5, 10, 30)
    fill = eng.simulate_fill(order, exec_bar)
    assert fill.source == "simulator" and fill.units == pytest.approx(100.0)
    assert fill.fill_price == pytest.approx(100.0 + 0.5 * 0.0002 * 100.0 + 0.05 * (2.0 / 100.5) * 100.0)
    assert fill.mirror["status"] == "partially_filled" and fill.mirror["final"] is False
    assert order.order_id in eng.pending_mirrors                                   # partial fills are not final
    client.status = "filled"
    eng.poll_mirrors()
    assert order.order_id not in eng.pending_mirrors
    assert any(e["event"] == "ORDER_MIRROR_FINAL" and e["status"] == "filled" for e in eng.events)
    rec = eng.reconcile_mirror("SYN", 100.0)
    assert rec["drift"] == 0.0 and not rec["flagged"]
    small = eng.build_order("SYN", T0 + timedelta(minutes=60), 100.0, 0.1, 0.1004, 100000.0, 100.0,
                            __import__("trading_bot.types", fromlist=["BotState"]).BotState.POSITIONED)
    assert small is not None
    eng.queue_for_next_bar(small)
    assert eng.mirror_acks[small.order_id]["status"] == "skipped"                # rounds to zero shares


# ------------------------------------------------------------ bot level
def _events(bot):
    p = bot.artifacts_dir / "audit" / f"{bot.run_id}_events.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_first_bar_rejected_does_not_crash(fast_cfg, tmp_path):
    bot = TradingBot(fast_cfg, run_id="first", artifacts_dir=tmp_path, log=None)
    b = generate_synthetic_bars(3, seed=1, instrument="SYN")[0]
    bot.on_bar(Bar(b.instrument, b.timestamp, b.open, b.low - 1, b.high + 1, b.close, b.volume, 30))
    assert bot.rejected_bars == 1 and len(bot.store) == 0
    bars = [json.loads(l) for l in (tmp_path / "audit" / "first_bars.jsonl").read_text().splitlines()]
    assert bars[0]["portfolio"] is None and bars[0]["validation"]["storable"] is False


def test_audit_features_are_json_objects_with_timestamps(bot_run):
    bars = [json.loads(l) for l in (bot_run.artifacts_dir / "audit" / "test_run_bars.jsonl").read_text().splitlines()]
    rec = next(b for b in bars if b["features"])
    assert isinstance(rec["features"], dict) and "fd_adaptive_z" in rec["features"]
    assert rec["feature_timestamp"] is not None and rec["latest_source_timestamp"] is not None
    assert rec["latest_source_timestamp"] <= rec["feature_timestamp"]
    assert all(v is None or isinstance(v, (int, float)) for v in rec["features"].values())


def test_attribution_contexts_match_closed_trades(bot_run):
    bot = bot_run
    assert len(bot.trade_contexts) == len(bot.ledger.trades)
    for ctx, trade in zip(bot.trade_contexts, bot.ledger.trades):
        if "direction" in ctx:
            assert ctx["direction"] == trade.direction
        assert ctx["pnl"] == pytest.approx(trade.net_pnl)


class _AlwaysLong:
    """Stub model that always forecasts a strong positive edge (exercises re-entry and suppression)."""

    def __init__(self, feature_names):
        self.feature_names = tuple(feature_names)
        self.version = "stub"
        self.d_star = 0.4

    def predict(self, features):
        sigma = features.get("sigma_h")
        return Prediction(features.timestamp, 3.0, 3.0 * sigma * 2.0, 0.9, 0.8, "stub")


def test_max_holding_reentry_and_turnover_suppression_live(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1520, seed=31, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1400, "window_bars": 1400}}),
                     run_id="stub", artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:1450])
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    for b in bars[1450:1520]:
        bot.on_bar(b)
    ev = _events(bot)
    assert any(e["event"] == "MAX_HOLDING_REENTRY" for e in ev)
    reentries = [t for t in bot.ledger.trades if t.exit_reason == "max_holding_reentry"]
    assert reentries and all(t.bars_held <= 12 for t in reentries)
    assert bot.ledger.holding_bars <= 13
    recs = [json.loads(l) for l in (tmp_path / "audit" / "stub_bars.jsonl").read_text().splitlines()]
    suppressed = [r for r in recs if r["risk"] and r["risk"]["reason"] == "TURNOVER_SUPPRESSED"]
    assert suppressed and all(r["order"] is None for r in suppressed)
    for f in bot.ledger.fills:
        assert f.fill_timestamp >= f.signal_timestamp


def test_grid_keys_validated_and_applied(fast_cfg, fractional):
    bad = fast_cfg.with_overrides({"training": {"hyperparameter_grid": {"n_estimators": [10], "bogus": [1]}}})
    with pytest.raises(ValueError):
        ModelTrainer(bad, FeatureEngine(bad, fractional, NY), fractional, CostModel.from_config(bad))
    good = fast_cfg.with_overrides({"training": {"hyperparameter_grid": {"n_estimators": [10], "learning_rate": [0.01, 0.05]}}})
    tr = ModelTrainer(good, FeatureEngine(good, fractional, NY), fractional, CostModel.from_config(good))
    assert len(tr.grid) == 2 and tr._params(tr.grid[1]).learning_rate == 0.05


def test_report_records_baseline_params_and_config(fast_cfg, fractional):
    bars = generate_synthetic_bars(1500, seed=12, instrument="SYN")
    tr = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    report = tr.retrain(BarStore("SYN", 30, bars))
    assert report.error is None
    assert report.baseline_params == report.best_params                      # single-point grid in the fast config
    assert report.model.metadata.extra["config"]["training"]["window_bars"] == fast_cfg.training.window_bars
    assert report.model.metadata.validation_metrics["baseline_grid"]
