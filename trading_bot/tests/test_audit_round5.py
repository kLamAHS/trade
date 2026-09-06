"""Regression tests for the residual findings of the audit-response verification round."""

import json
from datetime import date, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.feed import AlpacaBarFeed
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.orders import OrderBuilder
from trading_bot.execution.simulator import AlpacaPaperBroker, ExecutionEngine, ExecutionSimulator, FillDeferred
from trading_bot.features.engine import FeatureEngine
from trading_bot.main import run_live_loop
from trading_bot.risk.limits import apply_position_rules
from trading_bot.risk.manager import RiskEngine
from trading_bot.training.trainer import ModelTrainer, git_commit
from trading_bot.training.validation import SimulationParams, simulate_validation
from trading_bot.types import Bar, BotState, Order, Prediction

NY = SessionCalendar()
UTC = timezone.utc


class _AlwaysLong:
    def __init__(self, names):
        self.feature_names = tuple(names)
        self.version = "stub"
        self.d_star = 0.4

    def predict(self, features):
        sigma = features.get("sigma_h")
        return Prediction(features.timestamp, 3.0, 3.0 * sigma * 2.0, 0.9, 0.8, "stub")


def _mk_bot(fast_cfg, tmp_path, run_id, seed=71, n=1500, boot=1450):
    bars = generate_synthetic_bars(n, seed=seed, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1400, "window_bars": 1400}}),
                     run_id=run_id, artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:boot])
    return bot, bars


def _live(bar, now, bid=None, ask=None, qts=None):
    return Bar(bar.instrument, bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume, 30, bid, ask,
               quote_timestamp=qts, observed_at=now)


# ------------------------------------------------------- live decision time never at the close
def test_live_bar_without_quote_never_fills_at_the_next_open(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "noquote")
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    b = bars[1450]
    received = b.close_time + timedelta(seconds=25)               # quote endpoint failed: no bid/ask
    bot.on_bar(_live(b, received))
    assert bot.execution.has_pending()
    order = bot.execution.queue.peek()[0]
    assert order.signal_timestamp == received                     # stamped at receipt, not at the close
    nxt = bars[1451]
    bot.on_bar(_live(nxt, nxt.close_time + timedelta(seconds=25)))   # its open print predates the decision
    assert not bot.ledger.fills and bot.execution.has_pending()
    ev = [json.loads(l) for l in (tmp_path / "audit" / "noquote_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "ORDER_DEFERRED" for e in ev)
    after = bars[1452]
    bot.on_bar(_live(after, after.close_time + timedelta(seconds=25)))
    fill = bot.ledger.fills[0]
    assert fill.price_source == "next_open" and fill.reference_price == after.open
    assert fill.fill_timestamp >= order.signal_timestamp


def test_feed_stamps_receipt_time_and_standing_quote_time():
    d = date(2026, 9, 3)
    starts = [s.astimezone(UTC) for s in NY.regular_session_starts(d)]
    raw = [SimpleNamespace(timestamp=starts[0], open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0)]
    quote = SimpleNamespace(bid_price=100.4, ask_price=100.6, timestamp=starts[0] + timedelta(minutes=29))  # stale NBBO time

    class C:
        def get_stock_bars(self, req):
            return SimpleNamespace(data={"SYN": raw})

        def get_stock_latest_quote(self, req):
            return {"SYN": quote}

    now = starts[0] + timedelta(minutes=30, seconds=22)
    feed = AlpacaBarFeed("SYN", NY, api_key="k", secret_key="s", data_client=C(), clock=lambda: now)
    got = feed.poll_new_bars(now)
    assert got[0].observed_at == now and got[0].quote_timestamp == now      # standing quote observed at fetch time
    assert got[0].latest_source_time == now
    late = starts[0] + timedelta(minutes=75)                                # delayed delivery still gets receipt time
    feed2 = AlpacaBarFeed("SYN", NY, api_key="k", secret_key="s", data_client=C(), clock=lambda: late)
    got2 = feed2.poll_new_bars(late)
    assert got2[0].observed_at == late and got2[0].latest_source_time == late


def test_flatten_orders_carry_the_real_decision_time(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "flat")
    from trading_bot.types import Fill
    last = bot.store.last()
    bot.ledger.apply(Fill("f", "SYN", last.close_time, last.close_time, "buy", 50.0, last.close, last.close, 0, 0, 0,
                          new_entry=True, entry_sigma=0.01))
    b = bars[1450]
    received = b.close_time + timedelta(seconds=30)
    bad = Bar(b.instrument, b.timestamp, b.open, b.low - 1, b.high + 1, b.close, b.volume, 30, observed_at=received)
    bot.on_bar(bad)                                                # rejected bar arriving live
    order = bot.execution.queue.peek()[0]
    assert order.target_exposure == 0.0 and order.signal_timestamp == received
    # a delayed bar with a receipt time also stamps the halt-flatten path at receipt
    bot2, bars2 = _mk_bot(fast_cfg, tmp_path / "b2", "flat2")
    bot2.ledger.apply(Fill("f", "SYN", bot2.store.last().close_time, bot2.store.last().close_time, "buy", 50.0,
                           bot2.store.last().close, bot2.store.last().close, 0, 0, 0, new_entry=True, entry_sigma=0.01))
    b2 = bars2[1450]
    bot2.on_bar(_live(b2, b2.close_time + timedelta(seconds=25)))
    if bot2.execution.has_pending():
        assert bot2.execution.queue.peek()[0].signal_timestamp >= b2.close_time + timedelta(seconds=25)


def test_catch_up_bar_after_successor_opened_cannot_trade(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "late")
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)

    class Feed:
        def __init__(self):
            self.calls = 0

        def poll_new_bars(self, now):
            self.calls += 1
            return [bars[1450]] if self.calls == 1 else []

    feed = Feed()
    from datetime import datetime
    import trading_bot.main as m
    real_now = datetime.now
    delivered_at = bars[1450].close_time + timedelta(minutes=40)   # a full bar late
    m.datetime = SimpleNamespace(now=lambda tz=None: delivered_at)
    try:
        run_live_loop(bot, feed, poll_seconds=1, log=lambda *_: None, should_stop=lambda: feed.calls >= 2, sleep=lambda s: None)
    finally:
        m.datetime = datetime
    recs = [json.loads(l) for l in (tmp_path / "audit" / "late_bars.jsonl").read_text().splitlines()]
    assert recs[-1]["extra"]["note"] == "catch-up bar: no orders" and not bot.execution.has_pending()


# ------------------------------------------------------- mirror consistency
class _Broker:
    def __init__(self):
        self.status = "accepted"
        self.submitted = []
        self.orders = {}
        self.position = 10.0

    def submit_order(self, req):
        self.submitted.append(req)
        oid = f"a{len(self.submitted)}"
        self.orders[oid] = req.qty
        return SimpleNamespace(id=oid, status="accepted", filled_avg_price=None, filled_qty="0")

    def get_order_by_id(self, oid):
        if self.status == "filled":
            return SimpleNamespace(id=oid, status="filled", filled_avg_price="101.25", filled_qty=str(self.orders.get(oid, 0)))
        return SimpleNamespace(id=oid, status="accepted", filled_avg_price=None, filled_qty="0")

    def get_open_position(self, symbol):
        return SimpleNamespace(qty=str(self.position))

    def cancel_order_by_id(self, oid):
        pass


def test_deferred_order_keeps_mirror_ack_and_uses_broker_fill_on_retry():
    client = _Broker()
    client.position = 0.0
    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=client)
    eng = ExecutionEngine(CostModel(), OrderBuilder(id_prefix="d"), broker=broker, live_fill_source="broker")
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    order = eng.build_order("SYN", T + timedelta(minutes=30, seconds=20), 0.0, 0.0, 0.03, 100000.0, 100.0, BotState.READY)
    eng.queue_for_next_bar(order)
    exec_bar = Bar("SYN", T + timedelta(minutes=30), 100.0, 101.0, 99.0, 100.5, 10, 30)
    with pytest.raises(FillDeferred):                     # broker not filled yet, no quote -> defer
        eng.simulate_fill(order, exec_bar, None)
    assert order.order_id in eng.mirror_acks               # the acknowledgement survives the deferral
    client.status = "filled"
    fill = eng.simulate_fill(order, exec_bar, None)
    assert fill.price_source == "broker" and fill.fill_price == 101.25 and fill.units == pytest.approx(order.units)
    assert order.order_id not in eng.mirror_acks


def test_flip_fill_quantity_sums_both_legs():
    client = _Broker()
    client.position = 10.0
    client.status = "filled"
    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=client)
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    ack = broker.submit(Order("SYN", T, "sell", 25.0, -0.15, 0.1, 100.0, order_id="f-1"))
    assert ack["flip"] and [r.qty for r in client.submitted] == [10, 15]
    assert ack["filled_qty"] == pytest.approx(10.0) and not ack["final"]        # open leg acknowledged, not yet filled
    refreshed = broker.refresh_flip(ack)
    assert refreshed["filled_qty"] == pytest.approx(25.0) and refreshed["status"] == "filled" and refreshed["final"]
    # the engine refreshes a non-final flip itself and books the full quantity at the broker price
    eng = ExecutionEngine(CostModel(), OrderBuilder(id_prefix="f"), broker=broker, live_fill_source="broker")
    eng.mirror_acks["f-1"] = ack
    eng.pending_mirrors["f-1"] = ack["id"]
    order = Order("SYN", T + timedelta(minutes=30, seconds=20), "sell", 25.0, -0.15, 0.1, 100.0, order_id="f-1")
    fill = eng.simulate_fill(order, Bar("SYN", T + timedelta(minutes=30), 100.0, 101.0, 99.0, 100.5, 10, 30), None)
    assert fill.units == pytest.approx(25.0) and fill.price_source == "broker"


def test_quote_older_than_decision_is_rejected():
    sim = ExecutionSimulator(CostModel())
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    exec_bar = Bar("SYN", T + timedelta(minutes=30), 100, 101, 99, 100.5, 10, 30)
    order = Order("SYN", T + timedelta(minutes=30, seconds=21), "buy", 1.0, 0.5, 0.0, 100.0)
    sb = Bar("SYN", T, 99.5, 100.6, 99.4, 100.0, 10, 30, bid=100.2, ask=100.4, quote_timestamp=T + timedelta(minutes=30, seconds=20))
    with pytest.raises(FillDeferred):
        sim.simulate_fill(order, exec_bar, sb)


# ------------------------------------------------------- re-entry at the next open, hold-to-horizon
def test_reentry_is_booked_at_next_open(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "reenter", n=1520)
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    for b in bars[1450:1470]:
        bot.on_bar(b)
    re = [t for t in bot.ledger.trades if t.exit_reason == "max_holding_reentry"]
    assert re
    opens = {b.timestamp: b.open for b in bars}
    for t in re:
        assert t.exit_price in opens.values()             # booked at a bar open, never at the decision bar's close
        assert t.exit_time in opens                       # stamped at that bar's start
    assert bot.ledger.entry_price in opens.values()


def test_hold_to_horizon_rule_and_simulator_cadence():
    q = 0.0
    holding = 0
    changes = []
    for i in range(12):
        if q != 0:
            holding += 1
        target = (0.3 + 0.2 * (i // 4)) if i % 2 == 0 else 0.9      # a different target at every 4-bar boundary
        r = apply_position_rules(target, q, holding, 12, False, 0.15, reevaluate_every=4)
        if r.exposure != q:
            changes.append(i)
        if r.new_entry:
            holding = 0
        q = r.exposure
    assert changes == [0, 4, 8]                            # re-targeted only at 4-bar boundaries
    n = 40
    E = np.tile([3.0, -3.0, -3.0, -3.0], 10)             # flips every bar if re-evaluated every bar
    sigma = np.full(n, 0.01); sigma_ref = np.full(n, 0.01)
    args = (E, np.ones(n), sigma, sigma_ref, np.full(n, 0.0005), np.full(n, 0.00025), np.log(np.full(n, 100.0)),
            np.full(n, 100.0), np.full(n, 100.0))
    m1 = simulate_validation(*args, SimulationParams(horizon=4, bars_per_day=13, reevaluate_every=1))
    m4 = simulate_validation(*args, SimulationParams(horizon=4, bars_per_day=13, reevaluate_every=4))
    assert m1.n_trades > m4.n_trades and m4.mean_bars_held > m1.mean_bars_held
    assert m1.mean_bars_held <= 2.0 and m4.mean_bars_held >= 4.0        # flips every bar vs held across boundaries


# ------------------------------------------------------- ablation counting, provenance
def test_ablation_failures_ignore_overlapping_holdouts():
    r = RiskEngine(ablation_failures_to_halt=3)
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    span = lambda a, b: (T + timedelta(hours=a), T + timedelta(hours=b))
    r.record_retrain(True, -0.5, span(0, 100))
    r.record_retrain(True, -0.5, span(10, 110))            # overlaps -> repeat, not counted
    r.record_retrain(True, -0.5, span(50, 150))            # overlaps the *counted* failure -> repeat
    assert r.ablation_failures == 1 and not r.ablation_halted
    assert any(e["event"] == "FRACTIONAL_EDGE_FAILURE_REPEAT" for e in r.events)
    r.record_retrain(True, -0.5, span(101, 200))           # disjoint from the counted failure
    r.record_retrain(True, -0.5, span(201, 300))
    assert r.ablation_failures == 3 and r.ablation_halted
    r.record_retrain(True, 0.2, span(250, 350))
    assert r.ablation_failures == 0 and not r.ablation_halted


def test_git_commit_is_package_relative_and_marks_dirty(tmp_path, monkeypatch):
    import subprocess
    sha = git_commit()
    assert sha != "nogit" and len(sha.split("-")[0]) == 40
    foreign = tmp_path / "other"
    foreign.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=foreign, check=True)
    monkeypatch.chdir(foreign)
    assert git_commit() == sha                              # launch directory does not matter
    monkeypatch.undo()


def test_production_model_uses_holdout_validated_d(fast_cfg, fractional):
    tr = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    bars = generate_synthetic_bars(1700, seed=81, instrument="SYN", memory_d=0.45, amplitude=6.0)
    report = tr.retrain(BarStore("SYN", 30, bars))
    assert report.error is None
    assert report.model.d_star == report.holdout_d_star
    assert report.model.metadata.extra["d_production"] == report.holdout_d_star
    assert report.model.metadata.extra["d_full"] == report.d_full == report.stationarity.d_star
    assert report.holdout_span is not None and report.holdout_span[0] < report.holdout_span[1]
    # with fold_local_d off the holdout d* is still inner-block only
    cfg2 = fast_cfg.with_overrides({"training": {"fold_local_d": False}})
    tr2 = ModelTrainer(cfg2, FeatureEngine(cfg2, fractional, NY), fractional, CostModel.from_config(cfg2))
    r2 = tr2.retrain(BarStore("SYN", 30, bars))
    window = BarStore("SYN", 30, bars).last(tr2.window_bars)
    ds = tr2.builder.build(window, r2.stationarity.d_star)
    inner, holdout, _ = tr2._layout(len(ds))
    last_inner = tr2._last_label_bar(ds, int(inner[-1]))
    assert r2.holdout_d_star == fractional.estimate_stationarity(window.log_close()[: last_inner + 1]).d_star


def test_inner_calibration_uses_its_own_d(fast_cfg, fractional):
    tr = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    bars = generate_synthetic_bars(1600, seed=82, instrument="SYN")
    store = BarStore("SYN", 30, bars)
    ds = tr.builder.build(store, 0.4)
    _, _, folds = tr._layout(len(ds))
    sets = tr.build_fold_sets(store, ds, folds)
    A, Y, rows = tr._inner_calibration_set(sets[0], tr.grid[0], ds.feature_names)
    train = folds[0].train
    split = int(len(train) * (1 - tr.inner_calibration_fraction))
    inner_train = train[: split - tr.purge]
    cut = tr._last_label_bar(ds, int(inner_train[-1])) + 1
    assert cut <= int(ds.bar_index[rows[0]])                  # inner d* prefix ends before the calibration block
    wb = list(store.bars)
    mutated = wb[:cut] + [Bar(b.instrument, b.timestamp, b.open, b.high * 1.02, b.low * 0.98,
                              b.close * (1 + 0.004 * ((i % 5) - 2)), b.volume, 30) for i, b in enumerate(wb[cut:])]
    store2 = BarStore("SYN", 30, mutated)
    ds2 = tr.builder.build(store2, 0.4)
    sets2 = tr.build_fold_sets(store2, ds2, folds)
    A2, _, rows2 = tr._inner_calibration_set(sets2[0], tr.grid[0], ds.feature_names)
    assert np.array_equal(rows, rows2)
    # the *models* saw identical data; only inputs at the mutated (calibration) rows differ, never the fit
    Xa = sets[0].dataset.X[inner_train]
    Xb = sets2[0].dataset.X[inner_train]
    assert np.array_equal(np.nan_to_num(Xa), np.nan_to_num(Xb))
