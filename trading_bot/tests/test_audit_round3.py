"""Regression tests for the final audit round (live-loop robustness, mirror, overrides)."""

import json
from datetime import date, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.feed import AlpacaBarFeed
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.orders import OrderBuilder
from trading_bot.execution.simulator import AlpacaPaperBroker, ExecutionEngine
from trading_bot.main import parse_scalar, run_live_loop
from trading_bot.training.validation import SimulationParams, simulate_validation
from trading_bot.types import Bar, BotState, Fill, Order

NY = SessionCalendar()
UTC = timezone.utc


def test_parse_scalar_handles_scientific_notation_and_yaml():
    assert parse_scalar("1e-3") == 0.001 and isinstance(parse_scalar("1e-3"), float)
    assert parse_scalar("500") == 500 and isinstance(parse_scalar("500"), int)
    assert parse_scalar("true") is True and parse_scalar("[1, 2]") == [1, 2] and parse_scalar("iex") == "iex"
    from trading_bot.gui.settings import GuiSettings
    s = GuiSettings(overrides="training.acceptance.min_net_pnl=1e-3\nrisk.daily_loss_limit=0.02\n# comment\n")
    assert s.override_dict() == {"training": {"acceptance": {"min_net_pnl": 0.001}}, "risk": {"daily_loss_limit": 0.02}}


def test_validation_holding_clock_matches_live_cadence():
    """Constant long signal: first max-holding exit decided at row entry+12 -> 12 bars held, like the ledger."""
    n = 40
    params = SimulationParams(horizon=4, bars_per_day=13, max_holding_bars=12, rebalance_threshold=0.15)
    E = np.full(n, 3.0)
    sigma = np.full(n, 0.01); sigma_ref = np.full(n, 0.01)
    cost_rt = np.full(n, 0.0005); cost_side = np.full(n, 0.00025)
    opens = np.full(n + 2, 100.0)
    m = simulate_validation(E, np.ones(n), sigma, sigma_ref, cost_rt, cost_side, np.log(opens[:n]), opens[1:n + 1],
                            opens[2:n + 2], params)
    # exposure constant from row 0; re-entries at rows 12, 24, 36 leave exposure unchanged, so count trades
    assert m.n_trades == 3 or m.n_trades == 4        # 3 closed re-entries (+ the open one counted at the end)
    # confirm the re-entry cadence explicitly with the shared rule
    from trading_bot.risk.limits import apply_position_rules
    holding = 0
    fired = []
    q = 0.0
    for i in range(30):
        if q != 0:
            holding += 1
        r = apply_position_rules(0.5, q, holding, 12, False, 0.15)
        if r.new_entry:
            fired.append(i)
            holding = 0
        q = r.exposure
    assert fired == [0, 12, 24]


def _mk_bot(fast_cfg, tmp_path, run_id):
    bars = generate_synthetic_bars(1500, seed=41, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1400, "window_bars": 1400}}),
                     run_id=run_id, artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:1450])
    return bot, bars


def _events(bot):
    p = bot.artifacts_dir / "audit" / f"{bot.run_id}_events.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_late_quote_order_is_deferred_not_crashing(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "late")
    b = bars[1450]
    order = bot.execution.build_order("SYN", b.close_time + timedelta(minutes=35), 0.0, 0.0, 0.3, 100000.0, b.close,
                                      BotState.READY)
    bot.execution.queue_for_next_bar(order)
    bot.on_bar(bars[1451])                              # bar closed before the signal time -> deferred, no exception
    assert bot.execution.has_pending() and not bot.ledger.fills
    bot.on_bar(bars[1452])                              # opened before the decision and no quote -> deferred again
    assert bot.execution.has_pending() and not bot.ledger.fills
    assert sum(1 for e in _events(bot) if e["event"] == "ORDER_DEFERRED") >= 2
    bot.on_bar(bars[1453])                              # first bar whose open print is after the decision -> fills
    assert len(bot.ledger.fills) == 1
    assert all(o.order_id != order.order_id for o in bot.execution.queue.peek())   # the model may queue a new one
    fill = bot.ledger.fills[0]
    assert fill.fill_timestamp == bars[1453].timestamp and fill.reference_price == bars[1453].open
    assert fill.price_source == "next_open" and fill.fill_timestamp >= fill.signal_timestamp


def test_no_new_order_while_one_is_pending(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "pend")
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    b = bars[1450]
    # an order whose signal time lies far in the future stays pending for several bars
    order = bot.execution.build_order("SYN", b.close_time + timedelta(hours=2), 0.0, 0.0, 0.2, 100000.0, b.close, BotState.READY)
    bot.execution.queue_for_next_bar(order)
    for nb in bars[1451:1455]:
        bot.on_bar(nb)
    assert len(bot.execution.queue) == 1 and bot.execution.queue.peek()[0].order_id == order.order_id
    recs = [json.loads(l) for l in (tmp_path / "audit" / "pend_bars.jsonl").read_text().splitlines()]
    tail = [r for r in recs if r["order"] is None and r["extra"]["note"] == "order pending"]
    assert len(tail) >= 3


class _AlwaysLong:
    def __init__(self, names):
        self.feature_names = tuple(names)
        self.version = "stub"
        self.d_star = 0.4

    def predict(self, features):
        from trading_bot.types import Prediction
        sigma = features.get("sigma_h")
        return Prediction(features.timestamp, 3.0, 3.0 * sigma * 2.0, 0.9, 0.8, "stub")


def test_catch_up_bars_do_not_queue_orders(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "catchup")
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    bot.on_bar(bars[1450], allow_orders=False)
    assert not bot.execution.has_pending()
    bot.on_bar(bars[1451], allow_orders=True)
    assert bot.execution.has_pending()
    recs = [json.loads(l) for l in (tmp_path / "audit" / "catchup_bars.jsonl").read_text().splitlines()]
    assert recs[-2]["extra"]["note"] == "catch-up bar: no orders" and recs[-2]["risk"] is not None


class _FlakyFeed:
    """Feed stub: first poll raises, second returns two bars, third returns nothing and asks to stop."""

    def __init__(self, bars):
        self.bars = bars
        self.calls = 0

    def poll_new_bars(self, now):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("boom")
        if self.calls == 2:
            return list(self.bars)
        return []


def test_live_loop_survives_feed_errors_and_batches(fast_cfg, tmp_path):
    bot, bars = _mk_bot(fast_cfg, tmp_path, "live")
    bot.registry._current = _AlwaysLong(bot.feature_engine.schema.model_names)
    feed = _FlakyFeed(bars[1450:1453])
    slept = []
    run_live_loop(bot, feed, poll_seconds=1, log=lambda *_: None, should_stop=lambda: feed.calls >= 3,
                  sleep=lambda s: slept.append(s))
    ev = _events(bot)
    assert any(e["event"] == "FEED_ERROR" for e in ev)
    assert bot.risk.data_halted                       # feed failure arms the data halt
    assert len(bot.store) == 1453
    assert slept and slept[0] == 1
    # only the newest bar of the batch may queue orders, and the halt blocks them anyway: none queued
    recs = [json.loads(l) for l in (tmp_path / "audit" / "live_bars.jsonl").read_text().splitlines()][-3:]
    assert recs[0]["extra"]["note"] == "catch-up bar: no orders" and recs[1]["extra"]["note"] == "catch-up bar: no orders"


def test_feed_does_not_attach_stale_quote():
    d = date(2026, 9, 3)
    starts = [s.astimezone(UTC) for s in NY.regular_session_starts(d)]
    raw = [SimpleNamespace(timestamp=starts[0], open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0)]
    quote = SimpleNamespace(bid_price=100.4, ask_price=100.6, timestamp=None)

    class C:
        def get_stock_bars(self, req):
            return SimpleNamespace(data={"SYN": raw})

        def get_stock_latest_quote(self, req):
            return {"SYN": quote}

    late = starts[0] + timedelta(minutes=75)         # the 09:30 bar is delivered after the 10:00 bar closed
    feed = AlpacaBarFeed("SYN", NY, api_key="k", secret_key="s", data_client=C(), clock=lambda: late)
    got = feed.poll_new_bars(late)
    assert len(got) == 1 and got[0].bid is None and got[0].latest_source_time == got[0].close_time
    fresh = starts[0] + timedelta(minutes=31)
    feed2 = AlpacaBarFeed("SYN", NY, api_key="k", secret_key="s", data_client=C(), clock=lambda: fresh)
    got2 = feed2.poll_new_bars(fresh)
    assert got2[0].bid == 100.4 and got2[0].quote_timestamp == fresh


class _FlipClient:
    def __init__(self, position=100.0):
        self.position = position
        self.submitted = []
        self.cancelled = []

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id=f"a{len(self.submitted)}", status="filled", filled_avg_price="100.0", filled_qty=str(req.qty))

    def get_order_by_id(self, oid):
        return SimpleNamespace(id=oid, status="filled", filled_avg_price="100.0", filled_qty="1")

    def get_open_position(self, symbol):
        if self.position == 0:
            raise RuntimeError("no position")
        return SimpleNamespace(qty=str(self.position))

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)


def test_mirror_flip_is_split_into_two_legs():
    client = _FlipClient(position=100.0)
    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=client)
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    ack = broker.submit(Order("SYN", T, "sell", 250.0, -0.15, 0.1, 100.0, order_id="r-1"))
    assert ack["flip"] is True and [r.qty for r in client.submitted] == [100, 150]
    assert client.submitted[0].client_order_id == "r-1c" and client.submitted[1].client_order_id == "r-1o"
    ack2 = broker.submit(Order("SYN", T, "sell", 50.0, 0.05, 0.1, 100.0, order_id="r-2"))    # reduce only: one leg
    assert "flip" not in ack2 and client.submitted[-1].qty == 50


def test_cancel_pending_cancels_mirror_and_flags_drift():
    class PartialClient(_FlipClient):
        def submit_order(self, req):
            self.submitted.append(req)
            return SimpleNamespace(id="p1", status="accepted", filled_avg_price=None, filled_qty="0")

        def get_order_by_id(self, oid):
            return SimpleNamespace(id=oid, status="canceled", filled_avg_price="100.0", filled_qty="30")

    client = PartialClient(position=0.0)
    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=client)
    eng = ExecutionEngine(CostModel(), OrderBuilder(id_prefix="c"), broker=broker)
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    order = eng.build_order("SYN", T, 0.0, 0.0, 0.1, 100000.0, 100.0, BotState.READY)
    eng.queue_for_next_bar(order)
    assert order.order_id in eng.pending_mirrors
    cancelled = eng.cancel_pending()
    assert cancelled == [order] and client.cancelled == ["p1"] and not eng.pending_mirrors
    kinds = [e["event"] for e in eng.events]
    assert "ORDER_MIRROR_CANCELLED" in kinds and "MIRROR_DRIFT_RISK" in kinds
