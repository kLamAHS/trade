"""Execution, cost model and ledger tests (spec sections 24, 42-44)."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.orders import OrderBuilder, OrderQueue
from trading_bot.execution.simulator import ExecutionEngine, ExecutionSimulator
from trading_bot.portfolio.ledger import PortfolioLedger
from trading_bot.types import Bar, BotState, Fill, Order

T0 = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)   # 09:30 New York


def bar(i, o=100.0, h=101.0, l=99.0, c=100.5, bid=None, ask=None):
    return Bar("SYN", T0 + timedelta(minutes=30 * i), o, h, l, c, 1000.0, 30, bid, ask)


def test_cost_model_round_trip_decomposition():
    cm = CostModel(commission_per_side=0.0001, default_spread=0.0002, slippage_range_fraction=0.05)
    est = cm.estimate(range_rel=0.01, spread_rel=0.0004)
    assert est.commission == pytest.approx(0.0002)
    assert est.spread == pytest.approx(0.0004)
    assert est.slippage == pytest.approx(2 * 0.05 * 0.01)
    assert est.total == pytest.approx(0.0002 + 0.0004 + 0.001)
    assert cm.estimate(0.01, None).spread == 0.0002
    assert cm.estimate(0.01, float("nan")).spread == 0.0002


def test_fill_model_buy_and_sell():
    cm = CostModel(0.0, 0.0002, 0.05)
    sim = ExecutionSimulator(cm)
    o = Order("SYN", T0 + timedelta(minutes=30), "buy", 10.0, 0.5, 0.0, 100.0)
    exec_bar = bar(1, o=200.0, h=204.0, l=196.0, c=200.0)
    fill = sim.simulate_fill(o, exec_bar)
    slip = 0.05 * (204 - 196) / 200 * 200.0
    assert fill.fill_price == pytest.approx(200.0 + 0.5 * 0.0002 * 200.0 + slip)
    assert fill.spread_cost == pytest.approx(0.5 * 0.0002 * 200.0 * 10)
    assert fill.slippage_cost == pytest.approx(slip * 10)
    s = Order("SYN", T0 + timedelta(minutes=30), "sell", 10.0, -0.5, 0.0, 100.0)
    fill_s = sim.simulate_fill(s, exec_bar)
    assert fill_s.fill_price == pytest.approx(200.0 - 0.5 * 0.0002 * 200.0 - slip)
    # quotes on the execution bar override the default spread
    quoted = bar(1, o=200.0, h=204.0, l=196.0, c=200.0, bid=199.9, ask=200.1)
    f2 = sim.simulate_fill(o, quoted)
    assert f2.fill_price == pytest.approx(200.0 + 0.5 * (0.2 / 200.0) * 200.0 + slip)


def test_fill_cannot_precede_signal():
    sim = ExecutionSimulator(CostModel())
    o = Order("SYN", T0 + timedelta(minutes=60), "buy", 1.0, 0.5, 0.0, 100.0)
    with pytest.raises(ValueError):
        sim.simulate_fill(o, bar(1))


def test_order_builder_units_and_min_notional():
    ob = OrderBuilder(min_order_notional=50.0)
    o = ob.build("SYN", T0, current_units=0.0, current_exposure=0.0, target_exposure=0.5, equity=100000.0,
                 estimated_price=200.0)
    assert o.side == "buy" and o.units == pytest.approx(250.0)
    o2 = ob.build("SYN", T0, current_units=250.0, current_exposure=0.5, target_exposure=-0.25, equity=100000.0,
                  estimated_price=200.0)
    assert o2.side == "sell" and o2.units == pytest.approx(375.0)
    assert ob.build("SYN", T0, 250.0, 0.5, 0.5001, 100000.0, 200.0) is None
    assert ob.build("SYN", T0, 0.0, 0.0, 0.5, 100000.0, float("nan")) is None


def test_execution_engine_state_consistency():
    eng = ExecutionEngine(CostModel())
    kw = dict(instrument="SYN", signal_timestamp=T0, current_units=0.0, current_exposure=0.0, equity=1e5,
              estimated_price=100.0)
    assert eng.build_order(target_exposure=0.5, state=BotState.INITIALIZING, **kw) is None
    assert eng.build_order(target_exposure=0.5, state=BotState.RISK_HALTED, **kw) is None
    assert eng.build_order(target_exposure=0.5, state=BotState.DATA_HALTED, **kw) is None
    assert eng.build_order(target_exposure=0.5, state=BotState.READY, **kw) is not None
    flat = eng.build_order(instrument="SYN", signal_timestamp=T0, current_units=500.0, current_exposure=0.5,
                           target_exposure=0.0, equity=1e5, estimated_price=100.0, state=BotState.RISK_HALTED)
    assert flat is not None and flat.side == "sell"
    assert len(eng.rejected) == 3


def test_order_queue_is_consumed_at_next_bar():
    q = OrderQueue()
    o = Order("SYN", T0, "buy", 1.0, 0.1, 0.0, 100.0)
    q.push(o)
    assert len(q) == 1
    assert q.pop_all() == [o]
    assert len(q) == 0 and q.pop_all() == []


def _fill(ts, side, units, price, spread=0.0, slip=0.0, comm=0.0, new_entry=False, sigma=0.01):
    return Fill("x", "SYN", ts - timedelta(minutes=30), ts, side, units, price, price, spread, slip, comm,
                new_entry=new_entry, entry_sigma=sigma)


def test_ledger_round_trip_pnl_and_costs():
    led = PortfolioLedger(100000.0, "SYN")
    d = T0.date()
    led.mark(T0, 100.0, 0, d)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 100.0, 101.0, spread=5.0, slip=2.0, comm=1.0, new_entry=True))
    led.mark(T0 + timedelta(minutes=30), 102.0, 1, d)
    assert led.units == 100.0
    assert led.exposure == pytest.approx(100 * 102.0 / led.equity)
    assert led.holding_bars == 1
    assert led.entry_price == 101.0 and led.entry_direction == 1
    assert led.position_return == pytest.approx(math.log(102.0 / 101.0))
    led.apply(_fill(T0 + timedelta(minutes=60), "sell", 100.0, 103.0, spread=5.0, slip=2.0, comm=1.0))
    led.mark(T0 + timedelta(minutes=60), 103.0, 2, d)
    assert led.units == 0.0
    assert led.realized_pnl == pytest.approx(200.0)
    assert led.equity == pytest.approx(100000.0 + 200.0 - 2.0)   # commissions explicit; spread/slippage in price
    assert led.total_costs == pytest.approx(16.0)
    assert len(led.trades) == 1 and led.trades[0].direction == 1 and led.trades[0].realized_pnl == pytest.approx(200.0)
    assert led.holding_bars == 0 and led.entry_price is None


def test_ledger_flip_opens_new_trade():
    led = PortfolioLedger(100000.0, "SYN")
    d = T0.date()
    led.mark(T0, 100.0, 0, d)
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 100.0, 100.0, new_entry=True))
    led.apply(_fill(T0 + timedelta(minutes=60), "sell", 300.0, 110.0, new_entry=True))
    assert led.units == -200.0
    assert led.realized_pnl == pytest.approx(1000.0)
    assert led.entry_direction == -1 and led.entry_price == 110.0
    assert len(led.trades) == 1
    led.mark(T0 + timedelta(minutes=60), 105.0, 2, d)
    assert led.unrealized_pnl == pytest.approx(-200 * (105.0 - 110.0))
    assert led.position_return == pytest.approx(-math.log(105.0 / 110.0))


def test_daily_return_resets_each_session():
    led = PortfolioLedger(1000.0, "SYN")
    led.mark(T0, 10.0, 0, T0.date())
    led.apply(_fill(T0 + timedelta(minutes=30), "buy", 50.0, 10.0, new_entry=True))
    led.mark(T0 + timedelta(minutes=30), 9.0, 1, T0.date())
    assert led.daily_return == pytest.approx(-50.0 / 1000.0)
    nxt = T0 + timedelta(days=1)
    led.mark(nxt, 9.0, 2, nxt.date())
    assert led.daily_return == pytest.approx(0.0)
    assert led.drawdown == pytest.approx(-50.0 / 1000.0)
