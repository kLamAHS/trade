"""Sizing, risk-rule, halt, validator and state-machine tests (spec sections 25-35, 44)."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.validator import DataValidator
from trading_bot.risk.limits import apply_position_rules, stop_distance, stop_triggered
from trading_bot.risk.manager import RiskEngine
from trading_bot.strategy.sizing import (confidence_from_edge, direction_from_edge, net_edge, raw_exposure,
                                         turnover_suppressed, volatility_multiplier)
from trading_bot.strategy.signal import SignalEngine
from trading_bot.types import Bar, BotState, FeatureVector, PortfolioSnapshot, Prediction, CostEstimate

T0 = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)


def test_edge_threshold_direction_and_confidence():
    cost = 0.001
    assert direction_from_edge(0.0031, cost) == 1
    assert direction_from_edge(-0.0031, cost) == -1
    assert direction_from_edge(0.003, cost) == 0
    assert direction_from_edge(float("nan"), cost) == 0
    assert net_edge(0.005, cost) == pytest.approx(0.002)
    assert confidence_from_edge(0.003, cost) == pytest.approx(0.5)
    assert confidence_from_edge(0.012, cost) == 1.0


def test_volatility_multiplier_and_raw_exposure():
    assert volatility_multiplier(0.01, 0.02) == 0.5
    assert volatility_multiplier(0.01, 0.001) == 1.5
    assert volatility_multiplier(0.01, 0.1) == 0.25
    assert volatility_multiplier(float("nan"), 0.1) == 0.25
    assert raw_exposure(1, 0.8, 1.5) == 1.0
    assert raw_exposure(-1, 0.5, 0.5) == -0.25
    assert raw_exposure(0, 1.0, 1.5) == 0.0


def test_turnover_suppression():
    assert turnover_suppressed(0.50, 0.40) == 0.40
    assert turnover_suppressed(0.56, 0.40) == 0.56
    assert turnover_suppressed(-0.10, 0.40) == -0.10   # sign change always rebalances
    assert turnover_suppressed(0.0, 0.10) == 0.0        # going flat is a direction change
    assert turnover_suppressed(0.12, 0.0) == 0.12


def test_position_rules_stop_and_max_holding():
    r = apply_position_rules(0.5, 0.4, holding_bars=3, max_holding_bars=12, stop_hit=True, rebalance_threshold=0.15)
    assert r.exposure == 0.0 and r.reason == "STOP_LOSS" and r.stop_status == "TRIGGERED"
    r = apply_position_rules(0.0, 0.4, 12, 12, False, 0.15)
    assert r.exposure == 0.0 and r.reason == "MAX_HOLDING_EXIT" and r.max_holding_status == "EXPIRED"
    r = apply_position_rules(0.45, 0.4, 12, 12, False, 0.15)
    assert r.exposure == 0.45 and r.new_entry and r.reason == "MAX_HOLDING_REENTRY"
    r = apply_position_rules(0.45, 0.4, 5, 12, False, 0.15)
    assert r.exposure == 0.4 and not r.new_entry and r.reason == "TURNOVER_SUPPRESSED"
    r = apply_position_rules(0.3, 0.0, 0, 12, False, 0.15)
    assert r.exposure == 0.3 and r.new_entry
    assert stop_distance(0.01, 4) == pytest.approx(0.08)
    assert stop_triggered(-0.081, 0.01, 4) and not stop_triggered(-0.079, 0.01, 4)


def _snapshot(exposure=0.0, units=0.0, daily=0.0, dd=0.0, holding=0, pos_ret=0.0, sigma=0.01):
    return PortfolioSnapshot(T0, 1e5, units, 100.0, 1e5, exposure, 1e5, dd, 1e5, daily, 0.0, 0.0,
                             0 if units else None, 100.0 if units else None, sigma if units else None, holding, pos_ret)


def _features(sigma=0.01):
    return FeatureVector("SYN", T0, T0, 0, 0.5, 500, {"sigma_h": sigma, "range_rel": 0.005, "spread_rel": 1e-4})


def _signal(engine, er, cost=0.001, sigma=0.01):
    return engine.build(T0, er, cost, sigma, sigma)


def test_signal_engine_pipeline():
    eng = SignalEngine(cost_multiplier=3.0, confidence_cost_multiplier=6.0, horizon=4)
    pred = Prediction(T0, expected_normalized_return=0.5, expected_raw_return=float("nan"), probability_up=0.7,
                      model_confidence=0.4, model_version="m")
    eng.seed_sigma_history([0.01] * 10)
    sig = eng.create(pred, _features(0.01), CostEstimate(0.0, 0.0002, 0.0005))
    assert sig.expected_return == pytest.approx(0.5 * 0.01 * 2)
    assert sig.direction == 1
    assert sig.expected_net_edge == pytest.approx(0.01 - 3 * 0.0007)
    assert sig.confidence == pytest.approx(min(1.0, 0.01 / (6 * 0.0007)))
    assert sig.target_exposure == pytest.approx(sig.confidence * 1.0)
    weak = Prediction(T0, 0.05, float("nan"), 0.55, 0.1, "m")
    assert eng.create(weak, _features(0.01), CostEstimate(0.0, 0.0002, 0.0005)).target_exposure == 0.0


def test_risk_engine_daily_loss_halt_and_recovery():
    risk = RiskEngine(daily_loss_limit=0.025)
    eng = SignalEngine()
    day = T0.date()
    d = risk.evaluate(_signal(eng, 0.01), _snapshot(exposure=0.5, units=500, daily=-0.03), _features(), day)
    assert d.approved_exposure == 0.0 and d.daily_loss_status == "DAILY_RISK_HALT"
    assert any(e["event"] == "DAILY_RISK_HALT" for e in risk.events)
    assert risk.state_for(0.0) == BotState.RISK_HALTED
    d2 = risk.evaluate(_signal(eng, 0.01), _snapshot(), _features(), day)
    assert d2.approved_exposure == 0.0
    d3 = risk.evaluate(_signal(eng, 0.01), _snapshot(), _features(), day + timedelta(days=1))
    assert d3.approved_exposure > 0 and d3.daily_loss_status == "OK"


def test_risk_engine_drawdown_halt_until_retrain_accepted():
    risk = RiskEngine(drawdown_halt=0.10)
    eng = SignalEngine()
    d = risk.evaluate(_signal(eng, 0.01), _snapshot(dd=-0.11), _features(), T0.date())
    assert d.approved_exposure == 0.0 and d.drawdown_status == "DRAWDOWN_HALT"
    assert risk.drawdown_halted
    risk.record_retrain(accepted=False, delta_score=1.0)
    assert risk.drawdown_halted
    risk.record_retrain(accepted=True, delta_score=1.0)
    assert not risk.drawdown_halted
    assert risk.evaluate(_signal(eng, 0.01), _snapshot(), _features(), T0.date()).approved_exposure > 0


def test_fractional_edge_halt_after_three_failures():
    risk = RiskEngine(ablation_failures_to_halt=3)
    for _ in range(2):
        risk.record_retrain(True, -0.5)
    assert not risk.ablation_halted
    risk.record_retrain(True, 0.0)
    assert risk.ablation_halted
    assert any(e["event"] == "FRACTIONAL_EDGE_NOT_DETECTED" for e in risk.events)
    risk.record_retrain(True, 0.2)
    assert not risk.ablation_halted and risk.ablation_failures == 0


def test_risk_engine_stop_and_max_holding_through_evaluate():
    risk = RiskEngine(maximum_holding_bars=12, stop_sigma_multiple=4.0, horizon=4)
    eng = SignalEngine()
    d = risk.evaluate(_signal(eng, 0.01), _snapshot(exposure=0.5, units=500, pos_ret=-0.09, sigma=0.01), _features(), T0.date())
    assert d.approved_exposure == 0.0 and d.stop_status == "TRIGGERED"
    d = risk.evaluate(_signal(eng, 0.0), _snapshot(exposure=0.5, units=500, holding=12), _features(), T0.date())
    assert d.approved_exposure == 0.0 and d.max_holding_status == "EXPIRED"
    d = risk.evaluate(_signal(eng, 0.01), _snapshot(exposure=0.5, units=500, holding=12), _features(), T0.date())
    assert d.approved_exposure > 0 and d.new_entry


def _bar(ts, o=100.0, h=101.0, l=99.0, c=100.5, v=1000.0, bid=None, ask=None):
    return Bar("SYN", ts, o, h, l, c, v, 30, bid, ask)


def test_data_validator_rejects_and_halts():
    cal = SessionCalendar()
    v = DataValidator(cal, max_abs_log_jump=0.10, instrument="SYN")
    assert v.validate(_bar(T0)).ok
    assert not v.validate(_bar(T0)).ok                                      # duplicate
    assert "duplicate timestamp" in v.check(_bar(T0)).reasons
    assert not v.check(_bar(T0 - timedelta(minutes=30))).ok                 # backward
    r = v.check(_bar(T0 + timedelta(minutes=30), h=98.0))                   # high < low
    assert not r.ok and not r.storable
    r = v.check(_bar(T0 + timedelta(minutes=30), c=-1.0))
    assert not r.ok and "zero/negative price" in r.reasons
    r = v.check(_bar(T0 + timedelta(minutes=30), c=float("nan"), o=float("nan"), h=float("nan"), l=float("nan")))
    assert not r.ok and "non-finite price" in r.reasons
    r = v.check(_bar(T0 + timedelta(minutes=60)))                           # missing 10:00 bar
    assert not r.ok and r.storable and r.halt_only
    r = v.check(_bar(T0 + timedelta(minutes=30), c=120.0, h=121.0))         # extreme jump
    assert not r.ok and r.storable and any("extreme" in x for x in r.reasons)
    r = v.check(_bar(T0 + timedelta(minutes=30), bid=101.0, ask=100.0))
    assert not r.ok and "crossed quote: ask < bid" in r.reasons
    assert not v.check(_bar(T0 + timedelta(hours=8))).ok                     # outside session
    assert v.check(_bar(T0 + timedelta(minutes=30))).ok
    # next session must start at the open
    nxt = T0 + timedelta(days=1)
    v.accept(_bar(T0 + timedelta(minutes=30 * 12)))
    assert v.check(_bar(nxt)).ok
    assert not v.check(_bar(nxt + timedelta(minutes=30))).ok


def test_state_machine_transitions(bot_run):
    bot = bot_run
    events = [e for e in _read_events(bot) if e["event"] == "STATE"]
    assert events[0]["from"] == "INITIALIZING" and events[0]["to"] in ("READY", "POSITIONED")
    transitions = {(e["from"], e["to"]) for e in events}
    assert ("READY", "POSITIONED") in transitions and ("POSITIONED", "READY") in transitions
    assert bot.state in (BotState.READY, BotState.POSITIONED, BotState.RISK_HALTED)
    assert sum(1 for e in events if e["from"] == "INITIALIZING") == 1     # INITIALIZING is left exactly once
    assert all(e["to"] != "INITIALIZING" for e in events)


def _read_events(bot):
    import json
    p = bot.artifacts_dir / "audit" / f"{bot.run_id}_events.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
