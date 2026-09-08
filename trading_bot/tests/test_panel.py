"""Pooled training rows from other instruments (training.panel).

The bot trades one symbol.  These tests pin the three properties that make pooling legitimate:
rows join the *fit* only, they are causally eligible, and they are put on the primary's scale.
"""

import math

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.config import load_config
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_market
from trading_bot.training.panel import (PanelRows, TrainingPanel, build_panel_rows, label_end_times,
                                        panel_forecast_quality)

SYMS = ("SPY", "QQQ", "IWM")
FAST = {"market": {"instrument": "SPY"},
        "training": {"window_bars": 2200, "minimum_bars": 1800, "retrain_every_bars": 250,
                     "hyperparameter_grid": {"n_estimators": [40], "min_child_samples": [60]},
                     "acceptance": {"min_accuracy": 0.0, "min_correlation": -1.0, "min_net_pnl": -1.0,
                                    "min_profit_factor": 0.0, "max_drawdown": 1.0,
                                    "min_folds_beating_baseline": 0, "require_holdout_edge": False}},
        "models": {"regression": {"num_threads": 1}}}


@pytest.fixture(scope="module")
def market():
    return generate_synthetic_market(2600, seed=11, calendar=SessionCalendar(), symbols=SYMS, primary="SPY",
                                     amplitude=4.0, memory_d=0.45)


@pytest.fixture(scope="module")
def stores(market):
    return (BarStore("SPY", 30, market.bars["SPY"]),
            {s: BarStore(s, 30, market.bars[s]) for s in SYMS[1:]})


def _cfg(enabled: bool, **panel):
    p = {"enabled": enabled, "symbols": list(SYMS[1:])}
    p.update(panel)
    return load_config(overrides=FAST).with_overrides({"training": {"panel": p}})


def test_label_end_times_follow_the_horizon(stores):
    from trading_bot.training.dataset import TrainingDatasetBuilder
    from trading_bot.execution.cost_model import CostModel
    from trading_bot.features.engine import FeatureEngine
    from trading_bot.fractional.engine import FractionalEngine

    store, _ = stores
    cfg = _cfg(False)
    cal = SessionCalendar.from_config(cfg)
    fe = FeatureEngine(cfg, FractionalEngine.from_config(cfg), cal)
    ds = TrainingDatasetBuilder.from_config(cfg, fe, CostModel.from_config(cfg)).build(store, 0.45)
    H = int(cfg.prediction.horizon_bars)
    ends = label_end_times(ds, store, H)
    # a row's label is complete only once bar_index + H + 1 has closed, and never before its own decision
    for i in (0, len(ds) // 2, len(ds) - 1):
        idx = min(int(ds.bar_index[i]) + H + 1, len(store) - 1)
        assert ends[i] == pytest.approx(store[idx].close_time.timestamp())
        assert ends[i] >= ds.close_times[i].timestamp()
    assert np.all(np.diff(ends) >= 0)


def test_panel_rows_require_the_primary_feature_set(stores):
    """A symbol that cannot produce the primary's columns is skipped, never padded: a pooled row has
    to mean the same thing as a primary row, column for column."""
    from trading_bot.training.dataset import TrainingDatasetBuilder
    from trading_bot.execution.cost_model import CostModel
    from trading_bot.features.engine import FeatureEngine
    from trading_bot.fractional.engine import FractionalEngine

    store, panel = stores
    cfg = _cfg(False)
    fe = FeatureEngine(cfg, FractionalEngine.from_config(cfg), SessionCalendar.from_config(cfg))
    ds = TrainingDatasetBuilder.from_config(cfg, fe, CostModel.from_config(cfg)).build(panel["QQQ"], 0.45)
    names = tuple(ds.feature_names)
    assert build_panel_rows("QQQ", ds, panel["QQQ"], 4, names) is not None
    assert build_panel_rows("QQQ", ds, panel["QQQ"], 4, names + ("no_such_feature",)) is None


def test_eligibility_never_admits_a_row_whose_label_is_not_complete():
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
    ends = np.array([(t0 + timedelta(minutes=30 * k)).timestamp() for k in range(10)])
    rows = PanelRows("AAA", np.zeros((10, 2)), np.arange(10.0), np.arange(10.0), ends, ("a", "b"))
    panel = TrainingPanel([rows], align=False)
    cutoff = t0 + timedelta(minutes=30 * 4)
    X, yn, yr = panel.eligible(cutoff, np.zeros((5, 2)))
    assert len(yn) == 5 and yn.max() == 4.0                       # rows 0..4 complete at or before the cutoff
    assert panel.eligible(t0 - timedelta(days=1), np.zeros((5, 2))) is None
    # the out-of-sample view is the complement, half open on the left so no row is counted twice
    oos = panel.out_of_sample(cutoff, t0 + timedelta(minutes=30 * 9))
    assert oos and len(oos[0][2]) == 5 and oos[0][2].min() == 5.0


def test_alignment_puts_a_panel_symbol_on_the_primary_scale():
    """Level-dependent features (log volatility) identify the symbol.  Left alone a tree can split on
    that level and silently un-pool the data, so each symbol is mapped onto the primary's moments."""
    rng = np.random.default_rng(0)
    primary = rng.normal(loc=[-6.7, 0.0], scale=[0.3, 1.0], size=(500, 2))
    panel_raw = rng.normal(loc=[-5.5, 0.0], scale=[0.8, 1.0], size=(500, 2))
    ends = np.full(500, 0.0)
    rows = PanelRows("VOLATILE", panel_raw, rng.normal(size=500), rng.normal(size=500), ends, ("vol_10", "return_1"))
    from datetime import datetime, timezone
    cutoff = datetime(1970, 1, 2, tzinfo=timezone.utc)

    aligned = TrainingPanel([rows], align=True).eligible(cutoff, primary)[0]
    assert aligned.mean(axis=0) == pytest.approx(primary.mean(axis=0), abs=0.05)
    assert aligned.std(axis=0) == pytest.approx(primary.std(axis=0), abs=0.05)
    unaligned = TrainingPanel([rows], align=False).eligible(cutoff, primary)[0]
    assert abs(unaligned[:, 0].mean() - primary[:, 0].mean()) > 0.5      # the raw level still identifies the symbol


def test_panel_leaves_the_primary_untouched_when_disabled(stores):
    """With the panel off, training is bit-for-bit what it was: the pooled path is opt-in."""
    store, panel = stores
    off = TradingBot(_cfg(False), run_id="off", artifacts_dir=None, log=None).trainer
    rep_off = off.retrain(store, None, panel=panel)                      # stores supplied but the family is disabled
    ignored = TradingBot(_cfg(False), run_id="ign", artifacts_dir=None, log=None).trainer.retrain(store, None)
    assert rep_off.panel_diagnosis["pooled_rows"] == 0
    assert rep_off.model is not None and ignored.model is not None
    assert rep_off.model.version == ignored.model.version                # identical fitted artefact
    assert np.array_equal(rep_off.holdout_metrics.equity_curve, ignored.holdout_metrics.equity_curve)


def test_pooled_rows_enlarge_the_fit_but_not_the_evaluation(stores):
    store, panel = stores
    bot = TradingBot(_cfg(True), run_id="panel", artifacts_dir=None, log=None)
    rep = bot.trainer.retrain(store, None, panel=panel)
    pd = rep.panel_diagnosis
    assert pd["enabled"] and set(pd["symbols"]) == set(SYMS[1:])
    assert pd["rows_in_final_fit"] > pd["primary_rows_in_final_fit"]      # the sample really did grow
    # evaluation stayed on the primary: the acceptance sample is the primary's own holdout rows
    assert rep.holdout_rows == rep.holdout_diagnosis["rows"] > 0
    assert rep.holdout_metrics.n <= rep.holdout_rows
    # and the panel is scored out of sample by the holdout model, which never saw those rows
    oos = pd["out_of_sample"]
    assert oos["rows"] > 0 and set(oos["symbols"]) == set(SYMS[1:])
    assert all(-1.0 <= v["correlation"] <= 1.0 for v in oos["symbols"].values())


def test_panel_symbols_are_never_traded(stores):
    """The panel is a training input.  It must not reach the ledger, the risk engine or any order."""
    store, panel = stores
    bot = TradingBot(_cfg(True), run_id="notrade", artifacts_dir=None, log=None)
    for sym, ps in panel.items():
        for b in ps.bars[:200]:
            bot.on_panel_bar(b)
    assert set(bot.panel) == set(SYMS[1:]) and bot.ledger.instrument == "SPY"
    assert set(bot.panel) & set(bot.context) == set()          # panel and cross-asset context are separate roles
    for f in bot.ledger.fills:
        assert f.instrument == "SPY"
    assert bot._panel_snapshot() is not None and len(bot._panel_snapshot()["QQQ"]) == 200


def test_panel_forecast_quality_reports_per_symbol():
    class _Model:
        feature_names = ("a",)

        def predict_arrays(self, X):
            return {"E": X[:, 0]}

    y = np.array([1.0, 2.0, -1.0, -2.0])
    rows = [("AAA", y.reshape(-1, 1), y), ("BBB", (-y).reshape(-1, 1), y)]
    out = panel_forecast_quality(_Model(), rows)
    assert out["symbols"]["AAA"]["correlation"] == pytest.approx(1.0)
    assert out["symbols"]["BBB"]["correlation"] == pytest.approx(-1.0)
    assert out["rows"] == 8 and out["correlation"] == pytest.approx(0.0, abs=1e-9)
    assert out["forecast_constant"] is False


def test_panel_quality_says_no_forecast_rather_than_zero_correlation():
    """A collapsed calibrator forecasts the same number everywhere.  Reporting a correlation of zero
    would read as a measured result; there is nothing to measure."""
    class _Constant:
        feature_names = ("a",)

        def predict_arrays(self, X):
            return {"E": np.zeros(len(X))}

    y = np.array([1.0, -2.0, 3.0])
    out = panel_forecast_quality(_Constant(), [("AAA", y.reshape(-1, 1), y)])
    assert out["forecast_constant"] is True and out["signal_rows"] == 0
    assert math.isnan(out["correlation"]) and math.isnan(out["accuracy"])
    assert out["symbols"]["AAA"]["forecast_constant"] is True


def test_a_panel_symbol_from_the_future_contributes_nothing(stores):
    """The causality guarantee at the level that matters: an instrument whose history lies entirely
    after the primary's training cutoff must not reach a single fit, and the fitted artefact must be
    identical to training with no panel at all."""
    from datetime import timedelta

    store, _ = stores
    shift = store.last().timestamp - store[0].timestamp + timedelta(days=365)
    future = generate_synthetic_market(2600, seed=11, calendar=SessionCalendar(), symbols=SYMS, primary="SPY",
                                       amplitude=4.0, memory_d=0.45)
    from trading_bot.types import Bar
    shifted = BarStore("QQQ", 30, [Bar(b.instrument, b.timestamp + shift, b.open, b.high, b.low, b.close,
                                       b.volume, b.bar_minutes) for b in future.bars["QQQ"]])

    with_future = TradingBot(_cfg(True, symbols=["QQQ"]), run_id="fut", artifacts_dir=None, log=None).trainer
    rep_future = with_future.retrain(store, None, panel={"QQQ": shifted})
    rep_none = TradingBot(_cfg(False), run_id="none", artifacts_dir=None, log=None).trainer.retrain(store, None)

    assert rep_future.panel_diagnosis["pooled_rows"] > 0            # the symbol was built...
    assert rep_future.panel_diagnosis["rows_in_final_fit"] == 0     # ...and every row was refused as incomplete
    assert rep_future.model is not None and rep_none.model is not None
    # the fitted behaviour is identical: no future row reached any model.  (The model *id* differs only
    # because the configuration digest records that a panel was configured at all.)
    assert rep_future.model.feature_names == rep_none.model.feature_names
    rng = np.random.default_rng(3)
    X = rng.normal(size=(64, len(rep_none.model.feature_names)))
    a, b = rep_future.model.predict_arrays(X), rep_none.model.predict_arrays(X)
    for k in ("M", "P", "E"):
        assert np.array_equal(a[k], b[k])
    assert np.array_equal(rep_future.holdout_metrics.equity_curve, rep_none.holdout_metrics.equity_curve)
