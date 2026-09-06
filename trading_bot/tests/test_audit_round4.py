"""Regression tests for the September 2026 external source audit (docs/AUDIT_RESPONSE.md)."""

import json
from datetime import date, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.orders import OrderBuilder
from trading_bot.execution.simulator import (AlpacaPaperBroker, ExecutionEngine, FillDeferred, LiveTradingNotSupported,
                                             paper_trading_client)
from trading_bot.features.engine import FeatureEngine
from trading_bot.gui.controller import BotController
from trading_bot.gui.settings import GuiSettings
from trading_bot.training.trainer import ModelTrainer
from trading_bot.types import Bar, BotState, Order

NY = SessionCalendar()
UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ 1. paper only
def test_live_trading_is_impossible(fast_cfg, tmp_path):
    with pytest.raises(LiveTradingNotSupported):
        AlpacaPaperBroker(api_key="k", secret_key="s", paper=False)
    with pytest.raises(LiveTradingNotSupported):
        TradingBot(fast_cfg.with_overrides({"alpaca": {"paper": False}}), run_id="live", artifacts_dir=tmp_path, log=None)
    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"overrides": "alpaca.paper=false"})
    with pytest.raises(RuntimeError):
        ctl.build_config()
    s = GuiSettings()
    assert s.update({"paper": False, "symbol": "SPY"}) == []           # 'paper' is not a setting any more
    assert not hasattr(s, "paper") and s.public()["paper"] is True
    html = (ROOT / "trading_bot" / "gui" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="paper"' not in html and "Paper account only" in html
    # the only TradingClient factory targets the paper endpoint and verifies it
    client = paper_trading_client("k", "s")
    assert "paper" in str(getattr(client, "_base_url", "paper")).lower()


# ------------------------------------------------------------ 2. fill timing (bot level, live-like bars)
def test_live_like_bar_with_quote_fills_at_quote_not_at_earlier_open(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1500, seed=51, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1400, "window_bars": 1400}}),
                     run_id="q", artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:1450])
    b = bars[1450]
    qts = b.close_time + timedelta(seconds=25)                      # decision 25 s after the close, live style
    live_bar = Bar(b.instrument, b.timestamp, b.open, b.high, b.low, b.close, b.volume, 30, bid=b.close - 0.05,
                   ask=b.close + 0.05, quote_timestamp=qts)
    order = bot.execution.build_order("SYN", qts, 0.0, 0.0, 0.3, 100000.0, b.close, BotState.READY)
    bot.execution.queue_for_next_bar(order)
    bot._pending_signal_bar = live_bar
    bot.on_bar(bars[1451])
    fill = bot.ledger.fills[0]
    assert fill.price_source == "quote" and fill.fill_timestamp == qts
    assert fill.fill_price == pytest.approx((b.close + 0.05) + 0.05 * (b.high - b.low) / b.close * b.close)
    assert fill.reference_price != bars[1451].open


def test_broker_fill_source_uses_alpaca_price(fast_cfg):
    class Client:
        def submit_order(self, req):
            return SimpleNamespace(id="a1", status="filled", filled_avg_price="101.25", filled_qty=str(req.qty))

        def get_order_by_id(self, oid):
            return SimpleNamespace(id=oid, status="filled", filled_avg_price="101.25", filled_qty="30")

        def get_open_position(self, symbol):
            raise RuntimeError("none")

    broker = AlpacaPaperBroker(api_key="k", secret_key="s", trading_client=Client())
    eng = ExecutionEngine(CostModel(), OrderBuilder(id_prefix="b"), broker=broker, live_fill_source="broker")
    T = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    order = eng.build_order("SYN", T + timedelta(minutes=30, seconds=20), 0.0, 0.0, 0.03, 100000.0, 100.0, BotState.READY)
    eng.queue_for_next_bar(order)
    exec_bar = Bar("SYN", T + timedelta(minutes=30), 100.0, 101.0, 99.0, 100.5, 10, 30)
    fill = eng.simulate_fill(order, exec_bar, None)                # no quote: the simulator alone would defer
    assert fill.price_source == "broker" and fill.fill_price == 101.25 and fill.source == "broker"
    assert fill.units == pytest.approx(30.0) and fill.fill_timestamp >= order.signal_timestamp
    with pytest.raises(ValueError):
        ExecutionEngine(CostModel(), live_fill_source="nope")


# ------------------------------------------------------------ 3-5. training protocol
def test_holdout_is_untouched_and_drives_acceptance(fast_cfg, fractional):
    cfg = fast_cfg.with_overrides({"training": {"acceptance": {"require_holdout_edge": True}}})
    tr = ModelTrainer(cfg, FeatureEngine(cfg, fractional, NY), fractional, CostModel.from_config(cfg))
    bars = generate_synthetic_bars(1700, seed=61, instrument="SYN", memory_d=0.45, amplitude=6.0)
    store = BarStore("SYN", 30, bars)
    report = tr.retrain(store)
    assert report.error is None
    ds = tr.builder.build(store.last(tr.window_bars), report.stationarity.d_star)
    inner, holdout, folds = tr._layout(len(ds))
    assert len(holdout) == report.holdout_rows > 0
    for f in folds:                                                    # no fold ever touches the holdout rows
        assert f.train[-1] < inner[-1] + 1 and f.validate[-1] <= inner[-1]
        assert not np.intersect1d(np.concatenate([f.train, f.validate]), holdout).size
    assert holdout[0] - inner[-1] - 1 >= tr.purge + tr.embargo        # purged and embargoed gap
    assert report.holdout_metrics is not None and report.baseline_holdout_metrics is not None
    assert report.delta_score == pytest.approx(report.holdout_metrics.score - report.baseline_holdout_metrics.score)
    assert report.full_score == report.holdout_metrics.score
    assert "holdout_edge" in report.acceptance.checks
    assert report.acceptance.values["accuracy"] == report.holdout_metrics.accuracy       # section-39 metrics from the holdout
    assert len(report.fold_d_stars) == cfg.training.walk_forward_folds
    meta = report.model.metadata
    assert meta.validation_metrics["holdout_rows"] == report.holdout_rows
    assert meta.validation_metrics["fold_d_stars"] == report.fold_d_stars


def test_holdout_can_be_disabled_for_fold_only_evaluation(fast_cfg, fractional):
    cfg = fast_cfg.with_overrides({"training": {"outer_holdout_fraction": 0, "fold_local_d": False}})
    tr = ModelTrainer(cfg, FeatureEngine(cfg, fractional, NY), fractional, CostModel.from_config(cfg))
    bars = generate_synthetic_bars(1500, seed=62, instrument="SYN")
    report = tr.retrain(BarStore("SYN", 30, bars))
    assert report.error is None and report.holdout_metrics is None and report.holdout_rows == 0
    assert report.full_score == pytest.approx(report.fold_full_score)
    assert "holdout_edge" not in report.acceptance.checks
    assert all(d == report.stationarity.d_star for d in report.fold_d_stars)      # whole-window d everywhere


def test_fold_local_d_changes_fold_features_when_it_differs(fast_cfg, fractional):
    """Folds with a different d* than the window get their adaptive channel rebuilt with it."""
    from trading_bot.training.trainer import FoldSet

    tr = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    bars = generate_synthetic_bars(1500, seed=63, instrument="SYN")
    store = BarStore("SYN", 30, bars)
    ds = tr.builder.build(store, 0.40)
    _, _, folds = tr._layout(len(ds))
    sets = tr.build_fold_sets(store, ds, folds, fixed_d=0.75)
    assert all(fs.d_star == 0.75 for fs in sets)
    j = ds.feature_names.index("fd_adaptive_z")
    assert not np.allclose(sets[0].dataset.X[:, j], ds.X[:, j])
    k = ds.feature_names.index("fd_025_z")
    assert np.allclose(sets[0].dataset.X[:, k], ds.X[:, k])          # fixed channels are unaffected
    assert np.array_equal(sets[0].dataset.bar_index, ds.bar_index)


# ------------------------------------------------------------ 6-7. reproducibility, CI
def test_artifact_records_environment_and_lock_file_exists(fast_cfg, fractional):
    tr = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    report = tr.retrain(BarStore("SYN", 30, generate_synthetic_bars(1500, seed=64, instrument="SYN")))
    env = report.model.metadata.environment
    assert env["python"] and env["packages"]["lightgbm"] and env["packages"]["scikit-learn"] and env["git_commit"]
    lock = ROOT / "requirements.lock"
    assert lock.exists()
    pinned = [l.split("==")[0].lower() for l in lock.read_text().splitlines() if "==" in l]
    for pkg in ("numpy", "pandas", "scikit-learn", "lightgbm", "statsmodels", "alpaca-py"):
        assert pkg in pinned, pkg
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.exists() and "pytest" in ci.read_text() and "requirements.lock" in ci.read_text()
