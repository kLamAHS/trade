"""Training pipeline, reproducibility, registry, calibration and end-to-end bot tests."""

import json
import math

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.feed import ReplayFeed
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.execution.cost_model import CostModel
from trading_bot.features.engine import FeatureEngine
from trading_bot.models.calibration import Calibrator
from trading_bot.models.combined import combine
from trading_bot.models.registry import ModelRegistry
from trading_bot.training.trainer import ModelTrainer
from trading_bot.training.validation import SimulationParams, simulate_validation
from trading_bot.types import BotState

# Small windows and a permissive acceptance gate so the integration run always exercises the
# trading path; the acceptance criteria themselves are unit-tested in test_training_and_bot/test_risk.
FAST = {"market": {"instrument": "SYN"},
        "training": {"window_bars": 1600, "minimum_bars": 1400, "retrain_every_bars": 250,
                     "hyperparameter_grid": {"n_estimators": [40], "min_child_samples": [50]},
                     "acceptance": {"min_accuracy": 0.0, "min_correlation": -1.0, "min_net_pnl": -1.0,
                                    "min_profit_factor": 0.0, "max_drawdown": 1.0, "min_folds_beating_baseline": 0,
                                    "require_holdout_edge": False}},
        "models": {"regression": {"num_threads": 1}}}


@pytest.fixture(scope="session")
def fast_cfg(cfg):
    return cfg.with_overrides(FAST)


@pytest.fixture(scope="session")
def bot_run(fast_cfg, tmp_path_factory):
    bars = generate_synthetic_bars(2000, seed=21, instrument="SYN", memory_d=0.45, amplitude=6.0)
    bot = TradingBot(fast_cfg, run_id="test_run", artifacts_dir=tmp_path_factory.mktemp("artifacts"), log=None)
    bot.run(ReplayFeed(bars, bot.calendar))
    return bot


def test_combine_rules():
    A, D = combine(np.array([0.5, -0.5, 0.5, 0.0]), np.array([0.8, 0.8, 0.2, 0.5]))
    assert A[0] == pytest.approx(0.5 * 0.6)
    assert A[1] == 0.0        # magnitude says down, direction says up
    assert A[2] == 0.0
    assert A[3] == 0.0


def test_calibrator_monotone_and_zero_maps_to_zero():
    rng = np.random.RandomState(0)
    A = rng.randn(2000)
    y = 0.5 * A + rng.randn(2000) * 0.3
    for method in ("isotonic", "binned"):
        c = Calibrator(method).fit(A, y)
        xs = np.linspace(-2, 2, 41)
        ys = c.predict(xs)
        assert np.all(np.diff(ys[xs != 0]) >= -1e-12)
        assert c.predict(np.array([0.0]))[0] == 0.0
        assert c.predict(np.array([]).reshape(0)).shape == (0,)


def test_validation_simulator_costs_and_trades():
    n = 60
    params = SimulationParams(horizon=4, bars_per_day=13)
    E = np.zeros(n); E[5:20] = 2.0; E[30:40] = -2.0
    sigma = np.full(n, 0.01); sigma_ref = np.full(n, 0.01)
    cost_rt = np.full(n, 0.001); cost_side = np.full(n, 0.0005)
    opens = 100.0 * np.exp(np.cumsum(np.r_[0, np.full(n + 1, 0.001)]))
    open_next, open_next2 = opens[1:n + 1], opens[2:n + 2]
    log_close = np.log(opens[:n])
    y = np.ones(n)
    m = simulate_validation(E, y, sigma, sigma_ref, cost_rt, cost_side, log_close, open_next, open_next2, params)
    assert m.n_trades >= 2
    assert m.total_cost > 0
    assert m.net_pnl == pytest.approx(m.equity_curve[-1] - 1.0)
    assert m.accuracy == pytest.approx(np.mean(np.sign(E[E != 0]) == 1))
    assert 0 <= m.max_drawdown <= 1


def test_trainer_reproducible_and_registry_roundtrip(fast_cfg, fractional, calendar, tmp_path):
    bars = generate_synthetic_bars(1700, seed=5, instrument="SYN", memory_d=0.45, amplitude=6.0)
    store = BarStore("SYN", 30, bars)
    cm = CostModel.from_config(fast_cfg)

    def run():
        eng = FeatureEngine(fast_cfg, fractional, calendar)
        tr = ModelTrainer(fast_cfg, eng, fractional, cm)
        return tr.retrain(store)

    r1, r2 = run(), run()
    assert r1.error is None, r1.error
    assert r1.model.version == r2.model.version
    assert r1.stationarity.d_star == r2.stationarity.d_star
    X = np.random.RandomState(0).randn(5, len(r1.model.feature_names))
    assert np.array_equal(r1.model.predict_arrays(X)["E"], r2.model.predict_arrays(X)["E"])
    meta = r1.model.metadata
    for key in ("model_id", "training_start", "training_end", "feature_schema_version", "source_data_checksum",
                "fractional_d", "fractional_kernel_size", "normalization", "model_params", "random_seed",
                "validation_metrics", "software_version"):
        assert getattr(meta, key) is not None
    assert meta.source_data_checksum == store.last(fast_cfg.training.window_bars).checksum()
    reg = ModelRegistry(tmp_path / "models")
    reg.promote(r1.model)
    loaded = ModelRegistry(tmp_path / "models").load_current()
    assert loaded.version == r1.model.version
    assert np.array_equal(loaded.predict_arrays(X)["E"], r1.model.predict_arrays(X)["E"])
    assert (tmp_path / "models" / r1.model.version / "metadata.json").exists()
    # ablation baseline evaluated with the same hyperparameters and no fractional features
    assert len(r1.baseline_fold_metrics) == len(r1.full_fold_metrics) == fast_cfg.training.walk_forward_folds
    assert math.isfinite(r1.delta_score)


def test_bot_end_to_end(bot_run, fast_cfg):
    bot = bot_run
    assert bot.retrain_count >= 2
    assert bot.registry.current is not None
    assert bot.state != BotState.INITIALIZING
    assert bot.ledger.fills, "expected at least one fill"
    assert bot.ledger.equity > 0
    audit_dir = bot.artifacts_dir / "audit"
    bars = [json.loads(l) for l in (audit_dir / "test_run_bars.jsonl").read_text().splitlines()]
    assert len(bars) == 2000
    traded = [b for b in bars if b["order"]]
    assert traded
    rec = traded[0]
    for key in ("bar", "features", "prediction", "cost_estimate", "signal", "risk", "order", "portfolio",
                "fractional_d", "fractional_kernel_size", "model_version", "state"):
        assert rec[key] is not None, key
    assert rec["risk"]["approved_exposure"] == rec["order"]["target_exposure"]
    fills = [json.loads(l) for l in (audit_dir / "test_run_fills.jsonl").read_text().splitlines()]
    assert len(fills) == len(bot.ledger.fills)
    summary = json.loads((audit_dir / "test_run_summary.json").read_text())
    assert summary["metrics"]["trade_count"] == len(bot.ledger.trades)
    assert (bot.artifacts_dir / "diagnostics" / "fractional_contribution.csv").exists()
    assert (bot.artifacts_dir / "diagnostics" / "stationarity_vs_d.csv").exists()
    # exposure never exceeds the configured cap
    for b in bars:
        if b["portfolio"]:
            assert abs(b["portfolio"]["exposure"]) <= fast_cfg.risk.max_absolute_exposure + 0.05
    # every fill was queued at the previous bar and executed at the next bar open
    ts_to_bar = {b["timestamp"]: b for b in bars}
    for f in fills:
        exec_bar = ts_to_bar[f["fill"]["fill_timestamp"]]
        assert f["fill"]["reference_price"] == exec_bar["bar"]["open"]


def test_bot_halts_on_corrupt_bar(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1500, seed=8, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1450, "window_bars": 1450}}),
                     run_id="halt", artifacts_dir=tmp_path, log=None)
    from trading_bot.types import Bar
    b = bars[1470]
    bars[1470] = Bar(b.instrument, b.timestamp, b.open, b.low - 1, b.high + 1, b.close, b.volume, 30)  # high < low
    bot.run(ReplayFeed(bars, bot.calendar))
    assert bot.rejected_bars == 1
    assert len(bot.store) == 1499
    events = [json.loads(l) for l in (tmp_path / "audit" / "halt_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "DATA_HALT" for e in events)
    assert any(e["event"] == "DATA_HALT_CLEARED" for e in events)
