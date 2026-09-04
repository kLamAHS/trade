"""Leakage tests (spec section 54).  Failure of any test invalidates the backtest."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from trading_bot.data.store import BarStore
from trading_bot.execution.cost_model import CostModel
from trading_bot.fractional.engine import FractionalEngine
from trading_bot.fractional.stationarity import StationaryOrderEstimator
from trading_bot.models.direction import DirectionModel
from trading_bot.training.dataset import TrainingDatasetBuilder
from trading_bot.training.walkforward import walk_forward_folds
from trading_bot.types import Bar


def _mutate_future(bars, start):
    out = list(bars[:start])
    for b in bars[start:]:
        out.append(Bar(b.instrument, b.timestamp, b.open * 1.5, b.high * 1.7, b.low * 1.2, b.close * 1.4,
                       b.volume * 3, b.bar_minutes, b.bid, b.ask))
    return out


def test_future_bar_mutation_leaves_past_features_identical(feature_engine, bars_1500):
    cut = 1300
    fm_a = feature_engine.compute_matrix(BarStore("SYN", 30, bars_1500))
    fm_b = feature_engine.compute_matrix(BarStore("SYN", 30, _mutate_future(bars_1500, cut)))
    a, b = fm_a.values[:cut], fm_b.values[:cut]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.array_equal(a[~both_nan], b[~both_nan])
    assert not np.array_equal(np.nan_to_num(fm_a.values[cut:]), np.nan_to_num(fm_b.values[cut:]))


def test_future_bar_mutation_leaves_past_labels_identical(cfg, feature_engine, bars_1500):
    builder = TrainingDatasetBuilder.from_config(cfg, feature_engine, CostModel.from_config(cfg))
    cut = 1400
    ds_a = builder.build(BarStore("SYN", 30, bars_1500), 0.35)
    ds_b = builder.build(BarStore("SYN", 30, _mutate_future(bars_1500, cut)), 0.35)
    # rows whose label horizon (t+H+1) ends before the cut must be identical
    H = cfg.prediction.horizon_bars
    keep = ds_a.bar_index + H + 2 < cut
    keep_b = ds_b.bar_index + H + 2 < cut
    assert np.array_equal(ds_a.bar_index[keep], ds_b.bar_index[keep_b])
    assert np.array_equal(ds_a.y_norm[keep], ds_b.y_norm[keep_b])
    assert np.array_equal(ds_a.X[keep], ds_b.X[keep_b])


def test_label_definition_is_executable_forward_return(cfg, feature_engine, bars_1500):
    builder = TrainingDatasetBuilder.from_config(cfg, feature_engine, CostModel.from_config(cfg))
    store = BarStore("SYN", 30, bars_1500)
    ds = builder.build(store, 0.35)
    H = cfg.prediction.horizon_bars
    p = np.log(store.arrays()["close"])
    for k in (0, 10, len(ds) - 1):
        t = ds.bar_index[k]
        assert ds.y_raw[k] == pytest.approx(p[t + H + 1] - p[t + 1])
        assert ds.y_norm[k] == pytest.approx(ds.y_raw[k] / (ds.sigma[k] * math.sqrt(H) + 1e-12))
        assert ds.open_next[k] == store[t + 1].open
        assert ds.open_next2[k] == store[t + 2].open
    assert ds.bar_index.max() <= len(store) - (H + 2)


def test_walk_forward_folds_are_chronological_with_purge_and_embargo():
    folds = walk_forward_folds(2000, 5, 0.40, 0.10, purge=5, embargo=5)
    assert len(folds) == 5
    for f in folds:
        assert f.train[0] == 0
        assert f.train[-1] < f.validate[0]
        assert f.gap >= 10
        assert np.all(np.diff(f.train) == 1) and np.all(np.diff(f.validate) == 1)
    for a, b in zip(folds, folds[1:]):
        assert a.validate[-1] < b.validate[0]
        assert len(b.train) > len(a.train)
    # fold 5 trains on the earliest 80% and validates the following 10% (spec section 38 example)
    assert folds[-1].validate[-1] == int(2000 * 0.9) - 1
    assert folds[0].train[-1] == int(2000 * 0.4) - 5 - 1


def test_scaling_statistics_exclude_validation_rows():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 4)
    y = rng.randn(500)
    train, val = np.arange(0, 300), np.arange(320, 500)
    X[val] += 50.0  # validation rows are wildly shifted
    m = DirectionModel().fit(X[train], y[train])
    assert np.allclose(m.scaler_mean, X[train].mean(axis=0))
    assert np.allclose(m.scaler_scale, X[train].std(axis=0))


def test_fractional_order_uses_only_training_interval():
    rng = np.random.RandomState(6)
    p = np.cumsum(rng.randn(2500)) / 100 + 5
    est = StationaryOrderEstimator(candidates=[0.2, 0.5, 0.8], adf_maxlag=5)
    interval = p[:1800]
    d_a = est.estimate(interval).d_star
    q = p.copy()
    q[1800:] += np.linspace(0, 50, 700)   # corrupt the future
    d_b = est.estimate(q[:1800]).d_star
    assert d_a == d_b
    assert est.estimate(interval).to_dict() == est.estimate(q[:1800]).to_dict()


def test_execution_shift_all_fills_after_signal(bot_run):
    bot = bot_run
    assert bot.ledger.fills, "the integration run must produce fills"
    index_by_start = {b.timestamp: i for i, b in enumerate(bot.store)}
    index_by_close = {b.close_time: i for i, b in enumerate(bot.store)}
    for fill in bot.ledger.fills:
        assert fill.fill_timestamp >= fill.signal_timestamp
        exec_idx = index_by_start[fill.fill_timestamp]
        signal_idx = index_by_close[fill.signal_timestamp]
        # fills happen at the open of the very next stored bar after the signal bar (across session gaps too)
        assert exec_idx == signal_idx + 1
        assert bot.store[exec_idx].timestamp >= fill.signal_timestamp
        assert fill.reference_price == bot.store[exec_idx].open


def test_label_columns_never_enter_feature_code():
    root = Path(__file__).resolve().parents[1]
    forbidden = re.compile(r"\b(y_norm|y_raw|label|forward_return|open_next)\b")
    for path in (root / "features").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), f"label-like symbol found in {path.name}"
    for path in (root / "fractional").glob("*.py"):
        assert not forbidden.search(path.read_text(encoding="utf-8")), path.name


def test_feature_values_do_not_depend_on_rows_beyond_window(feature_engine, bars_1500):
    fm_short = feature_engine.compute_matrix(BarStore("SYN", 30, bars_1500[:1300]))
    fm_full = feature_engine.compute_matrix(BarStore("SYN", 30, bars_1500))
    a, b = fm_short.values, fm_full.values[:1300]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.array_equal(a[~both_nan], b[~both_nan])
