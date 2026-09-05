"""Leakage tests (spec section 54).  Failure of any test invalidates the backtest."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.execution.cost_model import CostModel
from trading_bot.fractional.engine import FractionalEngine
from trading_bot.fractional.stationarity import StationaryOrderEstimator
from trading_bot.models.direction import DirectionModel
from trading_bot.training.dataset import TrainingDataset, TrainingDatasetBuilder
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


def test_scaling_statistics_exclude_validation_rows(fast_cfg, fractional, bars_1500):
    """The trainer's per-fold direction models scale with training-row statistics only."""
    from trading_bot.training.trainer import ModelTrainer
    from trading_bot.features.engine import FeatureEngine
    from trading_bot.data.calendar import SessionCalendar

    store = BarStore("SYN", 30, bars_1500)
    trainer = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, SessionCalendar()), fractional,
                           CostModel.from_config(fast_cfg))
    ds = trainer.builder.build(store, 0.4)
    folds = walk_forward_folds(len(ds), trainer.n_folds, trainer.first_train_fraction, trainer.fold_validation_fraction,
                               trainer.purge, trainer.embargo)
    ev = trainer.evaluate_candidate(ds, folds, trainer.grid[0], ds.feature_names)
    Xall = ds.columns(ds.feature_names)
    for fold, fp in zip(folds, ev.fold_predictions):
        assert np.allclose(fp.direction_model.scaler_mean, Xall[fold.train].mean(axis=0))
        assert np.allclose(fp.direction_model.scaler_scale, Xall[fold.train].std(axis=0))
    # Shifting a fold's *own* validation rows must not change that fold's fitted scaler or regressor
    # (validation blocks are training data for *later* folds, so only the fold itself is checked).
    for i, fold in enumerate(folds):
        ds_shift = TrainingDataset(**{**ds.__dict__, "X": ds.X.copy()})
        ds_shift.X[fold.validate] += 25.0
        ev2 = trainer.evaluate_candidate(ds_shift, folds[: i + 1], trainer.grid[0], ds.feature_names)
        fp, fp2 = ev.fold_predictions[i], ev2.fold_predictions[i]
        assert np.allclose(fp.direction_model.scaler_mean, fp2.direction_model.scaler_mean)
        assert np.allclose(fp.direction_model.scaler_scale, fp2.direction_model.scaler_scale)
        assert np.allclose(fp.regressor.predict(Xall[:5]), fp2.regressor.predict(Xall[:5]))
        assert np.allclose(ev.calibrator_fold[i].predict(np.array([0.3, -0.3])), ev2.calibrator_fold[i].predict(np.array([0.3, -0.3])))


def test_fractional_order_uses_only_training_interval(fast_cfg, fractional):
    """d* of a retrain equals the estimate on the training window alone and ignores bars outside it."""
    from trading_bot.training.trainer import ModelTrainer
    from trading_bot.features.engine import FeatureEngine
    from trading_bot.data.calendar import SessionCalendar

    cfg = fast_cfg.with_overrides({"training": {"window_bars": 1500, "minimum_bars": 1400},
                                   "fractional": {"adaptive_min": 0.2, "adaptive_step": 0.25}})
    bars = generate_synthetic_bars(1900, seed=9, instrument="SYN")
    est = FractionalEngine.from_config(cfg)
    trainer = ModelTrainer(cfg, FeatureEngine(cfg, est, SessionCalendar()), est, CostModel.from_config(cfg))
    store = BarStore("SYN", 30, bars)
    report = trainer.retrain(store)
    window = store.last(1500)
    assert report.stationarity.d_star == est.estimate_stationarity(window.log_close()).d_star
    # corrupt everything *before* the window: d* must not change
    corrupted = [Bar(b.instrument, b.timestamp, b.open * 3, b.high * 3.2, b.low * 2.9, b.close * 3, b.volume, 30)
                 for b in bars[:400]] + bars[400:]
    report2 = trainer.retrain(BarStore("SYN", 30, corrupted))
    assert report2.stationarity.d_star == report.stationarity.d_star
    assert report2.stationarity.to_dict() == report.stationarity.to_dict()
    # corrupt the window itself: the diagnostic curve changes
    inside = bars[:1300] + [Bar(b.instrument, b.timestamp, b.open, b.high * 1.01, b.low, b.close * (1 + 0.002 * ((i % 7) - 3)),
                                b.volume, 30) for i, b in enumerate(bars[1300:])]
    report3 = trainer.retrain(BarStore("SYN", 30, inside))
    assert report3.stationarity.to_dict() != report.stationarity.to_dict()


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
