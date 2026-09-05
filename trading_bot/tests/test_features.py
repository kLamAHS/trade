"""Feature-engine tests: batch/stream equality, formulas, schema, look-ahead guard."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading_bot.config import load_config
from trading_bot.features.engine import FeatureEngine
from trading_bot.features.rolling import ewma_variance, robust_zscore, rolling_corr_with_index, rolling_std
from trading_bot.features.schema import build_schema
from trading_bot.types import FeatureVector


def test_batch_and_streaming_features_identical(feature_engine, store_1500):
    fm = feature_engine.compute_matrix(store_1500)
    for i in (feature_engine.required_history - 1, 1200, len(store_1500) - 1):
        sub = store_1500.slice(0, i + 1)
        fv = feature_engine.compute_latest(sub)
        row = fm.row(i)
        for name in feature_engine.schema.all_names:
            a, b = row[name], fv.values[name]
            if math.isnan(a) or math.isnan(b):
                assert math.isnan(a) and math.isnan(b), name
            else:
                assert a == pytest.approx(b, abs=1e-12), name
        assert fv.timestamp == store_1500[i].close_time
        assert fv.latest_source_timestamp <= fv.timestamp


def test_all_features_finite_after_required_history(feature_engine, store_1500):
    fm = feature_engine.compute_matrix(store_1500)
    valid = fm.valid_mask(feature_engine.schema.model_names)
    first = int(np.argmax(valid))
    assert first <= feature_engine.required_history
    assert valid[first:].all()
    assert not np.isinf(fm.values[first:]).any()


def test_feature_vector_rejects_lookahead():
    t = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        FeatureVector("SYN", t, t + timedelta(minutes=1), 0, 0.5, 500, {"x": 1.0})
    fv = FeatureVector("SYN", t, t, 0, 0.5, 500, {"x": 1.0})
    assert fv["x"] == 1.0


def test_robust_zscore_matches_manual():
    x = np.random.RandomState(0).randn(400)
    z = robust_zscore(x, 50)
    window = x[350:400]
    med = np.median(window)
    mad = np.median(np.abs(window - med))
    assert z[399] == pytest.approx((x[399] - med) / (1.4826 * mad + 1e-12))
    assert np.isnan(z[:49]).all()


def test_ewma_variance_matches_recursion():
    r = np.random.RandomState(1).randn(700) * 0.01
    lam, W = 0.94, 450
    v = ewma_variance(r, lam, W)
    t = 650
    seed = r[t - W + 1] ** 2
    acc = seed
    for j in range(t - W + 2, t + 1):
        acc = lam * acc + (1 - lam) * r[j] ** 2
    assert v[t] == pytest.approx(acc, rel=1e-9)
    assert np.isnan(v[: W - 1]).all()


def test_rolling_helpers():
    x = np.arange(100, dtype=float)
    assert rolling_corr_with_index(x, 10)[-1] == pytest.approx(1.0)
    assert rolling_corr_with_index(-x, 10)[-1] == pytest.approx(-1.0)
    assert rolling_std(np.ones(20), 5)[-1] == 0.0


def test_time_features_cyclic(feature_engine, store_1500):
    fm = feature_engine.compute_matrix(store_1500)
    ts = store_1500[0].timestamp
    assert feature_engine.calendar.minutes_since_open(ts) == 0
    assert fm.column("time_sin")[0] == pytest.approx(0.0)
    assert fm.column("time_cos")[0] == pytest.approx(1.0)
    phase = 2 * math.pi * 30 / 390
    assert fm.column("time_sin")[1] == pytest.approx(math.sin(phase))


def test_formula_spot_checks(feature_engine, store_1500):
    fm = feature_engine.compute_matrix(store_1500)
    i = len(store_1500) - 1
    row = fm.row(i)
    arrays = store_1500.arrays()
    p = np.log(arrays["close"])
    r = np.diff(p)
    sigma50 = np.std(r[i - 50: i])
    assert row["sigma_h"] == pytest.approx(sigma50)
    assert row["return_4"] == pytest.approx((p[i] - p[i - 4]) / (sigma50 * 2 + 1e-12))
    assert row["vol_50"] == pytest.approx(math.log(sigma50 + 1e-12))
    bar = store_1500[i]
    assert row["close_location"] == pytest.approx((2 * bar.close - bar.high - bar.low) / (bar.high - bar.low + 1e-12))
    assert row["fd_cross_sm"] == pytest.approx(row["fd_025_z"] - row["fd_050_z"])
    assert row["fd_curvature_050"] == pytest.approx(fm.column("fd_050")[i] - 2 * fm.column("fd_050")[i - 1] + fm.column("fd_050")[i - 2])
    assert row["fractional_state"] == pytest.approx(abs(row["fd_025_z"]) / (abs(row["fd_075_z"]) + 0.1))
    assert -1.0 <= row["trend_state"] <= 1.0


def test_schema_groups_and_volume_toggle():
    s = build_schema(volume_enabled=True)
    assert "volume_z" in s.model_names and "volume_z" not in s.fractional_names
    assert set(s.fractional_names) <= set(s.model_names)
    assert "fd_025" not in s.model_names  # raw levels are recorded, not modelled
    assert "fd_025_z" in s.fractional_names and "fractional_volatility" in s.fractional_names
    assert "time_sin" in s.baseline_names and "fd_cross_sf" not in s.baseline_names
    s2 = build_schema(volume_enabled=False)
    assert "volume_z" not in s2.all_names


def test_volume_disabled_engine(cfg, fractional, calendar, store_1500):
    cfg2 = cfg.with_overrides({"data": {"volume_enabled": False}})
    eng = FeatureEngine(cfg2, fractional, calendar, adaptive_d=0.4)
    fv = eng.compute_latest(store_1500)
    assert "volume_z" not in fv.values
