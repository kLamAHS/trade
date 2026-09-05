"""Numerical tests for the fractional operator (spec section 55)."""

import numpy as np
import pytest

from trading_bot.fractional.stationarity import StationaryOrderEstimator
from trading_bot.fractional.transform import fractional_latest, fractional_transform
from trading_bot.fractional.weights import build_weights, kernel_length, weight_via_binomial


def test_first_weight_is_one_and_recursion_matches_binomial():
    from scipy.special import binom

    for d in (0.05, 0.25, 0.4, 0.5, 0.75, 0.95):
        w = build_weights(d)
        assert w[0] == 1.0
        for k in range(1, min(len(w), 120)):
            assert abs(w[k] - weight_via_binomial(d, k)) < 1e-12
            # independent oracle: (-1)^k * C(d, k) via the gamma-function binomial
            assert w[k] == pytest.approx(((-1.0) ** k) * binom(d, k), rel=1e-9, abs=1e-14)


def test_d_zero_is_identity_and_d_one_is_first_difference():
    x = np.cumsum(np.random.RandomState(0).randn(600))
    k0 = kernel_length(0.0)
    assert np.allclose(fractional_transform(x, 0.0)[k0:], x[k0:])
    k1 = kernel_length(1.0)
    assert np.allclose(fractional_transform(x, 1.0)[k1:], (x - np.roll(x, 1))[k1:])


def test_truncation_rule_is_deterministic_and_bounded():
    w1, w2 = build_weights(0.5, 1e-5, 10, 500), build_weights(0.5, 1e-5, 10, 500)
    assert np.array_equal(w1, w2)
    assert len(w1) - 1 <= 500
    # d = 0.75 decays fast enough to stop before the cap; 0.25 hits the cap
    assert kernel_length(0.75) < 500
    assert kernel_length(0.25) == 500
    w = build_weights(0.75)
    tail = np.abs(w[-10:])
    assert (tail < 1e-5).all()


def test_transform_no_nan_no_inf_after_warmup_and_reproducible():
    x = np.cumsum(np.random.RandomState(1).randn(2000)) / 50 + 5
    for d in (0.25, 0.5, 0.75):
        f = fractional_transform(x, d)
        k = kernel_length(d)
        assert np.isnan(f[:k]).all()
        assert np.isfinite(f[k:]).all()
        assert np.array_equal(f, fractional_transform(x, d), equal_nan=True)
        assert abs(fractional_latest(x, d) - f[-1]) < 1e-12


def test_transform_is_causal_convolution():
    x = np.random.RandomState(2).randn(700)
    d = 0.5
    w = build_weights(d)
    f = fractional_transform(x, d)
    t = 650
    manual = sum(w[k] * x[t - k] for k in range(len(w)))
    assert abs(f[t] - manual) < 1e-10


def test_nan_inside_window_propagates():
    x = np.cumsum(np.random.RandomState(3).randn(800))
    x[600] = np.nan
    f = fractional_transform(x, 0.75)
    k = kernel_length(0.75)
    assert np.isnan(f[600: 600 + k + 1]).all()
    assert np.isfinite(f[600 + k + 1:]).all()


def test_stationary_order_estimator_prefers_smallest_acceptable_d():
    rng = np.random.RandomState(4)
    white = rng.randn(1500)
    est = StationaryOrderEstimator(candidates=[0.05, 0.5, 0.95], adf_maxlag=5)
    res = est.estimate(white)
    assert res.d_star == 0.05
    assert res.selected_by == "min_acceptable"
    assert len(res.candidates) == 3


def test_stationary_order_estimator_random_walk_needs_more_differencing():
    rng = np.random.RandomState(5)
    walk = np.cumsum(rng.randn(2500))
    est = StationaryOrderEstimator(candidates=[0.05, 0.35, 0.65, 0.95], adf_maxlag=5)
    res = est.estimate(walk)
    assert res.d_star > 0.05
    assert res.d_star <= 0.95
    for c in res.candidates:
        assert np.isfinite(c.adf_stat) and np.isfinite(c.correlation)
