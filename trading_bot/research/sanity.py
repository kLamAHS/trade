"""Leakage and sanity tests: the framework actively tries to disprove the strategy (sections 23-24).

Refit tests (light protocol, see ``ModelTrainer.light_refit``) compare a *destroyed* training
signal with the light refit of the intact one:

    shuffled_labels    training labels permuted           -> edge must vanish
    shuffled_features  every feature column permuted      -> edge must vanish
    target_shift       label moved +N bars into the future -> edge must degrade or vanish

Simulation tests re-trade the walk-forward forecasts:

    reversed_forecasts  E -> -E          -> must be worse than the strategy (ideally negative)
    random_forecasts    E permuted       -> the strategy must sit in the upper tail of the permutation distribution
    zero_cost / double_cost               -> sanity of the cost model's direction and size

Label alignment is verified directly against the bar store: the label of a decision at bar t is
``log O[t+1+H] - log O[t+1]``, i.e. it starts at the execution price and uses no earlier bar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .stress import _row, light_refit_series


def label_alignment_check(runner, oos, label_price: str, horizon: int) -> dict[str, Any]:
    arrays = runner.store.arrays()
    price = np.log(arrays["open"] if label_price == "open" else arrays["close"])
    idx = np.asarray(oos.bar_index, dtype=int)
    expected = price[idx + 1 + horizon] - price[idx + 1]
    diff = np.abs(expected - oos.y_raw)
    exec_open = arrays["open"][idx + 1]
    return {"rows": int(len(idx)), "max_abs_label_error": float(diff.max()) if len(diff) else 0.0,
            "labels_aligned": bool(len(diff) == 0 or diff.max() < 1e-9),
            "entry_price_is_execution_open": bool(np.allclose(exec_open, oos.open_next)),
            "label_first_bar_offset": 1, "label_last_bar_offset": 1 + horizon}


def run_sanity(runner, result, target_shift_bars: int = 20, n_random: int = 20, seed: int = 0,
               max_destroyed_sharpe: float = 1.0, log=None) -> dict[str, Any]:
    oos = result.oos
    cfg = runner.cfg
    strategy = _row(runner, oos, result.sims["full"], "strategy")
    tests: dict[str, Any] = {}

    def light(label: str, **kw) -> dict[str, Any]:
        E = light_refit_series(runner, result.windows, lambda wr: wr.d_star, log=log, **kw)
        return _row(runner, oos, runner.simulate(oos, "full", E=E), label, protocol="light_refit")

    ref = light("light_refit_reference")
    tests["light_refit_reference"] = {**ref, "expectation": "context for the refit tests", "passed": None}
    for label, kw, expect in (("shuffled_labels", {"shuffle_labels": True}, "edge must vanish"),
                              ("shuffled_features", {"shuffle_features": True}, "edge must vanish"),
                              ("target_shift", {"label_offset_bars": int(target_shift_bars)}, "edge must degrade or vanish")):
        r = light(label, **kw)
        r["expectation"] = expect
        if label == "target_shift":
            r["shift_bars"] = int(target_shift_bars)
            r["passed"] = bool(r["sharpe"] < ref["sharpe"] or r["total_return"] <= 0)
        else:
            r["passed"] = bool(r["sharpe"] < max_destroyed_sharpe and (r["sharpe"] < ref["sharpe"] or ref["sharpe"] <= 0))
        tests[label] = r
    # simulation-side
    E = oos.forecasts["full"]["E"]
    rev = _row(runner, oos, runner.simulate(oos, "full", E=-E), "reversed_forecasts")
    rev.update({"expectation": "worse than the strategy", "passed": bool(rev["sharpe"] < strategy["sharpe"])})
    tests["reversed_forecasts"] = rev
    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_random):
        r = _row(runner, oos, runner.simulate(oos, "full", E=E[rng.permutation(len(E))]), "random")
        rand.append(r["sharpe"])
    rand = np.asarray(rand)
    pct = float(np.mean(rand < strategy["sharpe"]))
    tests["random_forecasts"] = {"label": "random_forecasts", "n": int(n_random), "sharpe_median": float(np.median(rand)),
                                 "sharpe_p95": float(np.quantile(rand, 0.95)), "sharpe_max": float(rand.max()),
                                 "strategy_percentile": pct, "expectation": "strategy above the 95th percentile of permutations",
                                 "passed": bool(pct >= 0.95 and float(np.median(rand)) < max_destroyed_sharpe)}
    zero = _row(runner, oos, runner.simulate(oos, "full", cost_bps=0.0), "zero_cost")
    double = _row(runner, oos, runner.simulate(oos, "full", cost_scale=2.0), "double_cost")
    tests["zero_cost"] = {**zero, "expectation": "at least as good as the strategy", "passed": bool(zero["total_return"] >= strategy["total_return"] - 1e-12)}
    tests["double_cost"] = {**double, "expectation": "no better than the strategy", "passed": bool(double["total_return"] <= strategy["total_return"] + 1e-12)}
    checks = [t["passed"] for t in tests.values() if t.get("passed") is not None]
    return {"strategy": strategy, "tests": tests, "n_tests": len(checks), "n_failed": int(sum(not c for c in checks)),
            "passed": bool(all(checks))}


def run_leakage(runner, result, target_shift_bars: int = 20, log=None) -> dict[str, Any]:
    cfg = runner.cfg
    align = label_alignment_check(runner, result.oos, str(cfg.prediction.get("label_price", "open")), int(cfg.prediction.horizon_bars))
    audit = dict(result.timestamp_audit)
    # feature timestamp invariant: every FeatureVector construction enforces latest_source <= timestamp (types.py);
    # here the OOS rows are checked again explicitly.
    fa, da, ea = result.oos.feature_available_at, result.oos.decision_at, result.oos.execution_at
    violations = int(sum(1 for a, b, c in zip(fa, da, ea) if a > b or c < b))
    audit["violations"] = violations
    passed = bool(align["labels_aligned"] and align["entry_price_is_execution_open"] and violations == 0)
    return {"timestamps": audit, "label_alignment": align, "passed": passed,
            "checks": ["feature_available_at <= decision_at", "decision_at <= execution_at",
                       "label starts at the execution price and ends 1+H bars after the decision bar",
                       "d*, hyper-parameters and calibration selected inside the training block only (trainer protocol)",
                       f"target shift +{target_shift_bars} bars degrades the edge (see sanity.target_shift)"]}


__all__ = ["run_sanity", "run_leakage", "label_alignment_check"]
