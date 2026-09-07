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


def _mutated_store(store, from_bar: int, factor: float = 1.37, seed: int = 0):
    """Copy of the history whose bars from ``from_bar`` onwards are replaced by a different price path
    (scaled, re-noised, volume doubled).  Bars before ``from_bar`` are untouched."""
    from ..data.store import BarStore
    from ..types import Bar

    rng = np.random.default_rng(seed)
    bars = list(store.bars[:from_bar])
    for b in store.bars[from_bar:]:
        k = factor * (1.0 + 0.01 * rng.standard_normal())
        o, c = b.open * k, b.close * k * (1.0 + 0.002 * rng.standard_normal())
        hi, lo = max(o, c) * 1.001, min(o, c) * 0.999
        bars.append(Bar(b.instrument, b.timestamp, o, hi, lo, c, b.volume * 2.0, b.bar_minutes,
                        None if b.bid is None else b.bid * k, None if b.ask is None else b.ask * k,
                        b.quote_timestamp, b.observed_at))
    return BarStore(store.instrument, store.bar_minutes, bars)


def forward_mutation_check(runner, result, n_samples: int = 4, log=None) -> dict[str, Any]:
    """Change every bar *after* a decision bar and re-run the fitted model: the forecast at that bar (and
    at every earlier bar of the block) must be bit-identical.  A model or feature that sees the fill bar
    or any later bar fails here even if the timestamps look right (section 23)."""
    windows = [w for w in result.windows if w.model is not None]
    if not windows:
        return {"passed": None, "samples": [], "note": "no fitted model"}
    picks = sorted({int(round(k * (len(windows) - 1) / max(1, n_samples - 1))) for k in range(n_samples)})
    samples = []
    for k in picks:
        wr = windows[k]
        w = wr.window
        ds, mask, offset = runner._oos_dataset(w, wr.model.d_star, {})
        rows = np.flatnonzero(mask)
        cut = rows[len(rows) // 2]                                   # decision bar in the middle of the block
        cut_bar = int(ds.bar_index[cut] + offset)
        E_ref = runner._forecast(wr.model, ds, mask)["E"]
        mutated = _mutated_store(runner.store, cut_bar + 1, seed=k)  # everything after the decision bar changes
        ds_m, mask_m, offset_m = runner._oos_dataset(w, wr.model.d_star, {}, store=mutated)
        E_mut = runner._forecast(wr.model, ds_m, mask_m)["E"]
        rows_m = np.flatnonzero(mask_m)
        upto = int(np.sum((ds.bar_index[rows] + offset) <= cut_bar))
        upto_m = int(np.sum((ds_m.bar_index[rows_m] + offset_m) <= cut_bar))
        same_rows = upto == upto_m
        identical = bool(same_rows and np.array_equal(E_ref[:upto], E_mut[:upto_m]))
        changed_after = bool(len(E_mut) > upto_m and len(E_ref) > upto and not np.array_equal(E_ref[upto:upto + 1], E_mut[upto_m:upto_m + 1]))
        samples.append({"window": w.index, "model_id": wr.model_id, "decision_bar": cut_bar,
                        "decision_at": runner.store[cut_bar].close_time.isoformat(),
                        "mutated_from_bar": cut_bar + 1, "rows_compared": upto, "identical_up_to_decision": identical,
                        "max_abs_difference": float(np.max(np.abs(E_ref[:upto] - E_mut[:upto_m]))) if same_rows and upto else 0.0,
                        "forecast_after_cut_changed": changed_after})
        if log:
            log(f"forward mutation window {w.index}: bars > {cut_bar} rewritten -> forecasts up to the decision "
                f"{'identical' if identical else 'CHANGED'}")
    return {"passed": bool(all(s["identical_up_to_decision"] for s in samples)), "samples": samples,
            "note": "bars after the decision bar were replaced by a different price path; forecasts at and before it must not move"}


def timestamp_chain(runner, result, n_trades: int = 10) -> list[dict[str, Any]]:
    """Human-verifiable chain for a sample of trades (section 23):
    newest bar used -> feature / forecast / order timestamp -> fill bar -> fill price."""
    trades = result.sims["full"].trades
    if not trades:
        return []
    idx = sorted({int(round(k * (len(trades) - 1) / max(1, n_trades - 1))) for k in range(min(n_trades, len(trades)))})
    out = []
    store = runner.store
    for i in idx:
        t = trades[i]
        row = t["entry_row"]
        b = int(result.oos.bar_index[row])
        dec, fill = store[b], store[b + 1]
        out.append({"trade_id": t["trade_id"], "row": row, "decision_bar_index": b,
                    "newest_bar_used": {"index": b, "start": dec.timestamp.isoformat(), "close_time": dec.close_time.isoformat(),
                                        "close": dec.close, "latest_source_time": dec.latest_source_time.isoformat()},
                    "feature_timestamp": result.oos.decision_at[row].isoformat(),
                    "forecast_timestamp": result.oos.decision_at[row].isoformat(),
                    "order_timestamp": result.oos.decision_at[row].isoformat(),
                    "fill_bar": {"index": b + 1, "start": fill.timestamp.isoformat(), "close_time": fill.close_time.isoformat(),
                                 "open": fill.open},
                    "fill_price": t["entry_price"], "fill_price_is_next_bar_open": bool(abs(t["entry_price"] - fill.open) < 1e-9),
                    "fill_bar_starts_at_decision_bar_close": bool(fill.timestamp == dec.close_time),
                    "fill_bar_is_after_newest_bar_used": bool(fill.timestamp >= dec.close_time and b + 1 > b)})
    return out


def run_leakage(runner, result, target_shift_bars: int = 20, log=None, mutation_samples: int = 4) -> dict[str, Any]:
    cfg = runner.cfg
    align = label_alignment_check(runner, result.oos, str(cfg.prediction.get("label_price", "open")), int(cfg.prediction.horizon_bars))
    audit = dict(result.timestamp_audit)
    # feature timestamp invariant: every FeatureVector construction enforces latest_source <= timestamp (types.py);
    # here the OOS rows are checked again explicitly.
    fa, da, ea = result.oos.feature_available_at, result.oos.decision_at, result.oos.execution_at
    violations = int(sum(1 for a, b, c in zip(fa, da, ea) if a > b or c < b))
    audit["violations"] = violations
    mutation = forward_mutation_check(runner, result, mutation_samples, log=log) if mutation_samples > 0 else {"passed": None, "samples": []}
    chain = timestamp_chain(runner, result)
    chain_ok = all(c["fill_price_is_next_bar_open"] and c["fill_bar_is_after_newest_bar_used"] for c in chain)
    passed = bool(align["labels_aligned"] and align["entry_price_is_execution_open"] and violations == 0
                  and mutation["passed"] in (True, None) and chain_ok)
    return {"timestamps": audit, "label_alignment": align, "forward_mutation": mutation, "timestamp_chain": chain,
            "chain_consistent": bool(chain_ok), "passed": passed,
            "checks": ["feature_available_at <= decision_at", "decision_at <= execution_at",
                       "label starts at the execution price and ends 1+H bars after the decision bar",
                       "forecasts are bit-identical when every bar after the decision bar is rewritten (forward mutation)",
                       "sampled trades: fill price == open of the bar that starts at the decision bar's close",
                       "d*, hyper-parameters and calibration selected inside the training block only (trainer protocol)",
                       f"target shift +{target_shift_bars} bars degrades the edge (see sanity.target_shift)"]}


__all__ = ["run_sanity", "run_leakage", "label_alignment_check", "forward_mutation_check", "timestamp_chain"]
