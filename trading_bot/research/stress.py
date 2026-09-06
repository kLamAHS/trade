"""Cost, timing and parameter stress tests (research spec sections 11, 20-22).

Simulation-side tests re-trade the *same* OOS forecasts under a changed assumption (cost level,
execution delay, position-rule parameter).  The fractional-order perturbation refits the models
because ``d`` defines the features: every window is refitted with the light protocol at
``d* + k * step`` and compared with the light refit at ``d*`` itself, so the comparison is
apples to apples.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np

from .simulate import SimResult


def _row(runner, oos, sim: SimResult, label: str, **extra) -> dict[str, Any]:
    m = runner.metrics(oos, sim, "full")
    return {"label": label, "sharpe": m["sharpe"], "total_return": m["total_return"], "cagr": m["cagr"],
            "max_drawdown": m["max_drawdown"], "profit_factor": m["profit_factor"], "trade_count": m["trade_count"],
            "total_cost": m["total_cost"], "win_rate": m["win_rate"], "time_invested": m["time_invested"], **extra}


def cost_curve(runner, oos, levels_bps=(0, 1, 2, 5, 10), scales=(1.0, 2.0, 3.0)) -> dict[str, Any]:
    rows = []
    for s in scales:
        rows.append(_row(runner, oos, runner.simulate(oos, "full", cost_scale=float(s)), f"model_cost_x{s:g}",
                         kind="model", scale=float(s)))
    for bps in levels_bps:
        rows.append(_row(runner, oos, runner.simulate(oos, "full", cost_bps=float(bps)), f"flat_{bps:g}bps",
                         kind="flat_bps", bps=float(bps)))
    flat = [(r["bps"], r["total_return"]) for r in rows if r["kind"] == "flat_bps"]
    breakeven = None
    for (b0, r0), (b1, r1) in zip(flat[:-1], flat[1:]):
        if r0 > 0 >= r1 and r0 != r1:
            breakeven = float(b0 + (b1 - b0) * r0 / (r0 - r1))
            break
    if breakeven is None and flat and flat[-1][1] > 0:
        breakeven = float("inf")
    ref = next(r for r in rows if r["kind"] == "model" and r["scale"] == 1.0)
    x2 = next((r for r in rows if r["kind"] == "model" and r["scale"] == 2.0), None)
    mean_cost_bps = float(np.mean(oos.cost_roundtrip)) * 1e4
    return {"rows": rows, "breakeven_flat_bps": breakeven, "mean_model_roundtrip_bps": mean_cost_bps,
            "reference": ref, "profitable_at_2x_cost": bool(x2 is not None and x2["total_return"] > 0),
            "sharpe_at_2x_cost": x2["sharpe"] if x2 else None}


def timing_delays(runner, oos, delays=(0, 1, 2)) -> dict[str, Any]:
    rows = [_row(runner, oos, runner.simulate(oos, "full", delay=int(d)), f"delay_{d}", delay=int(d)) for d in delays]
    ref = rows[0]
    for r in rows:
        r["sharpe_retained"] = (r["sharpe"] / ref["sharpe"]) if ref["sharpe"] > 0 else None
    one = next((r for r in rows if r["delay"] == 1), None)
    return {"rows": rows, "viable_at_plus_1": bool(one is not None and one["total_return"] > 0 and one["sharpe"] > 0),
            "sharpe_at_plus_1": one["sharpe"] if one else None}


PERTURBATIONS = [
    ("cost_multiplier", 0.5), ("cost_multiplier", 2.0),
    ("confidence_cost_multiplier", 0.5), ("confidence_cost_multiplier", 2.0),
    ("rebalance_threshold", 0.5), ("rebalance_threshold", 2.0),
    ("max_holding_bars", 0.5), ("max_holding_bars", 2.0),
    ("stop_sigma_multiple", 0.5), ("stop_sigma_multiple", 2.0),
    ("vol_multiplier_max", 0.5), ("vol_multiplier_max", 2.0),
]


def parameter_perturbations(runner, oos, collapse_fraction: float = 0.75) -> dict[str, Any]:
    base = runner.sim_params
    ref = _row(runner, oos, runner.simulate(oos, "full"), "reference")
    rows = [ref]
    for name, factor in PERTURBATIONS:
        value = getattr(base, name) * factor
        if isinstance(getattr(base, name), int):
            value = max(1, int(round(value)))
        params = dataclasses.replace(base, **{name: value})
        r = _row(runner, oos, runner.simulate(oos, "full", params=params), f"{name}_x{factor:g}", parameter=name,
                 factor=factor, value=value)
        r["sharpe_change"] = r["sharpe"] - ref["sharpe"]
        r["collapsed"] = bool(ref["sharpe"] > 0 and (r["sharpe"] < (1.0 - collapse_fraction) * ref["sharpe"]) and r["total_return"] <= 0)
        rows.append(r)
    pert = rows[1:]
    return {"rows": rows, "reference": ref, "n_collapsed": int(sum(r["collapsed"] for r in pert)),
            "any_collapse": bool(any(r["collapsed"] for r in pert)),
            "min_sharpe": float(min(r["sharpe"] for r in pert)) if pert else None,
            "positive_fraction": float(np.mean([r["total_return"] > 0 for r in pert])) if pert else None}


def light_refit_series(runner, windows, d_of: Callable[[Any], float], names_of=None, log=None, **refit_kwargs) -> np.ndarray:
    """OOS forecasts E for every window from a light refit (see ``ModelTrainer.light_refit``)."""
    parts = []
    for wr in windows:
        w = wr.window
        d = float(d_of(wr))
        names = names_of(wr) if names_of else (wr.model.feature_names if wr.model is not None else runner.trainer.fe.schema.model_names)
        history = runner.store.slice(w.train_start, w.train_end)
        params = wr.best_params or runner.trainer.grid[0]
        try:
            model = runner.trainer.light_refit(history, d, params, names, **refit_kwargs)
        except ValueError as exc:
            if log:
                log(f"light refit window {w.index} failed: {exc}")
            parts.append(np.zeros(wr.n_oos_rows))
            continue
        ds, mask, _ = runner._oos_dataset(w, d, {})
        parts.append(runner._forecast(model, ds, mask)["E"])
    return np.concatenate(parts)


def d_perturbation(runner, result, steps=(-2, -1, 0, 1, 2), log=None) -> dict[str, Any]:
    f = runner.cfg.fractional
    step, lo, hi = float(f.adaptive_step), float(f.adaptive_min), float(f.adaptive_max)
    oos = result.oos
    rows = []
    ref = None
    for k in steps:
        E = light_refit_series(runner, result.windows, lambda wr, k=k: float(np.clip(wr.d_star + k * step, lo, hi)), log=log)
        sim = runner.simulate(oos, "full", E=E)
        r = _row(runner, oos, sim, f"d_star{k:+d}step", step=int(k), delta_d=float(k * step))
        rows.append(r)
        if k == 0:
            ref = r
    if ref is None:
        ref = rows[0]
    for r in rows:
        r["sharpe_change"] = r["sharpe"] - ref["sharpe"]
    non_ref = [r for r in rows if r["step"] != 0]
    return {"rows": rows, "reference_light_refit": ref, "protocol": "light refit per window (no d*/grid search), identical for every step",
            "any_collapse": bool(ref["sharpe"] > 0 and any(r["total_return"] <= 0 and r["sharpe"] < 0.25 * ref["sharpe"] for r in non_ref)),
            "min_sharpe": float(min(r["sharpe"] for r in non_ref)) if non_ref else None}


__all__ = ["cost_curve", "timing_delays", "parameter_perturbations", "d_perturbation", "light_refit_series", "PERTURBATIONS"]
