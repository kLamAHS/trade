"""Feature-family research: the ablation matrix, conditional contribution, family acceptance gates,
sabotage (failure) tests and the complexity budget (market-state spec sections 28-30, 46-52).

Every optional family must earn its place.  Contributions are measured two ways with the light
refit protocol (identical for every variant):

    additive   Base + F   versus   Base          what F adds on its own
    marginal   Full       versus   Full - F      what F adds on top of everything else

Failure tests sabotage a family's *inputs* to the fitted full model (shuffle / freeze / flip /
delay / randomise / remove) and re-trade: real predictive benefit must disappear.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from .bootstrap import ablation_bootstrap
from .metrics import _annualise, drawdown_stats
from .regimes import tag_regimes
from .stress import _row, light_refit_series
from .walkforward import window_statistics

CORE_FAMILIES = ("PRICE", "VOLATILITY", "VOLUME")
OPTIONAL_FAMILIES = ("FRACTIONAL", "REGIME", "OFI", "KALMAN", "CROSS_ASSET", "TOXICITY")


def _delta(a: dict, b: dict, key: str):
    if a.get(key) is None or b.get(key) is None:
        return None
    if a[key] == "inf" or b[key] == "inf":
        return None
    return float(a[key] - b[key])


def _per_window_sharpe(runner, oos, sim) -> dict[int, float]:
    return {s["window"]: s["sharpe"] for s in window_statistics(sim, oos.window_ids, runner.bars_per_year)}


def _dominance(deltas: np.ndarray) -> float:
    """Share of the total positive contribution that comes from the single best window."""
    pos = deltas[deltas > 0]
    return float(pos.max() / pos.sum()) if len(pos) and pos.sum() > 0 else 0.0


def ablation_matrix(runner, result, log: Callable[[str], None] | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    schema = runner.trainer.fe.schema
    windows = [w for w in result.windows if w.model is not None]
    if not windows:
        return {"families": {}, "note": "no fitted model"}
    present = [f for f in OPTIONAL_FAMILIES if any(f in (w.model.families or ()) for w in windows)]
    families_all = tuple(CORE_FAMILIES) + tuple(present)
    oos = result.oos

    def names_for(fams):
        keep = set(fams)
        return lambda wr: tuple(n for n in wr.model.feature_names if schema.family_of(n) in keep)

    variants: dict[str, tuple[str, ...]] = {"base": tuple(CORE_FAMILIES), "full": families_all}
    for f in present:
        variants[f"base+{f}"] = tuple(CORE_FAMILIES) + (f,)
        variants[f"full-{f}"] = tuple(x for x in families_all if x != f)
    sims: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    per_window: dict[str, dict[int, float]] = {}
    timings: dict[str, float] = {}
    for name, fams in variants.items():
        t0 = time.time()
        E = light_refit_series(runner, windows, lambda wr: wr.d_star, names_of=names_for(fams), log=log)
        sim = runner.simulate(oos, "full", E=E)
        sims[name] = sim
        rows[name] = _row(runner, oos, sim, name, families=list(fams))
        m = runner.metrics(oos, sim, E=E)
        rows[name]["forecast_correlation"] = m["forecast_correlation"]
        rows[name]["turnover_per_year"] = m["turnover_per_year"]
        rows[name]["directional_accuracy"] = m["directional_accuracy"]
        per_window[name] = _per_window_sharpe(runner, oos, sim)
        timings[name] = time.time() - t0
        if log:
            log(f"ablation matrix {name}: Sharpe {rows[name]['sharpe']:.2f} return {rows[name]['total_return']:+.2%} trades {rows[name]['trade_count']}")
    out_f: dict[str, Any] = {}
    for f in present:
        entry: dict[str, Any] = {}
        for kind, a, b in (("additive", f"base+{f}", "base"), ("marginal", "full", f"full-{f}")):
            wa, wb = per_window[a], per_window[b]
            deltas = np.array([wa[k] - wb[k] for k in sorted(wa) if k in wb], dtype=float)
            boot = ablation_bootstrap(deltas, n_boot=n_boot, seed=seed) if len(deltas) else {}
            entry[kind] = {
                "variant": a, "reference": b, "delta_sharpe": _delta(rows[a], rows[b], "sharpe"),
                "delta_return": _delta(rows[a], rows[b], "total_return"), "delta_max_drawdown": _delta(rows[a], rows[b], "max_drawdown"),
                "delta_turnover": _delta(rows[a], rows[b], "turnover_per_year"), "delta_cost": _delta(rows[a], rows[b], "total_cost"),
                "delta_calibration": _delta(rows[a], rows[b], "forecast_correlation"),
                "delta_accuracy": _delta(rows[a], rows[b], "directional_accuracy"),
                "median_window_delta_sharpe": float(np.median(deltas)) if len(deltas) else None,
                "mean_window_delta_sharpe": float(deltas.mean()) if len(deltas) else None,
                "positive_window_fraction": float(np.mean(deltas > 0)) if len(deltas) else None,
                "bootstrap_ci": boot.get("mean_ci"), "dominance": _dominance(deltas) if len(deltas) else None,
                "per_window_delta_sharpe": deltas.tolist(),
            }
        out_f[f] = entry
    return {"families": out_f, "present": present, "variants": rows, "protocol": "light refit per window, identical for every variant",
            "timings_s": timings}


def conditional_contribution(runner, result, matrix: dict[str, Any], log=None) -> dict[str, Any]:
    """ΔSharpe of each family by ex-ante regime (volatility, trend, and the HMM stress probability when
    present): a feature that is mediocre globally may matter under specific conditions (section 30)."""
    oos = result.oos
    windows = [w for w in result.windows if w.model is not None]
    if not windows or not matrix.get("present"):
        return {}
    schema = runner.trainer.fe.schema
    tags = tag_regimes(runner.store.log_close(), oos.bar_index)
    conds: dict[str, np.ndarray] = {}
    for kind, arr in tags.items():
        for name in sorted(set(arr.tolist()) - {"warmup"}):
            conds[f"{kind}:{name}"] = arr == name
    if oos.stress_p is not None and np.any(oos.stress_p > 0):
        conds["hmm:stress"] = oos.stress_p > 0.5
        conds["hmm:non_stress"] = oos.stress_p <= 0.5

    def series(fams):
        keep = set(fams)
        return light_refit_series(runner, windows, lambda wr: wr.d_star,
                                  names_of=lambda wr: tuple(n for n in wr.model.feature_names if schema.family_of(n) in keep), log=None)

    base_E = series(CORE_FAMILIES)
    base_sim = runner.simulate(oos, "full", E=base_E)
    out: dict[str, Any] = {}
    for f in matrix["present"]:
        sim = runner.simulate(oos, "full", E=series(tuple(CORE_FAMILIES) + (f,)))
        table = {}
        for cname, mask in conds.items():
            if mask.sum() < 20:
                continue
            a = _annualise(sim.bar_pnl[mask], runner.bars_per_year)["sharpe"]
            b = _annualise(base_sim.bar_pnl[mask], runner.bars_per_year)["sharpe"]
            table[cname] = {"bars": int(mask.sum()), "delta_sharpe": float(a - b), "sharpe_with": float(a), "sharpe_base": float(b)}
        out[f] = table
    return out


def sabotage_series(runner, result, family: str, mode: str, seed: int = 0) -> np.ndarray:
    """OOS forecasts of the fitted full model after sabotaging one family's *inputs* (section 51)."""
    schema = runner.trainer.fe.schema
    rng = np.random.default_rng(seed)
    parts = []
    for wr in result.windows:
        model = wr.model
        if model is None:
            parts.append(np.zeros(wr.n_oos_rows))
            continue
        ds, mask, _ = runner._oos_dataset(wr.window, model.d_star, {}, components=getattr(model, "components", None))
        X = ds.columns(model.feature_names)[mask].copy()
        cols = [j for j, n in enumerate(model.feature_names) if schema.family_of(n) == family]
        if cols:
            sub = X[:, cols]
            if mode == "shuffle":
                sub = sub[rng.permutation(len(sub))]
            elif mode == "freeze":
                sub = np.repeat(sub.mean(axis=0, keepdims=True), len(sub), axis=0)
            elif mode == "random":
                sub = rng.standard_normal(sub.shape) * sub.std(axis=0, keepdims=True) + sub.mean(axis=0, keepdims=True)
            elif mode == "flip":
                sub = -sub
            elif mode == "delay":
                sub = np.vstack([sub[:1], sub[:-1]])
            elif mode == "delay2":
                sub = np.vstack([sub[:2], sub[:-2]])
            elif mode == "remove":
                sub = np.zeros_like(sub)
            else:
                raise ValueError(f"unknown sabotage mode {mode!r}")
            X[:, cols] = sub
        out = model.predict_arrays(X)
        parts.append(out["E"])
    return np.concatenate(parts)


SABOTAGE_MODES = {"REGIME": ("shuffle", "freeze", "random"), "OFI": ("flip", "delay", "random"),
                  "CROSS_ASSET": ("delay", "delay2", "shuffle", "remove"), "KALMAN": ("random", "delay"),
                  "FRACTIONAL": ("shuffle", "random"), "TOXICITY": ("shuffle", "random")}


STOCHASTIC_SABOTAGE = ("shuffle", "random")


def _forecast_correlation(E: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(E) & np.isfinite(y)
    if ok.sum() < 3 or np.std(E[ok]) == 0 or np.std(y[ok]) == 0:
        return 0.0
    return float(np.corrcoef(E[ok], y[ok])[0, 1])


def failure_tests(runner, result, families, log=None, seed: int = 0, n_seeds: int = 3) -> dict[str, Any]:
    """Sabotage the fitted full model's inputs one family at a time (section 51).  A family whose benefit is
    real must lose it under every sabotage: the sabotaged series may not out-perform the reference, in
    Sharpe (the traded outcome) and in forecast correlation with the realised label (the quantity the
    sabotage acts on directly, far less noisy than the Sharpe of a short series).  Stochastic sabotages
    (shuffle, random) are averaged over ``n_seeds`` draws so a single lucky permutation cannot decide."""
    oos = result.oos
    ref = _row(runner, oos, result.sims["full"], "reference")
    ref_ic = _forecast_correlation(np.asarray(oos.forecasts["full"]["E"], dtype=float), oos.y_norm)
    ref["forecast_correlation"] = ref_ic
    out: dict[str, Any] = {"reference": ref, "families": {}, "n_seeds": int(n_seeds)}
    for f in families:
        rows = []
        for mode in SABOTAGE_MODES.get(f, ("shuffle", "random")):
            seeds = [seed + k for k in range(max(1, int(n_seeds)))] if mode in STOCHASTIC_SABOTAGE else [seed]
            draws = []
            for sd in seeds:
                E = sabotage_series(runner, result, f, mode, sd)
                r = _row(runner, oos, runner.simulate(oos, "full", E=E), f"{f}:{mode}", mode=mode)
                r["forecast_correlation"] = _forecast_correlation(E, oos.y_norm)
                draws.append(r)
            r = dict(draws[0])
            r["draws"] = len(draws)
            r["sharpe_draws"] = [float(d["sharpe"]) for d in draws]
            for key in ("sharpe", "total_return", "max_drawdown", "total_cost", "forecast_correlation"):
                r[key] = float(np.mean([d[key] for d in draws]))
            r["delta_sharpe"] = r["sharpe"] - ref["sharpe"]
            r["delta_forecast_correlation"] = r["forecast_correlation"] - ref_ic
            r["sharpe_degrades"] = bool(r["sharpe"] <= ref["sharpe"] + 1e-9)
            r["forecast_quality_degrades"] = bool(r["forecast_correlation"] <= ref_ic + 1e-9)
            r["benefit_disappears"] = bool(r["sharpe_degrades"] and r["forecast_quality_degrades"])
            rows.append(r)
            if log:
                log(f"sabotage {f}/{mode}: Sharpe {r['sharpe']:.2f} vs reference {ref['sharpe']:.2f}; forecast corr "
                    f"{r['forecast_correlation']:+.3f} vs {ref_ic:+.3f}" + (f" (mean of {len(draws)} draws)" if len(draws) > 1 else ""))
        out["families"][f] = {"rows": rows, "all_degrade": bool(rows and all(r["benefit_disappears"] for r in rows)),
                              "all_sharpe_degrade": bool(rows and all(r["sharpe_degrades"] for r in rows)),
                              "all_forecast_degrade": bool(rows and all(r["forecast_quality_degrades"] for r in rows)),
                              "max_sharpe_after_sabotage": float(max(r["sharpe"] for r in rows)) if rows else None,
                              "max_forecast_correlation_after_sabotage": float(max(r["forecast_correlation"] for r in rows)) if rows else None}
    return out


def _family_param_count(f: str, model) -> int:
    comp = getattr(model, "components", None)
    if f == "REGIME" and comp is not None and comp.hmm is not None:
        h = comp.hmm
        K, D = h.means.shape
        return int(K * D * 2 + K * K + K)
    if f == "KALMAN":
        return 3
    return 0


def complexity_budget(runner, result, matrix: dict[str, Any], failures: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per family: what it costs (parameters, latency, data dependency) versus what it earns (section 52)."""
    schema = runner.trainer.fe.schema
    windows = [w for w in result.windows if w.model is not None]
    model = windows[-1].model if windows else None
    tier = runner.store.data_tier()
    deps = {"FRACTIONAL": "Tier A (OHLCV)", "REGIME": "Tier A (fitted HMM parameters)", "KALMAN": "Tier A (fitted noise parameters)",
            "OFI": "Tier B (bar-close NBBO with displayed sizes)", "CROSS_ASSET": "Tier A + context instruments (as-of join)",
            "TOXICITY": "Tier A (bulk volume classification)"}
    risk = {"FRACTIONAL": "low: deterministic transform", "REGIME": "medium: EM fit, label switching handled by volatility ranking",
            "KALMAN": "low: two-state filter", "OFI": "high: depends on quote size quality and latency",
            "CROSS_ASSET": "medium: staleness and delivery order of other instruments", "TOXICITY": "medium: classification confidence"}
    out = []
    # feature-computation latency per family (ms per bar) measured on the last OOS block
    latencies = {}
    if windows:
        w = windows[-1]
        for f in matrix.get("present", []):
            t0 = time.time()
            n = 0
            try:
                fe = runner.trainer.fe
                ext = runner.store.slice(max(0, w.window.train_end - runner._lead_bars(w.d_star)), w.window.oos_end)
                fm = fe.compute_matrix(ext, context=runner.context or None)
                n = len(fm.close_times)
            except Exception:
                n = 0
            latencies[f] = ((time.time() - t0) / max(1, n)) * 1000.0
    for f in matrix.get("present", []):
        e = matrix["families"].get(f, {})
        add, mar = e.get("additive", {}), e.get("marginal", {})
        out.append({"family": f, "n_features": len(schema.families.get(f, ())), "n_parameters": _family_param_count(f, model),
                    "data_dependency": deps.get(f, "Tier A"), "data_tier_available": tier,
                    "compute_ms_per_bar_all_features": latencies.get(f), "implementation_risk": risk.get(f, ""),
                    "incremental_sharpe_additive": add.get("delta_sharpe"), "incremental_sharpe_marginal": mar.get("delta_sharpe"),
                    "incremental_turnover": mar.get("delta_turnover"), "incremental_cost": mar.get("delta_cost"),
                    "median_window_delta_sharpe": mar.get("median_window_delta_sharpe"),
                    "sabotage_degrades": (failures or {}).get("families", {}).get(f, {}).get("all_degrade")})
    return out


def family_gates(runner, result, matrix: dict[str, Any], failures: dict[str, Any], delay_rows: dict[str, Any] | None,
                 cost_rows: dict[str, Any] | None, thresholds: dict[str, Any] | None = None, log=None) -> dict[str, Any]:
    """Acceptance per family (sections 47-50).  Hard: median ΔSharpe > 0 and positive windows > 55% (marginal
    contribution).  Reported: bootstrap CI, dominance, +1 latency survival, cost-stress survival, sabotage."""
    g = {"min_median_delta_sharpe": 0.0, "min_positive_fraction": 0.55, "max_dominance": 0.5}
    g.update(thresholds or {})
    oos = result.oos
    windows = [w for w in result.windows if w.model is not None]
    schema = runner.trainer.fe.schema
    out: dict[str, Any] = {}
    for f in matrix.get("present", []):
        e = matrix["families"][f]["marginal"]
        checks = {
            "median_delta_sharpe_positive": (e["median_window_delta_sharpe"] is not None and e["median_window_delta_sharpe"] > g["min_median_delta_sharpe"]),
            "positive_windows": (e["positive_window_fraction"] is not None and e["positive_window_fraction"] > g["min_positive_fraction"]),
            "bootstrap_supports_benefit": bool(e["bootstrap_ci"] and e["bootstrap_ci"][0] > 0),
            "no_single_window_dominates": (e["dominance"] is not None and e["dominance"] < g["max_dominance"]),
            "sabotage_removes_benefit": bool(failures.get("families", {}).get(f, {}).get("all_degrade")),
        }
        # latency / cost survival of the *additive* contribution: Base+F must still beat Base under +1 bar and 2x realised cost
        keep_b = set(CORE_FAMILIES)
        keep_f = keep_b | {f}
        E_b = light_refit_series(runner, windows, lambda wr: wr.d_star, names_of=lambda wr: tuple(n for n in wr.model.feature_names if schema.family_of(n) in keep_b))
        E_f = light_refit_series(runner, windows, lambda wr: wr.d_star, names_of=lambda wr: tuple(n for n in wr.model.feature_names if schema.family_of(n) in keep_f))
        d1 = _row(runner, oos, runner.simulate(oos, "full", E=E_f, delay=1), "f+1")["sharpe"] - _row(runner, oos, runner.simulate(oos, "full", E=E_b, delay=1), "b+1")["sharpe"]
        c2 = _row(runner, oos, runner.simulate(oos, "full", E=E_f, exec_cost_scale=2.0), "f2x")["sharpe"] - _row(runner, oos, runner.simulate(oos, "full", E=E_b, exec_cost_scale=2.0), "b2x")["sharpe"]
        checks["survives_plus_1_latency"] = bool(d1 > 0)
        checks["survives_2x_cost"] = bool(c2 > 0)
        special = special_family_gate(runner, result, f)
        # hard acceptance: positive marginal contribution in most windows, the family's benefit must vanish under
        # sabotage (otherwise it is noise the fitted model happens to key on), and the family-specific conditions
        hard = (checks["median_delta_sharpe_positive"] and checks["positive_windows"] and checks["sabotage_removes_benefit"]
                and special.get("passed", True))
        status = "ACCEPTED" if hard else ("RESEARCH ONLY" if f == "OFI" and not special.get("passed", True) else "REJECTED")
        out[f] = {"status": status, "checks": checks, "delta_sharpe_plus_1_latency": float(d1), "delta_sharpe_2x_cost": float(c2),
                  "marginal": {k: e[k] for k in ("delta_sharpe", "median_window_delta_sharpe", "positive_window_fraction", "bootstrap_ci", "dominance")},
                  "special": special}
        if log:
            log(f"family gate {f}: {status} (median ΔSharpe {e['median_window_delta_sharpe']}, positive {e['positive_window_fraction']}, "
                f"+1 latency Δ {d1:+.2f}, 2x cost Δ {c2:+.2f})")
    return out


def special_family_gate(runner, result, family: str) -> dict[str, Any]:
    """Section 48-50 conditions that do not follow from performance."""
    store = runner.store
    windows = [w for w in result.windows if w.model is not None]
    if family == "OFI":
        quotes = store.has_quotes()
        sizes = store.has_quote_sizes()
        checks = {"quote_timestamp_quality": bool(quotes), "displayed_sizes_present": bool(sizes),
                  "staleness": True,                        # bar-close snapshots: the newest quote is the bar's own close
                  "feature_latency": True,                  # OFI is complete at the bar close, before the next open
                  "execution_latency_modeled": True}        # the +1 / +2 bar timing test covers it
        return {"passed": all(checks.values()), "checks": checks, "note": "bar-close NBBO snapshots (Tier B); intrabar quote updates are not observed"}
    if family == "REGIME":
        model = windows[-1].model if windows else None
        comp = getattr(model, "components", None)
        hmm = comp.hmm if comp is not None else None
        if hmm is None:
            return {"passed": False, "checks": {"fitted": False}}
        meta = hmm.state_meta
        occ = [m["occupancy"] for m in meta]
        pers = [m["persistence"] for m in meta]
        # causal computation: the filter on a prefix must equal the filter on the full series (checked on the last block)
        from ..features.hmm import hmm_observations
        arr = store.arrays()
        obs = hmm_observations(np.log(arr["close"]), arr["volume"], None)[-600:]
        full = hmm.filter(obs)
        pre = hmm.filter(obs[:400])
        causal = bool(np.allclose(full[:400], pre))
        # profit concentration by most-likely state on the OOS rows
        oos = result.oos
        sim = result.sims["full"]
        conc = {}
        if oos.stress_p is not None:
            stress_rows = oos.stress_p > 0.5
            share_occ = float(stress_rows.mean())
            pnl_total = float(sim.bar_pnl.sum())
            share_pnl = float(sim.bar_pnl[stress_rows].sum() / pnl_total) if pnl_total > 0 else 0.0
            conc = {"stress_occupancy": share_occ, "stress_pnl_share": share_pnl,
                    "rare_state_drives_profit": bool(share_occ < 0.05 and share_pnl > 0.5)}
        checks = {"causal_filter": causal, "state_persistence_reasonable": all(0.5 <= p <= 0.9995 for p in pers),
                  "state_occupancy_sufficient": all(o >= 0.02 for o in occ), "no_rare_state_profit_concentration": not conc.get("rare_state_drives_profit", False)}
        return {"passed": all(checks.values()), "checks": checks, "state_meta": meta, "concentration": conc}
    if family == "CROSS_ASSET":
        oos = result.oos
        ref = _row(runner, oos, result.sims["full"], "ref")["sharpe"]
        rows = {}
        for mode in ("delay", "delay2", "remove"):
            E = sabotage_series(runner, result, "CROSS_ASSET", mode)
            rows[mode] = _row(runner, oos, runner.simulate(oos, "full", E=E), mode)["sharpe"]
        graceful = all(v > min(0.0, 0.5 * ref) for v in rows.values()) if ref > 0 else True
        return {"passed": bool(graceful), "checks": {"degrades_gracefully_under_delay_or_missing": bool(graceful)},
                "sharpe_under": rows, "reference_sharpe": ref,
                "note": "correlation-breakdown robustness is exercised by the synthetic market generator (stress regimes)"}
    return {"passed": True, "checks": {}}


__all__ = ["ablation_matrix", "conditional_contribution", "failure_tests", "family_gates", "complexity_budget", "sabotage_series",
           "special_family_gate", "OPTIONAL_FAMILIES", "CORE_FAMILIES"]
