"""Acceptance gates and result classification (research spec sections 30-31).

Classification ladder (each level requires the previous one):

    EXPERIMENTAL          anything that ran
    CANDIDATE             the walk-forward gates pass (windows, trades, Sharpe, drawdown, PF, consistency)
    VALIDATED CANDIDATE   + ablation, cost / timing / parameter stress, bootstrap, sanity and leakage gates
    HOLDOUT PASSED        + the locked holdout was opened and is profitable
    PAPER ELIGIBLE        + reproducibility verified (and the synthetic engineering validation, when run)

Synthetic runs never classify above EXPERIMENTAL: they are engineering validation, not evidence.
Thresholds live in ``research.gates`` of the strategy configuration.
"""

from __future__ import annotations

from typing import Any

DEFAULT_GATES: dict[str, Any] = {
    "min_windows": 20, "min_trades": 200, "min_sharpe": 1.0, "max_drawdown": 0.25, "min_profit_factor": 1.2,
    "min_positive_window_fraction": 0.55, "min_median_delta_sharpe": 0.0, "min_positive_delta_fraction": 0.5,
    "require_profitable_at_2x_cost": True, "require_viable_at_plus_1_bar": True, "allow_perturbation_collapse": False,
    "min_bootstrap_sharpe_lower": 0.0, "require_sanity_pass": True, "require_leakage_pass": True,
    "holdout_min_return": 0.0, "holdout_min_sharpe": 0.0, "require_reproducible": True,
}

LEVELS = ["EXPERIMENTAL", "CANDIDATE", "VALIDATED CANDIDATE", "HOLDOUT PASSED", "PAPER ELIGIBLE"]


def _gate(name: str, group: str, value, threshold, passed: bool | None, evidence: str = "", op: str = ">=") -> dict[str, Any]:
    return {"gate": name, "group": group, "value": value, "threshold": threshold, "op": op,
            "passed": passed, "evidence": evidence}


def evaluate_gates(summary: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    g = dict(DEFAULT_GATES)
    g.update(thresholds or {})
    dev = summary.get("development") or {}
    m = (dev.get("metrics") or {}).get("full") or {}
    per_window = dev.get("per_window") or []
    abl = summary.get("ablation") or {}
    cost = summary.get("cost_curve") or {}
    timing = summary.get("timing") or {}
    pert = summary.get("parameter_perturbations") or {}
    dpert = summary.get("d_perturbation") or {}
    boot = summary.get("bootstrap") or {}
    sanity = summary.get("sanity") or {}
    leak = summary.get("leakage") or {}
    hold = summary.get("holdout") or {}
    repro = summary.get("reproducibility") or {}
    syn = summary.get("synthetic_ensemble") or {}
    gates: list[dict[str, Any]] = []

    def have(x) -> bool:
        return bool(x)

    # --- walk-forward ------------------------------------------------------------------------
    n_w = dev.get("n_windows", 0)
    gates.append(_gate("OOS windows", "walkforward", n_w, g["min_windows"], n_w >= g["min_windows"], f"{n_w} walk-forward windows"))
    trades = m.get("trade_count", 0)
    gates.append(_gate("OOS trades", "walkforward", trades, g["min_trades"], trades >= g["min_trades"]))
    sharpe = m.get("sharpe")
    gates.append(_gate("OOS Sharpe", "walkforward", sharpe, g["min_sharpe"], (sharpe is not None) and sharpe > g["min_sharpe"], op=">"))
    dd = m.get("max_drawdown")
    gates.append(_gate("OOS max drawdown", "walkforward", dd, g["max_drawdown"], (dd is not None) and dd < g["max_drawdown"], op="<"))
    pf = m.get("profit_factor")
    gates.append(_gate("OOS profit factor", "walkforward", pf, g["min_profit_factor"], (pf is not None) and pf > g["min_profit_factor"], op=">"))
    pos = [w.get("full_net_return") for w in per_window if w.get("full_net_return") is not None]
    frac = (sum(1 for x in pos if x > 0) / len(pos)) if pos else None
    gates.append(_gate("positive OOS windows", "walkforward", frac, g["min_positive_window_fraction"],
                       (frac is not None) and frac > g["min_positive_window_fraction"], f"{sum(1 for x in pos if x > 0)}/{len(pos)}", op=">"))
    # --- ablation ---------------------------------------------------------------------------
    med = abl.get("median_delta_sharpe")
    gates.append(_gate("median ΔSharpe (fractional − baseline)", "ablation", med, g["min_median_delta_sharpe"],
                       None if not have(abl) else med > g["min_median_delta_sharpe"], op=">"))
    pfrac = abl.get("positive_delta_fraction")
    gates.append(_gate("windows where fractional beats baseline", "ablation", pfrac, g["min_positive_delta_fraction"],
                       None if not have(abl) else pfrac > g["min_positive_delta_fraction"], op=">"))
    # --- stress ---------------------------------------------------------------------------
    gates.append(_gate("profitable at 2× model cost", "stress", cost.get("profitable_at_2x_cost"), True,
                       None if not have(cost) else (bool(cost.get("profitable_at_2x_cost")) or not g["require_profitable_at_2x_cost"]), op="=="))
    gates.append(_gate("viable at +1 bar execution delay", "stress", timing.get("viable_at_plus_1"), True,
                       None if not have(timing) else (bool(timing.get("viable_at_plus_1")) or not g["require_viable_at_plus_1_bar"]), op="=="))
    collapse = None
    if have(pert) or have(dpert):
        collapse = bool(pert.get("any_collapse")) or bool(dpert.get("any_collapse"))
    gates.append(_gate("no parameter-perturbation collapse", "stress", collapse, False,
                       None if collapse is None else (not collapse or g["allow_perturbation_collapse"]),
                       ("" if collapse is None else f"position rules: {pert.get('n_collapsed', 'n/a')} collapsed; d±steps min Sharpe "
                        f"{dpert.get('min_sharpe') if have(dpert) else 'not run'}"), op="=="))
    # --- bootstrap --------------------------------------------------------------------------
    lower = ((boot.get("block") or {}).get("sharpe") or {}).get("ci_low")
    gates.append(_gate("bootstrap Sharpe CI lower bound", "bootstrap", lower, g["min_bootstrap_sharpe_lower"],
                       None if lower is None else lower > g["min_bootstrap_sharpe_lower"], op=">"))
    # --- sanity / leakage --------------------------------------------------------------------
    gates.append(_gate("sanity tests", "sanity", sanity.get("passed"), True,
                       None if not have(sanity) else (bool(sanity.get("passed")) or not g["require_sanity_pass"]),
                       f"{sanity.get('n_failed')} of {sanity.get('n_tests')} failed" if have(sanity) else "not run", op="=="))
    gates.append(_gate("leakage tests", "leakage", leak.get("passed"), True,
                       None if not have(leak) else (bool(leak.get("passed")) or not g["require_leakage_pass"]), op="=="))
    # --- holdout ---------------------------------------------------------------------------
    hm = ((hold.get("metrics") or {}).get("full") or {}) if have(hold) else {}
    hr, hs = hm.get("total_return"), hm.get("sharpe")
    gates.append(_gate("holdout opened", "holdout", bool(have(hold)), True, None if not have(hold) else True,
                       (f"opened {hold.get('access', {}).get('opened_at', '')} (opening #{hold.get('access', {}).get('opening_number', '?')})"
                        if have(hold) else "locked: run with --open-holdout once development is finished"), op="=="))
    gates.append(_gate("holdout return", "holdout", hr, g["holdout_min_return"], None if hr is None else hr > g["holdout_min_return"], op=">"))
    gates.append(_gate("holdout Sharpe", "holdout", hs, g["holdout_min_sharpe"], None if hs is None else hs > g["holdout_min_sharpe"], op=">"))
    # --- reproducibility ---------------------------------------------------------------------
    rstat = repro.get("status")
    gates.append(_gate("reproducibility", "reproducibility", rstat, "IDENTICAL",
                       None if rstat is None else (rstat == "IDENTICAL" or not g["require_reproducible"]), op="=="))
    if have(syn):
        ok = syn.get("engineering_checks", {}).get("passed")
        gates.append(_gate("synthetic engineering validation", "synthetic", ok, True, None if ok is None else bool(ok), op="=="))

    def group_ok(*groups: str) -> bool:
        rows = [x for x in gates if x["group"] in groups]
        return bool(rows) and all(x["passed"] is True for x in rows)

    level = "EXPERIMENTAL"
    if group_ok("walkforward"):
        level = "CANDIDATE"
        if group_ok("ablation", "stress", "bootstrap", "sanity", "leakage"):
            level = "VALIDATED CANDIDATE"
            if group_ok("holdout"):
                level = "HOLDOUT PASSED"
                if group_ok("reproducibility") and (not have(syn) or group_ok("synthetic")):
                    level = "PAPER ELIGIBLE"
    synthetic = (summary.get("manifest") or {}).get("kind") == "synthetic"
    if synthetic:
        level_out = "EXPERIMENTAL (SYNTHETIC)"
    else:
        level_out = level
    n_pass = sum(1 for x in gates if x["passed"] is True)
    n_fail = sum(1 for x in gates if x["passed"] is False)
    n_skip = sum(1 for x in gates if x["passed"] is None)
    return {"classification": level_out, "classification_if_real": level, "levels": LEVELS, "gates": gates,
            "passed": n_pass, "failed": n_fail, "not_evaluated": n_skip, "thresholds": g,
            "paper_eligible": bool(level == "PAPER ELIGIBLE" and not synthetic)}


__all__ = ["evaluate_gates", "DEFAULT_GATES", "LEVELS"]
