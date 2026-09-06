"""Multi-seed synthetic engineering validation (research spec section 19).

Runs the research pipeline on an ensemble of synthetic price paths with varied drift, volatility,
autocorrelation, jumps, volatility regimes and long-memory strength.  The purpose is to prove the
*machinery* (no look-ahead, labels aligned, reproducible, the strategy does not manufacture edge
from a pure random walk), never to estimate performance:

    SYNTHETIC / ENGINEERING VALIDATION — NOT PERFORMANCE EVIDENCE
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..data.calendar import SessionCalendar
from ..data.store import BarStore
from ..data.synthetic import generate_synthetic_bars
from .artifacts import write_json
from .metrics import summarize_distribution
from .runner import SYNTHETIC_STAGES, ResearchRun

LABEL = "SYNTHETIC / ENGINEERING VALIDATION — NOT PERFORMANCE EVIDENCE"


def draw_path_params(rng: np.random.Generator, k: int) -> dict[str, Any]:
    """Path 0 is always a pure random walk (no exploitable structure): the strategy must not profit there."""
    if k == 0:
        return {"amplitude": 0.0, "memory_d": 0.0, "drift": 0.0, "base_vol": 0.0025, "autocorrelation": 0.0,
                "jump_intensity": 0.0, "jump_size": 0.0, "regime_bars": 0, "vol_clustering": 0.0}
    return {"amplitude": float(rng.choice([0.0, 2.0, 4.0, 6.0])), "memory_d": float(rng.uniform(0.2, 0.5)),
            "drift": float(rng.uniform(-0.15, 0.15)), "base_vol": float(rng.uniform(0.0015, 0.0035)),
            "autocorrelation": float(rng.uniform(-0.15, 0.15)), "jump_intensity": float(rng.choice([0.0, 0.002, 0.01])),
            "jump_size": float(rng.uniform(2.0, 5.0)), "regime_bars": int(rng.choice([0, 65, 260])),
            "vol_clustering": float(rng.uniform(0.0, 0.5))}


def run_synthetic_ensemble(cfg, n_seeds: int = 5, n_bars: int = 4000, master_seed: int = 0, artifacts_root: str | Path | None = None,
                           log: Callable[[str], None] | None = None, on_progress: Callable[[str, str, float], None] | None = None,
                           should_stop: Callable[[], bool] | None = None, stages: list[str] | None = None) -> dict[str, Any]:
    _log = log or (lambda *_: None)
    progress = on_progress or (lambda *_: None)
    root = Path(artifacts_root) if artifacts_root is not None else Path(cfg.paths.artifacts_dir)
    cal = SessionCalendar.from_config(cfg)
    rng = np.random.default_rng(master_seed)
    t0 = time.time()
    ensemble_id = f"synthetic_ensemble_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _log(LABEL)
    runs = []
    for k in range(n_seeds):
        if should_stop and should_stop():
            raise InterruptedError("synthetic ensemble interrupted")
        params = draw_path_params(rng, k)
        seed = master_seed * 1000 + k
        progress("synthetic", f"synthetic path {k + 1}/{n_seeds}", k / max(1, n_seeds))
        _log(f"--- synthetic path {k} (seed {seed}): {params}")
        bars = generate_synthetic_bars(n_bars, seed=seed, instrument=str(cfg.market.instrument), calendar=cal, **params)
        store = BarStore(str(cfg.market.instrument), int(cfg.market.bar_minutes), bars)
        run = ResearchRun(cfg, store, {"source": "synthetic", "seed": seed, "path_params": params}, root,
                          run_id=f"{ensemble_id}_path{k}", log=_log, kind="synthetic", stages=stages or SYNTHETIC_STAGES,
                          should_stop=should_stop,
                          on_progress=lambda st, msg, f, k=k: progress("synthetic", f"path {k + 1}/{n_seeds}: {msg}", (k + f) / max(1, n_seeds)))
        summary = run.execute()
        dev = summary["development"]
        m = dev["metrics"]["full"]
        rand_pct = ((summary.get("baselines") or {}).get("_comparison", {}).get("strategy", {}) or {}).get("random_signal_percentile")
        runs.append({"path": k, "seed": seed, "run_id": summary["run_id"], "params": params, "n_windows": dev["n_windows"],
                     "total_return": m["total_return"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "trade_count": m["trade_count"], "median_delta_sharpe": (summary.get("ablation") or {}).get("median_delta_sharpe"),
                     "timestamp_violations": dev["timestamp_audit"]["violations"],
                     "labels_aligned": (summary.get("leakage") or {}).get("label_alignment", {}).get("labels_aligned"),
                     "reproducible": (summary.get("reproducibility") or {}).get("status"),
                     "random_signal_percentile": rand_pct, "run_dir": str(run.run_dir)})
    no_structure = [r for r in runs if r["params"]["amplitude"] == 0.0]
    structured = [r for r in runs if r["params"]["amplitude"] > 0.0]
    checks = {
        "all_paths_completed": True,
        "no_timestamp_violations": all(r["timestamp_violations"] == 0 for r in runs),
        "labels_aligned_everywhere": all(r["labels_aligned"] in (True, None) for r in runs),
        "no_structure_paths": len(no_structure),
        "no_structure_median_sharpe": float(np.median([r["sharpe"] for r in no_structure])) if no_structure else None,
        "no_edge_on_random_walk": (bool(np.median([r["sharpe"] for r in no_structure]) < 1.0) if no_structure else None),
        "structured_paths": len(structured),
        "structured_positive_fraction": float(np.mean([r["total_return"] > 0 for r in structured])) if structured else None,
    }
    checks["passed"] = bool(checks["no_timestamp_violations"] and checks["labels_aligned_everywhere"]
                            and checks["no_edge_on_random_walk"] in (True, None)
                            and all(r["reproducible"] in ("IDENTICAL", None) for r in runs))
    out = {"label": LABEL, "ensemble_id": ensemble_id, "n_seeds": n_seeds, "n_bars": n_bars, "master_seed": master_seed,
           "runs": runs, "engineering_checks": checks,
           "distribution": {**summarize_distribution(np.array([r["total_return"] for r in runs]), "return_"),
                            **summarize_distribution(np.array([r["sharpe"] for r in runs]), "sharpe_"),
                            **summarize_distribution(np.array([r["max_drawdown"] for r in runs]), "max_drawdown_"),
                            **summarize_distribution(np.array([r["median_delta_sharpe"] for r in runs if r["median_delta_sharpe"] is not None]), "delta_sharpe_")},
           "elapsed_seconds": time.time() - t0}
    write_json(root / "runs" / f"{ensemble_id}.json", out)
    _log(f"synthetic ensemble: {n_seeds} paths, Sharpe median {out['distribution'].get('sharpe_median')}, "
         f"random-walk paths median Sharpe {checks['no_structure_median_sharpe']}, engineering checks passed: {checks['passed']}")
    return out


__all__ = ["run_synthetic_ensemble", "draw_path_params", "LABEL"]
