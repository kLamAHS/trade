"""The research pipeline (spec section 34): one entry point that runs the validation stages in
order, writes the run directory and classifies the result.

    walkforward -> ablation -> baselines -> regimes -> cost -> timing -> perturbation -> d_perturbation
    -> sanity -> leakage -> bootstrap -> reproducibility -> [holdout] -> gates -> artifacts

Every stage tries to disprove the strategy; the summary records what was run so that a result
with skipped stages can never be mistaken for a fully validated one.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..config import FrozenConfig
from ..data.store import BarStore
from .artifacts import write_run
from .baselines import run_baselines
from .bootstrap import ablation_bootstrap, block_bootstrap, monte_carlo_trades, multiple_testing
from .gates import evaluate_gates
from .manifest import RunManifest, results_hash
from .metrics import summarize_distribution
from .regimes import regime_attribution, tag_regimes
from .sanity import run_leakage, run_sanity
from .stress import cost_curve, d_perturbation, parameter_perturbations, timing_delays
from .walkforward import WalkForwardResult, WalkForwardRunner

STAGES = ["walkforward", "ablation", "baselines", "regimes", "cost", "timing", "perturbation", "d_perturbation",
          "sanity", "leakage", "bootstrap", "reproducibility", "holdout", "gates"]
QUICK_STAGES = ["walkforward", "ablation", "baselines", "regimes", "cost", "timing", "perturbation", "bootstrap", "leakage",
                "reproducibility", "gates"]
SYNTHETIC_STAGES = ["walkforward", "ablation", "cost", "timing", "perturbation", "bootstrap", "leakage", "gates"]


def resolve_stages(spec: str | list[str] | None, open_holdout: bool = False) -> list[str]:
    if spec is None or spec == "full":
        stages = list(STAGES)
    elif spec == "quick":
        stages = list(QUICK_STAGES)
    elif isinstance(spec, str):
        stages = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        stages = list(spec)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown research stage(s) {unknown}; known: {STAGES}")
    if "walkforward" not in stages:
        stages.insert(0, "walkforward")
    if open_holdout and "holdout" not in stages:
        stages.append("holdout")
    if not open_holdout and "holdout" in stages:
        stages.remove("holdout")
    if "gates" not in stages:
        stages.append("gates")
    return [s for s in STAGES if s in stages]


class ResearchRun:
    def __init__(self, cfg: FrozenConfig, store: BarStore, data_info: dict[str, Any] | None = None,
                 artifacts_root: str | Path | None = None, run_id: str | None = None, log: Callable[[str], None] | None = None,
                 kind: str = "real", stages: str | list[str] | None = None, open_holdout: bool = False,
                 on_progress: Callable[[str, str, float], None] | None = None, should_stop: Callable[[], bool] | None = None,
                 full_reproducibility: bool = False, synthetic_summary: dict[str, Any] | None = None):
        self.cfg = cfg
        self.store = store
        self.data_info = dict(data_info or {})
        self.root = Path(artifacts_root) if artifacts_root is not None else Path(cfg.paths.artifacts_dir)
        self.kind = kind
        self.stages = resolve_stages(stages, open_holdout)
        self.open_holdout = bool(open_holdout)
        self.on_progress = on_progress or (lambda *_: None)
        self.should_stop = should_stop or (lambda: False)
        self.full_reproducibility = bool(full_reproducibility)
        self.synthetic_summary = synthetic_summary
        self.lines: list[str] = []
        self._log_cb = log or (lambda *_: None)
        self.r = (cfg.get("research", {}) or {})
        self.runner = WalkForwardRunner(cfg, store, log=self.log)
        self.manifest = RunManifest.create(cfg, store, self.data_info, self.runner.schedule().to_dict(), self.stages,
                                           kind=kind, run_id=run_id)
        self.run_id = self.manifest.run_id
        self.run_dir = self.root / "runs" / self.run_id
        self.summary: dict[str, Any] = {}
        self.dev: Optional[WalkForwardResult] = None
        self.hold: Optional[WalkForwardResult] = None

    # ----------------------------------------------------------------- utils
    def log(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {message}"
        self.lines.append(line)
        self._log_cb(message)

    def _sub(self, key: str, default=None):
        v = self.r.get(key, {}) or {}
        return v if v else (default or {})

    def _check_stop(self) -> None:
        if self.should_stop():
            raise InterruptedError("research run interrupted")

    def _stage(self, name: str, i: int) -> None:
        self._check_stop()
        self.on_progress(name, f"stage {i + 1}/{len(self.stages)}: {name}", i / len(self.stages))
        self.log(f"=== stage {name} ===")

    # --------------------------------------------------------------- execute
    def execute(self) -> dict[str, Any]:
        t0 = time.time()
        s = self.summary
        s.update({"run_id": self.run_id, "kind": self.kind, "evidence_label": self.manifest.evidence_label,
                  "manifest": {k: v for k, v in self.manifest.to_dict().items() if k != "config"},
                  "stages": self.stages, "schedule": self.runner.schedule().to_dict(), "stages_completed": []})
        self.log(f"research run {self.run_id}: {self.kind} data, {len(self.store)} bars of {self.cfg.market.instrument}, "
                 f"config {self.manifest.config_hash} / model config {self.manifest.model_config_hash}, code {self.manifest.code_commit[:12]}")
        self.log(self.manifest.evidence_label)
        for i, stage in enumerate(self.stages):
            self._stage(stage, i)
            getattr(self, f"stage_{stage}")()
            s["stages_completed"].append(stage)
        s["elapsed_seconds"] = time.time() - t0
        self.on_progress("artifacts", "writing run directory", 0.98)
        paths = write_run(self.run_dir, self.manifest, s, self.dev, self.hold, self.cfg, self.lines)
        s["artifacts"] = paths
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as fh:   # rewrite with the artifact paths
            from .artifacts import _clean
            json.dump(_clean(s), fh, indent=2, default=str)
        self.log(f"run directory: {self.run_dir}")
        self.on_progress("done", s.get("gates", {}).get("classification", "done"), 1.0)
        return s

    # ---------------------------------------------------------------- stages
    def stage_walkforward(self) -> None:
        n = len(self.runner.schedule().windows)

        def on_window(wr):
            self._check_stop()
            self.on_progress("walkforward", f"walk-forward window {wr.window.index + 1}/{n}", 0.05 + 0.45 * (wr.window.index + 1) / max(1, n))

        self.dev = self.runner.run(on_window=on_window)
        dev = self.dev.summary()
        s = self.summary
        s["development"] = dev
        s["equity"] = {"development": self._equity_payload(self.dev)}
        s["results_hash"] = results_hash({v: self.dev.oos.forecasts[v]["E"] for v in ("full", "baseline")},
                                         self.dev.sims["full"].equity, [w.fitted_model_hash for w in self.dev.windows])
        m = dev["metrics"]["full"]
        self.log(f"development OOS: {dev['n_windows']} windows, {dev['n_rows']} rows, return {m['total_return']:+.2%}, "
                 f"Sharpe {m['sharpe']:.2f}, max DD {m['max_drawdown']:.2%}, trades {m['trade_count']}, "
                 f"accepted windows {dev['accepted_windows']}/{dev['n_windows']}; results hash {s['results_hash']}")

    def _equity_payload(self, res: WalkForwardResult) -> dict[str, Any]:
        oos = res.oos
        return {"timestamps": [t.isoformat() for t in oos.decision_at], "window": oos.window_ids.tolist(),
                "full": res.sims["full"].equity[1:].tolist(), "baseline": res.sims["baseline"].equity[1:].tolist(),
                "production": res.sims["production"].equity[1:].tolist(), "exposure": res.sims["full"].exposures.tolist(),
                "bar_pnl": res.sims["full"].bar_pnl.tolist(), "model_id": list(oos.model_ids)}

    def stage_ablation(self) -> None:
        dev = self.dev
        pw = dev.per_window
        deltas = np.array([w["delta_sharpe"] for w in pw if w.get("delta_sharpe") is not None], dtype=float)
        dret = np.array([w["delta_return"] for w in pw if w.get("delta_return") is not None], dtype=float)
        full_pos = int(sum(1 for w in pw if (w.get("full_net_return") or 0) > 0))
        base_pos = int(sum(1 for w in pw if (w.get("baseline_net_return") or 0) > 0))
        boot = ablation_bootstrap(deltas, seed=int(self._sub("bootstrap").get("seed", 0)))
        trainer_deltas = np.array([w.holdout_delta for w in dev.windows if np.isfinite(w.holdout_delta)], dtype=float)
        mf, mb = dev.metrics["full"], dev.metrics["baseline"]
        self.summary["ablation"] = {
            "cycles": len(pw), "positive_cycles_full": full_pos, "positive_cycles_baseline": base_pos,
            "mean_delta_sharpe": float(deltas.mean()) if len(deltas) else None,
            "median_delta_sharpe": float(np.median(deltas)) if len(deltas) else None,
            "positive_delta_fraction": float(np.mean(deltas > 0)) if len(deltas) else None,
            "mean_delta_return": float(dret.mean()) if len(dret) else None,
            "delta_sharpe_ci": boot.get("mean_ci"), "bootstrap": boot,
            "oos_full": {k: mf[k] for k in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "profit_factor", "trade_count")},
            "oos_baseline": {k: mb[k] for k in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "profit_factor", "trade_count")},
            "trainer_holdout_delta": {**summarize_distribution(trainer_deltas, ""), "note": "S_F - S_0 on each window's untouched training holdout (the live promotion signal)"},
            "per_window_delta_sharpe": deltas.tolist(),
        }
        a = self.summary["ablation"]
        self.log(f"ablation: ΔSharpe mean {a['mean_delta_sharpe']:+.2f} median {a['median_delta_sharpe']:+.2f} "
                 f"(positive in {a['positive_delta_fraction']:.0%} of {a['cycles']} windows, CI {boot.get('mean_ci')}); "
                 f"OOS Sharpe full {mf['sharpe']:.2f} vs baseline {mb['sharpe']:.2f}")

    def stage_baselines(self) -> None:
        b = self._sub("baselines")
        out = run_baselines(self.runner, self.dev.oos, self.dev.sims, n_random=int(b.get("random_permutations", 20)),
                            seed=int(self._sub("bootstrap").get("seed", 0)), vol_target=float(b.get("vol_target", 0.10)),
                            momentum_bars=int(b.get("momentum_bars", 65)))
        self.summary["baselines"] = out
        self.summary["equity"]["baselines"] = {k: v["equity"][1:] for k, v in out.items() if not k.startswith("_")}
        for k, v in out.items():
            if not k.startswith("_"):
                v.pop("equity", None)
        comp = out["_comparison"]
        self.log("baselines (Sharpe): " + ", ".join(f"{k} {v['sharpe']:.2f}" for k, v in comp.items()))

    def stage_regimes(self) -> None:
        rg = self._sub("regimes")
        tags = tag_regimes(self.store.log_close(), self.dev.oos.bar_index, int(rg.get("short_bars", 20)), int(rg.get("long_bars", 200)))
        sim = self.dev.sims["full"]
        self.summary["regimes"] = {"protocol": "ex-ante: trailing 20-bar vs 200-bar realised volatility; close vs trailing 200-bar mean",
                                   "attribution": regime_attribution(sim.bar_pnl, sim.exposures, sim.trades, tags, self.runner.bars_per_year)}
        self.summary["equity"]["development"]["regime_vol"] = tags["volatility"].tolist()
        self.summary["equity"]["development"]["regime_trend"] = tags["trend"].tolist()

    def stage_cost(self) -> None:
        c = self._sub("cost")
        out = cost_curve(self.runner, self.dev.oos, tuple(c.get("flat_bps_levels", (0, 1, 2, 5, 10))),
                         tuple(c.get("model_cost_scales", (1.0, 2.0, 3.0))))
        self.summary["cost_curve"] = out
        self.log(f"cost curve: mean model round trip {out['mean_model_roundtrip_bps']:.2f} bps, breakeven flat cost "
                 f"{out['breakeven_flat_bps']} bps, profitable at 2x model cost: {out['profitable_at_2x_cost']}")

    def stage_timing(self) -> None:
        t = self._sub("timing")
        out = timing_delays(self.runner, self.dev.oos, tuple(int(x) for x in t.get("delays", (0, 1, 2))))
        self.summary["timing"] = out
        self.log("timing: " + ", ".join(f"+{r['delay']} bar Sharpe {r['sharpe']:.2f}" for r in out["rows"]))

    def stage_perturbation(self) -> None:
        st = self._sub("stress")
        out = parameter_perturbations(self.runner, self.dev.oos, float(st.get("collapse_fraction", 0.75)))
        self.summary["parameter_perturbations"] = out
        self.log(f"parameter perturbations: min Sharpe {out['min_sharpe']:.2f}, collapsed {out['n_collapsed']}/{len(out['rows']) - 1}")

    def stage_d_perturbation(self) -> None:
        st = self._sub("stress")
        out = d_perturbation(self.runner, self.dev, tuple(int(x) for x in st.get("d_steps", (-2, -1, 0, 1, 2))), log=self.log)
        self.summary["d_perturbation"] = out
        self.log("d perturbation (light refit): " + ", ".join(f"{r['label']} Sharpe {r['sharpe']:.2f}" for r in out["rows"]))

    def stage_sanity(self) -> None:
        sa = self._sub("sanity")
        out = run_sanity(self.runner, self.dev, int(sa.get("target_shift_bars", 20)), int(sa.get("random_permutations", 20)),
                         int(self._sub("bootstrap").get("seed", 0)), float(sa.get("max_destroyed_sharpe", 1.0)), log=self.log)
        self.summary["sanity"] = out
        self.log(f"sanity: {out['n_tests'] - out['n_failed']}/{out['n_tests']} passed; " +
                 ", ".join(f"{k} Sharpe {v['sharpe']:.2f}" for k, v in out["tests"].items() if "sharpe" in v))

    def stage_leakage(self) -> None:
        sa = self._sub("sanity")
        out = run_leakage(self.runner, self.dev, int(sa.get("target_shift_bars", 20)), log=self.log)
        self.summary["leakage"] = out
        self.log(f"leakage: timestamps violations {out['timestamps']['violations']}, labels aligned {out['label_alignment']['labels_aligned']}")

    def stage_bootstrap(self) -> None:
        b = self._sub("bootstrap")
        sim = self.dev.sims["full"]
        seed = int(b.get("seed", 0))
        block = block_bootstrap(sim.bar_pnl, self.runner.bars_per_year, int(b.get("block_bars", 65)), int(b.get("n_boot", 1000)), seed)
        mc = monte_carlo_trades(sim.trade_pnls(), int(b.get("monte_carlo_paths", 2000)), seed)
        prior = int(self._sub("multiple_testing").get("prior_trials", 0))
        mt = multiple_testing(self.dev.metrics["full"]["sharpe"], len(sim.bar_pnl), self.runner.bars_per_year,
                              self.dev.configurations_tested + prior)
        mt["configurations_tested_in_run"] = self.dev.configurations_tested
        mt["prior_trials_declared"] = prior
        self.summary["bootstrap"] = {"block": block}
        self.summary["monte_carlo"] = mc
        self.summary["multiple_testing"] = mt
        self.log(f"bootstrap: Sharpe {block['sharpe']['point']:.2f} CI [{block['sharpe']['ci_low']:.2f}, {block['sharpe']['ci_high']:.2f}], "
                 f"P(loss) {block['total_return']['p_loss']:.1%}; Monte Carlo P(DD>20%) {mc.get('max_drawdown', {}).get('p_over_20pct')}; "
                 f"Bonferroni p {mt.get('bonferroni_p_value')}")

    def stage_reproducibility(self) -> None:
        rp = self._sub("reproducibility")
        n = int(rp.get("windows", 1))
        checks = [self.runner.reproduce_window(w) for w in self.dev.windows[:n]]
        out: dict[str, Any] = {"windows_checked": [c["window"] for c in checks], "checks": checks,
                               "status": "IDENTICAL" if all(c["identical"] for c in checks) else "REPRODUCIBILITY FAILURE"}
        if self.full_reproducibility:
            self.log("full reproducibility check: re-running the development walk-forward")
            second = WalkForwardRunner(self.cfg, self.store).run()
            h2 = results_hash({v: second.oos.forecasts[v]["E"] for v in ("full", "baseline")}, second.sims["full"].equity,
                              [w.fitted_model_hash for w in second.windows])
            out["full_rerun"] = {"results_hash": [self.summary["results_hash"], h2], "identical": h2 == self.summary["results_hash"]}
            if not out["full_rerun"]["identical"]:
                out["status"] = "REPRODUCIBILITY FAILURE"
        out["manifest_hash"] = self.manifest.manifest_hash
        out["results_hash"] = self.summary["results_hash"]
        self.summary["reproducibility"] = out
        self.log(f"reproducibility: {out['status']} ({len(checks)} window(s) retrained and compared bit for bit)")

    def stage_holdout(self) -> None:
        access_path = self.root / "holdout_access.jsonl"
        previous = 0
        if access_path.exists():
            previous = sum(1 for line in access_path.read_text(encoding="utf-8").splitlines() if line.strip())
        sched = self.runner.holdout_schedule()
        record = {"holdout_opened_at": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                  "strategy_commit_hash": self.manifest.code_commit, "config_hash": self.manifest.config_hash,
                  "model_config_hash": self.manifest.model_config_hash, "data_hash": self.manifest.data["data_hash"],
                  "model_version": self.dev.windows[-1].model_id if self.dev.windows else None,
                  "holdout_start_bar": sched.holdout_start, "holdout_bars": len(self.store) - (sched.holdout_start or 0),
                  "opening_number": previous + 1, "kind": self.kind}
        access_path.parent.mkdir(parents=True, exist_ok=True)
        with open(access_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if previous:
            self.log(f"WARNING: the holdout has been opened {previous} time(s) before; this is no longer a first look")
        n = len(sched.windows)
        self.hold = self.runner.run_holdout(self.dev, on_window=lambda wr: self.on_progress(
            "holdout", f"holdout window {wr.window.index + 1}/{n}", 0.85 + 0.1 * (wr.window.index + 1) / max(1, n)))
        h = self.hold.summary()
        h["access"] = {"opened_at": record["holdout_opened_at"], "opening_number": record["opening_number"],
                       "previous_openings": previous, "record": record, "log": str(access_path)}
        self.summary["holdout"] = h
        self.summary["equity"]["holdout"] = self._equity_payload(self.hold)
        m = h["metrics"]["full"]
        self.log(f"HOLDOUT (opening #{previous + 1}): {h['n_windows']} windows, return {m['total_return']:+.2%}, Sharpe {m['sharpe']:.2f}, "
                 f"max DD {m['max_drawdown']:.2%}, trades {m['trade_count']}")

    def stage_gates(self) -> None:
        if self.synthetic_summary is not None:
            self.summary["synthetic_ensemble"] = self.synthetic_summary
        out = evaluate_gates(self.summary, dict(self._sub("gates")))
        self.summary["gates"] = out
        self.log(f"classification: {out['classification']} ({out['passed']} gates passed, {out['failed']} failed, "
                 f"{out['not_evaluated']} not evaluated)")
        for gt in out["gates"]:
            if gt["passed"] is False:
                self.log(f"  FAILED gate: {gt['gate']} = {gt['value']} (needs {gt['op']} {gt['threshold']}) {gt['evidence']}")


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "summary.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_runs(artifacts_root: str | Path) -> list[dict[str, Any]]:
    root = Path(artifacts_root) / "runs"
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        p = d / "summary.json"
        if not p.exists():
            continue
        try:
            s = load_summary(d)
        except (OSError, json.JSONDecodeError):
            continue
        m = ((s.get("development") or {}).get("metrics") or {}).get("full") or {}
        out.append({"run_id": s.get("run_id", d.name), "kind": s.get("kind"), "created_at": (s.get("manifest") or {}).get("created_at"),
                    "instrument": (s.get("manifest") or {}).get("instrument"), "classification": (s.get("gates") or {}).get("classification"),
                    "sharpe": m.get("sharpe"), "total_return": m.get("total_return"), "max_drawdown": m.get("max_drawdown"),
                    "trades": m.get("trade_count"), "windows": (s.get("development") or {}).get("n_windows"),
                    "holdout": bool(s.get("holdout")), "stages": s.get("stages_completed"), "path": str(d)})
    return out


__all__ = ["ResearchRun", "STAGES", "QUICK_STAGES", "SYNTHETIC_STAGES", "resolve_stages", "load_summary", "list_runs"]
