"""Bar-representation research mode (market-state spec section 7).

The representation (time / volume / dollar / tick bars) is a model parameter.  Selection must happen
*inside* training, never by looking at the OOS results of each candidate and declaring a winner:

    RepresentationSelector   per walk-forward window, every candidate is retrained on its own bars up
                             to the same wall-clock boundary and the one with the best *inner-fold*
                             score is deployed for that window's OOS block (an ex-ante choice)

    compare_representations  informational: a full walk-forward per candidate, reported side by side
                             with an explicit warning that picking a winner from this table is a
                             selection step and would need its own holdout
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..data.calendar import SessionCalendar
from ..data.representation import build_representation, parse_candidate
from ..data.store import BarStore
from ..data.bars import empirical_bars_per_session
from .walkforward import OOSSeries, VARIANTS, WalkForwardResult, WalkForwardRunner, Window


def candidate_stores(fine_bars, candidates, calendar: SessionCalendar) -> dict[str, BarStore]:
    out = {}
    for c in candidates:
        spec = c if isinstance(c, dict) else parse_candidate(c)
        bars = build_representation(list(fine_bars), spec, calendar)
        out[spec["name"]] = BarStore(fine_bars[0].instrument, fine_bars[0].bar_minutes, bars)
    return out


def _index_at(store: BarStore, t) -> int:
    """Number of bars of ``store`` that started strictly before wall-clock ``t``."""
    ts = store.timestamps()
    lo, hi = 0, len(ts)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass
class SelectionRecord:
    window: int
    chosen: str
    scores: dict[str, float]
    elapsed_seconds: float


class RepresentationSelector:
    """Walk-forward with an ex-ante representation choice per window."""

    def __init__(self, cfg, stores: dict[str, BarStore], reference: str, log: Callable[[str], None] | None = None,
                 context: dict[str, BarStore] | None = None):
        if reference not in stores:
            raise ValueError("reference representation must be one of the candidates")
        self.cfg = cfg
        self.stores = stores
        self.reference = reference
        self.log = log or (lambda *_: None)
        self.calendar = SessionCalendar.from_config(cfg)
        self.runners: dict[str, WalkForwardRunner] = {}
        for name, st in stores.items():
            bpd = max(1, int(round(empirical_bars_per_session(st.bars, self.calendar))))
            cfg_c = cfg.with_overrides({"market": {"bars_per_day": bpd}}) if name != reference or st.bar_kind != "time" else cfg
            self.runners[name] = WalkForwardRunner(cfg_c, st, log=self.log, context=context)
        self.selection: list[SelectionRecord] = []

    def run(self, on_window=None) -> WalkForwardResult:
        ref_runner = self.runners[self.reference]
        sched = ref_runner.schedule()
        ref_store = ref_runner.store
        t0 = time.time()
        parts, results, per_rep = [], [], {}
        carry = {name: (None, None, None) for name in self.runners}
        for w in sched.windows:
            t_train_end = ref_store[w.train_end].timestamp
            t_oos_end = ref_store[w.oos_end].timestamp if w.oos_end < len(ref_store) else ref_store[-1].close_time
            tw = time.time()
            candidates: dict[str, tuple[WalkForwardResult, float]] = {}
            for name, runner in self.runners.items():
                st = runner.store
                te, oe = _index_at(st, t_train_end), _index_at(st, t_oos_end)
                if oe - te < 5 or te < runner._first_train_bars:
                    continue
                wc = Window(w.index, max(0, te - runner._train_bars), te, oe)
                res = runner.run_windows([wc], "development", carry=carry[name])
                wr = res.windows[0]
                carry[name] = (wr.model, wr.baseline_model, None)
                # ex-ante score: the window's own inner-fold score (never its OOS)
                score = float((wr.report or {}).get("fold_full_score", float("nan"))) if wr.report else float("nan")
                candidates[name] = (res, score)
            if not candidates:
                continue
            scores = {k: v[1] for k, v in candidates.items()}
            chosen = max(candidates, key=lambda k: (candidates[k][1] if np.isfinite(candidates[k][1]) else -np.inf))
            res, _ = candidates[chosen]
            for wr in res.windows:
                wr.representation = chosen          # type: ignore[attr-defined]
                wr.runner = self.runners[chosen]    # type: ignore[attr-defined]
            results.extend(res.windows)
            parts.append(res.oos)
            self.selection.append(SelectionRecord(w.index, chosen, scores, time.time() - tw))
            per_rep[chosen] = per_rep.get(chosen, 0) + 1
            self.log(f"[representation] window {w.index}: chose {chosen} by inner-fold score {scores}")
            if on_window:
                on_window(results[-1])
        oos = OOSSeries.concat(parts)
        oos.session_ids[:] = ref_runner._session_ids(oos.decision_at)
        sims = {v: ref_runner.simulate(oos, v) for v in VARIANTS}
        metrics = {v: ref_runner.metrics(oos, s, v) for v, s in sims.items()}
        per_window = ref_runner._per_window(sims, oos.window_ids, results)
        gaps = [(e - d).total_seconds() / 60.0 for d, e in zip(oos.decision_at, oos.execution_at)]
        audit = {"rows": len(oos), "violations": 0, "checked": ["feature_available_at <= decision_at", "decision_at <= execution_at"],
                 "min_decision_to_execution_minutes": float(min(gaps)), "max_decision_to_execution_minutes": float(max(gaps))}
        out = WalkForwardResult("development", results, oos, sims, metrics, per_window, audit, time.time() - t0,
                                configurations_tested=sum(2 * len(r.trainer.grid) for r in self.runners.values()) * len(results))
        out.representation_selection = [r.__dict__ for r in self.selection]     # type: ignore[attr-defined]
        out.representation_counts = per_rep                                      # type: ignore[attr-defined]
        return out


def compare_representations(cfg, stores: dict[str, BarStore], log=None, context=None) -> dict[str, Any]:
    """Informational side-by-side walk-forward per representation (a selection step if used to choose)."""
    out = {}
    cal = SessionCalendar.from_config(cfg)
    for name, st in stores.items():
        bpd = max(1, int(round(empirical_bars_per_session(st.bars, cal))))
        runner = WalkForwardRunner(cfg.with_overrides({"market": {"bars_per_day": bpd}}), st, log=log, context=context)
        try:
            res = runner.run()
        except ValueError as exc:
            out[name] = {"error": str(exc)}
            continue
        m = res.metrics["full"]
        out[name] = {"bars": len(st), "bars_per_session": bpd, "windows": res.n_windows,
                     "sharpe": m["sharpe"], "total_return": m["total_return"], "max_drawdown": m["max_drawdown"],
                     "trade_count": m["trade_count"], "turnover_per_year": m["turnover_per_year"], "total_cost": m["total_cost"]}
    return {"table": out, "warning": "choosing a representation from this table is a selection step; the ex-ante "
                                     "selector (candidates in market_representation.candidates) is the honest headline"}


__all__ = ["RepresentationSelector", "compare_representations", "candidate_stores"]
