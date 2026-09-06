"""Diagnostic curves (spec section 53):

* d -> ADF(d)                (stationarity versus fractional order)
* d -> Corr(p, D^d p)        (memory preservation)
* d -> OOSScore(d)           (out-of-sample score of a model trained with adaptive order d)
* time -> S_F - S_0          (fractional-feature contribution over retraining cycles)

Curves are written as CSV/JSON; PNG plots are produced when matplotlib is available.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


class FractionalDiagnostics:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.contribution: list[dict[str, Any]] = []
        self.stationarity_curves: list[dict[str, Any]] = []
        self.oos_curves: list[dict[str, Any]] = []

    # ------------------------------------------------------------- record
    def record_stationarity(self, stationarity, timestamp: Optional[datetime] = None) -> None:
        rows = [c.to_dict() for c in stationarity.candidates]
        self.stationarity_curves.append({"timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                                         "d_star": stationarity.d_star, "curve": rows})
        self._write_csv("stationarity_vs_d.csv", rows, ["d", "adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue",
                                                         "correlation", "n_obs", "kernel_size"])

    def record_contribution(self, timestamp: datetime, full_score: float, baseline_score: float, delta: float,
                            accepted: bool, d_star: float) -> None:
        row = {"timestamp": timestamp.isoformat(), "full_score": full_score, "baseline_score": baseline_score,
               "delta_score": delta, "accepted": accepted, "d_star": d_star}
        self.contribution.append(row)
        self._write_csv("fractional_contribution.csv", self.contribution,
                        ["timestamp", "full_score", "baseline_score", "delta_score", "accepted", "d_star"])

    def record_oos_by_d(self, rows: list[dict[str, Any]], timestamp: Optional[datetime] = None) -> None:
        self.oos_curves.append({"timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(), "curve": rows})
        self._write_csv("oos_score_vs_d.csv", rows, ["d", "score", "sharpe", "accuracy", "net_pnl", "max_drawdown"])

    # ------------------------------------------------------------ compute
    @staticmethod
    def oos_score_by_d(trainer, history, candidates, combo=None, log=None) -> list[dict[str, Any]]:
        """Train/validate the full model for each fixed adaptive order d with the given hyperparameters
        (the production model's when called from a retrain; the first grid point otherwise)."""
        window = history.last(trainer.window_bars)
        rows: list[dict[str, Any]] = []
        previous_d = trainer.fe.adaptive_d
        try:
            for d in candidates:
                ds = trainer.builder.build(window, float(d))
                _, _, folds = trainer._layout(len(ds))
                fold_sets = trainer.build_fold_sets(window, ds, folds, fixed_d=float(d))
                ev = trainer.evaluate_candidate(fold_sets, combo or trainer.grid[0], ds.feature_names)
                agg = ev.aggregate
                rows.append({"d": float(d), "score": ev.mean_score, "sharpe": agg.sharpe, "accuracy": agg.accuracy,
                             "net_pnl": agg.net_pnl, "max_drawdown": agg.max_drawdown})
                if log:
                    log(f"d={d:.2f}: score={ev.mean_score:.3f} sharpe={agg.sharpe:.2f} acc={agg.accuracy:.3f}")
        finally:
            trainer.fe.set_adaptive_d(previous_d)
        return rows

    # -------------------------------------------------------------- output
    def _write_csv(self, name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with open(self.dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def save_json(self) -> Path:
        p = self.dir / "diagnostics.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"stationarity": self.stationarity_curves, "contribution": self.contribution,
                       "oos_by_d": self.oos_curves}, fh, indent=2, default=str)
        return p

    def plot(self) -> list[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return []
        out: list[Path] = []
        if self.stationarity_curves:
            curve = self.stationarity_curves[-1]["curve"]
            d = [c["d"] for c in curve]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(d, [c["adf_stat"] for c in curve], marker="o"); ax[0].set_title("ADF statistic vs d"); ax[0].set_xlabel("d")
            ax[1].plot(d, [c["correlation"] for c in curve], marker="o"); ax[1].set_title("Corr(p, D^d p) vs d"); ax[1].set_xlabel("d")
            p = self.dir / "stationarity_vs_d.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)
        if self.oos_curves:
            curve = self.oos_curves[-1]["curve"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([c["d"] for c in curve], [c["score"] for c in curve], marker="o"); ax.set_title("OOS score vs d"); ax.set_xlabel("d")
            p = self.dir / "oos_score_vs_d.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)
        if self.contribution:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(self.contribution)), [c["delta_score"] for c in self.contribution], marker="o")
            ax.axhline(0, color="k", lw=0.5); ax.set_title("Fractional contribution S_F - S_0 over retrains"); ax.set_xlabel("retrain cycle")
            p = self.dir / "fractional_contribution.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); out.append(p)
        return out


__all__ = ["FractionalDiagnostics"]
