"""Run artifact writer (research spec section 27):

    artifacts/runs/<run_id>/
        manifest.yaml  summary.json  equity.csv  trades.csv  fills.csv  decisions.csv  retrains.csv
        models/        one JSON per fitted model (metadata + fitted-model hash)
        diagnostics/   one JSON per analysis (ablation, baselines, cost curve, ...)
        plots/         dependency-free SVG plots (equity with segments, underwater, rolling Sharpe)
        logs/run.log
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .metrics import drawdown_series, rolling_sharpe

TRADE_FIELDS = ["trade_id", "segment", "entry_row", "exit_row", "decision_timestamp", "execution_timestamp", "exit_timestamp",
                "model_id", "direction", "forecast", "expected_return", "probability_up", "confidence", "target_exposure",
                "approved_exposure", "max_exposure", "entry_price", "exit_price", "bars_held", "exit_reason", "gross_pnl",
                "cost", "net_pnl", "gross_pnl_currency", "cost_currency", "net_pnl_currency", "equity_before", "delay"]


def _clean(v):
    """JSON-safe values: NaN -> null, ±inf -> "inf"/"-inf" (browsers reject the Infinity literal)."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(payload), fh, indent=2, default=str, allow_nan=True)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k)) for k in fields})


def decision_rows(result, variant: str = "full", segment: str = "development", equity_offset: float = 1.0) -> list[dict[str, Any]]:
    oos, sim = result.oos, result.sims[variant]
    f = oos.forecasts[variant]
    rows = []
    for i in range(len(oos)):
        rows.append({"row": i, "segment": segment, "bar_index": int(oos.bar_index[i]), "window": int(oos.window_ids[i]),
                     "session": int(oos.session_ids[i]), "model_id": oos.model_ids[i],
                     "feature_available_at": oos.feature_available_at[i], "decision_at": oos.decision_at[i],
                     "execution_at": oos.execution_at[i], "M": f["M"][i], "P": f["P"][i], "E": f["E"][i],
                     "sigma": oos.sigma[i], "sigma_ref": oos.sigma_ref[i], "cost_roundtrip": oos.cost_roundtrip[i],
                     "cost_side_exec": oos.cost_side_exec[i], "close": math.exp(oos.log_close[i]), "open_next": oos.open_next[i],
                     "open_next2": oos.open_next2[i], "label_y_norm": oos.y_norm[i], "target_exposure": sim.targets[i],
                     "exposure": sim.exposures[i], "halted": bool(sim.halted[i]), "gross_pnl": sim.gross_bar_pnl[i],
                     "cost": sim.cost_bar[i], "net_pnl": sim.bar_pnl[i], "equity": sim.equity[i + 1] * equity_offset})
    return rows


DECISION_FIELDS = ["row", "segment", "bar_index", "window", "session", "model_id", "feature_available_at", "decision_at",
                   "execution_at", "M", "P", "E", "sigma", "sigma_ref", "cost_roundtrip", "cost_side_exec", "close",
                   "open_next", "open_next2", "label_y_norm", "target_exposure", "exposure", "halted", "gross_pnl", "cost",
                   "net_pnl", "equity"]


def fill_rows(result, variant: str = "full", segment: str = "development") -> list[dict[str, Any]]:
    """One fill per exposure change (market order at the next open, always filled; section 12)."""
    oos, sim = result.oos, result.sims[variant]
    rows = []
    prev = 0.0
    trade_by_row = {}
    for t in sim.trades:
        for r in range(t["entry_row"], t.get("exit_row", t["entry_row"]) + 1):
            trade_by_row[r] = t["trade_id"]
    for i in range(len(oos)):
        q = float(sim.exposures[i])
        if q != prev:
            rows.append({"fill_id": len(rows) + 1, "segment": segment, "row": i, "trade_id": trade_by_row.get(i),
                         "submitted_at": oos.decision_at[i], "filled_at": oos.execution_at[i], "status": "FILLED",
                         "order_type": "market", "side": "buy" if q > prev else "sell", "from_exposure": prev, "to_exposure": q,
                         "quantity_exposure": abs(q - prev), "price": oos.open_next[i], "slippage_cost": sim.cost_bar[i],
                         "delay_bars": sim.delay, "model_id": oos.model_ids[i]})
            prev = q
    return rows


FILL_FIELDS = ["fill_id", "segment", "row", "trade_id", "submitted_at", "filled_at", "status", "order_type", "side",
               "from_exposure", "to_exposure", "quantity_exposure", "price", "slippage_cost", "delay_bars", "model_id"]


def equity_rows(result, segment: str, offsets: dict[str, float]) -> list[dict[str, Any]]:
    oos = result.oos
    sims = result.sims
    rows = []
    dd = drawdown_series(sims["full"].equity)
    for i in range(len(oos)):
        rows.append({"decision_at": oos.decision_at[i], "segment": segment, "window": int(oos.window_ids[i]),
                     "equity_full": sims["full"].equity[i + 1] * offsets.get("full", 1.0),
                     "equity_baseline": sims["baseline"].equity[i + 1] * offsets.get("baseline", 1.0),
                     "equity_production": sims["production"].equity[i + 1] * offsets.get("production", 1.0),
                     "exposure": sims["full"].exposures[i], "drawdown": dd[i + 1], "net_pnl": sims["full"].bar_pnl[i]})
    return rows


EQUITY_FIELDS = ["decision_at", "segment", "window", "equity_full", "equity_baseline", "equity_production", "exposure",
                 "drawdown", "net_pnl"]


def retrain_rows(results: list, segment: str) -> list[dict[str, Any]]:
    rows = []
    for res in results:
        for w in res.windows:
            d = w.to_dict()
            rows.append({"segment": segment if res.label == "development" else res.label, **{k: v for k, v in d.items() if k != "window"},
                         "window": w.window.index, "train_start": w.window.train_start, "train_end": w.window.train_end, "oos_end": w.window.oos_end,
                         "train_from": w.train_span[0], "train_to": w.train_span[1], "oos_from": w.oos_span[0], "oos_to": w.oos_span[1]})
    return rows


RETRAIN_FIELDS = ["segment", "window", "train_start", "train_end", "oos_end", "train_from", "train_to", "oos_from", "oos_to",
                  "model_id", "baseline_model_id", "deployed_model_id", "accepted", "carried", "error", "d_star", "d_full",
                  "fold_d_stars", "best_params", "baseline_params", "holdout_score", "baseline_holdout_score", "holdout_delta",
                  "n_training_rows", "n_oos_rows", "fitted_model_hash", "baseline_fitted_model_hash", "elapsed_seconds"]


# ------------------------------------------------------------------ SVG plots (no dependencies)
def _svg_line_chart(series: dict[str, list[float]], title: str, width: int = 960, height: int = 300, log_scale: bool = False,
                    segments: list[tuple[int, int, str]] | None = None, zero_line: bool = False) -> str:
    pad_l, pad_r, pad_t, pad_b = 60, 16, 28, 24
    colors = ["#3987e5", "#d95926", "#7fd47f", "#c3c2b7", "#fab219"]
    n = max(len(v) for v in series.values()) if series else 0
    vals = np.concatenate([np.asarray(v, dtype=float) for v in series.values()]) if series else np.array([1.0])
    vals = vals[np.isfinite(vals)]
    if log_scale:
        vals = np.log(np.maximum(vals, 1e-9))
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span

    def x(i):
        return pad_l + (width - pad_l - pad_r) * (i / max(1, n - 1))

    def y(v):
        v = math.log(max(v, 1e-9)) if log_scale else v
        return pad_t + (height - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'style="background:#1a1a19;font:11px system-ui,sans-serif">',
           f'<text x="{pad_l}" y="16" fill="#c3c2b7" font-size="13">{title}</text>']
    for a, b, label in (segments or []):
        fill = {"TRAIN": "#2a2a28", "VALIDATION": "#26302a", "OOS": "#1f2a3a", "HOLDOUT": "#3a2a1f"}.get(label, "#222")
        out.append(f'<rect x="{x(a):.1f}" y="{pad_t}" width="{max(1, x(b) - x(a)):.1f}" height="{height - pad_t - pad_b}" fill="{fill}"/>')
        out.append(f'<text x="{x(a) + 4:.1f}" y="{pad_t + 12}" fill="#8a897f" font-size="10">{label}</text>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = pad_t + (height - pad_t - pad_b) * (1 - k / 4)
        label = f"{math.exp(v):.3g}" if log_scale else f"{v:.3g}"
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#2e2e2b"/>')
        out.append(f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" fill="#8a897f" text-anchor="end">{label}</text>')
    if zero_line and lo < 0 < hi:
        out.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#8a897f" stroke-dasharray="3 3"/>')
    for k, (name, v) in enumerate(series.items()):
        pts = " ".join(f"{x(i):.1f},{y(float(val)):.1f}" for i, val in enumerate(v) if val is not None and np.isfinite(val))
        out.append(f'<polyline fill="none" stroke="{colors[k % len(colors)]}" stroke-width="{2 if k == 0 else 1.2}" points="{pts}"/>')
        out.append(f'<text x="{width - pad_r - 8}" y="{pad_t + 14 + 14 * k}" fill="{colors[k % len(colors)]}" text-anchor="end">{name}</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_plots(run_dir: Path, dev, hold, bars_per_day: int, bars_per_year: int) -> list[Path]:
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written = []
    eq_full = list(dev.sims["full"].equity)
    eq_base = list(dev.sims["baseline"].equity)
    segments = [(0, len(eq_full) - 1, "OOS")]
    if hold is not None:
        off = eq_full[-1]
        eq_full += [x * off for x in hold.sims["full"].equity[1:]]
        eq_base += [x * eq_base[-1] for x in hold.sims["baseline"].equity[1:]]
        segments.append((segments[0][1], len(eq_full) - 1, "HOLDOUT"))
    for name, log_scale in (("equity", False), ("equity_log", True)):
        p = plots / f"{name}.svg"
        p.write_text(_svg_line_chart({"fractional": eq_full, "baseline (no fractional)": eq_base},
                                     f"OOS equity ({'log' if log_scale else 'linear'} scale)", log_scale=log_scale, segments=segments))
        written.append(p)
    dd = drawdown_series(np.asarray(eq_full))
    p = plots / "underwater.svg"
    p.write_text(_svg_line_chart({"drawdown": list(dd)}, "Underwater (drawdown from peak)", height=200, segments=segments))
    written.append(p)
    rs = rolling_sharpe(np.asarray(dev.sims["full"].bar_pnl), 20 * bars_per_day, bars_per_year)
    p = plots / "rolling_sharpe.svg"
    p.write_text(_svg_line_chart({"rolling Sharpe (20 sessions)": [float(v) if np.isfinite(v) else None for v in rs]},
                                 "Rolling Sharpe", height=200, zero_line=True))
    written.append(p)
    return written


def write_run(run_dir: Path, manifest, summary: dict[str, Any], dev, hold, cfg, log_lines: list[str]) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(run_dir / "manifest.yaml")
    write_json(run_dir / "summary.json", summary)
    offsets = {v: 1.0 for v in ("full", "baseline", "production")}
    eq = equity_rows(dev, "development", offsets)
    trades = [{**t, "segment": "development"} for t in dev.sims["full"].trades]
    fills = fill_rows(dev, "full", "development")
    decisions = decision_rows(dev, "full", "development")
    if hold is not None:
        offs = {v: float(dev.sims[v].equity[-1]) for v in offsets}
        eq += equity_rows(hold, "holdout", offs)
        base_id = len(trades)
        base_fill = len(fills)
        base_row = len(decisions)
        for t in hold.sims["full"].trades:
            trades.append({**t, "trade_id": t["trade_id"] + base_id, "segment": "holdout",
                           "entry_row": t["entry_row"] + base_row, "exit_row": t.get("exit_row", 0) + base_row})
        for f in fill_rows(hold, "full", "holdout"):
            fills.append({**f, "fill_id": f["fill_id"] + base_fill, "row": f["row"] + base_row,
                          "trade_id": (f["trade_id"] + base_id) if f["trade_id"] is not None else None})
        for d in decision_rows(hold, "full", "holdout", offs["full"]):
            decisions.append({**d, "row": d["row"] + base_row})
    write_csv(run_dir / "equity.csv", eq, EQUITY_FIELDS)
    write_csv(run_dir / "trades.csv", trades, TRADE_FIELDS)
    write_csv(run_dir / "fills.csv", fills, FILL_FIELDS)
    write_csv(run_dir / "decisions.csv", decisions, DECISION_FIELDS)
    write_csv(run_dir / "retrains.csv", retrain_rows([dev] + ([hold] if hold else []), "development"), RETRAIN_FIELDS)
    models = run_dir / "models"
    models.mkdir(exist_ok=True)
    for res in [dev] + ([hold] if hold else []):
        for w in res.windows:
            for mdl, h in ((w.model, w.fitted_model_hash), (w.baseline_model, w.baseline_fitted_model_hash)):
                if mdl is not None and mdl.metadata is not None:
                    write_json(models / f"{mdl.version}.json", {**mdl.metadata.to_dict(), "fitted_model_hash": h,
                                                                "segment": res.label, "window": w.window.index})
    diag = run_dir / "diagnostics"
    for key in ("ablation", "baselines", "cost_curve", "timing", "parameter_perturbations", "d_perturbation", "bootstrap",
                "monte_carlo", "multiple_testing", "regimes", "sanity", "leakage", "reproducibility", "gates", "synthetic_ensemble"):
        if summary.get(key):
            write_json(diag / f"{key}.json", summary[key])
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    write_plots(run_dir, dev, hold, int(cfg.market.bars_per_day), int(cfg.market.bars_per_day) * int(cfg.market.trading_days_per_year))
    return {"run_dir": str(run_dir), "summary": str(run_dir / "summary.json")}


__all__ = ["write_run", "write_json", "write_csv", "decision_rows", "fill_rows", "equity_rows", "retrain_rows", "write_plots"]
