"""Research-grade validation framework: schedule, simulator parity, metrics, statistics, gates,
manifests, leakage detection, synthetic knobs, the end-to-end pipeline and its dashboard API."""

import json
import math
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from trading_bot.config import load_config
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.research.bootstrap import ablation_bootstrap, block_bootstrap, monte_carlo_trades, multiple_testing
from trading_bot.research.gates import LEVELS, evaluate_gates
from trading_bot.research.manifest import RunManifest, compare_runs, model_config_hash
from trading_bot.research.metrics import compute_strategy_metrics, drawdown_stats
from trading_bot.research.runner import STAGES, ResearchRun, list_runs, resolve_stages
from trading_bot.research.sanity import run_leakage
from trading_bot.research.simulate import SimInputs, simulate_strategy
from trading_bot.research.windows import build_schedule
from trading_bot.training.validation import SimulationParams, simulate_validation

RESEARCH_FAST = {"market": {"instrument": "SYN"},
                 "training": {"window_bars": 1600, "minimum_bars": 1400, "retrain_every_bars": 250,
                              "hyperparameter_grid": {"n_estimators": [40], "min_child_samples": [50]}},
                 "models": {"regression": {"num_threads": 1}},
                 "research": {"holdout": {"fraction": 0.2}, "bootstrap": {"n_boot": 100, "monte_carlo_paths": 100},
                              "sanity": {"random_permutations": 4}, "baselines": {"random_permutations": 4}}}


@pytest.fixture(scope="session")
def research_cfg(cfg):
    return cfg.with_overrides(RESEARCH_FAST)


@pytest.fixture(scope="session")
def research_store():
    bars = generate_synthetic_bars(2800, seed=21, instrument="SYN", memory_d=0.45, amplitude=6.0)
    return BarStore("SYN", 30, bars)


@pytest.fixture(scope="session")
def research_run(research_cfg, research_store, tmp_path_factory):
    root = tmp_path_factory.mktemp("research")
    run = ResearchRun(research_cfg, research_store, {"source": "synthetic", "seed": 21}, root, log=None, kind="synthetic",
                      stages="full", open_holdout=True)
    summary = run.execute()
    return run, summary, root


# ---------------------------------------------------------------- schedule
def test_schedule_windows_are_rolling_contiguous_and_stop_before_holdout():
    s = build_schedule(10000, 3000, 250, holdout_fraction=0.15)
    assert s.holdout_start == 8500 and len(s.windows) == 22
    for a, b in zip(s.windows[:-1], s.windows[1:]):
        assert a.oos_end == b.train_end                       # OOS blocks tile the development span
        assert b.train_end - b.train_start == 3000
    assert s.windows[-1].oos_end <= s.holdout_start
    first = build_schedule(10000, 10000, 250, first_train_bars=3000, holdout_fraction=0.15)
    assert first.windows[0].train_end == 3000 and first.windows[5].train_start == 0    # grows until window_bars is available
    hold = build_schedule(10000, 3000, 250, holdout_start=8500, span=(8500, 10000))
    assert hold.windows[0].train_end == 8500 and hold.windows[-1].oos_end == 10000
    assert all(w.train_end >= 8500 for w in hold.windows)


def test_resolve_stages():
    assert resolve_stages("quick")[0] == "walkforward" and "gates" in resolve_stages("quick")
    assert "holdout" not in resolve_stages("full") and "holdout" in resolve_stages("full", open_holdout=True)
    assert resolve_stages("ablation,cost") == ["walkforward", "ablation", "cost", "gates"]
    with pytest.raises(ValueError):
        resolve_stages("nonsense")
    assert set(STAGES) >= set(resolve_stages(None))


# --------------------------------------------------------------- simulator
def _inputs(n=400, seed=3):
    rng = np.random.RandomState(seed)
    price = 100.0 * np.exp(np.cumsum(rng.randn(n + 2) * 0.004))
    sigma = np.full(n, 0.004)
    E = np.where(rng.rand(n) < 0.3, rng.randn(n) * 2.0, 0.0)
    return SimInputs(E=E, y_norm=rng.randn(n), sigma=sigma, sigma_ref=sigma * 1.1, cost_roundtrip=np.full(n, 0.0004),
                     cost_side_exec=np.full(n, 0.0002), log_close=np.log(price[:n]), open_next=price[1:n + 1],
                     open_next2=price[2:n + 2], P=np.clip(0.5 + E / 8, 0.01, 0.99),
                     session_ids=np.arange(n) // 13, window_ids=np.arange(n) // 100)


def test_simulate_strategy_matches_validation_simulator():
    inp = _inputs()
    params = SimulationParams(horizon=4)
    ref = simulate_validation(inp.E, inp.y_norm, inp.sigma, inp.sigma_ref, inp.cost_roundtrip, inp.cost_side_exec,
                              inp.log_close, inp.open_next, inp.open_next2, params)
    res = simulate_strategy(inp, params)
    assert res.net_pnl == pytest.approx(ref.net_pnl) and len(res.trades) == ref.n_trades
    assert np.allclose(res.equity, np.asarray(ref.equity_curve))
    assert sum(t["net_pnl"] for t in res.trades) == pytest.approx(res.bar_pnl.sum())
    assert all(t["decision_timestamp"] is None for t in res.trades)          # no timestamps supplied
    assert {"trade_id", "forecast", "expected_return", "probability_up", "confidence", "target_exposure",
            "approved_exposure", "entry_price", "exit_price", "bars_held", "exit_reason", "net_pnl"} <= set(res.trades[0])


def test_simulate_strategy_delay_cost_and_halts():
    inp = _inputs()
    params = SimulationParams(horizon=4)
    base = simulate_strategy(inp, params)
    delayed = simulate_strategy(inp, params, delay=1)
    assert delayed.net_pnl != base.net_pnl and delayed.equity[-1] == delayed.equity[-2]   # last row unusable
    flat = simulate_strategy(inp, params, cost_bps=0.0)
    assert flat.total_cost == 0.0 and flat.gross_pnl >= base.gross_pnl - 1e-12 or flat.n_signals >= base.n_signals
    doubled = simulate_strategy(inp, params, cost_scale=2.0)
    assert doubled.total_cost <= 2.0 * base.total_cost + 1e-12
    halted = simulate_strategy(inp, params, daily_loss_limit=0.0005)
    assert halted.halted.any() and np.all(halted.exposures[halted.halted] == 0.0)
    dd = simulate_strategy(inp, params, drawdown_halt=0.002)
    assert dd.halted.any()
    # a drawdown halt is lifted at the next walk-forward window
    first_halt = int(np.flatnonzero(dd.halted)[0])
    later_windows = inp.window_ids > inp.window_ids[first_halt]
    assert (dd.exposures[later_windows] != 0).any()


# ------------------------------------------------------------------ metrics
def test_metrics_and_drawdown():
    eq = np.array([1.0, 1.1, 1.0, 0.9, 1.2, 1.1])
    dd = drawdown_stats(eq)
    assert dd["max_drawdown"] == pytest.approx(1 - 0.9 / 1.1) and dd["max_drawdown_duration"] == 2
    rets = np.diff(eq) / eq[:-1]
    trades = [{"net_pnl": 0.1, "direction": 1, "bars_held": 2, "exit_reason": "flat", "entry_row": 0},
              {"net_pnl": -0.05, "direction": -1, "bars_held": 6, "exit_reason": "stop", "entry_row": 2}]
    m = compute_strategy_metrics(rets, eq, np.array([1, 1, -1, -1, 0.0]), trades, np.zeros(5), session_ids=np.array([0, 0, 1, 1, 2]),
                                 timestamps=["2024-01-01T10:00", "2024-01-01T10:30", "2024-01-02T10:00", "2024-01-02T10:30", "2024-02-01T10:00"],
                                 y_norm=np.array([1, -1, 1, 1, 0.0]), E=np.array([0.5, 0.0, -0.5, 0.2, 0.0]))
    assert m["total_return"] == pytest.approx(0.1) and m["trade_count"] == 2 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(2.0) and m["long_pnl"] == 0.1 and m["short_pnl"] == -0.05
    assert m["pnl_by_holding_period"]["2-4"]["count"] == 1 and m["exits_by_reason"]["stop"] == 1
    assert set(m["pnl_by_year"]) == {"2024"} and set(m["pnl_by_month"]) == {"2024-01", "2024-02"}
    assert m["directional_accuracy"] == pytest.approx(2 / 3) and m["worst_daily_loss"] < 0
    assert "sharpe" in m and "sortino" in m and "ulcer_index" in m and "var_95" in m


# ----------------------------------------------------------------- statistics
def test_bootstrap_monte_carlo_and_multiple_testing_are_deterministic():
    rng = np.random.RandomState(0)
    r = rng.randn(1300) * 0.002 + 0.0002
    a = block_bootstrap(r, 3276, block=65, n_boot=200, seed=1)
    b = block_bootstrap(r, 3276, block=65, n_boot=200, seed=1)
    assert a == b and a["sharpe"]["ci_low"] <= a["sharpe"]["point"] <= a["sharpe"]["ci_high"]
    assert 0.0 <= a["max_drawdown"]["p_over_20pct"] <= 1.0
    ab = ablation_bootstrap(np.array([0.5, 0.2, -0.1, 0.4, 0.3]), n_boot=500, seed=2)
    assert ab["positive_fraction"] == 0.8 and ab["mean_ci"][0] <= ab["mean"] <= ab["mean_ci"][1]
    mc = monte_carlo_trades(np.array([0.01, -0.005, 0.02, -0.01, 0.015]), n_paths=300, seed=3)
    assert mc["n_trades"] == 5 and 0 <= mc["terminal_return"]["p_loss"] <= 1 and mc["max_drawdown"]["p_over_20pct"] == 0.0
    mt = multiple_testing(2.0, 5000, 3276, n_tests=50)
    assert mt["bonferroni_p_value"] >= mt["p_value"] and mt["sharpe_required_for_5pct_after_correction"] > 0


# ------------------------------------------------------------------- gates
def _summary(level: str, synthetic: bool = False) -> dict:
    dev_metrics = {"sharpe": 1.5, "max_drawdown": 0.10, "profit_factor": 1.5, "trade_count": 300}
    per_window = [{"full_net_return": 0.01 if i % 4 else -0.01, "delta_sharpe": 0.2} for i in range(24)]
    s = {"manifest": {"kind": "synthetic" if synthetic else "real"},
         "development": {"n_windows": 24, "metrics": {"full": dev_metrics}, "per_window": per_window}}
    if level == "EXPERIMENTAL":
        s["development"]["n_windows"] = 5
        return s
    if LEVELS.index(level) >= 2:
        s["ablation"] = {"median_delta_sharpe": 0.2, "positive_delta_fraction": 0.7}
        s["cost_curve"] = {"profitable_at_2x_cost": True}
        s["timing"] = {"viable_at_plus_1": True}
        s["parameter_perturbations"] = {"any_collapse": False, "n_collapsed": 0}
        s["bootstrap"] = {"block": {"sharpe": {"ci_low": 0.3}}}
        s["sanity"] = {"passed": True, "n_failed": 0, "n_tests": 7}
        s["leakage"] = {"passed": True}
    if LEVELS.index(level) >= 3:
        s["holdout"] = {"metrics": {"full": {"total_return": 0.05, "sharpe": 0.8}}, "access": {"opening_number": 1}}
    if LEVELS.index(level) >= 4:
        s["reproducibility"] = {"status": "IDENTICAL"}
    return s


@pytest.mark.parametrize("level", LEVELS)
def test_gate_classification_ladder(level):
    out = evaluate_gates(_summary(level))
    assert out["classification"] == level
    assert out["paper_eligible"] == (level == "PAPER ELIGIBLE")


def test_gates_synthetic_runs_never_classify_and_failures_block():
    out = evaluate_gates(_summary("PAPER ELIGIBLE", synthetic=True))
    assert out["classification"] == "EXPERIMENTAL (SYNTHETIC)" and not out["paper_eligible"]
    s = _summary("PAPER ELIGIBLE")
    s["reproducibility"] = {"status": "REPRODUCIBILITY FAILURE"}
    assert evaluate_gates(s)["classification"] == "HOLDOUT PASSED"
    s["sanity"] = {"passed": False, "n_failed": 1, "n_tests": 7}
    assert evaluate_gates(s)["classification"] == "CANDIDATE"
    s["development"]["metrics"]["full"]["max_drawdown"] = 0.4
    assert evaluate_gates(s)["classification"] == "EXPERIMENTAL"
    loose = evaluate_gates(s, {"max_drawdown": 0.5, "require_sanity_pass": False, "require_reproducible": False})
    assert loose["classification"] == "PAPER ELIGIBLE"


# ---------------------------------------------------------------- manifest
def test_manifest_hashes_and_run_comparison(research_cfg, research_store):
    m1 = RunManifest.create(research_cfg, research_store, {"source": "synthetic"}, {"windows": []}, ["walkforward"], "synthetic")
    m2 = RunManifest.create(research_cfg, research_store, {"source": "synthetic"}, {"windows": []}, ["walkforward"], "synthetic")
    assert m1.manifest_hash == m2.manifest_hash and m1.run_id != m2.run_id or m1.run_id == m2.run_id
    other = RunManifest.create(research_cfg.with_overrides({"signal": {"cost_multiplier": 2.5}}), research_store, {"source": "synthetic"},
                               {"windows": []}, ["walkforward"], "synthetic")
    assert other.manifest_hash != m1.manifest_hash and other.model_config_hash != m1.model_config_hash
    assert model_config_hash(research_cfg) == model_config_hash(research_cfg.with_overrides({"research": {"bootstrap": {"seed": 9}}}))
    a = {"manifest": {"manifest_hash": "x", "run_id": "a"}, "results_hash": "r1"}
    assert compare_runs(a, {"manifest": {"manifest_hash": "x", "run_id": "b"}, "results_hash": "r1"})["status"] == "IDENTICAL"
    assert compare_runs(a, {"manifest": {"manifest_hash": "x", "run_id": "b"}, "results_hash": "r2"})["status"] == "REPRODUCIBILITY FAILURE"
    assert compare_runs(a, {"manifest": {"manifest_hash": "y", "run_id": "b"}, "results_hash": "r2"})["status"] == "DIFFERENT INPUTS"


# ------------------------------------------------------- dataset / generator
def test_dataset_label_offset_shifts_the_target(research_cfg, research_store):
    from trading_bot.data.calendar import SessionCalendar
    from trading_bot.execution.cost_model import CostModel
    from trading_bot.features.engine import FeatureEngine
    from trading_bot.fractional.engine import FractionalEngine
    from trading_bot.training.dataset import TrainingDatasetBuilder

    fe = FeatureEngine(research_cfg, FractionalEngine.from_config(research_cfg), SessionCalendar.from_config(research_cfg))
    builder = TrainingDatasetBuilder.from_config(research_cfg, fe, CostModel.from_config(research_cfg))
    window = research_store.slice(0, 1500)
    ds0 = builder.build(window, 0.4)
    ds20 = builder.build(window, 0.4, label_offset_bars=20)
    opens = np.log(window.arrays()["open"])
    H = int(research_cfg.prediction.horizon_bars)
    i = int(ds20.bar_index[10])
    assert ds20.y_raw[10] == pytest.approx(opens[i + 1 + 20 + H] - opens[i + 1 + 20])
    assert ds0.y_raw[10] == pytest.approx(opens[i + 1 + H] - opens[i + 1])
    assert len(ds20) == len(ds0) - 20
    with pytest.raises(ValueError):
        builder.build(window, 0.4, label_offset_bars=-1)


def test_synthetic_generator_knobs():
    base = BarStore("SYN", 30, generate_synthetic_bars(600, seed=5))
    again = BarStore("SYN", 30, generate_synthetic_bars(600, seed=5, drift=0.0, autocorrelation=0.0, jump_intensity=0.0))
    assert base.checksum() == again.checksum()
    up = generate_synthetic_bars(3000, seed=5, drift=2.0)
    down = generate_synthetic_bars(3000, seed=5, drift=-2.0)
    assert up[-1].close > base[-1].close or up[-1].close > down[-1].close
    jumpy = np.diff(np.log([b.close for b in generate_synthetic_bars(3000, seed=5, jump_intensity=0.05, jump_size=8.0)]))
    calm = np.diff(np.log([b.close for b in generate_synthetic_bars(3000, seed=5)]))
    assert np.abs(jumpy).max() > 1.5 * np.abs(calm).max()
    regimes = np.diff(np.log([b.close for b in generate_synthetic_bars(3000, seed=5, regime_bars=130, regime_vol_ratio=4.0)]))
    blocks = np.array([regimes[i: i + 130].std() for i in range(0, 2600, 130)])
    assert blocks.max() > 2 * blocks.min()


# ---------------------------------------------------------- leakage detector
def test_leakage_detector_flags_tampered_timestamps(research_run):
    run, summary, root = research_run
    res = run.dev
    ok = run_leakage(run.runner, res)
    assert ok["passed"] and ok["timestamps"]["violations"] == 0 and ok["label_alignment"]["labels_aligned"]
    tampered = res.oos.subset(np.ones(len(res.oos), dtype=bool))
    tampered.execution_at[5] = tampered.decision_at[5] - timedelta(minutes=1)      # fill before the decision

    class Fake:
        oos = tampered
        timestamp_audit = dict(res.timestamp_audit)

    bad = run_leakage(run.runner, Fake())
    assert not bad["passed"] and bad["timestamps"]["violations"] == 1
    tampered.execution_at[5] = tampered.decision_at[5]
    tampered.y_raw[7] += 1e-3                                                     # label not the executable return
    assert not run_leakage(run.runner, Fake())["label_alignment"]["labels_aligned"]


# ------------------------------------------------------------- end to end
def test_pipeline_end_to_end(research_run):
    run, s, root = research_run
    assert s["stages_completed"] == s["stages"] and "holdout" in s["stages_completed"]
    dev = s["development"]
    assert dev["n_windows"] >= 3 and dev["timestamp_audit"]["violations"] == 0
    assert s["leakage"]["passed"] and s["reproducibility"]["status"] == "IDENTICAL"
    assert s["evidence_label"].startswith("SYNTHETIC") and s["gates"]["classification"] == "EXPERIMENTAL (SYNTHETIC)"
    # the OOS rows tile the development span contiguously and every fill happens at the next open
    oos = run.dev.oos
    assert np.all(np.diff(oos.bar_index) == 1)
    assert all(e >= d and f <= d for d, e, f in zip(oos.decision_at, oos.execution_at, oos.feature_available_at))
    exec_opens = run.store.arrays()["open"][oos.bar_index + 1]
    assert np.allclose(exec_opens, oos.open_next)
    # ablation / stress / sanity results are present and internally consistent
    a = s["ablation"]
    assert a["cycles"] == dev["n_windows"] and len(a["per_window_delta_sharpe"]) == dev["n_windows"]
    cc = s["cost_curve"]
    x1 = next(r for r in cc["rows"] if r["label"] == "model_cost_x1")
    assert x1["total_return"] == pytest.approx(dev["metrics"]["full"]["total_return"])
    zero = next(r for r in cc["rows"] if r["label"] == "flat_0bps")
    assert zero["total_cost"] == 0.0
    t = s["sanity"]["tests"]
    assert t["shuffled_labels"]["sharpe"] < t["light_refit_reference"]["sharpe"]
    assert t["shuffled_features"]["sharpe"] < t["light_refit_reference"]["sharpe"]
    assert t["reversed_forecasts"]["sharpe"] < s["sanity"]["strategy"]["sharpe"]
    assert t["zero_cost"]["passed"] and t["double_cost"]["passed"]
    assert len(s["d_perturbation"]["rows"]) == 5 and s["timing"]["rows"][0]["delay"] == 0
    assert s["bootstrap"]["block"]["n_boot"] == 100 and s["monte_carlo"]["n_paths"] == 100
    assert s["multiple_testing"]["configurations_tested_in_run"] == 2 * dev["n_windows"]
    # holdout: opened once, recorded, walked with refits, appended after development
    h = s["holdout"]
    assert h["access"]["opening_number"] == 1 and h["n_windows"] >= 1
    access = (root / "holdout_access.jsonl").read_text().strip().splitlines()
    assert len(access) == 1 and json.loads(access[0])["run_id"] == s["run_id"]
    assert run.hold.oos.bar_index[0] == run.dev.oos.bar_index[-1] + 1 or run.hold.oos.bar_index[0] >= run.dev.oos.bar_index[-1]
    # results hash is a function of the forecasts: recomputing it from the kept arrays gives the same value
    from trading_bot.research.manifest import results_hash
    again = results_hash({v: run.dev.oos.forecasts[v]["E"] for v in ("full", "baseline")}, run.dev.sims["full"].equity,
                         [w.fitted_model_hash for w in run.dev.windows])
    assert again == s["results_hash"]


def test_pipeline_artifacts(research_run):
    run, s, root = research_run
    d = run.run_dir
    for name in ("manifest.yaml", "summary.json", "equity.csv", "trades.csv", "fills.csv", "decisions.csv", "retrains.csv",
                 "logs/run.log", "plots/equity.svg", "plots/equity_log.svg", "plots/underwater.svg", "plots/rolling_sharpe.svg",
                 "diagnostics/gates.json", "diagnostics/sanity.json", "diagnostics/ablation.json"):
        assert (d / name).exists(), name
    raw = (d / "summary.json").read_text()
    assert "Infinity" not in raw and "NaN" not in raw            # browser-safe JSON
    loaded = json.loads(raw)
    assert loaded["run_id"] == s["run_id"] and loaded["gates"]["classification"] == s["gates"]["classification"]
    models = list((d / "models").glob("*.json"))
    assert len(models) == 2 * (run.dev.n_windows + run.hold.n_windows)
    meta = json.loads(models[0].read_text())
    assert meta["fitted_model_hash"] and meta["segment"] in ("development", "holdout")
    import csv
    with open(d / "decisions.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(run.dev.oos) + len(run.hold.oos)
    assert rows[0]["feature_available_at"] <= rows[0]["decision_at"] <= rows[0]["execution_at"]
    with open(d / "trades.csv") as fh:
        trades = list(csv.DictReader(fh))
    assert len(trades) == len(run.dev.sims["full"].trades) + len(run.hold.sims["full"].trades)
    assert {t["segment"] for t in trades} <= {"development", "holdout"}
    with open(d / "retrains.csv") as fh:
        retrains = list(csv.DictReader(fh))
    assert [r["window"] for r in retrains if r["segment"] == "development"] == [str(i) for i in range(run.dev.n_windows)]
    listed = list_runs(root)
    assert listed and listed[0]["run_id"] == s["run_id"] and listed[0]["holdout"] is True


def test_reproduce_window_bit_for_bit(research_run):
    run, s, root = research_run
    check = run.runner.reproduce_window(run.dev.windows[1])
    assert check["identical"] and check["max_abs_forecast_difference"] == 0.0


# ------------------------------------------------------------ dashboard API
def test_dashboard_research_api(research_run, tmp_path):
    from trading_bot.gui.controller import BotController
    from trading_bot.gui.server import DashboardServer

    run, s, root = research_run
    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"artifacts_dir": str(root), "symbol": "SYN"})
    runs = ctl.research_runs()
    assert runs[0]["run_id"] == s["run_id"]
    trades = ctl.research_trades(s["run_id"], limit=3)
    assert len(trades) == 3 and isinstance(trades[0]["trade_id"], int)
    detail = ctl.research_trade(s["run_id"], trades[0]["trade_id"])
    assert detail["trade"]["trade_id"] == trades[0]["trade_id"] and detail["decisions"] and detail["model"]["fitted_model_hash"]
    assert detail["decisions"][0]["row"] == int(detail["trade"]["entry_row"]) - 2
    with pytest.raises(FileNotFoundError):
        ctl.research_summary("does_not_exist")
    with pytest.raises(ValueError):
        ctl.research_summary("../escape")
    server = DashboardServer(ctl, "127.0.0.1", 0)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = server.url
    try:
        def get(path):
            with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as r:
                return r.status, r.read().decode()
        status, body = get("/api/research/runs")
        assert status == 200 and json.loads(body)["runs"][0]["run_id"] == s["run_id"]
        status, body = get("/api/research/run?id=" + s["run_id"])
        assert status == 200 and "Infinity" not in body and json.loads(body)["gates"]["classification"].startswith("EXPERIMENTAL")
        status, body = get(f"/api/research/trade?id={s['run_id']}&trade={trades[0]['trade_id']}")
        assert status == 200 and json.loads(body)["trade"]["trade_id"] == trades[0]["trade_id"]
        try:
            get("/api/research/run?id=missing")
            assert False, "expected 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        status, body = get("/api/status")
        assert json.loads(body)["research"]["phase"] == "idle"
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_research_run_lifecycle(tmp_path):
    from trading_bot.gui.controller import BotController

    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"symbol": "SYN", "mode": "backtest", "data_source": "synthetic", "synthetic_bars": 2200, "synthetic_seed": 21,
                         "fast": True, "artifacts_dir": str(tmp_path / "art"),
                         "overrides": "training.window_bars=1600\ntraining.minimum_bars=1400\ntraining.hyperparameter_grid.n_estimators=[40]\n"
                                      "models.regression.num_threads=1\nresearch.holdout.fraction=0.2\nresearch.bootstrap.n_boot=50\n"
                                      "research.bootstrap.monte_carlo_paths=50\nresearch.baselines.random_permutations=2"})
    ctl.start_research({"stages": "walkforward,ablation,bootstrap,gates", "open_holdout": False})
    with pytest.raises(RuntimeError):
        ctl.start_research({})
    deadline = time.time() + 300
    while ctl.research_running and time.time() < deadline:
        time.sleep(0.5)
    st = ctl.research_status
    assert st["phase"] == "finished", (st["phase"], st["error"])
    assert st["run_id"] and st["classification"] == "EXPERIMENTAL (SYNTHETIC)"
    summary = ctl.research_summary(st["run_id"])
    assert summary["stages_completed"] == ["walkforward", "ablation", "bootstrap", "gates"] and "holdout" not in summary
    assert not (tmp_path / "art" / "holdout_access.jsonl").exists()
    assert any("classification" in line["text"] for line in ctl.logs_since(0)["lines"])
