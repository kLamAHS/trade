"""Market-state, microstructure and execution-intelligence update: synthetic market, event bars, feature
families (causality first), decision policy, execution model, trainer / bot integration, research stages."""

import json
import math
import time
from datetime import timedelta

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.config import load_config
from trading_bot.data.bars import BarBuilder, build_event_bars, empirical_bars_per_session, events_from_bars
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.representation import apply_representation, parse_candidate, representation_spec
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import REGIME_NAMES, generate_synthetic_bars, generate_synthetic_market
from trading_bot.data.validator import DataValidator
from trading_bot.execution.cost_model import CostModel
from trading_bot.execution.estimator import EXEC_INPUT_NAMES, ExecutionEstimator, ExecutionModel, exec_inputs_from_columns
from trading_bot.features.components import FeatureFamilyConfig, FittedComponents
from trading_bot.features.cross_asset import causal_asof_join
from trading_bot.features.engine import FeatureEngine
from trading_bot.features.hmm import GaussianHMM, hmm_observations, regime_feature_arrays
from trading_bot.features.kalman import fit_kalman, kalman_filter, rts_smooth
from trading_bot.features.ofi import ofi_events
from trading_bot.features.schema import build_schema
from trading_bot.features.toxicity import vpin_feature_arrays
from trading_bot.fractional.engine import FractionalEngine
from trading_bot.research.runner import ResearchRun
from trading_bot.research.simulate import SimInputs, simulate_strategy
from trading_bot.strategy.policy import DecisionPolicy, RegimeAdjustment
from trading_bot.strategy.sizing import confidence_from_edge, direction_from_edge
from trading_bot.training.validation import SimulationParams, simulate_validation
from trading_bot.types import Bar, ExecutionEstimate

NY = SessionCalendar()
FAST = {"market": {"instrument": "SPY"},
        "training": {"window_bars": 1600, "minimum_bars": 1400, "retrain_every_bars": 250,
                     "hyperparameter_grid": {"n_estimators": [40], "min_child_samples": [50]},
                     "acceptance": {"min_accuracy": 0.0, "min_correlation": -1.0, "min_net_pnl": -1.0, "min_profit_factor": 0.0,
                                    "max_drawdown": 1.0, "min_folds_beating_baseline": 0, "require_holdout_edge": False}},
        "models": {"regression": {"num_threads": 1}}}
FAMILIES = {"features": {"regime": {"enabled": True, "states": 3}, "ofi": {"enabled": True}, "kalman": {"enabled": True},
                         "cross_asset": {"enabled": True, "symbols": ["QQQ"], "max_staleness_seconds": 1800},
                         "vpin": {"enabled": True, "min_classification_confidence": 0.05}},
            "risk": {"regime_adjustment": {"enabled": True}}}


@pytest.fixture(scope="session")
def ms_cfg():
    return load_config(overrides=FAST).with_overrides(FAMILIES)


@pytest.fixture(scope="session")
def market(ms_cfg):
    return generate_synthetic_market(2400, seed=21, calendar=SessionCalendar.from_config(ms_cfg), symbols=("SPY", "QQQ"),
                                     amplitude=6.0, memory_d=0.45, lead_lag={"leader": "QQQ", "follower": "SPY", "beta": 0.4},
                                     regime_weights=(0.35, 0.15, 0.15, 0.1, 0.25))


@pytest.fixture(scope="session")
def stores(market):
    return BarStore("SPY", 30, market.bars["SPY"]), {"QQQ": BarStore("QQQ", 30, market.bars["QQQ"])}


@pytest.fixture(scope="session")
def engine(ms_cfg, stores):
    store, ctx = stores
    fe = FeatureEngine(ms_cfg, FractionalEngine.from_config(ms_cfg), SessionCalendar.from_config(ms_cfg))
    fe.set_components(fe.fit_components(store.slice(0, 1600)))
    return fe


@pytest.fixture(scope="session")
def family_bot(ms_cfg, market, tmp_path_factory):
    bot = TradingBot(ms_cfg, run_id="ms_bot", artifacts_dir=tmp_path_factory.mktemp("ms_bot"), log=None)
    for b_spy, b_qqq in zip(market.bars["SPY"], market.bars["QQQ"]):
        bot.on_context_bar(b_qqq)
        bot.on_bar(b_spy)
    summary = bot.finalize()
    return bot, summary


@pytest.fixture(scope="session")
def family_research(ms_cfg, stores, tmp_path_factory):
    store, ctx = stores
    cfg = ms_cfg.with_overrides({"research": {"holdout": {"fraction": 0.2}, "bootstrap": {"n_boot": 100, "monte_carlo_paths": 100},
                                              "families": {"n_boot": 200}}})
    run = ResearchRun(cfg, store, {"source": "synthetic"}, tmp_path_factory.mktemp("ms_research"), log=None, kind="synthetic",
                      stages="walkforward,ablation,execution_stress,families,leakage,gates", context=ctx)
    return run, run.execute()


# ------------------------------------------------------------ synthetic market
def test_synthetic_market_is_deterministic_hostile_and_informative(market):
    m2 = generate_synthetic_market(2400, seed=21, calendar=NY, symbols=("SPY", "QQQ"), amplitude=6.0, memory_d=0.45,
                                   lead_lag={"leader": "QQQ", "follower": "SPY", "beta": 0.4}, regime_weights=(0.35, 0.15, 0.15, 0.1, 0.25))
    assert BarStore("SPY", 30, market.bars["SPY"]).checksum() == BarStore("SPY", 30, m2.bars["SPY"]).checksum()
    t = market.truth
    assert set(np.unique(t["regime_name"])) <= set(REGIME_NAMES) and (t["regime_name"] == "stress").any()
    assert t["jump"].sum() >= 1 and t["gap"].sum() >= 1 and set(np.unique(t["correlation"])) == {0.4, 0.9}
    assert (t["spread_mult"][t["regime_name"] == "stress"] == 3.0).all()          # liquidity co-moves with stress
    a = BarStore("SPY", 30, market.bars["SPY"]).arrays()
    r = np.diff(np.log(a["close"]))
    imb = (a["bid_size"] - a["ask_size"]) / (a["bid_size"] + a["ask_size"])
    assert np.corrcoef(imb[:-1], r)[0, 1] > 0.15                                  # quotes carry information about the next bar
    rq = np.diff(np.log(BarStore("QQQ", 30, market.bars["QQQ"]).arrays()["close"]))
    assert np.corrcoef(rq[:-1], r[1:])[0, 1] > 0.1 > abs(np.corrcoef(r[:-1], rq[1:])[0, 1]) - 0.05   # QQQ leads SPY, not the reverse


def test_bar_store_persists_quote_sizes_and_event_fields(tmp_path, market):
    store = BarStore("SPY", 30, market.bars["SPY"][:200])
    assert store.data_tier() == "B" and store.has_quote_sizes()
    store.save(tmp_path / "b.csv")
    back = BarStore.load(tmp_path / "b.csv", "SPY", 30)
    assert back.checksum() == store.checksum() and back[3].bid_size == store[3].bid_size and back[0].bar_kind == "time"
    assert BarStore("SYN", 30, generate_synthetic_bars(100, seed=1)).data_tier() == "A"


# ------------------------------------------------------------- event bars
def test_event_bar_builder_contract_and_session_boundaries():
    fine = generate_synthetic_bars(1300, seed=3, calendar=NY)
    for kind, target in (("volume", 3.0 * np.median([b.volume for b in fine])), ("dollar", 1.2e9), ("tick", 3)):
        bars = build_event_bars(fine, kind, target, NY)
        assert bars and all(b.bar_kind == kind and b.end_time is not None and b.close_time >= b.timestamp for b in bars)
        assert all(bars[i + 1].timestamp >= bars[i].close_time for i in range(len(bars) - 1))
        assert all(NY.session_date(b.timestamp) == NY.session_date(b.close_time - timedelta(seconds=1)) for b in bars)  # never straddle
        assert all(b.available_at >= b.close_time and b.vwap > 0 and b.dollar_volume > 0 for b in bars)
        v = DataValidator(NY)
        assert all(v.validate(b).ok for b in bars)
        assert empirical_bars_per_session(bars, NY) >= 1
    vol = build_event_bars(fine, "volume", 3.0 * np.median([b.volume for b in fine]), NY)
    assert any("partial" in b.bar_id for b in vol)                                # session-end flush is flagged
    with pytest.raises(ValueError):
        BarBuilder("candle", 1)


def test_representation_config_and_candidates():
    assert parse_candidate("dollar_250m") == {"bar_type": "dollar", "target": 250e6, "name": "dollar_250m"}
    assert parse_candidate("volume_500k")["target"] == 500e3 and parse_candidate("time_30m")["bar_type"] == "time"
    cfg = load_config(overrides={"market_representation": {"bar_type": "volume", "target_shares": 1e6}})
    fine = generate_synthetic_bars(800, seed=2)
    bars, cfg2 = apply_representation(fine, cfg, NY)
    assert bars[0].bar_kind == "volume" and cfg2.market.bars_per_day != cfg.market.bars_per_day
    assert representation_spec(load_config())["bar_type"] == "time"
    with pytest.raises(ValueError):
        representation_spec(load_config(overrides={"market_representation": {"bar_type": "dollar", "target_dollar_volume": None}}))


# ------------------------------------------------------------ feature families
def test_hmm_filter_is_causal_and_smoother_is_not(stores):
    store, _ = stores
    a = store.arrays()
    obs = hmm_observations(np.log(a["close"]), a["volume"], None)
    hmm = GaussianHMM(3, seed=0).fit(obs[:1500])
    assert hmm.converged or hmm.n_fit > 0
    full = hmm.filter(obs)
    assert np.allclose(hmm.filter(obs[:900]), full[:900])                          # P(S_t | X_0..X_t): prefix invariant
    sm = hmm.smooth(obs)
    assert not np.allclose(hmm.smooth(obs[:900]), sm[:900], atol=1e-6)           # P(S_t | X_0..X_T) uses the future
    assert np.allclose(full.sum(axis=1), 1.0) and hmm.means[0, 2] <= hmm.means[-1, 2]   # states ordered by volatility
    names = [m["name"] for m in hmm.state_meta]
    assert names[0] == "calm" and names[-1] == "stress"
    feats = regime_feature_arrays(full, hmm.transition)
    assert set(feats) >= {"regime_p_0", "regime_entropy", "regime_age", "regime_transition_prob", "regime_most_likely", "regime_stress_p"}
    assert feats["regime_age"].max() > 10


def test_kalman_filter_causal_and_fitted(stores):
    store, _ = stores
    lc = np.log(store.arrays()["close"])
    p = fit_kalman(lc[:1500])
    assert p.q_velocity > 0 and p.r > 0 and math.isfinite(p.log_likelihood)
    f = kalman_filter(lc, p)
    g = kalman_filter(lc[:1000], p)
    assert np.allclose(f["level"][:1000], g["level"]) and np.allclose(f["innovation"][1:1000], g["innovation"][1:])
    s = rts_smooth(lc[:1000], p)
    assert not np.allclose(s, f["level"][:1000], atol=1e-8)                        # the smoother differs: never a feature


def test_ofi_formula_and_tier_requirement():
    bid = np.array([100.0, 100.0, 100.1, 100.1])
    ask = np.array([100.2, 100.2, 100.2, 100.3])
    bs = np.array([500.0, 700.0, 300.0, 300.0])
    asz = np.array([400.0, 400.0, 600.0, 200.0])
    e = ofi_events(bid, ask, bs, asz)
    assert np.isnan(e[0]) and e[1] == pytest.approx(700 - 500)                     # same prices: size deltas
    assert e[2] == pytest.approx(300 - 600 + 400)                                   # bid up: +q_b,n ; ask same: -q_a,n + q_a,n-1
    assert e[3] == pytest.approx(300 - 300 + 600)                                   # ask up: + q_a,n-1, bid same: +q_b,n - q_b,n-1
    cfg = load_config(overrides={"market": {"instrument": "SYN"}, "features": {"ofi": {"enabled": True}}})
    fe = FeatureEngine(cfg, FractionalEngine.from_config(cfg), NY)
    tier_a = BarStore("SYN", 30, generate_synthetic_bars(1200, seed=4))
    assert "OFI" in fe.schema.enabled_families and "OFI" not in fe.available_families(tier_a)
    fm = fe.compute_matrix(tier_a)
    assert np.isnan(fm.column("ofi_zscore")).all() and "OFI" not in fm.families_available     # never manufactured from Tier A


def test_causal_asof_join_never_takes_the_future_and_flags_staleness():
    from datetime import datetime, timezone
    t0 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    targets = [t0 + timedelta(minutes=30 * k) for k in range(4)]
    src = [t0 - timedelta(minutes=5), t0 + timedelta(minutes=29), t0 + timedelta(minutes=31), t0 + timedelta(minutes=200)]
    vals, age = causal_asof_join(targets, src, np.array([1.0, 2.0, 3.0, 4.0]))
    assert vals.tolist() == [1.0, 2.0, 3.0, 3.0]                                   # 31 min > 30 min target: not yet; never the next one
    vals2, _ = causal_asof_join(targets, src, np.array([1.0, 2.0, 3.0, 4.0]), max_staleness_seconds=60)
    assert np.isnan(vals2[3]) and vals2[1] == 2.0


def test_feature_families_are_causal_and_available_by_data(engine, stores):
    store, ctx = stores
    fm = engine.compute_matrix(store, context=ctx)
    assert set(fm.families_available) == {"PRICE", "VOLATILITY", "VOLUME", "FRACTIONAL", "REGIME", "OFI", "KALMAN", "CROSS_ASSET", "TOXICITY"}
    cut = 1900
    fm_p = engine.compute_matrix(store.slice(0, cut), context={s: c.slice(0, cut) for s, c in ctx.items()})
    a, b = fm.values[:cut], fm_p.values
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    assert np.nanmax(diff) < 1e-9                                                   # prefix invariance for every column
    fm_nc = engine.compute_matrix(store)                                            # missing context instrument
    assert "CROSS_ASSET" not in fm_nc.families_available and np.isnan(fm_nc.column("xa_qqq_rel_return_1")).all()
    valid = fm.valid_mask(engine.schema.names_for(fm.families_available))
    assert valid[engine.required_history + 10:].all()
    fv = engine.compute_latest(store, context=ctx)
    row = fm.row(len(store) - 1)
    for n in engine.schema.model_names:
        if math.isfinite(row[n]) and math.isfinite(fv.values[n]):
            assert abs(row[n] - fv.values[n]) < 1e-4, n                             # recursive filters converge within the warm-up


def test_schema_families_and_toggles():
    s = build_schema(regime_states=3, ofi=True, kalman_subset="innovation", cross_symbols=("QQQ",), toxicity=True)
    assert s.version == "2.0.0" and set(s.enabled_families) >= {"REGIME", "OFI", "KALMAN", "CROSS_ASSET", "TOXICITY"}
    assert s.family_of("regime_p_1") == "REGIME" and s.family_of("kalman_innovation") == "KALMAN" and s.family_of("sigma_h") == "AUX"
    assert "kalman_velocity" not in s.model_names and "kalman_innovation_z" in s.model_names
    assert set(s.names_for(("PRICE", "OFI"))) == set(s.families["PRICE"]) | set(s.families["OFI"])
    assert "regime_p_0" not in s.without("REGIME") and s.baseline_names == tuple(n for n in s.model_names if n not in s.fractional_names)
    default = build_schema()
    assert default.enabled_families == ("PRICE", "VOLATILITY", "VOLUME", "FRACTIONAL") and len(default.model_names) == 39


def test_vpin_disabled_without_classification_confidence():
    rng = np.random.RandomState(0)
    n = 600
    r = rng.randn(n) * 1e-6
    sigma = np.full(n, 1.0)                                                          # returns tiny relative to sigma: coin-flip classification
    feats, conf = vpin_feature_arrays(r, sigma, np.full(n, 1000.0))
    assert conf < 0.05
    comp = FittedComponents(vpin_confidence=conf, vpin_enabled=conf >= 0.15)
    assert not comp.vpin_enabled


# ------------------------------------------------------------- decision policy
def test_net_edge_policy_abstains_and_regime_adjusts():
    p = DecisionPolicy(mode="net_edge", minimum_net_edge=8e-4, uncertainty_buffer=3e-4,
                       regime=RegimeAdjustment(True, 0.7, 0.4, 1.5))
    d = p.decide(0.0030, 0.0004, 0.0002, adverse=0.0003, model_uncertainty=0.0)
    assert d.trade and d.direction == 1 and d.net_edge == pytest.approx(0.0030 - 0.0004 - 0.0002 - 0.0003 - 0.0003)
    d2 = p.decide(0.0015, 0.0004, 0.0002, adverse=0.0003)                          # 0.0003 net < 0.0008 minimum
    assert not d2.trade and d2.abstain_reason == "net edge below minimum"
    d3 = p.decide(0.0030, 0.0004, 0.0002, adverse=0.0003, stress_p=0.9)
    assert d3.exposure_cap_multiplier == 0.4 and d3.required_edge == pytest.approx(0.0012)
    assert not p.decide(-0.0018, 0.0004, 0.0002, stress_p=0.9).trade and p.decide(-0.0018, 0.0004, 0.0002).trade
    assert p.decide(float("nan"), 0.0004, 0.0002).abstain_reason == "non-finite forecast"


def test_legacy_policy_matches_the_original_sizing_rule():
    p = DecisionPolicy(mode="legacy", cost_multiplier=3.0, confidence_cost_multiplier=6.0)
    for er, cost in ((0.0031, 0.001), (-0.0031, 0.001), (0.003, 0.001), (0.012, 0.001), (0.0005, 0.001)):
        d = p.decide(er, cost, 0.0)
        assert d.direction == direction_from_edge(er, cost)
        if d.direction:
            assert d.confidence == pytest.approx(confidence_from_edge(er, cost))


def test_execution_model_target_and_estimator():
    rng = np.random.RandomState(1)
    n = 800
    X = np.zeros((n, len(EXEC_INPUT_NAMES)))
    X[:, EXEC_INPUT_NAMES.index("spread_rel")] = rng.uniform(1e-4, 5e-4, n)
    X[:, EXEC_INPUT_NAMES.index("range_rel")] = rng.uniform(0.001, 0.01, n)
    abs_er = rng.uniform(0.001, 0.004, n)
    conf = rng.uniform(0.2, 1.0, n)
    direction = np.where(rng.rand(n) < 0.5, 1.0, -1.0)
    # adverse first-bar move grows with the spread and the range: the model must learn it
    adverse_true = 4.0 * X[:, 0] + 0.2 * X[:, 1] + rng.randn(n) * 1e-4
    r_fill = direction * (abs_er / 4 - adverse_true)
    target = ExecutionModel.target(abs_er, direction, r_fill, 4)
    assert np.allclose(target, adverse_true)
    m = ExecutionModel(alpha=0.1).fit(X, abs_er, conf, direction, r_fill, 4)
    assert m.fitted and m.r2 > 0.5 and m.residual_std > 0
    pred = m.predict(X, abs_er, conf, direction)
    assert np.corrcoef(pred, adverse_true)[0, 1] > 0.7
    est = ExecutionEstimator(CostModel(0.0, 1e-4, 0.05), m, 4, 3e-4, 0.0, 0.0, True, 0.0)
    e = est.estimate(0.005, 2e-4, X[0], 0.003, 0.8)
    assert isinstance(e, ExecutionEstimate) and e.adverse_selection_source == "model" and e.adverse_selection_bps >= 0
    assert e.total_expected_bps == pytest.approx(e.spread_bps + e.slippage_bps + e.adverse_selection_bps + e.fee_bps)
    assert e.net_edge_bps == pytest.approx(30.0 - e.total_expected_bps - e.uncertainty_bps)
    floor = ExecutionEstimator(CostModel(0.0, 1e-4, 0.05), m, 4, 3e-4, 0.0, 0.0, True, 5e-4)
    assert floor.estimate(0.005, 2e-4, X[0], 0.003, 0.8).adverse_selection_bps >= 5.0


def test_simulators_agree_under_the_net_edge_policy():
    rng = np.random.RandomState(3)
    n = 400
    price = 100.0 * np.exp(np.cumsum(rng.randn(n + 2) * 0.004))
    sigma = np.full(n, 0.004)
    E = np.where(rng.rand(n) < 0.3, rng.randn(n) * 2.0, 0.0)
    adverse = np.abs(rng.randn(n)) * 1e-4
    unc = np.full(n, 5e-5)
    stress = rng.rand(n)
    params = SimulationParams(horizon=4, policy=DecisionPolicy(mode="net_edge", minimum_net_edge=2e-4, uncertainty_buffer=1e-4,
                                                                 regime=RegimeAdjustment(True, 0.7, 0.4, 1.5)))
    ref = simulate_validation(E, rng.randn(n), sigma, sigma * 1.1, np.full(n, 4e-4), np.full(n, 2e-4), np.log(price[:n]),
                              price[1:n + 1], price[2:n + 2], params, adverse=adverse, uncertainty=unc, stress_p=stress)
    inp = SimInputs(E=E, y_norm=rng.randn(n), sigma=sigma, sigma_ref=sigma * 1.1, cost_roundtrip=np.full(n, 4e-4),
                    cost_side_exec=np.full(n, 2e-4), log_close=np.log(price[:n]), open_next=price[1:n + 1], open_next2=price[2:n + 2],
                    adverse=adverse, uncertainty=unc, stress_p=stress)
    res = simulate_strategy(inp, params)
    assert res.net_pnl == pytest.approx(ref.net_pnl) and len(res.trades) == ref.n_trades and res.n_signals == ref.n_signals
    assert res.trades and "net_edge" in res.trades[0] and "stress_p" in res.trades[0]
    # stress regimes cap the exposure
    assert np.all(np.abs(res.exposures[stress > 0.7]) <= 0.4 + 1e-12)


# -------------------------------------------------------- trainer / bot integration
def test_bot_with_families_records_everything(family_bot):
    bot, s = family_bot
    assert s["retrains"] >= 3 and s["model_version"] != "none" and s["metrics"]["trade_count"] > 20
    assert set(s["feature_families"]) >= {"REGIME", "OFI", "KALMAN", "CROSS_ASSET"} and s["data_tier"] == "B"
    assert s["decision_policy"]["mode"] == "net_edge"
    rec = bot.last_record
    assert rec["execution"]["adverse_selection_source"] == "model" and rec["expected_net_edge_bps"] is not None
    assert rec["policy"]["mode"] == "net_edge" and "required_edge_bps" in rec["policy"]
    m = rec["market"]
    assert len(m["regime"]["probabilities"]) == 3 and abs(sum(m["regime"]["probabilities"].values()) - 1) < 1e-6
    assert m["ofi"]["zscore"] is not None and m["kalman"]["innovation_z"] is not None and "xa_qqq_rel_return_1" in m["cross_asset"]
    ec = s["execution_calibration"]
    assert ec["fills"] > 10 and 0.0 <= ec["within_p95_fraction"] <= 1.0 and ec["total_bps"]["realised"]["n"] == ec["fills"]
    assert ec["calibration"] and all("realised_mean_bps" in row for row in ec["calibration"])
    meta = bot.model.metadata.extra
    assert meta["components_digest"] and meta["execution_model"]["fitted"] and meta["execution_model"]["n_fit"] > 50
    assert bot.model.components.hmm is not None and bot.model.components.kalman is not None
    audit = [json.loads(l) for l in (bot.artifacts_dir / "audit" / "ms_bot_bars.jsonl").read_text().splitlines()]
    last = audit[-1]["extra"]
    assert last["execution"] is not None and last["market"] is not None and last["policy"] is not None
    events = [json.loads(l) for l in (bot.artifacts_dir / "audit" / "ms_bot_events.jsonl").read_text().splitlines()]
    realised = [e for e in events if e["event"] == "EXECUTION_REALISED"]
    assert realised and {"expected", "realised", "within_p95"} <= set(realised[0])


def test_fitted_model_hash_covers_components_and_execution_model(family_bot):
    from trading_bot.training.trainer import fitted_model_hash

    bot, _ = family_bot
    model = bot.model
    h = fitted_model_hash(model)
    ex = model.execution_model
    coef = ex.coef.copy()
    ex.coef = coef * 1.01
    assert fitted_model_hash(model) != h
    ex.coef = coef
    assert fitted_model_hash(model) == h


def test_execution_stress_and_family_research(family_research):
    run, s = family_research
    es = s["execution_stress"]
    ref = es["reference"]
    assert all(r["signals_equal_reference"] for r in es["frozen"])                  # frozen decisions: identical signals
    assert [r["multiplier"] for r in es["frozen"]] == [0.5, 1.0, 1.5, 2.0, 3.0] and [r["delay"] for r in es["delay"]] == [0, 1, 2, 3]
    assert es["frozen"][1]["sharpe"] == pytest.approx(ref["sharpe"])
    assert es["frozen"][0]["total_return"] >= es["frozen"][-1]["total_return"]      # cheaper fills, same trades: monotone
    fam = s["families"]
    assert set(fam["present"]) >= {"FRACTIONAL", "REGIME", "OFI", "KALMAN", "CROSS_ASSET"}
    mat = fam["matrix"]["families"]
    for f in fam["present"]:
        assert set(mat[f]) == {"additive", "marginal"} and "per_window_delta_sharpe" in mat[f]["marginal"]
        g, ft = fam["gates"][f], fam["failure_tests"]["families"][f]
        assert g["status"] in ("ACCEPTED", "REJECTED", "RESEARCH ONLY")
        # the verdict follows the hard checks: performance in most windows, sabotage removes the benefit, special gate
        hard = (g["checks"]["median_delta_sharpe_positive"] and g["checks"]["positive_windows"]
                and g["checks"]["sabotage_removes_benefit"] and g["special"].get("passed", True))
        assert (g["status"] == "ACCEPTED") == bool(hard)
        assert g["checks"]["sabotage_removes_benefit"] == ft["all_degrade"]
        assert ft["rows"] and all({"sharpe_degrades", "forecast_quality_degrades", "benefit_disappears", "forecast_correlation", "draws"} <= set(r)
                                  for r in ft["rows"])
        assert all(r["benefit_disappears"] == (r["sharpe_degrades"] and r["forecast_quality_degrades"]) for r in ft["rows"])
    # sabotage protocol (section 51): a no-op sabotage reproduces the reference forecasts bit for bit, a real one changes
    # them, and stochastic sabotages are averaged over several draws (two windows and ~50 trades are far too few for the
    # Sharpe deltas themselves to be evidence; the gate table above is only asserted to be self-consistent)
    from trading_bot.research.families import sabotage_series
    E_ref = run.dev.oos.forecasts["full"]["E"]
    assert np.array_equal(sabotage_series(run.runner, run.dev, "NONE", "shuffle"), E_ref)
    assert not np.allclose(sabotage_series(run.runner, run.dev, "CROSS_ASSET", "delay"), E_ref)
    shuffle_row = next(r for r in fam["failure_tests"]["families"]["CROSS_ASSET"]["rows"] if r["mode"] == "shuffle")
    assert shuffle_row["draws"] == fam["failure_tests"]["n_seeds"] == 3 and len(shuffle_row["sharpe_draws"]) == 3
    assert next(r for r in fam["failure_tests"]["families"]["CROSS_ASSET"]["rows"] if r["mode"] == "delay")["draws"] == 1
    assert fam["gates"]["REGIME"]["special"]["checks"]["causal_filter"] is True
    assert fam["gates"]["OFI"]["special"]["checks"]["displayed_sizes_present"] is True
    budget = {b["family"]: b for b in fam["complexity_budget"]}
    assert budget["REGIME"]["n_parameters"] > 0 and budget["KALMAN"]["n_parameters"] == 3 and "Tier B" in budget["OFI"]["data_dependency"]
    assert fam["conditional"]["CROSS_ASSET"] and any(k.startswith("hmm:") for k in fam["conditional"]["CROSS_ASSET"])
    assert s["leakage"]["passed"] and s["leakage"]["forward_mutation"]["passed"]
    assert (run.run_dir / "diagnostics" / "families.json").exists() and (run.run_dir / "diagnostics" / "execution_stress.json").exists()


def test_cross_asset_family_carries_the_planted_lead_causally(ms_cfg):
    """Section 37: the generator makes QQQ lead SPY by one bar.  The leader's last return, reconstructed from
    the causal as-of cross-asset feature and the primary's own return, must predict SPY's *next* bar with the
    relation and carry no such information without it (a leak would show up in both)."""
    cal = SessionCalendar.from_config(ms_cfg)
    cfg = ms_cfg.with_overrides({"features": {"regime": {"enabled": False}, "ofi": {"enabled": False}, "kalman": {"enabled": False},
                                              "vpin": {"enabled": False}}})

    def lead_corr(lead_lag):
        m = generate_synthetic_market(2400, seed=21, calendar=cal, symbols=("SPY", "QQQ"), amplitude=6.0, memory_d=0.45,
                                      lead_lag=lead_lag, regime_weights=(0.35, 0.15, 0.15, 0.1, 0.25))
        store, ctx = BarStore("SPY", 30, m.bars["SPY"]), {"QQQ": BarStore("QQQ", 30, m.bars["QQQ"])}
        fe = FeatureEngine(cfg, FractionalEngine.from_config(cfg), cal)
        fm = fe.compute_matrix(store, context=ctx)
        leader = fm.column("xa_qqq_rel_return_1") + fm.column("return_1")     # QQQ's own last return (sigma-normalised)
        nxt = np.full(len(store), np.nan)
        nxt[:-1] = np.diff(store.log_close())                                 # SPY's next-bar return (not a feature)
        ok = np.isfinite(leader) & np.isfinite(nxt)
        assert ok.sum() > 2000
        return float(np.corrcoef(leader[ok], nxt[ok])[0, 1])

    with_lead = lead_corr({"leader": "QQQ", "follower": "SPY", "beta": 0.4})
    without = lead_corr(None)
    assert with_lead > 0.15 and abs(without) < 0.05 and with_lead - without > 0.15


def test_representation_selector_chooses_by_inner_fold_score(ms_cfg, tmp_path):
    from trading_bot.research.representation import RepresentationSelector, candidate_stores

    cfg = load_config(overrides=FAST).with_overrides({"market_representation": {"candidates": ["time_30m", "tick_2"]},
                                                      "research": {"holdout": {"fraction": 0.2}}})
    fine = generate_synthetic_bars(2400, seed=21, instrument="SPY", memory_d=0.45, amplitude=6.0)
    stores = candidate_stores(fine, ["time_30m", "tick_2"], NY)
    assert set(stores) == {"time_30m", "tick_2"} and len(stores["tick_2"]) < len(stores["time_30m"])
    sel = RepresentationSelector(cfg, stores, "time_30m", log=None)
    res = sel.run()
    assert res.n_windows >= 2 and sel.selection and all(rec.chosen in stores for rec in sel.selection)
    assert all(math.isfinite(v) for rec in sel.selection for v in rec.scores.values())
    assert np.all(np.diff([d.timestamp() for d in res.oos.decision_at]) > 0)         # stitched OOS stays chronological


def test_dashboard_snapshot_exposes_market_state(tmp_path, family_bot):
    from trading_bot.gui.controller import BotController

    bot, s = family_bot
    ctl = BotController(tmp_path / "settings.json")
    ctl.bot = bot
    snap = ctl.snapshot()["bot"]
    assert snap["execution_calibration"]["fills"] > 0 and snap["policy"]["mode"] == "net_edge"
    assert "REGIME" in snap["feature_families"] and snap["data_tier"] == "B" and snap["context_symbols"] == ["QQQ"]
    assert snap["last_decision"]["market"]["regime"]["stress_p"] is not None and snap["recent_executions"]
