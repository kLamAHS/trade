"""Regression tests for the issues found in the adversarial audit."""

import json
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from trading_bot.bot import TradingBot
from trading_bot.data.calendar import SessionCalendar, nyse_early_closes
from trading_bot.data.feed import AlpacaBarFeed, ReplayFeed
from trading_bot.data.store import BarStore, bars_from_frame
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.data.validator import DataValidator
from trading_bot.execution.cost_model import CostModel
from trading_bot.features.engine import FeatureEngine
from trading_bot.features.rolling import rolling_corr_with_index
from trading_bot.fractional.stationarity import StationaryOrderEstimator
from trading_bot.models.calibration import Calibrator
from trading_bot.models.regression import BoostedRegressor, RegressionParams
from trading_bot.risk.manager import RiskEngine
from trading_bot.training.trainer import ModelTrainer
from trading_bot.training.walkforward import walk_forward_folds
from trading_bot.types import Bar, FeatureVector

NY = SessionCalendar()
UTC = timezone.utc


def _bar(ts, o=100.0, h=101.0, l=99.0, c=100.5, v=1000.0, **kw):
    return Bar("SYN", ts, o, h, l, c, v, 30, **kw)


# ----------------------------------------------------------------- calendar
def test_nyse_early_close_rules_and_session_length():
    assert date(2026, 11, 27) in nyse_early_closes(2026)      # day after Thanksgiving 2026
    assert date(2025, 11, 28) in nyse_early_closes(2025)
    assert date(2025, 12, 24) in nyse_early_closes(2025)      # Wednesday
    assert date(2026, 12, 24) in nyse_early_closes(2026)      # Thursday
    assert date(2025, 7, 3) in nyse_early_closes(2025)        # Thursday
    assert date(2026, 7, 3) not in nyse_early_closes(2026)    # Friday: 4 July observed, no early close
    d = date(2025, 11, 28)
    assert NY.is_early_close(d) and NY.session_minutes_for(d) == 210 and NY.bars_for(d) == 7
    last = NY.session_open_datetime(d) + timedelta(minutes=180)          # 12:30 bar
    assert NY.expected_next_start(last) is None
    assert not NY.is_regular_session_bar(NY.session_open_datetime(d) + timedelta(minutes=210))   # 13:00 bar
    assert NY.session_minutes_for(date(2025, 11, 26)) == 390
    cal2 = SessionCalendar(early_closes=frozenset({date(2025, 3, 5)}))
    assert cal2.is_early_close(date(2025, 3, 5)) and cal2.bars_for(date(2025, 3, 5)) == 7


def test_validator_accepts_early_close_session_end():
    v = DataValidator(NY, instrument="SYN")
    d = date(2025, 11, 28)
    starts = NY.regular_session_starts(d)
    assert len(starts) == 7
    for ts in starts:
        assert v.validate(_bar(ts.astimezone(UTC))).ok
    nxt = NY.session_open_datetime(date(2025, 12, 1)).astimezone(UTC)
    assert v.validate(_bar(nxt)).ok                       # no "session ended early" halt


def test_time_features_use_session_length_of_the_day(cfg, fractional):
    eng = FeatureEngine(cfg, fractional, NY, adaptive_d=0.4)
    bars = generate_synthetic_bars(1200, seed=2, instrument="SYN", start=date(2025, 10, 1))
    store = BarStore("SYN", 30, bars)
    fm = eng.compute_matrix(store)
    idx = [i for i, b in enumerate(bars) if NY.session_date(b.timestamp) == date(2025, 11, 28)]
    assert len(idx) == 7
    last = idx[-1]
    m = NY.minutes_since_open(bars[last].timestamp)
    assert fm.column("time_sin")[last] == pytest.approx(math.sin(2 * math.pi * m / 210))


# ----------------------------------------------------------------- validator
def test_is_stale_ignores_overnight_gap_and_detects_intraday_gap():
    v = DataValidator(NY, stale_feed_bars=2, instrument="SYN")
    thu_last = NY.session_open_datetime(date(2026, 9, 3)) + timedelta(minutes=360)     # 15:30 bar
    v.accept(_bar(thu_last.astimezone(UTC)))
    fri_open = NY.session_open_datetime(date(2026, 9, 4))
    for m in (0, 15, 45, 89):
        assert not v.is_stale((fri_open + timedelta(minutes=m, seconds=30)).astimezone(UTC))
    assert v.is_stale((fri_open + timedelta(minutes=91)).astimezone(UTC))
    assert not v.is_stale((fri_open - timedelta(hours=2)).astimezone(UTC))               # pre-market never stale
    v.accept(_bar((fri_open + timedelta(minutes=60)).astimezone(UTC)))                   # 10:30 bar (closes 11:00)
    assert not v.is_stale((fri_open + timedelta(minutes=179)).astimezone(UTC))
    assert v.is_stale((fri_open + timedelta(minutes=181)).astimezone(UTC))


def test_tolerant_csv_reader_defers_to_validator():
    import pandas as pd
    t0 = NY.session_open_datetime(date(2026, 3, 2)).astimezone(UTC)
    rows = [{"timestamp": (t0 + timedelta(minutes=30 * i)).isoformat(), "open": 100, "high": 101, "low": 99,
             "close": 100.5, "volume": 10, "bid": float("nan"), "ask": float("nan")} for i in (0, 1, 1, 0, 2)]
    bars = bars_from_frame(pd.DataFrame(rows), "SYN")
    assert len(bars) == 5 and bars[0].bid is None
    with pytest.raises(ValueError):
        BarStore("SYN", 30, bars)
    v = DataValidator(NY, instrument="SYN")
    results = [v.validate(b) for b in bars]
    assert [r.storable for r in results] == [True, True, False, False, True]
    assert "duplicate timestamp" in results[2].reasons and "timestamp moved backward" in results[3].reasons


# ----------------------------------------------------------------- feed
class _RawBar(SimpleNamespace):
    pass


class _FakeClient:
    def __init__(self, raw, quote=None):
        self.raw = raw
        self.quote = quote
        self.calls = 0

    def get_stock_bars(self, req):
        self.calls += 1
        start = req.start
        if start is not None and start.tzinfo is None:      # alpaca-py normalises request times to naive UTC
            start = start.replace(tzinfo=UTC)
        return SimpleNamespace(data={"SYN": [b for b in self.raw if start is None or b.timestamp >= start]})

    def get_stock_latest_quote(self, req):
        if self.quote is None:
            raise RuntimeError("no quote")
        return {"SYN": self.quote}


def _raw(ts, px):
    return _RawBar(timestamp=ts, open=px, high=px + 1, low=px - 1, close=px + 0.5, volume=100.0)


def test_alpaca_feed_drops_forming_bar_and_polls_incrementally():
    d = date(2026, 9, 3)
    starts = [s.astimezone(UTC) for s in NY.regular_session_starts(d)]
    pre = (NY.session_open_datetime(d) - timedelta(hours=2)).astimezone(UTC)
    raw = [_raw(pre, 90.0)] + [_raw(s, 100.0 + i) for i, s in enumerate(starts[:3])]     # 09:30, 10:00, 10:30 (+ pre-market)
    now = starts[2] + timedelta(minutes=20)          # 10:50 -> 10:30 bar still forming
    quote = SimpleNamespace(bid_price=101.0, ask_price=101.2, timestamp=now - timedelta(seconds=1))
    client = _FakeClient(raw, quote)
    feed = AlpacaBarFeed("SYN", NY, api_key="k", secret_key="s", data_client=client, clock=lambda: now)
    hist = feed.fetch_history(starts[0] - timedelta(days=1), now)
    assert [b.timestamp for b in hist] == starts[:2]                # forming 10:30 bar and pre-market bar excluded
    assert feed.last_timestamp is None                             # fetch_history never advances the cursor
    got = feed.poll_new_bars(now)
    assert [b.timestamp for b in got] == starts[:2]
    assert got[-1].bid == 101.0 and got[-1].ask == 101.2 and got[-1].quote_timestamp == quote.timestamp
    assert got[-1].latest_source_time == quote.timestamp
    assert feed.poll_new_bars(now) == []                            # nothing new
    later = starts[2] + timedelta(minutes=31)                       # 11:01 -> 10:30 bar complete
    got2 = feed.poll_new_bars(later)
    assert [b.timestamp for b in got2] == [starts[2]]
    assert feed.last_timestamp == starts[2]
    assert feed.poll_new_bars(later) == []


def test_feature_timestamp_reflects_late_quote(cfg, fractional):
    eng = FeatureEngine(cfg, fractional, NY, adaptive_d=0.4)
    bars = generate_synthetic_bars(1000, seed=3, instrument="SYN")
    last = bars[-1]
    qts = last.close_time + timedelta(seconds=7)
    bars[-1] = Bar(last.instrument, last.timestamp, last.open, last.high, last.low, last.close, last.volume, 30,
                   last.bid, last.ask, quote_timestamp=qts)
    fv = eng.compute_latest(BarStore("SYN", 30, bars))
    assert fv.latest_source_timestamp == qts and fv.timestamp == qts and fv.bar_close_time == last.close_time
    fv2 = eng.compute_latest(BarStore("SYN", 30, bars[:-1]))
    assert fv2.timestamp == fv2.latest_source_timestamp == bars[-2].close_time


# ----------------------------------------------------------------- features / types
def test_feature_vector_values_are_read_only():
    t = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
    fv = FeatureVector("SYN", t, t, 0, 0.5, 500, {"x": 1.0})
    with pytest.raises(TypeError):
        fv.values["x"] = 2.0        # type: ignore[index]
    assert fv["x"] == 1.0


def test_rolling_corr_propagates_nan():
    x = np.arange(60.0)
    x[30] = np.nan
    out = rolling_corr_with_index(x, 50)
    assert np.isnan(out[49:]).all()   # every window containing index 30
    x2 = np.arange(120.0)
    x2[30] = np.nan
    out2 = rolling_corr_with_index(x2, 50)
    assert np.isnan(out2[49:80]).all() and np.isfinite(out2[80:]).all() and out2[-1] == pytest.approx(1.0)


def test_sigma_window_independent_of_vol_windows(cfg, fractional, store_1500):
    cfg2 = cfg.with_overrides({"prediction": {"volatility_window": 30}})
    eng = FeatureEngine(cfg2, fractional, NY, adaptive_d=0.4)
    fv = eng.compute_latest(store_1500)
    r = np.diff(np.log(store_1500.arrays()["close"]))
    assert fv["sigma_h"] == pytest.approx(np.std(r[-30:]))


# ----------------------------------------------------------------- models
def test_binned_calibration_uses_pav_not_cummax():
    A = np.concatenate([np.full(100, -1.0), np.full(100, 0.0) + 1e-3, np.full(100, 1.0)])
    y = np.concatenate([np.full(100, 0.30), np.full(100, -0.10), np.full(100, 0.00)])
    c = Calibrator("binned", bins=3).fit(A, y)
    levels = c.predict(np.array([-1.0, 1e-3, 1.0]))
    assert np.all(np.diff(levels) >= -1e-12)
    assert levels.max() < 0.30 and levels.min() > -0.10        # pooled means, not a lifted envelope
    assert levels[0] == pytest.approx(0.2 / 3, abs=1e-6)


def test_sklearn_backend_has_no_early_stopping_and_records_effective_params():
    rng = np.random.RandomState(0)
    X = rng.randn(12000, 5)
    y = X[:, 0] + rng.randn(12000)
    m = BoostedRegressor(RegressionParams(backend="sklearn", n_estimators=25)).fit(X, y)
    assert m._model.n_iter_ == 25
    eff = m.effective_params()
    assert eff["subsample"] is None and eff["reg_alpha"] is None and eff["early_stopping"] is False


def test_stationarity_estimator_never_raises_on_degenerate_input():
    est = StationaryOrderEstimator(candidates=[0.2, 0.5], adf_maxlag=3)
    with pytest.raises(ValueError):
        est.estimate(np.ones(800))          # every candidate unusable -> explicit error
    x = np.cumsum(np.random.RandomState(1).randn(900))
    res = est.estimate(x)
    assert res.d_star in (0.2, 0.5)


def test_retrain_reports_error_instead_of_raising(fast_cfg, fractional):
    bars = generate_synthetic_bars(1500, seed=4, instrument="SYN")
    const = [Bar(b.instrument, b.timestamp, 100.0, 100.0, 100.0, 100.0, b.volume, 30) for b in bars]
    trainer = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    report = trainer.retrain(BarStore("SYN", 30, const))
    assert report.model is None and report.error and not report.accepted


def test_fold_calibration_is_chronological(fast_cfg, fractional):
    bars = generate_synthetic_bars(1700, seed=6, instrument="SYN", memory_d=0.45, amplitude=6.0)
    store = BarStore("SYN", 30, bars)
    trainer = ModelTrainer(fast_cfg, FeatureEngine(fast_cfg, fractional, NY), fractional, CostModel.from_config(fast_cfg))
    ds = trainer.builder.build(store, 0.4)
    _, _, folds = trainer._layout(len(ds))
    ev = trainer.evaluate_candidate(trainer.build_fold_sets(store, ds, folds, fixed_d=0.4), trainer.grid[0], ds.feature_names)
    H = fast_cfg.prediction.horizon_bars
    for fold, cal_rows in zip(folds, ev.calibration_rows):
        # every label used for calibration ends (t + H + 1) before the fold's first validation row
        assert cal_rows.max() + H + 1 < fold.validate[0]
        assert len(cal_rows) >= 20
    assert len(ev.calibration_rows[-1]) > len(ev.calibration_rows[0])


# ----------------------------------------------------------------- risk / bot
def test_data_halt_recovery_counter_resets_on_rearm():
    r = RiskEngine(halt_recovery_bars=3)
    r.set_data_halt(True, "gap")
    assert not r.note_clean_bar() and not r.note_clean_bar()
    r.set_data_halt(True, "stale")          # re-armed -> counter restarts
    assert r.clean_bars_since_halt == 0
    assert not r.note_clean_bar() and not r.note_clean_bar()
    assert r.note_clean_bar() and not r.data_halted and r.clean_bars_since_halt == 0
    assert not r.note_clean_bar()


def _events(bot):
    p = bot.artifacts_dir / "audit" / f"{bot.run_id}_events.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_gap_bar_while_positioned_flattens_at_next_bar_without_crash(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1700, seed=21, instrument="SYN", memory_d=0.45, amplitude=6.0)
    bot = TradingBot(fast_cfg, run_id="gap", artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:1450])
    assert bot.model is not None
    for b in bars[1450:1452]:
        bot.on_bar(b)
    bot.execution.pending_orders()                    # drop whatever the model queued: deterministic set-up
    # force a long position right before the gap, then feed a sequence with bar 1452 missing
    from trading_bot.types import Fill
    last = bot.store.last()
    bot.ledger.apply(Fill("f", "SYN", last.close_time, last.close_time, "buy", 100.0, last.close, last.close, 0, 0, 0,
                          new_entry=True, entry_sigma=0.01))
    units_before = bot.ledger.units                    # the model may already hold a position for this seed
    assert units_before >= 100.0
    n_fills = len(bot.ledger.fills)
    for b in bars[1453:1470]:                         # halt-only (gap) bar at 1453
        bot.on_bar(b)
    ev = _events(bot)
    assert any(e["event"] == "DATA_HALT" for e in ev)
    assert bot.halted_bars == 1 and bot.rejected_bars == 0
    assert bot.ledger.units == 0.0                       # flattened while halted
    flat = bot.ledger.fills[n_fills]
    assert flat.side == "sell" and flat.units == pytest.approx(units_before)
    assert flat.signal_timestamp == bars[1453].close_time      # decided on the gap bar ...
    assert flat.fill_timestamp == bars[1454].timestamp         # ... filled at the next bar's open
    for f in bot.ledger.fills:
        assert f.fill_timestamp >= f.signal_timestamp
    assert any(e["event"] == "DATA_HALT_CLEARED" for e in ev)
    # while DATA_HALTED only flattening fills may occur (section 35): everything after the flatten and before
    # recovery must have target exposure 0
    recovery_ts = bars[1453 + fast_cfg.data.halt_recovery_bars].timestamp
    for f in bot.ledger.fills[n_fills + 1:]:
        if f.fill_timestamp < recovery_ts:
            assert f.target_exposure == 0.0


def test_rejected_bar_while_positioned_flattens_at_next_clean_bar(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1500, seed=22, instrument="SYN")
    bot = TradingBot(fast_cfg.with_overrides({"training": {"minimum_bars": 1400, "window_bars": 1400}}),
                     run_id="rej", artifacts_dir=tmp_path, log=None)
    bot.bootstrap(bars[:1450])
    from trading_bot.types import Fill
    last = bot.store.last()
    bot.ledger.apply(Fill("f", "SYN", last.close_time, last.close_time, "sell", 50.0, last.close, last.close, 0, 0, 0,
                          new_entry=True, entry_sigma=0.01))
    b = bars[1450]
    corrupt = Bar(b.instrument, b.timestamp + timedelta(days=30), b.open, b.high, b.low, b.close, b.volume, 30)   # future ts, wrong session start
    bad = Bar(b.instrument, b.timestamp, b.open, b.low - 1, b.high + 1, b.close, b.volume, 30)                    # high < low
    bot.on_bar(bad)
    assert bot.rejected_bars == 1 and bot.execution.has_pending()
    for nb in bars[1450:1456]:
        bot.on_bar(nb)
    assert bot.ledger.units == 0.0
    for f in bot.ledger.fills:
        assert f.fill_timestamp >= f.signal_timestamp


def test_bootstrap_seeds_sigma_history_and_trains(fast_cfg, tmp_path):
    bars = generate_synthetic_bars(1500, seed=23, instrument="SYN", memory_d=0.45, amplitude=6.0)
    bot = TradingBot(fast_cfg, run_id="boot", artifacts_dir=tmp_path, log=None)
    stored = bot.bootstrap(bars)
    assert stored == 1500 and len(bot.store) == 1500
    assert bot.retrain_count == 1 and bot.model is not None
    assert bot.signal_engine.has_history
    assert math.isfinite(bot.signal_engine.reference_volatility())
    assert bot.ledger.fills == [] and bot.state.value in ("READY", "RISK_HALTED")
