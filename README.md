# Fractional-Memory Systematic Trading Bot (Alpaca paper trading + execution simulator)

A fully automated, research-grade trading system that tests whether **fractional-order
transformations of market data** carry exploitable information beyond ordinary returns.
It consumes 30-minute OHLCV bars (Alpaca Market Data or a stored/synthetic history),
builds fractional-memory features, forecasts the executable forward risk-adjusted return
with a boosted-tree magnitude model confirmed by a logistic direction model, sizes
positions by calibrated edge and volatility, enforces strict risk limits, simulates
realistic execution at the next bar's open, retrains on a walk-forward schedule using
only information available at each point, and logs enough to reconstruct every decision.

The fractional mathematics is not decorative: every retraining cycle trains a shadow
model **without** fractional features and the bot halts the strategy
(`FRACTIONAL_EDGE_NOT_DETECTED`) when the fractional contribution `S_F - S_0` is
non-positive for three consecutive cycles.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,plots]"

# 1. Synthetic end-to-end demo (random walk + stationary long-memory component)
python -m trading_bot.main backtest --synthetic 6000 --fast --symbol SYN

# 2. Real data: download 30-minute SPY bars from Alpaca, then backtest
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m trading_bot.main download --symbol SPY --days 1200 --out artifacts/data/SPY_30m.csv
python -m trading_bot.main backtest --csv artifacts/data/SPY_30m.csv --symbol SPY

# 3. Live paper trading against Alpaca (bootstraps history, trains, then polls for bars)
python -m trading_bot.main paper --symbol SPY            # add --no-mirror to keep orders internal

# 4. One retraining cycle / research diagnostics
python -m trading_bot.main train --csv artifacts/data/SPY_30m.csv
python -m trading_bot.main diagnose --csv artifacts/data/SPY_30m.csv --every 2

pytest            # unit, numerical, leakage, execution, risk and end-to-end tests
```

Always launch through `python -m trading_bot.main` (or the `trading-bot` console script):
the spec-mandated `trading_bot/logging` package must never shadow the standard library.

`--set key.path=value` overrides any configuration value from the command line, e.g.
`--set training.retrain_every_bars=500 --set risk.daily_loss_limit=0.02`.
`--fast` layers `config/strategy_fast.yaml` (smaller window and grid) for demos.

## Alpaca

* Credentials: `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (or `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).
* Data: `alpaca-py` `StockHistoricalDataClient`, 30-minute bars, `feed: iex` by default
  (`alpaca.feed: sip` if your subscription allows), split/dividend adjusted, restricted to
  regular-session bars (09:30-16:00 America/New_York). No synthetic bars are ever created.
* Start-up: history is bootstrapped into the store without simulated trading, one training cycle
  runs, and the bot goes live on the next completed bar.
* Live loop: polls for newly *completed* bars every `alpaca.poll_seconds` (the forming bar is never
  emitted), attaches the latest NBBO quote with its timestamp as `bid/ask`, feeds the bot, and flags
  a stale feed (no bar for `data.stale_feed_bars` intervals inside a session) as `DATA_HALTED`.
* Orders: the internal simulator (spec fill model) is the ledger's source of truth. With
  `alpaca.mirror_orders: true` (default) each exposure change is also submitted as a
  market order to the **paper** account; when Alpaca reports a fill, its average price
  replaces the simulated price in the audit trail (`source: alpaca`).
* Retraining in paper mode runs in a background thread; the live model is swapped
  atomically only after validation passes.

## Architecture (spec section 45)

```
MarketDataFeed (data/feed.py: ReplayFeed | AlpacaBarFeed)
  -> DataValidator (data/validator.py)     reject corrupt bars / halt on gaps & jumps
  -> BarStore (data/store.py)              append-only, checksummed
  -> FeatureEngine (features/engine.py)    FractionalEngine + VolatilityEngine + MarketStateEngine
  -> ModelEngine (models/combined.py)      boosted magnitude x logistic direction -> isotonic calibration
  -> SignalEngine (strategy/signal.py)     cost threshold, confidence, volatility scaling
  -> RiskEngine (risk/manager.py)          turnover suppression, max holding, stop, daily / drawdown halts
  -> ExecutionEngine (execution/)          next-open fills with spread + slippage, Alpaca mirror
  -> PortfolioLedger (portfolio/ledger.py) -> AuditLogger (logging/audit.py)

Trainer: HistoricalStore -> TrainingDatasetBuilder -> FractionalOrderEstimator
         -> WalkForwardValidator -> ModelTrainer -> ModelValidator -> ModelRegistry
```

The orchestrator is `trading_bot/bot.py` (`TradingBot.on_bar`, the five-state machine
`INITIALIZING / READY / POSITIONED / RISK_HALTED / DATA_HALTED`).

## Pipeline per bar (spec sections 57-59)

1. Orders queued at the previous bar are filled at **this bar's open** (`Open ± Spread/2 ± Slippage`).
2. The bar is validated, stored and marked; retraining is triggered every 250 stored bars.
3. Features are computed causally from the trailing history (batch and streaming paths share one
   implementation and are tested to be identical).
4. `M_t` (boosted regression) and `P_t^+` (logistic) are combined into `A_t = M_t |2P_t^+ - 1|`
   (zero on disagreement) and calibrated, `E_t = g(A_t)`; `ER_t = E_t sigma_50 sqrt(H)`.
5. Trade only when `|ER_t| > 3 Cost_t`; `Confidence = min(1, |ER|/(6 Cost))`;
   `VM = clip(sigma_ref / sigma_50, 0.25, 1.5)`; `Q = Direction x Confidence x VM`.
6. Risk: emergency stop at `-4 sigma sqrt(H)`, max holding 12 bars, rebalance threshold 0.15,
   daily loss halt at -2.5 %, drawdown halt at -10 % (until an accepted retrain).
7. An exposure-change order is queued for the next bar and everything is written to the audit log.

## Artifacts

```
artifacts/
  models/<model_id>/model.joblib + metadata.json     reproducibility record (spec 56), current.json pointer
  audit/<run>_bars.jsonl                             one record per bar: bar, features, d, kernel, prediction,
                                                     cost, signal, risk decision, order, ledger, state
  audit/<run>_fills.jsonl, _events.jsonl, _retrains.jsonl, _trades.jsonl, _summary.json
  diagnostics/stationarity_vs_d.csv, fractional_contribution.csv, oos_score_vs_d.csv (+ .png with matplotlib)
```

## Interpretation decisions (where the spec leaves room)

* **Cost model.** Round-trip cost = 2 x commission + one full spread + **2 x** per-side slippage
  `alpha (H-L)/C`, so the cost estimate is consistent with the fill model that charges slippage on
  both entry and exit. Simulated fills use the execution bar's range (`execution.slippage_reference`).
* **Model inputs.** Raw fractional levels `D^d p` are recorded in every FeatureVector but only the
  rolling-robust z-scored levels feed the models (section 9); toggle with
  `features.use_raw_fractional_levels`. Slopes and curvature are computed per channel (8 slope + 4
  curvature features, sections 10-11).
* **EWMA variance** uses a finite kernel of 450 bars (`lambda^450 ~ 1e-12`) so streaming and batch
  values are bit-identical.
* **Calibration** is strictly chronological: fold *k*'s calibrator is fitted on out-of-sample
  predictions that end before its validation window (earlier folds' validation predictions plus an
  inner chronological split of fold 1's training block); the production calibrator uses all pooled
  out-of-fold predictions. `A = 0` always maps to `E = 0`.
* **Adaptive order `d*`** is estimated once per retraining cycle on the whole training window before
  the folds are carved out, exactly as the section 58 pseudo-code prescribes (a single scalar with
  negligible leakage capacity); the leakage test checks it depends only on that window.
* **Sessions.** Early-close days (NYSE rules plus `market.early_closes`) have a 13:00 close: the time
  features use that day's session length and the validator does not flag the short session as a gap.
* **Live quotes.** In paper mode the NBBO quote attached to a completed bar carries its own timestamp;
  the FeatureVector's `latest_source_timestamp` is the newest source time and the feature timestamp
  is never earlier than it, so the section 3 guard is checked against real source times.
* **Alpaca history** is split-adjusted only (`alpaca.adjustment: split`); dividend adjustment would
  rescale history with information known only after each ex-dividend date and splice badly onto
  live bars. Only bars that have closed (plus a completion grace) are ever emitted.
* **Walk-forward** folds follow the section 38 example literally: fold k trains on the earliest
  `40 + 10(k-1)` % and validates the following 10 % after a 5-bar purge and 5-bar embargo.
* **Score** (section 41) = `Sharpe_net - 0.25 x (mean |dQ| x bars/day) - 0.25 x (maxDD / 10 %)`.
* **Turnover suppression** treats any change of direction (including to flat) as a rebalance.
* **Data halts.** Corrupt bars are discarded (never stored); gaps and extreme jumps store the bar but
  halt new orders; open positions are flattened at the next clean bar and trading resumes after
  `data.halt_recovery_bars` consecutive clean bars.
* **Synthetic data** (`--synthetic`) is a random walk plus a stationary ARFIMA(0, d, 0) component
  with U-shaped intraday volatility; it exists for tests and demos only.

## Tests (spec sections 54-55)

`trading_bot/tests/` covers: GL weight recursion vs. direct binomial evaluation, `d = 0` identity,
`d = 1` first difference, deterministic truncation, NaN/inf-free warm-up, causal convolution;
batch/streaming feature equality and formula spot checks; **future-bar mutation**, **execution-shift**,
**scaling**, **fractional-order** and **label-isolation** leakage tests plus purge/embargo checks;
fill model, order builder, state-consistent execution, ledger accounting; sizing arithmetic,
position rules, daily/drawdown/ablation halts, validator circuit breakers, state-machine transitions,
trainer reproducibility (same seed -> identical artifact id and predictions), registry round trip
and an end-to-end audit-trail check.
