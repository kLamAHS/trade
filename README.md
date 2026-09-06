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

## Dashboard (start.bat / start.sh)

Double-click **`start.bat`** on Windows (or run `./start.sh` on macOS/Linux). On first run it creates
`.venv`, installs the dependencies and then opens a local dashboard at `http://127.0.0.1:8765/`:

* **Alpaca credentials** — paste your API key ID and secret key, tick *Paper account*, press **Test
  connection** (shows the paper account's equity and buying power) and **Save settings**. Keys are stored
  in `settings.json` next to `start.bat`, owner-only permissions, git-ignored, never sent anywhere but
  Alpaca. Environment variables `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` still work for the CLI.
* **Run configuration** — symbol, backtest (synthetic bars or a CSV downloaded with the *Download* button)
  or Alpaca paper trading, fast preset, order mirroring, initial capital and free-form config overrides.
* **Start / Stop**, live portfolio tiles, equity chart (hover for values, table view for the numbers),
  the fractional-contribution chart per retrain, the last decision chain, trades, events and the log.
* **Validation tab** — runs the research-grade validation framework (below) on the selected data and
  renders the run directory: evidence banner, classification badge, OOS tiles (CAGR, max drawdown,
  Sharpe, Sortino, trades, costs), equity with baseline / buy-and-hold / production-policy overlays,
  OOS and HOLDOUT segments and a log-scale toggle, underwater and rolling-Sharpe charts, per-window
  ΔSharpe, the acceptance-gate table, cost / timing / parameter stress, sanity and leakage tests,
  bootstrap and Monte Carlo, baselines, regimes and a clickable trade audit trail.

The dashboard is the `gui` sub-command (`python -m trading_bot.main gui [--port 8765] [--no-browser]`).
It binds to localhost only.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps   # exact tested environment
# or: pip install -e ".[dev,plots]"                                # latest compatible versions

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

# 5. Research-grade validation (docs/VALIDATION_FRAMEWORK.md): walk-forward OOS, ablation, baselines,
#    cost / timing / parameter stress, sanity + leakage tests, bootstrap + Monte Carlo, reproducibility, gates
python -m trading_bot.main research-synthetic --seeds 5 --bars 4000          # engineering validation (not evidence)
python -m trading_bot.main research --csv artifacts/data/SPY_30m.csv         # development run, holdout stays locked
python -m trading_bot.main research --csv artifacts/data/SPY_30m.csv --open-holdout   # once, at the end
python -m trading_bot.main research-runs --compare RUN_A RUN_B               # identical manifest -> identical results

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

## Validation framework (research-grade backtesting)

A single backtest path is a demonstration, not evidence. `trading_bot/research/` implements the
validation framework described in `docs/VALIDATION_FRAMEWORK.md`, whose stages actively try to
disprove the strategy:

* **No-lookahead engine** — every OOS decision row carries `feature_available_at`, `decision_at` and
  `execution_at` (next open); the runner refuses any violation and the leakage stage re-checks the
  rows and the label alignment against the bar store.
* **Rolling walk-forward** — the production trainer is refit on every window (nested d* / grid /
  calibration selection inside the training block); the fitted full and no-fractional models forecast
  the unseen block; the blocks are concatenated into one continuous equity curve traded with the live
  position rules and circuit breakers. A **locked final holdout** (15%) is never touched until
  `--open-holdout`, which appends an access record to `artifacts/holdout_access.jsonl`.
* **Ablation with statistics** — positive cycles, mean / median ΔSharpe, bootstrap CI, sign test.
* **Baselines** — cash, buy & hold, vol-scaled, momentum, random permutations of the strategy's own
  forecasts, no-fractional model.
* **Stress** — cost curve (model ×1/×2/×3, flat 0-10 bps), execution delayed +1/+2 bars, position-rule
  parameters ×0.5/×2, fractional order d* ± 2 steps (light refit).
* **Sanity and leakage** — shuffled labels, shuffled features, target shifted +20 bars, reversed and
  random forecasts, zero / double cost, each with an expectation and a pass flag.
* **Statistics** — block bootstrap CIs, Monte Carlo trade resampling, multiple-testing bookkeeping.
* **Reproducibility** — manifests with separate data / config / model-config / code / fitted-model
  hashes; a window is retrained again and compared bit for bit (`REPRODUCIBILITY FAILURE` otherwise).
* **Gates and classification** — EXPERIMENTAL → CANDIDATE → VALIDATED CANDIDATE → HOLDOUT PASSED →
  PAPER ELIGIBLE, thresholds in `research.gates`. Synthetic runs are labelled
  *SYNTHETIC / ENGINEERING VALIDATION — NOT PERFORMANCE EVIDENCE* and never classify.

Runs are written to `artifacts/runs/<run_id>/` (`manifest.yaml`, `summary.json`, `equity.csv`,
`trades.csv`, `fills.csv`, `decisions.csv`, `retrains.csv`, `models/`, `diagnostics/`, `plots/`, `logs/`).

## Artifacts

```
artifacts/
  runs/<run_id>/                                     research runs (see above) + holdout_access.jsonl
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
* **Paper only.** The application can only talk to Alpaca's paper endpoint; there is no setting, flag
  or page control that enables live trading (`docs/AUDIT_RESPONSE.md`, finding 1).
* **Fill timing.** A fill may only use a price observed at or after the decision: the next bar's open
  in backtests; in live paper mode (where every bar carries its receipt time) the standing NBBO fetched
  with the bar, or a deferral to the first open after the decision, optionally the broker's actual fill
  (`execution.live_fill_source`).
* **Label.** `Y_t = log O_{t+1+H} - log O_{t+1}` by default (`prediction.label_price: open`): the
  H-bar return of a position entered at the first tradable price after the signal. By default the
  strategy re-evaluates that forecast every bar and re-targets exposure under turnover suppression and
  the 12-bar holding limit (`signal.reevaluate_every_bars: 4` holds each entry to the forecast horizon
  instead). The validation simulator applies the same position-level rules as the live bot; the
  portfolio-level circuit breakers (daily loss, drawdown, data halts) are not simulated.
* **Fold-local d\*.** Each walk-forward fold estimates the adaptive order on its own training block
  and rebuilds its features with it; the deployed model uses the holdout-validated d\* (estimated on
  the inner block), and the whole-window d\* is recorded for diagnostics only.
* **Outer holdout.** The newest 15 % of the training window is never used for selection; acceptance,
  the ablation score `S_F - S_0` and the `holdout_edge` check are read there once.
* **Calibration** is strictly chronological: fold *k*'s calibrator is fitted on out-of-sample
  predictions that end before its validation window (earlier folds' validation predictions plus an
  inner chronological split of fold 1's training block); the production calibrator uses all pooled
  out-of-fold predictions. `A = 0` always maps to `E = 0`.
* **Sessions.** Early-close days (NYSE rules plus `market.early_closes`) have a 13:00 close: the time
  features use that day's session length and the validator does not flag the short session as a gap.
* **Live quotes.** In paper mode the NBBO quote attached to a completed bar carries its own timestamp;
  the FeatureVector's `latest_source_timestamp` is the newest source time and the feature timestamp
  is never earlier than it, so the section 3 guard is checked against real source times.
* **Sessions and holidays.** The calendar carries rule-based NYSE holidays (plus `market.holidays`), so a
  bar that opens later than the next trading session is a *missing session* halt while a holiday gap is not.
* **Daily loss** is measured from the previous session's closing equity, so overnight gaps and fills at
  the open count toward the -2.5 % limit. After a drawdown halt is lifted by an accepted retrain the
  drawdown reference is re-based to the current equity (otherwise the halt would re-arm immediately).
* **Max-holding re-entry.** When a fresh signal re-opens a position in the same direction after 12 bars,
  the ledger closes the trade record and restarts the holding clock and the stop reference at the
  current mark; a turnover-suppressed position is maintained exactly (no order, no drift trimming).
* **Trade P&L** is net of commission only: spread and slippage are already inside the fill prices.
  The ablation baseline runs the same hyperparameter grid and is compared best-of-grid to best-of-grid.
* **Alpaca mirror.** Mirrored whole-share market orders are submitted the moment an order is queued (in
  live mode that is the open of the next bar); broker acknowledgements and final statuses are recorded
  as annotations and audit events, never as the ledger's fill, so paper and backtest accounting agree.
  Order ids are unique per run (`<run_id>-<n>`) and double as Alpaca client order ids.
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

## Reproducibility and CI

`requirements.lock` pins the tested environment. Every model artifact stores the data checksum,
configuration, fractional kernel, seed, effective model parameters and the software environment
(Python, platform, package versions, git commit), all of which enter the model id. GitHub Actions
(`.github/workflows/ci.yml`) runs the suite and a smoke backtest on every push. See
`docs/AUDIT_RESPONSE.md` for the changes made in response to the September 2026 source audit.

## Tests (spec sections 54-55)

`trading_bot/tests/` covers (`test_research.py` for the validation framework: schedule tiling and
holdout locking, simulator parity with the trainer's validation simulator, delay / cost / halt
semantics, metrics, bootstrap determinism, the gate ladder, manifest hashes, label offsets, synthetic
generator knobs, a leakage-detector mutation test, the end-to-end pipeline with artifacts, bit-for-bit
window reproduction and the dashboard research API): GL weight recursion vs. direct binomial evaluation, `d = 0` identity,
`d = 1` first difference, deterministic truncation, NaN/inf-free warm-up, causal convolution;
batch/streaming feature equality and formula spot checks; **future-bar mutation**, **execution-shift**,
**scaling**, **fractional-order** and **label-isolation** leakage tests plus purge/embargo checks;
fill model, order builder, state-consistent execution, ledger accounting; sizing arithmetic,
position rules, daily/drawdown/ablation halts, validator circuit breakers, state-machine transitions,
trainer reproducibility (same seed -> identical artifact id and predictions), registry round trip
and an end-to-end audit-trail check.
