# Rigorous backtesting / validation framework

The single-path backtest (`backtest` command, dashboard *Trading* tab) shows what the bot
does bar by bar. It is **not** evidence that the strategy works. Evidence comes from the
research framework in `trading_bot/research/`, whose job is to try to disprove the strategy:

```
python -m trading_bot.main research --csv artifacts/data/SPY_30m.csv            # development run (holdout stays locked)
python -m trading_bot.main research --csv artifacts/data/SPY_30m.csv --open-holdout   # once, at the end
python -m trading_bot.main research-synthetic --seeds 5 --bars 4000              # engineering validation only
python -m trading_bot.main research-runs [--compare RUN_A RUN_B]                 # list / reproducibility check
```

or the **Validation** tab of the dashboard (`start.bat`), which runs the same pipeline on the
data selected under *Run configuration* and renders the run directory.

## Run sequence (spec section 34)

| # | Stage | What it does | Module |
|---|-------|--------------|--------|
| 1 | `walkforward` | Rolling windows: retrain with the production trainer on the training block (nested d*/grid/calibration selection inside it, untouched outer holdout for acceptance), forecast the next unseen block with the fitted full **and** baseline models, concatenate all OOS blocks into one continuous decision series traded with the live position rules and circuit breakers. | `walkforward.py`, `simulate.py` |
| 2 | `ablation` | Fractional vs no-fractional per window: positive cycles, mean/median ΔSharpe, fraction of windows won, bootstrap CI of the mean ΔSharpe, sign test. | `runner.py`, `bootstrap.py` |
| 3 | `baselines` | Cash, buy & hold, vol-scaled long, momentum, random permutations of the strategy's own forecasts (distribution + percentile), no-fractional model. Same rows, timing and costs. | `baselines.py` |
| 4 | `regimes` | Ex-ante regime tags (trailing 20 vs 200-bar vol, close vs 200-bar mean) and P&L attribution. | `regimes.py` |
| 5 | `cost` | Model cost ×1/×2/×3 and flat 0/1/2/5/10 bps; break-even flat cost. | `stress.py` |
| 6 | `timing` | Execution delayed by +1 / +2 bars. | `stress.py` |
| 7 | `perturbation` | Position-rule parameters ×0.5 / ×2 (cost multipliers, rebalance threshold, max holding, stop, vol cap); collapse detection. | `stress.py` |
| 8 | `d_perturbation` | Fractional order d* ± 1, 2 steps with a light refit per window, compared with the light refit at d* itself. | `stress.py` |
| 9 | `sanity` | Shuffled labels, shuffled features, target shifted +20 bars (light refits), reversed forecasts, random forecasts, zero / double cost. Each has an expectation and a pass flag. | `sanity.py` |
| 10 | `leakage` | Timestamp audit of every OOS row (`feature_available_at <= decision_at <= execution_at`), label alignment against the bar store (label starts at the execution price), entry price = next open. | `sanity.py`, `walkforward.py` |
| 11 | `bootstrap` | Circular block bootstrap of bar returns (Sharpe / CAGR CIs, drawdown distribution), Monte Carlo trade resampling (terminal wealth, P(loss), P(DD > 20%)), multiple-testing bookkeeping (configurations tested, Bonferroni-adjusted p). | `bootstrap.py` |
| 12 | `reproducibility` | Retrain the first window(s) again and compare the fitted model hash and the OOS forecasts bit for bit; `--repro-full` re-runs the whole walk-forward and compares results hashes. Mismatch = `REPRODUCIBILITY FAILURE`. | `walkforward.py`, `manifest.py` |
| 13 | `holdout` | Only with `--open-holdout`: walk the locked final span with the same refit schedule, append an access record (`holdout_access.jsonl`: time, run id, code commit, config / model-config / data hashes, model version, opening number). Re-openings are flagged. | `runner.py` |
| 14 | `gates` | Acceptance gates and classification (below). | `gates.py` |

`--stages quick` runs 1-7, 11, 12, 14 (no light refits). The synthetic ensemble runs
walk-forward, ablation, cost, timing, perturbation, bootstrap, leakage and gates per path.

## No-lookahead engine (spec sections 3-5)

Each decision row carries three timestamps:

* `feature_available_at` — newest information time of the bars used (bar close, or a later quote /
  receipt time in live feeds);
* `decision_at` — the bar close at which the forecast is made;
* `execution_at` — the next bar's open, where the market order fills.

A 30-minute bar closing at 18:30 is followed by the open print at 18:30:00 or later, so the
invariant is `feature_available_at <= decision_at <= execution_at`; the runner raises on any row that
violates it and the leakage stage re-checks it. Latency beyond the boundary is measured by the
+1/+2 bar timing test rather than assumed away. The label of a decision at bar *t* is
`log O[t+1+H] − log O[t+1]`: it starts at the execution price and never touches an earlier bar.
Nothing selected inside a training block (d*, hyper-parameters, calibration) sees its OOS block;
the trainer's own protocol (fold-local d*, chronological calibration, untouched outer holdout) is
described in the README.

## Simulation (spec sections 11-13)

`simulate_strategy` is the one implementation of the trading rules used by model selection,
the OOS evaluation and every stress / sanity variant (a parity test keeps it identical to the
trainer's `simulate_validation`). Market orders fill at the next open; the per-side cost is the
execution bar's spread + range slippage from the cost model (or a flat bps override / scale).
Limit orders are not modelled: the strategy never uses them. Position sizing uses only
information available before execution (calibrated edge, σ at the decision bar, the trailing
σ reference, the estimated round-trip cost). The daily-loss halt (rest of session) and the
drawdown halt (until the next window's model, with the drawdown reference re-based, as the
live `RiskEngine` does after an accepted retrain) are applied when `research.simulation.portfolio_halts`
is true.

`research.walkforward.deploy_policy` selects which forecasts are traded: `always` (every refit,
the research default) or `production` (only accepted models, keeping the previous one
otherwise). Both series are always simulated; the *production policy* overlay in the dashboard
shows the second.

## Metrics (spec sections 14-15, 18)

Per run and per variant: total return, CAGR, annualised volatility, Sharpe, Sortino, Calmar,
max drawdown with duration and recovery, Ulcer index, skew / kurtosis, VaR / ES 95%, worst day
and worst 5-day loss, turnover, costs, gross / net exposure, time invested, trade statistics
(count, win rate, profit factor, expectancy, average win / loss, median, p05 / p95, long / short,
holding-period buckets, exits by reason), P&L by year / month / regime, forecast quality
(directional accuracy, correlation, calibration error), rolling Sharpe statistics.

## Statistical robustness (spec sections 16-17)

Block bootstrap (65-bar blocks by default) gives Sharpe and CAGR confidence intervals and the
probability of a negative Sharpe; the Monte Carlo trade resampler gives the distribution of
terminal wealth and maximum drawdown over paths with the same number of trades. The number
of configurations evaluated on the way to the result (grid × windows × feature sets, plus
`research.multiple_testing.prior_trials` declared by the researcher) is recorded together with a
Bonferroni-adjusted p-value. Trials are not independent, so the adjustment is conservative and
is reported, never used to inflate anything.

## Synthetic engineering validation (spec section 19)

`research-synthetic` generates N paths with varied drift, volatility, autocorrelation, jumps,
volatility regimes, clustering and long-memory strength (path 0 is always a pure random walk)
and runs the pipeline on each. Its output is labelled
**SYNTHETIC / ENGINEERING VALIDATION — NOT PERFORMANCE EVIDENCE** everywhere and checks the
machinery only: no timestamp violations, labels aligned, reproducible, and no edge manufactured
on the random-walk path. Synthetic runs never classify above `EXPERIMENTAL (SYNTHETIC)`.

## Reproducibility (spec sections 25-26)

Every run writes `manifest.yaml`: run id, data hash, whole-config hash, model-config hash
(the sections that define the model), code commit (`-dirty` when uncommitted changes exist),
seed, software environment, schedule and the stages run. Each fitted model has its own
`fitted_model_hash` (trees, logistic weights, calibration map). The `results_hash` covers the OOS
forecasts, the equity curve and the fitted model hashes. `research-runs --compare A B` reports
`IDENTICAL`, `REPRODUCIBILITY FAILURE` (same manifest content, different results) or
`DIFFERENT INPUTS`.

## Run directory (spec section 27)

```
artifacts/runs/<run_id>/
  manifest.yaml      inputs and environment
  summary.json       everything the dashboard shows (metrics, per-window, diagnostics, gates, equity series)
  equity.csv         equity of the fractional / baseline / production series, exposure, drawdown, segment, window
  trades.csv         trade audit trail: decision / execution / exit timestamps, model id, forecast, ER, P(up),
                     confidence, target and approved exposure, prices, bars held, exit reason, gross / cost / net
  fills.csv          one fill per exposure change (submitted_at, filled_at, status, price, slippage)
  decisions.csv      every OOS decision row with its three timestamps, forecasts, sizing, exposure, P&L
  retrains.csv       one row per window: spans, model ids, d*, acceptance, holdout ΔS, fitted model hashes
  models/            metadata + fitted hash of every model (full and baseline, development and holdout)
  diagnostics/       ablation, baselines, cost_curve, timing, parameter_perturbations, d_perturbation,
                     bootstrap, monte_carlo, multiple_testing, regimes, sanity, leakage, reproducibility, gates
  plots/             equity (linear / log, segments), underwater, rolling Sharpe (SVG, no dependencies)
  logs/run.log
artifacts/holdout_access.jsonl    append-only record of every holdout opening
```

## Gates and classification (spec sections 30-31)

Thresholds live in `research.gates` of `strategy.yaml`:

| Gate | Default |
|------|---------|
| OOS windows ≥ | 20 |
| OOS trades ≥ | 200 |
| OOS Sharpe > | 1.0 |
| OOS max drawdown < | 25% |
| OOS profit factor > | 1.2 |
| positive OOS windows > | 55% |
| median ΔSharpe (fractional − baseline) > | 0 |
| windows where fractional beats baseline > | 50% |
| profitable at 2× model cost | required |
| viable at +1 bar delay | required |
| no perturbation collapse | required |
| bootstrap Sharpe CI lower bound > | 0 |
| sanity tests, leakage tests | pass |
| holdout return / Sharpe > | 0 / 0 |
| reproducibility | IDENTICAL |

Classification ladder: **EXPERIMENTAL** → **CANDIDATE** (walk-forward gates) →
**VALIDATED CANDIDATE** (+ ablation, stress, bootstrap, sanity, leakage) → **HOLDOUT PASSED**
(+ holdout opened and profitable) → **PAPER ELIGIBLE** (+ reproducible, and the synthetic
engineering validation when it was run). The dashboard shows the gate table with the value,
threshold and evidence of every gate, and the classification badge.

## Runtime

Every window is a full production retrain (grid × folds × two feature sets + holdout), so a
development run on ~1200 days of 30-minute SPY bars (≈ 40 windows with the default
`retrain_every_bars: 250`) takes on the order of an hour on a laptop; the light-refit stages
(`d_perturbation`, `sanity`) add roughly a third. `--stages quick` skips the light refits, `--fast`
uses the smaller grid, and `research.walkforward.oos_bars` / `step_bars` trade resolution for speed.
The dashboard shows the stage and window progress while a run is in flight.

## Deliberate limitations

* The synthetic demo data has strong exploitable structure by construction; its results mean
  nothing about markets and are labelled accordingly.
* Perturbation and sanity refits use a *light* protocol (no d* search, no grid, chronological
  calibration split) for both the reference and the perturbed variant, so they compare like with
  like but their absolute numbers differ from the full walk-forward.
* Horizon perturbation is not implemented: the label definition depends on the horizon, so it
  would require a full re-run with a different configuration (`--set prediction.horizon_bars=...`).
* Fills are always at the next open with the modelled cost; partial fills and rejections are not
  simulated (the paper-trading mirror reports real Alpaca fills instead).
