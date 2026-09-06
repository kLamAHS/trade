# Response to the 5 September 2026 source audit

The audit rated the architecture B+/A- but research validity C and deployment safety D, on
seven findings. This document records what changed for each one. Every change ships with a
regression test in `trading_bot/tests/test_audit_round4.py` (and updated tests elsewhere).

| # | Finding | Status | Where |
|---|---------|--------|-------|
| 1 | "Paper-only" GUI could select a live account | **Fixed** | `execution/simulator.py`, `gui/*`, `bot.py`, config |
| 2 | Simulator could fill at a bar's open before the signal existed | **Fixed** | `execution/simulator.py`, `bot.py` |
| 3 | Adaptive order d* leaked future fold information | **Fixed** | `training/trainer.py` |
| 4 | Target and execution horizon did not match | **Fixed (label) / documented (re-targeting)** | `training/dataset.py`, README |
| 5 | Hyperparameters selected and accepted on the same folds | **Fixed** | `training/trainer.py`, `training/validation.py` |
| 6 | Reproducibility claims exceeded dependency/versioning support | **Fixed** | `requirements.lock`, `training/trainer.py`, `models/combined.py` |
| 7 | No CI, no published results | **CI added**; results still to be produced on real data | `.github/workflows/ci.yml` |

## 1. Live trading is impossible from this application

* `AlpacaPaperBroker` raises `LiveTradingNotSupported` unless `paper is True`, and the only
  constructor of an Alpaca `TradingClient` in the code base (`paper_trading_client`) forces the paper
  endpoint and verifies the client's base URL contains `paper`.
* `TradingBot` refuses any configuration with `alpaca.paper` not equal to `true`; the dashboard's
  `build_config` does the same after applying overrides.
* The dashboard no longer has a paper/live toggle: `paper` was removed from the settings model,
  incoming values are ignored, and the page states that live trading is not implemented.
* The connection test always reports the **PAPER** account.

Real-money execution, if ever wanted, would be a separately named mode with its own safeguards; it is
not a flag.

## 2. Fills use only prices observed at or after the decision

`ExecutionSimulator.simulate_fill` now applies one rule: the fill price must have been observed at or
after the order's decision time.

* Bar backtest: the decision is stamped at the bar close, which is the next bar's open print, so the
  next open is the first tradable price (spec section 42). Unchanged.
* Live paper mode: every bar emitted by the Alpaca feed carries its receipt time (`Bar.observed_at`),
  and a decision is stamped no earlier than that, so the next bar's open print (at the close) is never
  an available price for it: no quote, a stale quote timestamp or a delayed delivery cannot fall back
  to the close. The fill uses the standing NBBO fetched with the bar (buy at the ask, sell at the bid,
  plus range-based slippage), stamped at the fetch time, `price_source = "quote"`. Without a quote the
  order is **deferred** (`FillDeferred`, audited as `ORDER_DEFERRED`) to the first bar whose open is
  after the decision. Flatten orders (data halts, rejected bars) are stamped with the same decision
  time, a same-direction max-holding re-entry is booked at the next bar's open (as in the validation
  simulator), and a bar delivered after its successor opened is a catch-up bar that may not trade.
* `execution.live_fill_source: broker` makes the ledger use the actual Alpaca fill price when the
  mirrored order has filled (`price_source = "broker"`); the default `quote` keeps paper and backtest
  accounting on the same model. A deferred order keeps its broker acknowledgement across attempts, and
  a flip's filled quantity is the sum of its closing and opening legs.
* The old regression test that approved a fill at an earlier open was replaced by tests of the new rule.

## 3. d* is estimated per fold from the fold's own training block

`ModelTrainer.build_fold_sets` estimates d* for each walk-forward fold on the window's log prices up to
the newest bar that enters any *training* label of that fold, rebuilds the feature matrix with that d,
and trains/validates the fold on it. The first fold's inner calibration split estimates its own d* on
the inner-fit part only, so the calibration pairs are out of sample in every respect. The outer holdout
model always uses a d* estimated on the inner block (independently of `training.fold_local_d`, which
only controls the inner folds), and the **deployed** model is refit on the whole window with that same
holdout-validated d*; the whole-window d* is recorded for diagnostics (`d_full`) but never deployed.
The leakage test asserts that a fold's d* is unchanged when every bar after its training block is mutated.

## 4. The label is the executable open-to-open return

`prediction.label_price: open` (default) defines `Y_t = log O_{t+1+H} - log O_{t+1}`: entry at the
first tradable price after the signal, exit `H` bars later at an open. `close` restores the spec's
literal close-to-close formula. By default the strategy still re-evaluates the H-bar forecast every
bar and re-targets exposure (turnover-suppressed, 12-bar maximum holding), so the realised holding
period is shorter than H; the validation report now quantifies this (`mean_bars_held`,
`edge_to_realised_corr`). `signal.reevaluate_every_bars: 4` (= H) switches to hold-to-horizon: an
open position is re-targeted only at H-bar boundaries, making the traded return the trained one. The
validation simulator applies the same *position-level* rules as the live bot (sizing, turnover
suppression, stop, max holding, re-evaluation cadence); the portfolio-level circuit breakers (daily
loss, drawdown, data halts) are not simulated.

## 5. An untouched outer holdout decides acceptance

`training.outer_holdout_fraction` (default 15 %) removes the newest rows of the training window from
everything that is selected: fold construction, d*, hyperparameter grid, calibration. The best full
candidate and the best baseline candidate (each best-of-grid on the inner folds) are refit on the
inner block, calibrated on inner out-of-fold predictions and scored once on the holdout. The section 39
metrics are read from that holdout; `S_F - S_0` for the ablation halt is the holdout difference; and
`training.acceptance.require_holdout_edge` (default true) additionally requires that difference to be
positive for promotion. The fold-level "beats the baseline on 3 of 5 folds" criterion of the spec is
kept on the inner folds. Setting the fraction to 0 restores fold-only evaluation. Because consecutive
retrains' holdouts overlap heavily, a non-positive ΔS counts as a new ablation failure only when its
holdout does not overlap the last counted failure's (overlapping repeats are audited, not counted).

## 6. Reproducibility

* `requirements.lock` pins the exact environment the test suite was run in; `pip install -r
  requirements.lock && pip install -e . --no-deps` reproduces it.
* Every model artifact records `environment`: Python version and implementation, platform, machine,
  the versions of numpy / pandas / scipy / scikit-learn / lightgbm / statsmodels / PyYAML / joblib, the
  git commit of the repository that contains the package (with a `-dirty` suffix for uncommitted
  changes; `nogit` for wheel installs) and the software version. These enter the model-id hash, so an
  artifact produced under a different environment gets a different id. The lock file is for Python
  3.11 (the pinned numerical stack requires it, so the project now requires 3.11) and carries the
  Windows-only markers; both launchers install from it.

## 7. CI and evidence

`.github/workflows/ci.yml` runs the full test suite and a CLI smoke backtest on every push and pull
request with the pinned dependencies; the smoke step uses `--require-model` and asserts that a model
artifact with its environment record and holdout metrics was written, so it cannot pass vacuously. No empirical claim about SPY is made in this repository: the
synthetic generator contains a stationary long-memory component by construction and is a pipeline
test, not evidence. Producing walk-forward results on real 30-minute data, with the holdout and the
fold-local d* in place, is the next step and is deliberately left to a run the user controls.
