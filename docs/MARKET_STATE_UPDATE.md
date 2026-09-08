# Market-state, microstructure and execution intelligence update

This document describes how the design spec *Market-State, Microstructure, and Execution
Intelligence Update* is implemented, which parts of it the available data can support, and
how every new component is made to **earn its place** through out-of-sample ablation before
it is allowed near the trading decision.

The guiding rule of the update is unchanged from the validation framework: a component that
does not improve the net-of-cost, out-of-sample, latency-robust edge is removed. All new
feature families are therefore **off by default** (`features.*.enabled: false` except
`fractional`) and the research pipeline reports, per family, whether it is `ACCEPTED`,
`REJECTED` or `RESEARCH ONLY`.

```
python -m trading_bot.main research --synthetic 4000 --hostile --symbol SYN --fast \
    --set features.regime.enabled=true --set features.ofi.enabled=true \
    --set features.cross_asset.enabled=true --set features.cross_asset.symbols=[QQQ,IWM]
python -m trading_bot.main research --csv artifacts/data/SPY_30m.csv \
    --context QQQ=artifacts/data/QQQ_30m.csv --context IWM=artifacts/data/IWM_30m.csv \
    --set features.cross_asset.enabled=true --set features.cross_asset.symbols=[QQQ,IWM]
```

## 1. What the data can and cannot support

| Tier | Content | Available from | Enables |
|------|---------|----------------|---------|
| A | OHLCV time bars | Alpaca history, CSV, synthetic | PRICE, VOLATILITY, VOLUME, FRACTIONAL, REGIME, KALMAN, CROSS_ASSET, event bars, execution model without quote inputs |
| B | A + bar-close NBBO snapshot with **displayed sizes** | Alpaca live loop (`latest_quote_with_sizes`), hostile synthetic market, CSV with `bid_size`/`ask_size` | OFI, quote imbalance, spread-aware execution model, OFI special gate |
| C | Full order book / trade tape | not available | nothing is approximated from it |

`BarStore.data_tier()` reports the tier of the stored history. A family whose inputs are absent
is reported as **unavailable** (`FeatureEngine.available_families`) and its columns are NaN;
it is never synthesised. In particular OFI is never manufactured from Tier A data
(`test_ofi_formula_and_tier_requirement`), VPIN is disabled when bulk-volume classification
confidence is below `features.vpin.min_classification_confidence`, and the TOXICITY family is
marked experimental in the configuration.

## 2. Causality invariants (spec sections 3-5, 8, 12, 15, 29)

Every feature family shares the invariants of the validation framework:

* `feature_available_at <= decision_at <= execution_at` for every row (the leakage stage refuses
  violations); every feature column is **prefix-invariant** (the value at bar *t* computed on
  bars `0..t` equals the value computed on the full history) and the forward-mutation test
  rewrites all bars after a decision and requires bit-identical forecasts.
* **Regime (HMM).** `features/hmm.py` fits a Gaussian HMM on the training block by EM and
  emits **filtered** state probabilities only. The smoother (`smooth`) exists for the test
  that proves it leaks (`test_hmm_filter_is_causal_and_smoother_is_not`), and the schema
  forbids it as a feature (`causal_filter_only`). The stress state is the highest-variance
  state; `regime_stress_p` is an auxiliary column used by sizing and the execution model.
* **Kalman.** `features/kalman.py` is a local-linear-trend filter fitted per training block
  (maximum innovation likelihood over a grid scaled to the sample return variance). Features
  come from the **filter** only; no RTS smoothing enters a feature.
* **OFI.** `features/ofi.py` implements the Cont-Kukanov-Stoikov bar-to-bar order-flow
  imbalance from consecutive NBBO snapshots and their displayed sizes, normalised by displayed
  depth (`normalization: displayed_depth`). It requires Tier B data.
* **Cross-asset.** `features/cross_asset.py` joins context instruments with a strict causal
  **as-of** join (`asof_indices`): the newest context bar whose availability time is not after
  the decision time; rows older than `max_staleness_seconds` are treated as missing and the
  age is exported as an auxiliary column (`xa_<symbol>_age_s`).
* **Event bars.** `data/bars.py` aggregates fine time bars into volume / dollar / tick bars
  that never straddle a session boundary and carry `end_time`, `duration_seconds`,
  `dollar_volume` and `trade_count`; features use `available_at = end_time`.

Fitted components (HMM, Kalman, VPIN thresholds) are **frozen** per training window and
recorded in the model hash (`components_digest`) so a fold never sees parameters fitted on
its own validation block (`training.fold_local_components`).

## 3. Feature families (spec sections 8-31)

| Family | Model inputs | Requires | Notes |
|--------|--------------|----------|-------|
| PRICE | `return_{1,2,4,8,16}`, `range_z`, `close_location`, `time_sin/cos` (+ `duration_z`, `return_per_time`, `volume_intensity` on event bars) | A | core |
| VOLATILITY | `vol_{10,50,200}`, `vol_ratio_short/long`, `trend_state`, `volatility_state` | A | core |
| VOLUME | `volume_z`, `volume_change` | A | core |
| FRACTIONAL | 21 fractional-memory features (z-scored levels, slopes, curvature, cross terms, `fractional_volatility`, `fractional_state`) | A | optional; the original ablation |
| REGIME | `regime_p_k`, `regime_entropy`, `regime_age`, `regime_transition_prob`, `regime_most_likely` | A | filtered HMM only |
| OFI | `ofi_normalized`, `ofi_ema_short/long`, `ofi_zscore`, `ofi_acceleration`, `ofi_price_divergence`, `quote_imbalance`, `ofi_spread_zscore` | B | special gate |
| KALMAN | `kalman_price`, `kalman_velocity`, `kalman_acceleration`, `kalman_innovation(_z)`, `kalman_state_uncertainty`, `kalman_observed_minus_filtered` (subsets `latent_price`, `innovation`, `full`) | A | filter only |
| CROSS_ASSET | per symbol `rel_return_{1,4}`, `divergence`, `corr_short`, `corr_breakdown`, `rel_volume`, `vol_divergence`; `xa_breadth`, `xa_dispersion` | A + context | as-of join |
| TOXICITY | `vpin`, `vpin_zscore`, `vpin_change`, `vpin_percentile` | A (bulk volume classification) | experimental |

The schema (`features/schema.py`, version 2.0.0) records the family of every column; the
trainer's model inputs are `schema.names_for(families_available)` so the fitted model only
ever sees columns the data supports.

## 4. Execution intelligence (spec sections 1-3, 17-21)

The decision rule is

```
expected_net_edge = expected_gross_return
                  - spread_cost - slippage - adverse_selection - fees - uncertainty_buffer
trade only if expected_net_edge > minimum_required_edge, otherwise ABSTAIN
```

implemented by `strategy/policy.py` (`DecisionPolicy`). `ABSTAIN` is a first-class outcome
recorded in every decision. `execution.decision_policy: legacy` restores the original
`|ER| > k x cost` rule bit for bit (`test_legacy_policy_matches_the_original_sizing_rule`).

* **Adverse selection** (`execution/estimator.py`). A ridge model predicts the post-fill move
  against the position beyond what the forecast explains, `|ER|/H - direction x r_fill` with
  `r_fill = (C[t+1] - O[t+1]) / O[t+1]`, from `EXEC_INPUT_NAMES` (spread, range, sigma,
  volume z, time of day, stress probability, quote imbalance, OFI z-score, volatility state).
  It is fitted on **out-of-fold** rows only, its predictions are floored at
  `adverse_selection_floor_bps` (the model can add cost, never edge), and its residual
  standard deviation feeds the uncertainty buffer (`uncertainty_z`).
* **Cost used by the decision vs cost charged by the simulator.** `ExecutionEstimate` is what
  the strategy *believes*; the fill model charges what the bar *shows*. The two are kept
  apart in every simulator (`cost_scale` moves the belief, `exec_cost_scale` moves the
  charge) and reconciled in the execution calibration.
* **Regime-aware sizing** (`risk.regime_adjustment`). When the filtered stress probability
  exceeds the threshold, exposure is capped, the required edge is raised and confidence is
  scaled. The regime modifies sizing, never the algorithm.
* **Execution calibration** (`execution/calibration.py`). Every fill records expected vs
  realised spread, slippage, adverse selection and total in bps, the decision-to-fill move,
  mean / median / P90 / P95 / worst per component, the share of fills inside the expected P95,
  a calibration table by predicted-cost bucket and a split by price source (simulated /
  NBBO / broker). Paper mode therefore compares the model's expectation with Alpaca's fills.

## 5. Market representation (spec sections 6-7)

`market_representation.bar_type` selects `time | volume | dollar | tick` bars built from the
feed's time bars (`data/representation.py`). With `candidates` set (for example
`[time_30m, volume_500k, dollar_250m]`) the walk-forward runner runs a
`RepresentationSelector`: for every window, each candidate is trained on the training block
only, the one with the best inner-fold full-model score is chosen **ex ante**, and only that
candidate's OOS block enters the series (`summary["representation"]`). The choice is
recorded per window; the comparison of all candidates is diagnostic only.

## 6. Research stages added (spec sections 32-47)

| Stage | Output | Module |
|-------|--------|--------|
| `execution_stress` | Three separate experiments: **adaptive** (the strategy is told cost x0.5..x3 and re-decides), **frozen** (decisions fixed, fills charged x0.5..x3, break-even multiplier), **delay** (execution +0..+3 bars). `frozen_survives_2x/3x`, `adaptive_survives_2x`, `delay_survives_plus_1/2`. | `research/stress.py` |
| `families` | **Ablation matrix** (Base = core families; Base+F; Full-F; Full) with ΔSharpe, Δreturn, Δdrawdown, Δturnover, Δcost, Δcalibration (forecast correlation), Δaccuracy and window-level statistics (median ΔSharpe, positive-window fraction, bootstrap CI, dominance of the best window); **conditional contribution** by ex-ante regime tag (volatility, trend, and the filtered stress probability when the REGIME family is on); **sabotage failure tests** per family (`SABOTAGE_MODES`: shuffle, freeze, random, flip, delay, remove; stochastic modes averaged over `research.families.sabotage_seeds` draws) which must all remove the family's benefit, in Sharpe and in forecast correlation with the realised label; **complexity budget** (features, parameters, data dependency and tier availability, compute time per bar, implementation risk, additive and marginal incremental Sharpe, incremental turnover and cost, sabotage result); **family gates**. | `research/families.py` |

A family is `ACCEPTED` only when its marginal contribution (Full vs Full-F) has a positive
median ΔSharpe across windows, wins more than `min_positive_fraction` of the windows, **every
sabotage of its inputs removes its benefit** (a family whose sabotage leaves the fitted model
as good or better is noise the model happens to key on) and its special gate passes; the
report also shows whether the bootstrap CI excludes zero, whether a single window dominates
and whether the additive contribution (Base+F vs Base) survives +1 bar of latency and 2x
realised cost. With few windows all of these are noise: the `min_windows` gate of the
validation framework still applies before any family verdict is evidence. Special gates:

* **OFI**: quote timestamp quality, displayed sizes present, staleness, feature latency and
  modelled execution latency; failing it makes OFI `RESEARCH ONLY` rather than tradable.
* **REGIME**: the HMM must be fitted, its filter must be prefix-invariant on the stored
  history, state persistence and occupancy must be reasonable, and profit must not be
  concentrated in a rare state.
* **CROSS_ASSET**: the strategy must degrade gracefully when the context is delayed by one
  or two bars or removed (Sharpe must not collapse below half of the reference). The as-of
  join itself is covered by `test_causal_asof_join_never_takes_the_future_and_flags_staleness`,
  and `test_lead_lag_benefit_disappears_without_the_relation` shows the family's benefit
  vanishes when the synthetic lead / lag relation is removed.

## 7. Hostile synthetic market (spec section 48)

`data/synthetic.py::generate_synthetic_market` produces correlated instruments with a Markov
regime sequence (calm / trend / stress / mean-revert), stochastic volatility, jumps, opening
gaps, Student-t innovations, liquidity (spread, range, quote size) that deteriorates with
stress, a correlation that breaks down in stress, an optional lead / lag relation between a
leader and a follower, informative displayed quote sizes, volume bursts and per-instrument
Tier B quotes. The generating truth (regime path, volatility, jumps, correlation) is
returned with the bars so engineering checks can compare estimates with it. `--hostile`
selects it on the CLI; it is used automatically when the primary is synthetic and
`features.cross_asset.symbols` is set. As with all synthetic data, results are labelled
*engineering validation, not performance evidence*.

## 8. Decision audit record and dashboards (spec sections 49-53)

Each bar record in `artifacts/audit/<run>_bars.jsonl` now carries: the market state (regime
probabilities, entropy, age and stress probability, OFI z-score and quote imbalance, Kalman
innovation and velocity, cross-asset relative returns, liquidity including VPIN when
available, conventional trend / volatility states), the feature families the model used, the prediction, the
`ExecutionEstimate` (spread, slippage, adverse selection, fees, uncertainty, total, net edge)
and the policy decision (`trade` / `ABSTAIN`, reason, exposure cap multiplier). Realised
execution is appended as an `EXECUTION_REALISED` event when the fill arrives.

Dashboard panels (`gui/static/index.html`):

* **Trading tab** — *Expected edge* waterfall (gross → spread → slippage → adverse → fees →
  uncertainty → net, against the minimum required edge), *Market intelligence* (regime
  probabilities, stress, OFI, Kalman, cross-asset, data tier, available families),
  *Execution: expected vs realised* (calibration table, P95 coverage, recent fills).
* **Validation tab** — *Feature families* (ablation matrix, gates, verdicts, sabotage results)
  and *Execution stress* (adaptive vs frozen vs delay curves with break-even multipliers).

## 9. Configuration

```yaml
market_representation: {bar_type: time, target_minutes: 30, target_shares: 500000,
                        target_dollar_volume: 250000000, target_ticks: 20, candidates: []}
features:
  fractional: {enabled: true}
  regime:     {enabled: false, states: 3, causal_filter_only: true}
  ofi:        {enabled: false, normalization: displayed_depth}
  kalman:     {enabled: false, subset: full}
  cross_asset: {enabled: false, symbols: [], max_staleness_seconds: 1800}
  vpin:       {enabled: false, bucket_bars: 5, n_buckets: 50, min_classification_confidence: 0.15}
execution:
  decision_policy: net_edge            # or legacy
  minimum_net_edge_bps: 8
  uncertainty_buffer_bps: 3
  uncertainty_z: 0.0
  fee_bps: 0.0
  adverse_selection_model: true
  adverse_selection_floor_bps: 0.0
  adverse_selection_ridge_alpha: 1.0
risk:
  regime_adjustment: {enabled: false, stress_probability_threshold: 0.70,
                      exposure_multiplier: 0.40, edge_multiplier: 1.5, confidence_multiplier: 1.0}
training:
  fold_local_components: true
research:
  families: {n_boot: 1000, sabotage_seeds: 3,
             gates: {min_median_delta_sharpe: 0.0, min_positive_fraction: 0.55, max_dominance: 0.5}}
```

On the command line, `--context SYMBOL=path.csv` (repeatable) supplies real context histories and
`--hostile` selects the multi-asset synthetic market; in the dashboard the *Context CSVs* setting
does the same for CSV runs, and synthetic runs generate the configured context instruments.

Model metadata records `feature_families`, `components` (HMM / Kalman parameters),
`components_digest`, the execution model (coefficients, intercept, residual std) and the
`data_tier` of the training data; all of them enter the fitted-model hash used by the
reproducibility check.

## 10. Why a retrain is rejected, and how to read it

A rejected model leaves the bot in `INITIALIZING` with no position and no orders. That is the designed
behaviour, but with no explanation it is indistinguishable from a stall, so every retraining cycle now
records **why** its acceptance sample did or did not trade (`TrainingReport.holdout_diagnosis`,
`TradingBot.retrain_status()`, the *Model status* card, and `last_retrain` in the run summary).

An acceptance sample that takes no trades produces a row of zeros: net P&L exactly 0, profit factor
exactly 0, and an ablation delta of exactly 0 because the full and baseline models both score 0.000.
Four acceptance checks then fail for what is really one reason. The diagnosis separates the two causes:

* **The calibrator collapsed to a constant.** Isotonic calibration is monotone by construction, so when
  the pooled out-of-fold predictions carry no increasing relation to the label it fits a constant. Every
  bar then gets the same expected return, and no decision policy can produce a signal. This is the honest
  verdict of a model without edge, not a numerical failure; the diagnosis reports the calibration
  correlation and the number of distinct forecast values behind it.
* **The forecasts are real but never clear the trade threshold.** Reported as the |ER| distribution
  (median, p90, max, in bps) against the required edge in bps, plus the share of rows that clear it.

Both are visible before the acceptance table, so the numbers say whether the strategy has no signal or
a signal too small to pay for its own execution.

## 11. Known limits

* Intrabar quote updates are not observed: OFI and quote imbalance are computed from
  bar-close NBBO snapshots, which the OFI special gate states explicitly.
* VPIN uses bulk-volume classification on OHLCV bars; without a trade tape its
  classification confidence is measured and the family is disabled below the threshold.
* Event bars are built from 30-minute time bars, so `target_ticks` counts fine bars, not
  trades; finer source bars (`market.bar_minutes`) sharpen them.
* The adverse-selection model is linear (ridge) by design: it must be fitted on the few
  out-of-fold rows a window provides and must never add edge; a richer model is a research
  question for the ablation matrix, not a default.
* A context feed outage in paper mode makes the cross-asset columns non-finite; the bot treats
  that like any other non-finite feature (a data halt until `data.halt_recovery_bars` clean
  bars), so the strategy never trades on a partially available feature vector.
