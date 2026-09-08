# Pooled training across instruments (the training panel)

The bot trades **one** instrument. The training panel does not change that. It changes how many rows
the models are *fitted* on, by adding rows from other symbols that carry the same features and the
same scale-free label.

```bash
# real data: SPY is traded, the other four only enlarge the training sample
python -m trading_bot.main backtest --csv artifacts/data/SPY_30m.csv --symbol SPY \
    --panel QQQ=artifacts/data/QQQ_30m.csv --panel IWM=artifacts/data/IWM_30m.csv \
    --panel DIA=artifacts/data/DIA_30m.csv --panel MDY=artifacts/data/MDY_30m.csv \
    --set training.panel.enabled=true --set training.panel.symbols=[QQQ,IWM,DIA,MDY]

# synthetic: the correlated instruments of the hostile market become the panel automatically
python -m trading_bot.main backtest --synthetic 4000 --symbol SPY --fast \
    --set training.panel.enabled=true --set training.panel.symbols=[QQQ,IWM,DIA]
```

In the dashboard the *Panel CSVs* setting does the same for CSV runs, and synthetic runs generate the
configured instruments.

## Why

One instrument gives one experiment. Whatever a walk-forward concludes about SPY on 30-minute bars is
a single, non-replicable result: it cannot separate "the strategy works" from "this symbol behaved
that way over this span". Pooling addresses two separate problems.

* **The fit sees more rows.** A 4000-bar window with overlapping 4-bar labels is a thin sample for a
  39-feature boosted tree plus a logistic model plus a calibrator. Panel symbols multiply the rows for
  the same calendar span instead of reaching back into a different market regime.
* **The result can be checked against instruments it will not trade.** The holdout model is applied to
  the panel over the same span and its forecast quality is reported per symbol. A healthy primary
  correlation beside a pooled correlation near zero is the signature of a result that will not repeat.
  This is the better-powered test, and it is the main reason to turn the panel on.

## What is pooled, and what is not

| | Primary instrument | Panel instruments |
|---|---|---|
| Model fits (regression, direction) | yes | **yes** |
| Fold validation blocks | yes | no |
| Outer holdout and acceptance criteria | yes | no |
| Simulated P&L, ablation delta, gates | yes | no |
| Orders, positions, ledger, risk limits | yes | **never** |
| Out-of-sample forecast-quality diagnostic | yes | yes (reported separately) |

Acceptance therefore remains a statement about the instrument that will actually be traded. The panel
can only change *which model* is offered for acceptance, never the standard it is held to.

## The three properties that make pooling legitimate

1. **The label is already scale free.** `Y = (log P_{t+1+H} - log P_{t+1}) / (sigma_{t,50} sqrt(H))` is
   measured in each symbol's own volatility units, so a quiet index and a noisy mid-cap contribute
   comparable targets without any extra normalisation.
2. **Rows are pooled causally.** A panel row may join a fit only once the newest bar entering its label
   has closed no later than the newest bar entering the primary training block's label. Feature columns
   are prefix-invariant within each symbol, so filtering by label completion is sufficient for them.
   Fitted components are not prefix-invariant, so when any parametric family (REGIME, KALMAN, TOXICITY)
   is enabled each panel symbol's history is truncated at the cutoff before its components are fitted.
   A symbol whose history lies entirely after the primary's cutoff contributes nothing, and the fitted
   model is then identical to training with no panel at all.
3. **Level-dependent features are aligned.** Most features are ratios or rolling robust z-scores and are
   already comparable, but a few (`vol_10/50/200`, `fractional_volatility`) sit at a level that
   identifies the symbol. Left alone, a tree can split on that level and silently un-pool the data. Each
   panel symbol is mapped onto the primary's training distribution feature by feature, using statistics
   from eligible rows only. **The primary's own rows are never rescaled**, so the deployed model reads
   the live feature vector exactly as it is computed. Disable with `align_features: false` to measure
   what the alignment is worth.

A symbol that cannot produce the primary's exact feature columns (a family its data tier cannot support,
a cross-asset context the primary does not share) is skipped rather than padded: a pooled row has to mean
the same thing as a primary row, column for column.

## Configuration

```yaml
training:
  panel:
    enabled: false
    symbols: []             # e.g. [QQQ, IWM, DIA, MDY]
    align_features: true
    max_symbols: 20
    max_rows_per_fit: 0     # 0 = no cap; otherwise thin the pooled rows deterministically
```

`max_rows_per_fit` bounds cost: each fit grows with the pooled rows, and each distinct fractional order
across the folds triggers one feature build per panel symbol. With parametric families enabled the panel
is additionally rebuilt per cutoff, so keep `max_symbols` modest there.

## Reading the output

Every retrain logs the panel and records it in `TrainingReport.panel_diagnosis`, in
`TradingBot.retrain_status()`, in the run summary under `last_retrain.panel`, and on the dashboard's
*Model status* card:

```
training panel: 9780 pooled rows from 4 instruments joined the final fit of 2045 primary rows
panel out of sample: 1248 rows across 4 instruments, correlation -0.0154, accuracy 0.511
```

Read the second line first. It is the honest, better-powered answer to "does this feature set carry any
generalisable signal?", and it is independent of whether the primary happened to pass its gates.

## What pooling does not do

It does not manufacture edge. If the panel's out-of-sample correlation is flat, the pooled fit will
usually *dilute* a primary result rather than improve it, because the extra rows carry no signal to
learn from. That dilution is information, not a malfunction: it says the primary result rested on one
symbol over one span. Reach for a different horizon, a different instrument or a cross-sectional
formulation rather than turning the panel off to get the old number back.
