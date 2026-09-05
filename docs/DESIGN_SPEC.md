# Fractional-Memory Systematic Trading Bot — Design Specification

## 1. Objective
Build a fully automated trading system for a real-time market simulator whose purpose is to test whether fractional-order transformations of market data provide exploitable information beyond ordinary returns.
The bot shall: consume live OHLCV bar data; transform price and volatility using discrete fractional operators; estimate the current market state; forecast forward risk-adjusted return; produce long, flat, or short positions; size positions according to forecast confidence and volatility; simulate realistic execution costs; retrain automatically using only information available at each historical point; maintain strict separation between data, modeling, signal generation, risk management, and execution; produce enough logging to reconstruct every decision.
The design deliberately avoids discretionary indicators such as RSI/MACD as primary signals. The mathematical core is fractional-order time-series analysis.

## 2. Trading Universe
Use a single highly liquid instrument initially (continuous intraday liquidity, low bid/ask spread, reliable historical data, no illiquid small caps). The software shall support multiple instruments internally, but every model is trained independently per instrument unless cross-asset features are explicitly defined.
Primary bar frequency: **30-minute bars**. Trading session boundaries must be explicitly represented. No synthetic bars may be generated across market closures.

## 3. Information Timing
For each completed bar t the bot receives B_t = (O_t, H_t, L_t, C_t, V_t) plus, if available, Bid_t, Ask_t.
A signal generated using bar t may not trade at C_t. Earliest execution: **opening/tradable price of bar t+1**. This is a hard architectural constraint.
Every feature stored by the system must contain: feature_timestamp, latest_source_timestamp, instrument, feature_name, feature_value. The system rejects any feature where latest_source_timestamp > feature_timestamp.

## 4. Base Price Representation
p_t = log C_t; r_t = p_t − p_{t−1}. All price-domain transformations use log prices.

## 5. Fractional Operator
Discrete Grünwald-Letnikov: D^d p_t = Σ_{k=0}^{K} w_k(d) p_{t−k}, w_0 = 1, w_k = −w_{k−1} (d − k + 1)/k, 0 < d < 1.
Three fixed channels: slow F^(S) = D^{0.25} p, medium F^(M) = D^{0.50} p, fast F^(F) = D^{0.75} p. The system additionally estimates a data-driven order d*_t.

## 6. Fractional Weight Truncation
Weights are generated until either |w_k| < 1e−5 for 10 consecutive terms, or k = 500, whichever occurs first. The resulting kernel length is K(d). Weights must be generated once and cached. No fractional feature is considered valid until all required lag observations exist.

## 7. Adaptive Fractional Order
Candidates d ∈ {0.05, 0.10, …, 0.95}. For each candidate: fractionally transform the training-window log-price series; ADF test; KPSS test; correlation with the original log-price series. Acceptable when p_ADF < 0.05 and p_KPSS > 0.05. Choose d* = argmin d subject to the criteria. If none satisfies both, choose the candidate with the strongest ADF statistic subject to d ≤ 0.95. The selected value remains frozen until the next retraining event. F*_t = D^{d*} p_t.

## 8. Fractional Volatility Transformation
v_t = λ v_{t−1} + (1 − λ) r_t², λ = 0.94; q_t = log(v_t + ε), ε = 1e−12; FV_t = D^{0.40} q_t.

## 9. Primary Feature Vector
At the close of every bar t compute F*_t, F^(S)_t, F^(M)_t, F^(F)_t. Because absolute fractional values can depend on scale and sample history, each is normalized using rolling robust statistics: Z_t(X) = (X_t − median_N(X)) / (1.4826·MAD_N(X) + ε), N = 250 bars.

## 10. Fractional Slope
For every fractional series F: ΔF^(1)_t = F_t − F_{t−1}, ΔF^(4)_t = F_t − F_{t−4}. Use both → eight slope features across the four fractional price channels.

## 11. Fractional Curvature
C_t(F) = F_t − 2F_{t−1} + F_{t−2}, for F*, F^(S), F^(M), F^(F).

## 12. Fractional Cross-Scale Features
X_SM = Z(F^(S)) − Z(F^(M)); X_MF = Z(F^(M)) − Z(F^(F)); X_SF = Z(F^(S)) − Z(F^(F)).

## 13. Conventional Return Features
R_k = p_t − p_{t−k} for k ∈ {1, 2, 4, 8, 16}, each normalized by trailing realized volatility, e.g. NR_{4,t} = R_{4,t} / (σ_{t,50}√4 + ε).

## 14. Volatility Features
σ_{t,N} = StdDev(r_{t−N+1:t}) for N = 10, 50, 200. Features: log σ_10, log σ_50, log σ_200, σ_10/σ_50, σ_50/σ_200, and FV_t.

## 15. Range Features
TR_t = (H_t − L_t)/C_t; Z(TR_t); CL_t = (2C_t − H_t − L_t)/(H_t − L_t + ε).

## 16. Volume Features
LV_t = log(1 + V_t); Z_50(LV_t); ΔLV_t = LV_t − LV_{t−1}. If volume data is unreliable, this module can be disabled globally rather than silently filling values.

## 17. Time Features
m_t minutes since session open, M total session minutes: T_sin = sin(2π m_t/M), T_cos = cos(2π m_t/M). No raw hour-of-day categoricals.

## 18. Regime Estimation
Trend_t = rolling correlation between p_t and bar index over the previous 50 bars. VolRegime_t = σ_20/σ_200. FracRegime_t = |Z(F^(S)_t)| / (|Z(F^(F)_t)| + 0.1). Not converted into discrete labels; supplied directly to the model.

## 19. Forecast Target
H = 4 bars. Executable forward return Y_t = p_{t+5} − p_{t+1}. Normalized: Ỹ_t = Y_t / (σ_{t,50}√4 + ε). Predict ŷ_t = E[Ỹ_t | X_t].

## 20. Prediction Model
Gradient-boosted decision trees, shallow trees, heavy regularization: objective squared error; max_depth 3; learning_rate 0.03; n_estimators selected by validation; min_child_weight/min_samples_leaf high enough to suppress tiny leaves; subsample 0.8; column_sample 0.8; L1 and L2 regularization enabled. XGBoost, LightGBM, or equivalent. The same implementation must be used in backtesting and live simulation.

## 21. Secondary Direction Model
Logistic regression with L2 regularization predicting P(Y_t > 0) on identical features → P⁺_t.

## 22. Prediction Combination
D_t = 2P⁺_t − 1 ∈ [−1, 1]. A_t = M_t · |D_t|. If sign(M_t) disagrees with D_t, A_t = 0.

## 23. Forecast Calibration
Fit a monotonic calibration g(A) → E[Y | A] on validation data (isotonic regression or binned monotonic calibration). Live: E_t = g(A_t). Estimated entirely from training/validation history.

## 24. Transaction-Cost Model
Cost_t = Commission_t + Spread_t + Slippage_t. Spread_t = (Ask_t − Bid_t)/Mid_t if bid/ask available; entry and exit cross half of the spread each → approximately one full spread round trip. Slippage_t = α (H_t − L_t)/C_t, α = 0.05.

## 25. Required Edge
ER_t = E_t σ_{t,50} √H. Trading permitted only when |ER_t| > 3 Cost_t. NE_t = |ER_t| − 3 Cost_t. NE_t ≤ 0 ⇒ TargetPosition = 0.

## 26. Position Direction
ER_t > 3Cost_t ⇒ +1; ER_t < −3Cost_t ⇒ −1; otherwise 0.

## 27. Confidence Scaling
Confidence_t = min(1, |ER_t| / (6 Cost_t + ε)).

## 28. Volatility Scaling
σ_ref = median trailing 30-day intraday volatility. VM_t = clip(σ_ref/σ_{t,50}, 0.25, 1.5).

## 29. Raw Position
Q_t = Direction_t × Confidence_t × VM_t, clipped to [−1, 1]. No leverage.

## 30. Turnover Suppression
Rebalance only when |Q_t − Q_current| ≥ 0.15; otherwise maintain the existing position. If the signal changes sign, rebalance regardless of the threshold.

## 31. Maximum Holding Time
A position may remain open for at most 12 bars. After 12 bars TargetPosition = 0 unless a newly generated signal independently opens another position.

## 32. Stop Mechanism
At entry StopDistance = 4 σ_{t,50} √H. If cumulative mark-to-market return falls below −StopDistance the position is liquidated. Catastrophic-loss guard, not the primary exit.

## 33. Daily Risk Limit
If DailyReturn < −2.5 %: close all positions, prevent new trades for the remainder of the session, record DAILY_RISK_HALT.

## 34. Drawdown Circuit Breaker
DD_t = (E_t − E_peak)/E_peak. If DD_t < −10 %: close all positions and disable trading until a complete model retraining and diagnostic validation succeeds.

## 35. Data-Quality Circuit Breakers
Immediately stop generating new orders on: missing bar, duplicate timestamp, timestamp moves backward, OHLC inconsistency, zero/negative price, extreme unvalidated price jump, stale market feed, unavailable model, feature NaN, feature infinity, fractional kernel incomplete, execution simulator unavailable. Bad input produces no trade; it must never produce a guessed value.

## 36. Training Window
Rolling training window of 10,000 bars. When fewer exist, use all available history provided there are at least 3,000 valid bars. The bot may not operate before that minimum.

## 37. Retraining Schedule
Retrain every 250 new bars: rebuild fractional weights; estimate d*; rebuild all features; fit normalizers; fit regression model; fit logistic direction model; fit calibration; validate the candidate; replace the current model only if validation passes. Live trading continues using the previous accepted model while retraining; replacement is atomic.

## 38. Walk-Forward Validation
Never shuffle. Five chronological folds within the training window: Fold 1 train earliest 40 %, validate next 10 %; Fold 2 train 50 %, validate next 10 %; … Fold 5 train earliest 80 %, validate final 10 %. Purge window H + 1 = 5 bars between training and validation; embargo of 5 additional bars.

## 39. Model Acceptance Criteria
On the aggregated validation sample: directional accuracy > 51 %; Corr(Ŷ, Y) > 0; net simulated P&L > 0; profit factor > 1.05; maximum validation drawdown < 15 %. The combined model must outperform an otherwise identical baseline containing no fractional features on at least three of the five folds. If the candidate fails, retain the previous validated model.

## 40. Fractional Ablation Requirement
Every training cycle also trains a shadow model using returns, volatility, range, volume, time but no fractional features (score S_0). Full model score S_F. ΔS = S_F − S_0. Continue operating when ΔS > 0. If ΔS ≤ 0 for three consecutive retrain cycles, halt new trading and flag FRACTIONAL_EDGE_NOT_DETECTED.

## 41. Hyperparameter Selection
Fixed, small search space. Score = Sharpe_net − 0.25 × TurnoverPenalty − 0.25 × DrawdownPenalty. The same candidates are evaluated on every retraining cycle; no manual tweaking based on backtest results.

## 42. Execution Logic
ΔQ = Q_target − Q_current. Notional_t = Q_t Capital_t; Units_t = Notional_t / EstimatedExecutionPrice. Marketable simulated orders at the first tradable price of the next bar; never assume fills at the signal bar's close.

## 43. Fill Model
Buy: FillPrice = Open_{t+1} + Spread/2 + Slippage. Sell: FillPrice = Open_{t+1} − Spread/2 − Slippage. Applies when opening and closing.

## 44. State Machine
Exactly five states: INITIALIZING, READY, POSITIONED, RISK_HALTED, DATA_HALTED. Transitions explicit: INITIALIZING → READY when enough clean history exists and model validation passes; READY → POSITIONED when target exposure becomes nonzero; POSITIONED → READY when exposure returns to zero; ANY → RISK_HALTED when risk limits trigger; ANY → DATA_HALTED when data integrity fails. The execution engine must reject orders inconsistent with the current state.

## 45. Software Architecture
MarketDataFeed → DataValidator → BarStore → FeatureEngine (FractionalEngine, VolatilityEngine, MarketStateEngine) → ModelEngine → SignalEngine → RiskEngine → ExecutionEngine → PortfolioLedger → AuditLogger. Trainer: HistoricalStore → TrainingDatasetBuilder → FractionalOrderEstimator → WalkForwardValidator → ModelTrainer → ModelValidator → ModelRegistry.

## 46. Fractional Engine API
class FractionalEngine: build_weights(d, threshold=1e-5, max_lags=500); transform(series, d); latest(series, d); estimate_stationary_d(series). Weight calculations unit-tested independently.

## 47. Feature Engine Output
Immutable FeatureVector(instrument, timestamp, fd_adaptive, fd_025, fd_050, fd_075, fd_adaptive_z, fd_025_z, fd_050_z, fd_075_z, fd_slope_1, fd_slope_4, fd_curvature, fd_cross_sm, fd_cross_mf, fd_cross_sf, return_1, return_2, return_4, return_8, return_16, vol_10, vol_50, vol_200, vol_ratio_short, vol_ratio_long, fractional_volatility, range_z, close_location, volume_z, volume_change, trend_state, volatility_state, fractional_state, time_sin, time_cos). No model receives raw market objects.

## 48. Model Output
Prediction(timestamp, expected_normalized_return, expected_raw_return, probability_up, model_confidence, model_version).

## 49. Signal Object
Signal(timestamp, direction, expected_return, estimated_cost, expected_net_edge, confidence, target_exposure).

## 50. Risk Object
RiskDecision(proposed_exposure, approved_exposure, volatility_multiplier, daily_loss_status, drawdown_status, max_holding_status, reason). Execution may consume only approved_exposure.

## 51. Audit Log
For every completed bar store: timestamp, OHLCV, bid, ask, all feature values, fractional d, fractional kernel size, regression prediction, direction probability, calibrated forecast, cost estimate, target exposure, approved exposure, actual exposure, orders submitted, fills, slippage, commission, realized P&L, unrealized P&L, equity, drawdown, model version, risk state. Any trade must be completely reconstructable months later.

## 52. Performance Database
gross return, net return, transaction costs, turnover, Sharpe, Sortino, max drawdown, profit factor, average trade, median trade, win percentage, average winning trade, average losing trade, trade count, average exposure, long performance, short performance, performance by volatility regime, by time of day, by fractional d.

## 53. Research Diagnostics
Curves maintained automatically: d → ADF(d); d → Corr(p, D^d p); d → OOSScore(d); time → S_F − S_0.

## 54. Leakage Tests
Future-bar mutation test; execution-shift test (all fills strictly after the signal timestamp); scaling test (scaling statistics never include validation observations); fractional-order test (d* computed using only the training interval); label isolation test. Failure of any invalidates the backtest.

## 55. Numerical Tests
d = 0 → original series; d = 1 → first differencing; weight recursion vs. direct generalized-binomial calculation; no NaN after warm-up; no infinity; deterministic truncation; same input → identical output.

## 56. Reproducibility
Each model artifact stores: model ID, training start/end, feature schema version, source-data checksum, fractional d, fractional kernel, normalization parameters, model parameters, random seed, validation metrics, software version. All randomness uses a stored seed; a historical model must be exactly reproducible.

## 57. Main Event Loop
on_bar(bar): validate → append → (return if DATA_HALTED / not ready) → features → prediction → signal (with execution cost) → risk decision → build order from current vs approved exposure → queue for next bar → audit record.
on_next_bar_open(bar): for each queued order simulate fill, apply to ledger, record fill.

## 58. Retraining Process
Every 250 bars: history = last 10,000; d* = estimate; dataset = build(history, d*); folds = walk_forward(5, purge 5, embargo 5); boosted, logistic, calibration; candidate = CombinedModel(…, d*); evaluate; baseline without fractional features; evaluate; promote if acceptance tests pass.

## 59. Signal Lifecycle
bar closes → fractional transforms → feature state → forward return estimate → direction confirmation → calibration → cost estimate → edge threshold → confidence → volatility scaling → risk constraints → target exposure → order queued → next bar → spread + slippage → position entered → subsequent bars update → resize / maintain / close.

## 60. Core Hypothesis
Fractional transformations preserve predictive market memory more effectively than ordinary differencing alone. Success requires Performance(X_ordinary + X_fractional) > Performance(X_ordinary) on truly unseen chronological data after simulated trading costs.

## 61. Required Repository Layout
trading_bot/ config/strategy.yaml; data/{feed,validator,store}.py; fractional/{weights,transform,stationarity}.py; features/{price,volatility,volume,regime,engine}.py; models/{regression,direction,calibration,registry}.py; training/{dataset,walkforward,trainer,validation}.py; strategy/{signal,sizing}.py; risk/{manager,limits}.py; execution/{simulator,cost_model,orders}.py; portfolio/{ledger,metrics}.py; diagnostics/{fractional_analysis,attribution}.py; logging/audit.py; tests/{test_fractional,test_features,test_leakage,test_execution,test_risk}.py; main.py.

## 62. Central Configuration
All strategy parameters in a single immutable configuration file (market.bar_minutes 30; fractional fixed_orders 0.25/0.50/0.75, adaptive 0.05–0.95 step 0.05, weight_threshold 1e-5, max_lags 500; prediction.horizon_bars 4; training window 10000, minimum 3000, retrain_every 250, folds 5, purge 5, embargo 5; signal cost_multiplier 3.0, rebalance_threshold 0.15; risk max_absolute_exposure 1.0, maximum_holding_bars 12, daily_loss_limit 0.025, drawdown_halt 0.10; execution slippage_range_fraction 0.05). No strategy constant buried in application code.

## 63. Final Behavioral Definition
Each bar answers: what price memory exists (D^{0.25}p, D^{0.50}p, D^{0.75}p, D^{d*}p); does that state predict future risk-adjusted return (fitted models); is the predicted movement large enough to survive costs (|ER| > 3Cost); how much exposure is justified (confidence, volatility scaling, risk constraints). Q_t = RiskFilter[Direction × ForecastConfidence × VolatilityScaling]. The bot continuously measures whether fractional features add out-of-sample predictive value and halts the strategy if that edge disappears.
