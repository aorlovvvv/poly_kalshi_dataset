# Quant Autoresearch: Autonomous Tail-Risk Strategy Optimization

## What This Is

This is **not** a paper evaluation tool. This is an autonomous research engine that sits inside the paper's methodology section. The paper will describe this loop as:

> "We applied an automated hypothesis-testing framework (adapted from Karpathy 2025) to systematically explore the space of microstructure-based tail risk predictors. Over N iterations, the optimizer evaluated M feature combinations, K model architectures, and J threshold/horizon configurations, selecting the configuration that maximized out-of-sample risk-adjusted returns on a temporal hold-out set."

The autoresearch loop produces **the actual RQ3 results**: the best tail predictor, its SHAP decomposition, the calibration curves, and the backtest performance. These go directly into the paper as tables and figures.

---

## The Three Files

| File | Role | Editable? |
|------|------|-----------|
| `src/autoresearch/backtest.py` | **Fixed backtester + evaluator.** Loads temporal hold-out data, runs the agent's strategy, computes Sharpe/Sortino/drawdown/calibration. Defines the tail event label. Defines transaction costs. This is `prepare.py`. | **NO** |
| `src/autoresearch/strategy.py` | **Agent's playground.** Feature engineering, model choice (XGBoost/LightGBM/RL/linear), hyperparameters, threshold selection, position sizing logic. This is `train.py`. | **YES** |
| `src/autoresearch/program.md` | **Agent playbook.** | Reference |

---

## The Fitness Metric: Out-of-Sample Sharpe Ratio

A single scalar. Higher is better.

```
fitness = sharpe_ratio(strategy_returns, risk_free=0)
```

Where:
- `strategy_returns` = daily P&L of the tail-risk-adjusted trading strategy on the **temporal test set** (Sep–Nov 2025)
- The strategy uses the agent's model to predict tail events, then sizes positions accordingly
- Transaction costs are applied at realistic rates (Kalshi: 0.6% per contract, Polymarket: 0.02%)

**Secondary metrics** (reported but not optimized):
- Sortino ratio
- Maximum drawdown
- Out-of-sample AUC for tail prediction
- Brier score (calibration)
- SHAP feature importance ranking

**Why Sharpe, not AUC?** AUC measures statistical discrimination. Sharpe measures whether the predictions are economically useful after costs. A model with AUC 0.60 but terrible calibration might trade too often and bleed on transaction costs. Sharpe forces the agent to find signals that are both statistically real and economically exploitable — the core question of RQ3.

---

## What the Fixed Backtester Does (backtest.py)

### 1. Data Splits (Deterministic, Immutable)

```
TEMPORAL:
  Train:  all trades before 2025-06-01
  Val:    2025-06-01 to 2025-08-31
  Test:   2025-09-01 to 2025-11-25

MARKET SELECTION:
  Top 100 Kalshi markets by trade count (within overlap period)
  Top 100 Polymarket markets by trade count
  Seed = 42, deterministic ordering
```

### 2. Tail Event Definition (Fixed)

```python
# A tail event at trade t is: |Δp_t| > VaR_99(training set)
# VaR_99 computed once from training data, frozen for val+test
tail_threshold_kalshi = np.quantile(np.abs(train_kalshi_returns), 0.99)
tail_threshold_poly = np.quantile(np.abs(train_poly_returns), 0.99)
```

### 3. Strategy Execution (Fixed Protocol)

The backtester calls the agent's strategy at each time step:

```python
for t in test_timestamps:
    features_t = compute_features(data, t)  # fixed feature extraction

    # Agent's model predicts tail probability
    tail_prob = strategy.predict_tail_probability(features_t)

    # Agent's position sizing function
    position = strategy.size_position(tail_prob, current_position, portfolio_state)

    # Backtester applies transaction costs and computes P&L
    pnl_t = execute_trade(position, price_t, costs)
    daily_pnl[date(t)] += pnl_t
```

### 4. Evaluation Metrics (Fixed)

```python
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
sortino = mean(daily_returns) / downside_std(daily_returns) * sqrt(252)
max_dd = max_drawdown(cumulative_returns)
auc = roc_auc_score(test_labels, test_predictions)
brier = brier_score_loss(test_labels, test_predictions)
```

### 5. Transaction Cost Model (Fixed)

```python
KALSHI_FEE_RATE = 0.006      # 0.6% per contract
POLYMARKET_FEE_RATE = 0.0002  # 0.02% + gas (amortized to ~0.02%)
SLIPPAGE_BPS = 5              # 5 bps slippage per trade
```

---

## What the Agent Iterates On (strategy.py)

The agent's editable file must implement a `Strategy` class:

```python
class Strategy:
    def __init__(self):
        # Agent defines: model, features, hyperparameters, thresholds
        pass

    def train(self, train_features, train_labels, val_features, val_labels):
        # Agent implements: model training, feature selection, CV
        pass

    def predict_tail_probability(self, features):
        # Agent implements: model inference → P(tail event)
        pass

    def size_position(self, tail_prob, current_position, portfolio):
        # Agent implements: how to translate tail prediction into position
        # Examples:
        #   - Reduce position proportionally: pos *= (1 - tail_prob)
        #   - Binary: if tail_prob > threshold → exit entirely
        #   - Kelly criterion: size based on expected edge
        pass
```

### Dimensions the Agent Explores

**Feature Engineering (highest impact):**
- Lagged returns (how many lags? 1, 5, 10, 20?)
- Volatility windows (5-min, 1-hour, 1-day rolling std)
- Order imbalance (buy_fraction in last N trades)
- Temporal features (hour, day-of-week, distance-to-event-resolution)
- Spread estimate (Roll's implied spread, rolling)
- Cross-market signals (if multiple markets active simultaneously)
- Price level features (distance to 0/1 boundaries)
- Regime indicators (high/low volatility state from HMM or threshold)

**Model Architecture:**
- XGBoost (baseline — known to work from Phase 2)
- LightGBM (faster, potentially better calibration)
- Logistic regression (interpretable baseline)
- Neural network (MLP with dropout)
- Ensemble (stack XGBoost + LR + NN)
- RL agent (PPO/DQN for position sizing)

**Hyperparameters:**
- XGBoost: max_depth (3-8), learning_rate (0.01-0.3), n_estimators (50-500), class_weight
- Tail threshold: 95th vs 99th vs 99.5th percentile
- Feature selection: top K by mutual information, or LASSO regularization
- Position sizing: proportional reduction vs binary cutoff vs Kelly

**Strategy Logic:**
- When to enter: always in market (baseline), or only when tail_prob < X
- How to size: fixed, proportional to (1-tail_prob), Kelly criterion
- When to exit: tail_prob > Y, or trailing stop
- Rebalancing frequency: every trade, every 5 minutes, hourly

---

## The Autonomous Loop

```
INITIALIZE:
  git checkout -b autoresearch/tail-strategy-v1
  python3 -m src.autoresearch.backtest  # baseline Sharpe (random/naive)
  Log: baseline | sharpe=0.0 | auc=0.50 | keep | naive strategy

LOOP:
  1. cat artifacts/strategy_results.json  # current best metrics
  2. Identify weakest dimension:
     - If AUC < 0.55 → improve features or model
     - If Sharpe < 0.0 → improve position sizing or reduce costs
     - If calibration bad → add isotonic regression or Platt scaling
     - If drawdown > 20% → add risk management (stop-loss)
  3. Form hypothesis: "Adding 5-min rolling volatility should improve AUC by ~0.03"
  4. Edit strategy.py ONLY
  5. git commit -m "exp: add rolling_vol_5min feature"
  6. python3 -m src.autoresearch.backtest
  7. Read Sharpe from output
  8. KEEP if Sharpe improved ≥ 0.01 (or code got simpler at same Sharpe)
     REVERT if Sharpe decreased (git reset --hard HEAD~1)
  9. Log: commit | sharpe | auc | sortino | max_dd | status | description
  10. REPEAT — NEVER STOP
```

---

## What Goes Into the Paper

After N iterations, the autoresearch produces:

### Table: Strategy Optimization Results
| Iteration | Features | Model | AUC | Sharpe | Sortino | MaxDD |
|-----------|----------|-------|-----|--------|---------|-------|
| Baseline  | none     | random| 0.50| 0.00   | 0.00    | —     |
| 12        | ret_1, taker | XGB | 0.58 | 0.12 | 0.18 | -8% |
| 37        | +vol_5m, +imbalance | XGB | 0.63 | 0.31 | 0.42 | -6% |
| ...       |          |       |     |        |         |       |

### Figure: SHAP Summary (from best model)
- Feature importance ranking → validates/extends Phase 2 finding (taker_side = 84%)

### Figure: Calibration Curve (best model)
- Predicted vs actual tail probabilities → validates model isn't just fitting noise

### Figure: Cumulative Returns (strategy vs benchmark)
- Visual proof of economic significance

### Table: Cross-Platform Comparison
- Best strategy Sharpe on Kalshi vs Polymarket → directly answers RQ3

### Finding for Paper (Novel Contribution)
> "Using automated model search over 120+ configurations, we find that a gradient-boosted model using lagged taker-side imbalance, 5-minute rolling volatility, and price boundary distance achieves out-of-sample Sharpe of X.XX on Kalshi (Y.YY on Polymarket), net of realistic transaction costs. The dominant predictive signal is [feature], consistent with the informed trading hypothesis. This represents the first systematic evaluation of tail-risk-adjusted strategies in prediction markets."

---

## Why This IS the Research (Not Meta-Evaluation)

1. **The backtester is the methodology.** The paper says "we tested N feature/model combinations using temporal cross-validation with fixed transaction costs." The backtester implements exactly this.

2. **The Sharpe ratio is the research finding.** The paper's RQ3 asks: "Can microstructure features predict tail risk events?" The Sharpe answers: "Yes, and the economic magnitude is X."

3. **The iteration log is the ablation study.** The results.tsv shows which features/models helped and which didn't — this IS the ablation table in the paper.

4. **SHAP from the best model is the mechanistic insight.** Why does the strategy work? Because taker_side carries information (links to PIN), volatility clusters (links to EVT), etc.

5. **The cross-platform comparison is the structural finding.** Higher Sharpe on Kalshi (clustered hours, wider spreads) vs Polymarket (continuous, tighter) tells us about market design.

---

## Methods Used (From User's Specified Toolkit)

| Method | Where in Loop | Role |
|--------|--------------|------|
| **XGBoost** | `strategy.py` model | Primary tail predictor |
| **Martingales** | Backtest baseline | Benchmark: if markets are efficient, Sharpe ≈ 0 |
| **VaR/EVT** | Tail event definition + GPD threshold | The TARGET we're predicting |
| **RL** | Alternative position sizing agent | PPO/DQN learns when to trade |
| **Copulas** | Cross-platform feature: joint tail prob | Feature for multi-market strategy |
| **Linear factor pricing** | Baseline model comparison | Logistic regression on factors |
| **GMM** | Regime detection → volatility state feature | HMM/GMM identifies low/high vol regimes |
| **VAR** | Multi-market lagged return features | Price discovery signals |
| **Volatility smiles** | Feature: implied vol from price level | Distance-to-boundary captures option-like payoff |

---

## File Layout

```
src/autoresearch/
├── __init__.py
├── backtest.py          # FIXED — backtester + evaluator (prepare.py)
├── strategy.py          # EDITABLE — agent's model/features/sizing (train.py)
├── interfaces.py        # FIXED — Strategy ABC, result containers
├── data_loader.py       # FIXED — DuckDB temporal splits
└── program.md           # Agent playbook

artifacts/
├── strategy_results.json    # Latest backtest output
├── strategy_history.json    # All iterations
├── results.tsv              # Experiment log
├── shap_summary.png         # Best model SHAP (for paper)
├── calibration_curve.png    # Best model calibration (for paper)
└── cumulative_returns.png   # Strategy equity curve (for paper)
```
