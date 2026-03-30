# Autoresearch: Prediction Market Microstructure Strategy
## Version 2.0 — Aligned with research_design_v2.md

## Goal
Maximize **daily-sampled Sharpe ratio** of a microstructure-based trading strategy
on Kalshi prediction market trades. This is RQ3 of the paper.

**Primary metric:** `sharpe_ratio` (daily PnL aggregated, annualized with √252)
**Secondary metrics:** sortino_ratio, max_drawdown, calmar_ratio, win_rate

## Key Microstructure Findings (from Phase 1-3)
- **Mean reversion (VR ≈ 0.59):** Bid-ask bounce creates predictable reversal
- **Heavy tails (GPD ξ > 0):** Extreme moves are power-law distributed
- **Nonlinear dependence (BDS z > 10):** Sports markets have massive nonlinearity
- **Taker-side information (84% importance):** Order flow contains informed signal
- **Same efficiency, 3× different cost:** VR(2) same across platforms (p=0.63), but spreads differ 3×
- **Information arrival > venue design:** Sports vs political explains more than Kalshi vs Polymarket

## Fixed Harness (prepare.py — DO NOT MODIFY)
- **Data:** Top 20 Kalshi markets, max 50K trades each, ~500K total
- **Features (14):** price, trade_size, taker_buy, ret, lag_ret_1, lag_ret_2,
  lag_size_1, lag_taker_1, buy_imbalance_20, rolling_vol_10, rolling_vol_50,
  hour_of_day, day_of_week, price_boundary_dist
- **Target:** binary tail event (|ret| > 2σ)
- **Split:** 80% train / 20% test (temporal, ordered by timestamp)
- **Evaluation:** Walk-forward with 20bps transaction costs
- **Sharpe:** DAILY aggregation then √252 annualization (FIXED in v2)

## Strategy Interface (train.py — EDITABLE)
```python
class Strategy(ABC):
    def name(self) -> str: ...
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> None: ...
    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray: ...
    def size_position(self, tail_prob: float, state: PortfolioState) -> float: ...
    def get_feature_importances(self) -> dict[str, float]: ...
```

## Current Best Strategy (v37)
- LogisticRegression with sqrt-dampened mean-reversion signal
- 3-trade momentum: ret + 0.7*lag1 + 0.2*lag2
- Boundary boost for near-resolution contracts
- Direction model blend (45%)
- Tanh sizing with turnover threshold 0.75
- **Per-trade Sharpe: 3.66** (legacy metric, inflated by per-trade annualization)
- **Daily Sharpe: TBD** (needs re-evaluation with fixed prepare.py)

## Strategy Design Space
The following approaches are all fair game:

### Signal Construction
1. **Mean reversion:** Go against recent returns (works because VR ≈ 0.59)
2. **Momentum:** 3-trade, 5-trade, 10-trade weighted sums
3. **Volatility regime:** rolling vol, vol-of-vol, GARCH(1,1)
4. **Order flow:** taker-side imbalance, buy_imbalance rolling windows
5. **Temporal:** hour-of-day, day-of-week effects
6. **Price boundary:** distance to 0/1, nonlinear transforms (sqrt, tanh)
7. **VPIN-inspired:** volume-bucketed order imbalance (toxicity proxy)

### Model Architecture
1. **Logistic regression** (current winner)
2. **Gradient boosted trees** (sklearn GBM, XGBoost if available)
3. **Neural network** (1-2 hidden layers, ReLU/tanh)
4. **PPO reinforcement learning** (see rl_agent.py — state = features + portfolio, action = position)
5. **Ensemble** (stacking multiple models)

### Position Sizing
1. **Linear:** probability → position (simple)
2. **Nonlinear:** tanh, sqrt dampening (current winner)
3. **Kelly criterion:** edge/variance optimal fraction
4. **Risk parity:** inverse volatility weighting
5. **RL-learned:** PPO action space = continuous [-1, 1]
6. **Tail-conditioned:** reduce size when vol/VPIN exceeds threshold

### Risk Management
1. Turnover threshold (current: 0.75)
2. Maximum position size
3. Drawdown circuit breaker
4. Volatility scaling (target annualized vol)

## Experiment Protocol
1. Read current code state and results.tsv history
2. Form hypothesis (e.g., "Kelly sizing should improve risk-adjusted returns")
3. Edit train.py with proposed modification
4. `git commit` the change
5. Run 5-minute time-boxed experiment → get metrics
6. If **daily Sharpe improved** → keep (new baseline)
   If **daily Sharpe equal/worse** → `git reset --hard` (instant revert)
7. Log result to results.tsv: commit, sharpe, description, status
8. NEVER STOP — loop indefinitely

## Keep/Discard Criteria
- **KEEP** if daily Sharpe strictly improves (any amount)
- **KEEP** if daily Sharpe is equal but code is simpler (Occam's razor)
- **DISCARD** if daily Sharpe decreases
- **DISCARD** if code complexity increases without improvement
- **DISCARD** if feature leaks future information (no current-trade labels in features)

## Important Rules
1. NEVER modify prepare.py — it's the fixed evaluation harness
2. ALWAYS ensure features are strictly lagged (no information from current or future trades)
3. The `ret` feature is current-trade return — it IS legitimate as a mean-reversion signal
   for NEXT-trade prediction (position * next_ret), but be careful about AUC interpretation
4. Log EVERY experiment, including failures — the trajectory is part of the paper
5. Try RL (PPO) at some point — even if it loses to LogReg, that's a publishable finding
6. The paper's thesis is "information arrival > venue design" — strategy discoveries
   should be interpretable through this lens
