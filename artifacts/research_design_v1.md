# Research Design: Tail Risk and Optimal Execution in Prediction Markets
## An Empirical Study Using Extreme Value Theory and Reinforcement Learning

**Version:** 1.0
**Date:** March 29, 2026
**Dataset:** Kalshi (72M trades, Jun 2021–Nov 2025) + Polymarket (404M trades, Mar 2023–Jan 2026)
**Total trades analyzed:** ~476M

---

## 1. Research Questions

This study addresses three interconnected research questions regarding the microstructure, statistical properties, and exploitation potential of prediction market price processes.

**RQ1: Tail Risk Characterization**
Do prediction market price processes exhibit heavy tails? Do tail risk properties differ systematically between centralized (Kalshi) and decentralized (Polymarket) venues? Specifically:
- What are the empirically estimated shape parameters (ξ) of the generalized Pareto distribution (GPD) fit to price returns in each tail?
- Does Kalshi's centralized structure, regulatory oversight, and discrete pricing (cent-based) manifest in lighter tails than Polymarket's continuous, on-chain microstructure?
- How do extreme value statistics vary across market states (e.g., time-to-resolution, price level, volatility regime)?

**RQ2: Market Efficiency and Predictability**
Are prediction market prices martingales, or do microstructure features predict future price movements? Specifically:
- Do variance ratio tests (Lo-MacKinlay) at multiple horizons reject the random walk hypothesis?
- Is the price process dependent according to dimension-free BDS tests?
- Do conditional predictors (time-of-day, price level, trade size, trade flow imbalance) demonstrate statistically significant predictive power for returns beyond 2-sigma thresholds?
- Do findings differ materially across platforms or market subgroups?

**RQ3: Optimal Execution via Reinforcement Learning**
Can a reinforcement learning agent exploit the microstructure patterns identified in RQ1 and RQ2 for profitable execution? Specifically:
- Can an agent trained on historical data (PPO or DQN) achieve positive risk-adjusted returns (Sharpe ratio > 1) in backtests on unseen test data?
- What is the sensitivity of agent performance to reward specification, fee structure, and market regime?
- Does optimal execution strategy differ between centralized and decentralized venues?
- What is the out-of-sample generalization gap (train → validate → test)?

---

## 2. Data Description

### 2.1 Kalshi (Centralized, CFTC-Regulated)

- **Trading venue:** Binary options exchange regulated by the U.S. Commodity Futures Trading Commission (CFTC)
- **Time span:** June 2021 – November 2025 (~54 months)
- **Sample size:** ~72 million individual trades
- **Unique markets:** ~586,000 event contracts
- **Pricing model:**
  - Binary outcomes (yes/no)
  - Prices denominated in cents (0–99), representing probability × 100
  - Constraint: yes_price + no_price = 100 always
  - Settlement: $1 per contract for winning side, $0 for losing side
- **Data fields:** trade_id, ticker (market ID), yes_price, count (contract quantity), taker_side (yes or no), created_time
- **Microstructure features:**
  - Discrete pricing grid (0–99 cents)
  - Order-book based matching (visible spreads and depths available)
  - Regulatory oversight (position limits, reporting requirements)
  - Market hours: typically extended (limited overnight closures)

### 2.2 Polymarket (Decentralized, On-Chain)

- **Trading venue:** Polygon-based decentralized prediction market using the Conditional Token Framework (CTF)
- **Time span:** March 2023 – January 2026 (~34 months)
- **Sample size:** ~404 million on-chain trades
  - CTF Exchange: ~265M trades
  - NegRisk CTF Exchange: ~140M trades
- **Pricing model:**
  - Continuous prices (continuous support over [0, 1] representing implied probability)
  - Settlement: outcome tokens redeemable for $1 USDC
  - Amounts tracked in USDC atomic units (6 decimals); divide by 10^6 for USD values
- **Price derivation from trade data:**
  - If maker_asset_id = '0': maker pays USDC for outcome tokens → side = 'buy', price = maker_amount / taker_amount
  - If maker_asset_id ≠ '0': maker offers outcome tokens for USDC → side = 'sell', price = taker_amount / maker_amount
  - Market ID = the non-'0' asset_id (identifies the conditional token contract)
- **Data fields:** block_number, tx_hash, maker_asset_id, taker_asset_id, maker_amount, taker_amount, and joined block_timestamp
- **Microstructure features:**
  - Continuous pricing (no minimum tick constraint)
  - Automated market maker (AMM) + order book microstructure
  - Blockchain settlement (on-chain timestamps, fee-driven order flow)
  - 24/7 trading (no market hours restriction)
  - Global participant base (no geographic/regulatory restrictions)

### 2.3 Temporal Overlap and Preprocessing

- **Overlap period:** March 2023 – November 2025 (~32 months)
- **Data standardization:**
  - Kalshi: price = yes_price / 100, quantity = count, notional = price × count
  - Polymarket: joins blocks for timestamps, derives price from amount ratios, converts to USDC
  - Common schema: (timestamp, market_id, price, quantity, notional, side, platform, venue_type)
- **Cleaning criteria:**
  - Remove trades with price ≤ 0 or price ≥ 1 (both platforms)
  - Remove trades with quantity ≤ 0 or notional ≤ 0
  - Remove duplicates based on platform-specific identifiers
  - Handle timestamps: Polymarket null timestamps replaced via block-level joins
- **Total standardized trades:** ~476M across both platforms

---

## 3. Methodology

### 3.1 RQ1: Extreme Value Theory and Tail Risk Characterization

#### 3.1.1 Return Computation

For each market m with T_m trades:
1. **Log returns:** r_t = log(p_t / p_{t-1}) where p_t is the price of the t-th trade in market m
2. **Intra-market only:** Compute returns only for consecutive trades within the same market to maintain economic meaning
3. **Standardization:** Center and scale returns within each market to mitigate non-stationarity
4. **Outlier flagging:** Mark returns > 5 standard deviations for sensitivity analysis

#### 3.1.2 Threshold Selection and GPD Fitting

1. **Threshold selection strategy:**
   - Primary: 95th percentile of absolute returns within each market
   - Sensitivity: test 90th, 95th, 99th percentiles
   - Minimum data requirement: ≥20 exceedances per threshold to ensure GPD parameter stability

2. **Generalized Pareto Distribution (GPD) model:**
   - Fit GPD G(x; σ, ξ) to tail exceedances via maximum likelihood estimation (MLE)
   - Extract shape parameter ξ (heavy-tail indicator; ξ > 0 indicates power-law decay)
   - Extract scale parameter σ (tail thickness)
   - Compute tail index α = 1/ξ (inverse of shape; smaller α = heavier tail)

3. **Aggregation levels:**
   - **By platform:** Compare ξ distributions across Kalshi vs. Polymarket
   - **By market subgroup:** Stratify by resolution type (e.g., sports, politics, economics), market volume (quartiles), time-to-resolution
   - **By regime:** Separate high-volatility vs. normal-volatility periods using rolling 30-day volatility

#### 3.1.3 Goodness-of-Fit Assessment

1. **Quantile-Quantile (QQ) plots:**
   - Visually inspect fit quality for upper and lower tails separately
   - Produce platform-level and regime-level QQ plots

2. **Kolmogorov-Smirnov test:**
   - Test null hypothesis: observed tail exceedances ~ GPD(σ_hat, ξ_hat)
   - Report test statistic and p-value; flag poor fits (p < 0.05)

3. **Anderson-Darling test:**
   - More sensitive in tails than KS; report as secondary fit metric

#### 3.1.4 Machine Learning for Tail Prediction

1. **Classification task:**
   - **Target:** Binary indicator = 1 if |r_{t+1}| > 2σ (two standard deviations), 0 otherwise
   - **Features:**
     - Lagged returns: r_t, r_{t-1}, r_{t-2}
     - Trade flow: buy/sell ratio in preceding 5, 10, 20 trades
     - Volatility: 10-trade, 30-trade rolling standard deviation
     - Price level: quantile of current price within [0, 1] range for market
     - Time-to-resolution: days remaining
     - Hour-of-day (UTC): categorical, 0–23
     - Market subgroup (categorical): resolution type or volume quintile
   - **XGBoost classifier:** Baseline for predicting whether next return exceeds 2-sigma threshold
   - **Evaluation metrics:** AUC-ROC, precision, recall, F1-score, feature importance rankings
   - **Train/validate/test split:** 60% / 20% / 20% by date for each platform

### 3.2 RQ2: Market Efficiency and Martingale Testing

#### 3.2.1 Variance Ratio Test (Lo-MacKinlay)

1. **Test specification:**
   - Null hypothesis: price process is a random walk with or without drift
   - Alternative: mean reversion or mean aversion (irrational momentum)
   - Compute variance ratio: VR(k) = Var(r_t + r_{t-1} + ... + r_{t-k+1}) / (k × Var(r_t))
   - Under random walk: VR(k) = 1 for all k

2. **Horizons:** k ∈ {2, 5, 10, 20} trade intervals
3. **Heteroskedasticity-consistent standard errors:** Use Andrews heteroskedasticity-robust variance estimator
4. **Stratification:**
   - By platform (Kalshi vs. Polymarket)
   - By time-of-day (8-hour blocks: 00:00–08:00, 08:00–16:00, 16:00–24:00 UTC)
   - By price level (quintiles 0–20%, 20–40%, ..., 80–100%)
   - By trade size (terciles: small, medium, large)

#### 3.2.2 BDS Test (Brock-Dechert-Scheinkman)

1. **Test specification:**
   - Null hypothesis: price returns are i.i.d. (independent and identically distributed)
   - Detects nonlinear dependence missed by linear tests (e.g., variance ratio)
   - Compute correlation integral C_m(ε) and test statistic W_m

2. **Parameters:**
   - Embedding dimension m ∈ {2, 3, 4, 5}
   - Distance threshold ε ∈ {0.5, 1.0, 1.5} standard deviations of returns
   - Asymptotic distribution: standard normal under null

3. **Interpretation:**
   - p < 0.05: reject i.i.d., evidence of deterministic structure or nonlinear dependence
   - Large W_m values: nonlinearity increases with embedding dimension

4. **Stratification:** Same as variance ratio test (platform, time-of-day, price level, trade size)

#### 3.2.3 Runs Test for Randomness

1. **Test specification:**
   - Null hypothesis: sequence of signs (+ or −) in returns is random
   - Count runs R (maximal sequences of same sign)
   - Under null: E[R] = (2n₊n₋) / (n₊ + n₋) + 1, Var[R] = 2n₊n₋(2n₊n₋ − n₊ − n₋) / ((n₊ + n₋)²(n₊ + n₋ − 1))
   - Test statistic: Z = (R − E[R]) / √Var[R] ~ N(0,1) under null

2. **Stratification:** Same as above

#### 3.2.4 Conditional Prediction Analysis

1. **Logistic regression model:**
   - Target: 1 if r_{t+1} > median(r), 0 otherwise (classify next return as above/below median)
   - Predictors:
     - Price level bucket (categorical: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0])
     - Time-of-day (categorical: UTC hour)
     - Trade size relative to market average (continuous: scaled)
     - Trade flow imbalance (continuous: (count_buy − count_sell) / (count_buy + count_sell) over preceding 10 trades)
     - Platform (Kalshi vs. Polymarket)
   - Evaluation: Chi-squared test on coefficients, McFadden pseudo-R²

2. **Decision tree / random forest:**
   - Alternative specification to detect nonlinear interactions
   - Feature importance for predictors

### 3.3 RQ3: Reinforcement Learning for Optimal Execution

#### 3.3.1 RL Environment Design

1. **State space S:**
   - current_price: ∈ [0, 1] (normalized probability)
   - time_to_resolution: ∈ [0, max_days] (days until market closes)
   - recent_volatility: 20-trade rolling standard deviation of log returns
   - trade_flow_imbalance: (count_buy − count_sell) / (count_buy + count_sell) over preceding 10 trades, ∈ [−1, 1]
   - hour_of_day: UTC hour, categorical ∈ {0, 1, ..., 23}
   - price_level_bucket: discretized into 10 decile buckets (0–10%, 10–20%, ..., 90–100%)
   - position: agent's current inventory (number of outcome tokens held), ∈ [−max_position, max_position]

2. **Action space A (discrete):**
   - Actions ∈ {BUY_SMALL, BUY_MEDIUM, BUY_LARGE, HOLD, SELL_SMALL, SELL_MEDIUM, SELL_LARGE}
   - Quantity mapping:
     - SMALL: 1–10 contracts
     - MEDIUM: 11–50 contracts
     - LARGE: 51–200 contracts
   - Adaptive sizing: scale quantity proportionally to recent trade volume (5-minute rolling average)
   - Constraint: inventory never exceeds max_position; reject oversized actions

3. **Reward specification:**
   - **Immediate reward:**
     - Trade execution: r_t = −fee − slippage (cost of execution)
     - Resolution: r_terminal = final_pnl (profit/loss on closed position at market resolution)
   - **Aggregated episode reward:** cumulative risk-adjusted returns
     - Sharpe ratio over episode: (mean_return − risk_free_rate) / std_return
   - **Discount factor:** γ = 0.99 (balance immediate costs vs. terminal profit)
   - **Fee model:** realistic platform fees
     - Kalshi: ~0.2–0.4% bid-ask spread (mid-quote estimate)
     - Polymarket: ~0.2–0.5% AMM fee + gas costs (on-chain)

#### 3.3.2 Training and Validation Setup

1. **Training data:** 2023–2024 (12 months)
   - Subset selection: ~10M trades per market to reduce computational cost without losing diversity
   - Environment reset: each episode = one market life (from inception to resolution)

2. **Validation data:** January–June 2025 (6 months)
   - Evaluate agent on unseen markets and time periods
   - Out-of-sample Sharpe ratio, cumulative returns, max drawdown
   - Early stopping: if validation Sharpe ratio plateaus for 10 consecutive evaluation cycles, terminate training

3. **Test data:** July–November 2025 (5 months)
   - Final holdout set (completely unseen during training and validation)
   - Report test Sharpe ratio, returns distribution, and comparison to baseline strategies

#### 3.3.3 Agent Algorithms

1. **Primary algorithm: Proximal Policy Optimization (PPO)**
   - Advantage actor-critic (A2C) with clipped surrogate objective
   - Network architecture:
     - Shared encoder: 2 hidden layers (256 units each, ReLU activation)
     - Policy head: 2 hidden layers → output = softmax over 7 discrete actions
     - Value head: 2 hidden layers → output = scalar value function
   - Hyperparameters:
     - Batch size: 128
     - Learning rate: 5e−4 (with exponential decay schedule)
     - Entropy coefficient: 0.01 (encourage exploration)
     - Clip ratio: 0.2
     - Generalized Advantage Estimation (GAE) with λ = 0.95

2. **Baseline algorithm: Deep Q-Network (DQN)**
   - Q-learning with target network and experience replay
   - Network: 3 hidden layers (256, 256, 128 units, ReLU) → Q-values for each action
   - Hyperparameters:
     - Replay buffer size: 100k episodes
     - Target network update frequency: 1,000 steps
     - Epsilon decay: ε from 1.0 → 0.05 over 10k steps
   - Comparison: PPO vs. DQN on validation metrics

#### 3.3.4 Backtesting Protocol

1. **Implementation:**
   - Historical replay: reconstruct actual order book / price sequences from dataset
   - Realistic order matching: agent's buy/sell orders matched at realistic mid-quotes + slippage
   - Fee deduction: apply platform-specific fee schedule to each executed trade
   - Position management: track cumulative inventory and forced liquidation at market resolution

2. **Comparison baselines:**
   - **Buy-and-hold:** enter at market inception, hold until resolution (0 RL)
   - **Random execution:** uniformly random buy/sell/hold actions (null model)
   - **Time-weighted average price (TWAP):** divide order into equal time buckets, execute mechanically
   - **Volume-weighted average price (VWAP):** execute proportional to recent volume (simple execution rule)

3. **Performance metrics:**
   - Sharpe ratio (risk-adjusted return)
   - Cumulative return (%)
   - Maximum drawdown
   - Win rate (% of episodes with positive PnL)
   - Average P&L per trade
   - Trade slippage (realized price vs. mid-quote at execution time)

#### 3.3.5 Sensitivity and Generalization Analysis

1. **Fee sensitivity:**
   - Train separate agents with ±50% fee scaling
   - Report Sharpe ratio vs. fee level

2. **Market regime sensitivity:**
   - Evaluate on high-volatility vs. normal-volatility subsamples
   - Evaluate on different market types (e.g., sports vs. politics)

3. **Cross-platform transfer:**
   - Train agent on Kalshi, evaluate on Polymarket (and vice versa)
   - Measure generalization gap

4. **Reward specification ablation:**
   - Reward variant 1: risk-adjusted (Sharpe) only
   - Reward variant 2: cumulative P&L + inventory penalty
   - Reward variant 3: risk-adjusted with tail-risk penalty (penalize large losses)

---

## 4. Expected Contributions

### 4.1 Theoretical and Empirical Contributions

1. **Extreme Value Theory Application to Prediction Markets (RQ1)**
   - First systematic characterization of tail risk in binary prediction markets at scale (476M trades)
   - Quantification of shape parameters (ξ) for Kalshi vs. Polymarket, establishing whether decentralized venues exhibit heavier tails
   - Evidence on whether regulatory structure (centralized vs. decentralized) materially affects tail risk
   - Predictive models for identifying when tail events (>2σ moves) are likely, with platform-specific feature importance

2. **Market Efficiency Benchmarking (RQ2)**
   - Large-scale variance ratio, BDS, and runs tests on 476M trades—orders of magnitude larger than prior studies
   - Cross-platform efficiency comparison (centralized Kalshi vs. decentralized Polymarket)
   - Stratified analysis revealing time-of-day, price level, and trade size effects on market efficiency
   - Evidence of conditional predictability despite aggregate random walk properties

3. **Reinforcement Learning for Execution (RQ3)**
   - First application of modern RL (PPO / DQN) to prediction market execution
   - Quantification of exploitable profit opportunities via realistic backtesting (with fees, slippage)
   - Out-of-sample generalization analysis (train 2023–24 → test 2025)
   - Platform-specific execution strategies and cross-platform transfer insights

### 4.2 Methodological Contributions

1. Standardized data preprocessing pipeline for heterogeneous prediction market formats (discrete Kalshi vs. continuous Polymarket)
2. Integrated framework combining EVT, hypothesis testing, and RL for microstructure analysis
3. Benchmarking suite of classical (variance ratio, BDS, runs) and modern (XGBoost, RL) techniques on the same dataset

### 4.3 Practical Contributions

1. Actionable execution strategies for prediction market traders (risk-averse sizing, time-of-day effects)
2. Platform-specific recommendations (e.g., optimal order sizing for Kalshi's discrete grid vs. Polymarket's continuous AMM)
3. Risk assessment tools for practitioners (tail risk characterization, efficiency benchmarks)

---

## 5. Timeline and Milestones

### Phase 1: Data Profiling and Preparation (Current)
- **Duration:** Weeks 1–2 (March 29 – April 12, 2026)
- **Deliverables:**
  - Data profiling report (row counts, missing values, date ranges, market distributions)
  - Standardized parquet files for both platforms (data/processed/)
  - Validation checks (no negative prices, quantities, etc.)
- **Owner:** Data engineering / initial analysis

### Phase 2: Extreme Value Theory and Martingale Tests (RQ1 & RQ2)
- **Duration:** Weeks 3–5 (April 13 – May 3, 2026)
- **Deliverables:**
  - GPD fits to both platforms' return tails; QQ plots and KS test results
  - Variance ratio tests (k=2,5,10,20) with 95% confidence intervals
  - BDS test results (m=2–5, ε=0.5–1.5σ)
  - XGBoost classifier for 2σ tail prediction (AUC-ROC, feature importance)
  - Summary table: ξ, scale σ, tail index α by platform and regime
  - Summary table: VR(k) and p-values by platform and stratification
- **Owner:** Statistical analysis

### Phase 3: RL Environment and Agent Training (RQ3)
- **Duration:** Weeks 6–8 (May 4 – May 24, 2026)
- **Deliverables:**
  - RL environment implementation (state/action/reward definition)
  - PPO and DQN agent training on 2023–2024 data
  - Validation Sharpe ratios, learning curves (return vs. episode)
  - Hyperparameter sensitivity analysis
  - Baseline comparisons (buy-hold, random, TWAP, VWAP)
- **Owner:** RL / deep learning

### Phase 4: Backtesting and Cross-Platform Analysis
- **Duration:** Weeks 9–10 (May 25 – June 7, 2026)
- **Deliverables:**
  - Test-set performance (Sharpe, cumulative return, max drawdown)
  - Fee sensitivity analysis
  - Market regime sensitivity (high-vol vs. normal)
  - Cross-platform transfer analysis (train Kalshi → test Polymarket)
  - Agent learning curves and policy visualizations
- **Owner:** RL / backtesting

### Phase 5: Paper Writing and Revision
- **Duration:** Weeks 11–14 (June 8 – July 5, 2026)
- **Deliverables:**
  - Full draft: Introduction, Data & Methods, Results, Discussion, Conclusion
  - All figures (QQ plots, VR plots, RL learning curves, strategy comparison)
  - All tables (EVT summary, martingale tests, RL backtest results)
  - Supplementary appendix (sensitivity analyses, ablations, code availability statement)
  - Revision rounds incorporating feedback from internal review
- **Owner:** Lead author with collaborators

### Phase 6: Submission and Revision
- **Duration:** Weeks 15–20 (July 6 – August 16, 2026)
- **Target venues:**
  - Primary: ICAIF 2026 (deadline TBD, typically September)
  - Secondary: NeurIPS 2026 Datasets & Benchmarks (deadline TBD, typically May–June; may miss)
  - Journal: Journal of Financial Data Science (rolling submissions)
- **Owner:** Lead author

---

## 6. Target Venues

### 6.1 Primary: ACM International Conference on AI in Finance (ICAIF 2026)

- **Scope:** Intersection of machine learning, reinforcement learning, and finance
- **Strong fit for:**
  - RQ3 (RL execution strategy—novel RL application in prediction markets)
  - RQ2 (machine learning for predictability—XGBoost classifiers)
  - Cross-platform empirical comparison (centralized vs. decentralized)
- **Estimated timeline:** Submission deadline ~August–September 2026; notification Dec 2026; conference typically November–December
- **Audience:** quants, ML practitioners, finance technologists

### 6.2 Secondary: NeurIPS 2026 Datasets & Benchmarks Track

- **Scope:** Novel large-scale datasets, benchmarking, and reproducibility
- **Strong fit for:**
  - RQ1 & RQ2 (476M-trade benchmark for tail risk and efficiency testing)
  - Standardized preprocessing pipeline and code release
  - Cross-platform comparison framework
- **Note:** NeurIPS 2026 deadline likely passed (May 2026); consider NeurIPS 2027 or other ML conferences
- **Audience:** machine learning researchers, benchmark developers

### 6.3 Journal: Journal of Financial Data Science

- **Scope:** Empirical finance, data science applications, machine learning in finance
- **Strong fit for:**
  - All three research questions (comprehensive empirical study)
  - Data description and validation (476M trades, two platforms)
  - Reproducibility and code/data release emphasis
- **Timeline:** Rolling submissions; turnaround typically 3–6 months
- **Audience:** academic finance, practitioners

### 6.4 Alternative Venues (if primary venues reject)

- **NeurIPS 2027** (Datasets & Benchmarks or main conference)
- **AISTATS 2027** (probabilistic inference, statistics, ML)
- **Journal of Machine Learning Research (JMLR)** (RL algorithms, benchmarking)
- **Financial Data Science Conference** (academic + industry)

---

## 7. Data and Code Availability

### 7.1 Data Sharing

- **Public release:**
  - Standardized parquet files: `/data/processed/kalshi_trades.parquet`, `polymarket_trades.parquet`, `trades_all.parquet`
  - Data dictionary and schema documentation
  - Anonymized market metadata (market type, resolution date, final price, volume)

- **Licensing:**
  - Data: Creative Commons Attribution 4.0 (CC-BY-4.0) or CC0 (public domain)
  - Code: MIT or Apache 2.0

### 7.2 Code Availability

- **Repository:** GitHub (public)
  - All analysis scripts (`src/analysis/` directory)
  - RL environment and agent code (`src/agents/` + `src/rl/`)
  - Reproducibility: Makefile, requirements.txt, seed specification
  - Notebooks: Jupyter notebooks for exploratory analysis and result visualization

- **Documentation:**
  - README with setup and execution instructions
  - API documentation for RL environment
  - Hyperparameter specifications and tuning scripts

---

## 8. Ethical and Regulatory Considerations

### 8.1 Data Ethics

- **Privacy:** Kalshi and Polymarket trade data are already publicly available (Kalshi via API, Polymarket on-chain); no private trader data collected or exposed
- **Market impact:** Analysis is retrospective (backtesting on historical data); no live trading or market manipulation
- **Transparency:** All methods, data sources, and code released publicly for reproducibility and scrutiny

### 8.2 Regulatory Alignment

- **Kalshi:** Operates under CFTC oversight; no additional regulatory approval required for academic analysis of public data
- **Polymarket:** Decentralized, on-chain data; no regulatory approvals needed
- **Replicability:** Findings derived from historical data; no forward-testing of strategies on live markets without separate compliance review

---

## 9. Contingencies and Risk Mitigation

### 9.1 Data Risks

- **Risk:** Missing or corrupted timestamps in Polymarket data
- **Mitigation:** Pre-validation step; block-level joins for all timestamps; sensitivity analysis on timestamp precision

- **Risk:** Insufficient tail exceedances for reliable GPD estimation in some markets
- **Minimum threshold:** Markets with <100 trades excluded from EVT analysis; reported in limitations

### 9.2 Computational Risks

- **Risk:** 476M-trade dataset exceeds available memory for single-machine processing
- **Mitigation:** DuckDB (out-of-core processing), stratified sampling for intermediate analysis, distributed computing (if needed)

### 9.3 Methodological Risks

- **Risk:** RL agent overfitting to training data (2023–2024)
- **Mitigation:** Strict train/validate/test split by date; out-of-sample test set (July–Nov 2025); ablation studies

- **Risk:** Microstructure patterns change over time; agent trained on past may not generalize
- **Mitigation:** Report train/validate/test Sharpe ratios separately; cross-platform transfer tests; market regime sensitivity analysis

### 9.4 Publication Risks

- **Risk:** ICAIF submission rejected; venue deadlines missed
- **Mitigation:** Target multiple venues (ICAIF primary, NeurIPS/journal secondary); stagger submission timelines

---

## 10. Success Criteria

### 10.1 Quantitative Benchmarks

1. **RQ1 (EVT):**
   - Estimate ξ with 95% confidence intervals for both platforms
   - Report QQ plot visual fit and KS test p-value > 0.05 for ≥80% of market subgroups
   - XGBoost AUC-ROC > 0.65 for 2σ tail prediction

2. **RQ2 (Martingale tests):**
   - Variance ratio test results (reject/fail to reject RW) at k=2,5,10,20 for both platforms
   - BDS test p-values < 0.05 for ≥70% of platform/stratification combinations
   - Logistic regression: ≥2 statistically significant predictors (α=0.05) in each platform

3. **RQ3 (RL):**
   - Validation-set Sharpe ratio > 1.0 for PPO agent
   - Test-set Sharpe ratio > 0.8 (accounting for out-of-sample deterioration)
   - Win rate > 50% on test set (% episodes with positive PnL)
   - Outperformance vs. at least two baseline strategies (buy-hold, TWAP, random) on test set

### 10.2 Qualitative Benchmarks

1. **Clarity:** Methods section understandable to finance + ML audience without excessive notation
2. **Reproducibility:** All code, hyperparameters, random seeds specified; paper reproducible within 72 hours on reasonable hardware
3. **Novelty:** All three RQs address gaps in prior literature; contributions clearly articulated relative to existing work
4. **Rigor:** No overclaiming; limitations and caveats discussed transparently

---

## 11. References to Prior Work (Placeholder Section)

Key papers to cite:
- Extreme value theory: McNeil et al. (2015), "Quantitative Risk Management: Concepts, Techniques and Tools"
- Variance ratio tests: Lo & MacKinlay (1988), "Stock market prices do not follow random walks"
- BDS test: Brock, Dechert, Scheinkman (1987), "A test for independence"
- RL for finance: Almgren & Chriss (2001) on optimal execution; Nevmyvaka et al. (2006) on adaptive execution
- Prediction markets: Tetlock & Mellers (2014) on accuracy; Arrow et al. (2008) handbook
- Microstructure: O'Hara (1995), "Market Microstructure Theory"

---

## Appendix: Key Definitions

- **Tail Index (α):** 1/ξ where ξ is the GPD shape parameter; lower α indicates heavier tails
- **Sharpe Ratio:** (mean return − risk-free rate) / std(return); measures risk-adjusted performance
- **VR(k):** Variance ratio at lag k; = 1 under random walk hypothesis
- **BDS Statistic:** Correlation-integral-based test; high values suggest nonlinear dependence
- **Slippage:** Difference between execution price and mid-quote at time of order
- **Max Drawdown:** Largest peak-to-trough decline in cumulative profit during a test period

---

**Document Version:** 1.0
**Last Updated:** March 29, 2026
**Status:** Ready for review and execution
