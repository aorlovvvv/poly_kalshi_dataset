# Hypothesis Candidates for Prediction Market Microstructure Research

**Dataset:** Polymarket (404.5M trades, decentralized) + Kalshi (72.1M trades, centralized/CFTC-regulated)
**Overlap period:** March 2023 – November 2025 (~32 months)
**Profiled:** 2026-03-29 from `artifacts/data_profile.json`

---

## Striking Patterns from Data Profiling

### 1. Overwhelming Negative Serial Correlation (Bid-Ask Bounce)

All 10 most liquid Kalshi markets show strongly negative lag-1 autocorrelation in price changes, ranging from **−0.34 to −0.47**. The 2024 presidential markets (DJT: −0.475, KH: −0.446) exhibit the strongest mean reversion. This is a textbook signature of bid-ask bounce — trades alternating between the bid and ask sides of the order book — and constitutes direct evidence of microstructure noise contaminating the price signal. A naive efficiency test on raw trade prices would be severely biased.

### 2. Extreme Fat Tails (Kurtosis = 393)

Price changes across the 100 most liquid Kalshi markets exhibit a kurtosis of **393** (Gaussian = 3, i.e., 130× excess). The p01/p99 of returns (±2.2%) are compact, yet the distribution generates rare jumps exceeding ±10% (0.086% of trades). This is consistent with Pareto-type tails rather than any finite-variance distribution and demands Extreme Value Theory for proper characterization.

### 3. Divergent Price Concentration Between Platforms

Polymarket trades cluster at extreme prices (near 0 or 1) at **2.65× the rate** of Kalshi:

| Metric | Kalshi | Polymarket | Ratio |
|---|---|---|---|
| Frac trades at ≤5% or ≥95% | 12.0% | 31.7% | 2.65× |
| Frac trades at ≤10% or ≥90% | 21.4% | 39.1% | 1.83× |
| Frac trades in toss-up (40–60%) | 21.8% | 20.1% | 0.92× |
| Frac trades uncertain (25–75%) | 51.7% | 42.8% | 0.83× |

This suggests fundamentally different market compositions: Polymarket attracts more trading in near-resolved markets (high-conviction events), while Kalshi's regulated structure may discourage trading at extreme prices or attract more speculative uncertainty-zone activity.

### 4. Dramatically Different Intraday Rhythms

Kalshi shows a **24× peak-to-trough ratio** (Hour 20 UTC: 5.0M trades vs Hour 9 UTC: 228K trades), perfectly aligned with US Eastern business hours. Polymarket shows only a **1.37× ratio** (Hour 14 UTC: 19.8M vs Hour 23 UTC: 14.5M), consistent with a globally distributed, 24-hour participant base. This is a direct structural consequence of Kalshi's US-regulatory domain versus Polymarket's permissionless on-chain architecture.

### 5. Universal Buy-Side Bias

Both platforms show strong directional skew: Kalshi takers choose "yes" 68.9% of the time (49.7M/72.1M), and Polymarket makers pay USDC for tokens (buy) 74.1% of the time (299.6M/404.5M). This is consistent with the well-documented **favorite-longshot bias** and suggests retail participants systematically prefer "buying hope" (positive outcomes) over selling exposure.

### 6. Extreme Right-Skew in Trade Sizes

Kalshi trade sizes have a mean/median ratio of **6.33** (mean = 253 contracts, median = 40), with a maximum single trade of 3.1M contracts. The p95/p99 are 974/3,892 contracts. This power-law-like distribution indicates a dual-mode market: frequent small retail trades coexisting with rare institutional-scale orders.

### 7. Long-Tailed Market Activity Distribution

Of Kalshi's 586,025 unique markets, only **4 have ≥100K trades**, 1,105 have ≥10K trades, and 77,618 have ≥100 trades. Over 86% of markets have fewer than 100 trades. The distribution of activity across markets follows an extreme Zipf/power-law pattern, meaning liquidity is concentrated in a tiny fraction of listed events.

---

## Hypothesis 1: Microstructure-Corrected Martingale Testing Across Venue Types

### Motivation

The efficient markets hypothesis, applied to prediction markets, states that contract prices should follow a martingale: the best forecast of tomorrow's price is today's price. However, the profiling reveals that raw trade-level prices on Kalshi exhibit lag-1 autocorrelation between −0.34 and −0.47 — a textbook bid-ask bounce artifact that would cause naive martingale tests to **incorrectly reject** efficiency.

The research question is whether prediction market prices are informationally efficient *after* properly accounting for microstructure noise, and whether the answer differs between the centralized (Kalshi) and decentralized (Polymarket) venue.

### Methodology

1. **Roll's implied spread model** — Estimate the effective bid-ask spread from the negative autocovariance of price changes. The implied spread \(s = 2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}\) can be computed per-market and compared across platforms.

2. **Variance ratio tests (Lo-MacKinlay, 1988)** — Test whether \(VR(q) = \text{Var}(p_t - p_{t-q}) / [q \cdot \text{Var}(p_t - p_{t-1})]\) equals 1 at horizons \(q \in \{2, 5, 10, 20, 50\}\). Under a martingale, \(VR(q) = 1\). Apply both to raw prices (expecting rejection due to microstructure noise) and to mid-price reconstructions (if order book data is available) or time-sampled prices.

3. **BDS test (Brock, Dechert & Scheinkman)** — A non-parametric test for serial independence in the residuals after removing linear structure. This captures any remaining nonlinear predictability.

4. **Cross-platform comparison** — Run all tests separately for Kalshi and Polymarket, stratified by:
   - Market type (political, sports, crypto/finance, weather)
   - Market phase (early life, mid-life, approaching resolution)
   - Liquidity tier (top 1%, top 10%, remaining)

### Expected Contribution

- First large-scale martingale efficiency comparison between a centralized regulated exchange and a decentralized on-chain exchange using hundreds of millions of trades
- Quantification of effective bid-ask spreads across 586K+ Kalshi markets and Polymarket token pairs
- Evidence on whether decentralized markets are more or less efficient than regulated ones, with microstructure properly controlled
- Direct test of whether the strong negative autocorrelation (−0.34 to −0.47) is entirely explained by bid-ask bounce or if residual predictability remains

### Key Data Requirements

- Trade-level timestamps (Kalshi: `created_time`; Polymarket: via block join)
- Prices (Kalshi: `yes_price/100`; Polymarket: derived from amount ratios)
- Per-market stratification requires market metadata linkage

---

## Hypothesis 2: Extreme Value Theory for Prediction Market Tail Risk and the Regulatory-Structure Effect

### Motivation

The profiling reveals that Kalshi price changes exhibit kurtosis of **393** — a level of tail heaviness that is extraordinary even by financial market standards (equity index returns typically show kurtosis of 5–15). Approximately 0.086% of price changes exceed ±10 percentage points in a single trade, and the p01/p99 bounds (±2.2%) are tight relative to the tail mass. This distribution is inconsistent with Gaussian, Student-t, or even most stable distributions, and demands proper EVT characterization.

Simultaneously, Polymarket's price distribution is bimodal/U-shaped with 31.7% of trades at extreme prices (≤5% or ≥95%), compared to only 12.0% on Kalshi. This suggests the **tail behavior of prices themselves** (not just price changes) differs dramatically across venues.

### Methodology

1. **Peaks-Over-Threshold (POT) analysis** — For both platforms, fit the Generalized Pareto Distribution (GPD) to exceedances above high quantile thresholds. Estimate the shape parameter ξ (tail index) and scale parameter σ. Compare:
   - Kalshi ξ vs Polymarket ξ for price changes
   - Variation of ξ across market types (political events may have fatter tails than sports)
   - Variation of ξ across market phases (approaching resolution → tail compression or explosion?)

2. **Block maxima / GEV fitting** — Partition trade sequences into fixed-size blocks (e.g., 1000 trades per block), fit the Generalized Extreme Value distribution to block maxima. Test whether the Fréchet (ξ > 0, heavy-tailed), Gumbel (ξ = 0, exponential tail), or Weibull (ξ < 0, bounded tail) domain applies.

3. **Hill estimator for tail index** — Non-parametric estimation of the power-law exponent α in the tail P(X > x) ~ x^{−α}. A lower α means heavier tails. Compare α between platforms and market types.

4. **Conditional tail expectations** — Compute E[ΔP | ΔP > VaR_q] (Expected Shortfall / CVaR) at various quantile levels. This directly measures the "how bad can it get" risk for market makers and traders.

5. **Regulatory-structure decomposition** — Test whether the Kalshi-Polymarket divergence in tail behavior can be attributed to:
   - **Tick-size effects**: Kalshi's 1-cent grid (discrete) vs Polymarket's continuous pricing
   - **Participant composition**: Regulated (KYC/AML) vs permissionless participants
   - **Market design**: CLOB (Kalshi) vs AMM/hybrid (Polymarket)

### Expected Contribution

- First EVT characterization of prediction market returns at scale (476M+ trades)
- Quantification of how regulatory structure affects tail risk: are regulated markets safer or do they just hide risk differently?
- Practical implications for position limits, margin requirements, and risk management in prediction markets
- Evidence on whether extreme kurtosis (393) is a microstructure artifact (tick-size discretization) or reflects genuine information arrival patterns
- Novel "tail index surface" mapping ξ across (platform × market type × market phase) dimensions

### Key Data Requirements

- High-frequency price changes for both platforms
- Market categorization metadata (political, sports, crypto, weather)
- Market lifecycle stage (time-to-resolution)
- Sufficient depth in liquid markets for stable EVT estimation (the 100 markets with ≥1000 trades on Kalshi provide 5.6M return observations)

---

## Hypothesis 3: Cross-Platform Dependence Structure and ML-Based Arbitrage Detection

### Motivation

Kalshi and Polymarket frequently list contracts on the **same underlying events** (presidential elections, sports outcomes, economic indicators) but differ in every microstructural dimension: price concentration (2.65× more extreme trades on Polymarket), intraday patterns (24× vs 1.37× peak-to-trough), buy-side bias (69% vs 74%), and trade-size distributions. These structural differences create the possibility that **price discovery occurs at different speeds** on each platform, generating exploitable lead-lag relationships.

The 32-month overlap period (Mar 2023 – Nov 2025) with hundreds of millions of trades on each platform provides unprecedented statistical power to measure cross-platform dependence — something impossible in traditional finance where centralized exchanges dominate.

### Methodology

#### Part A: Copula-Based Cross-Platform Dependence

1. **Event matching** — Identify events listed on both Kalshi and Polymarket during the overlap period using market metadata (ticker descriptions, resolution criteria). Build a matched-event panel.

2. **Time-aligned price series** — For matched events, construct synchronized price series at common time intervals (1-minute, 5-minute, hourly, daily). Handle Polymarket's block-based timestamps by mapping to wall-clock time.

3. **Copula estimation** — Fit parametric copula families to the joint distribution of (Kalshi price, Polymarket price) for matched events:
   - **Gaussian copula** — baseline (symmetric dependence)
   - **Clayton copula** — lower tail dependence (do platforms crash together?)
   - **Gumbel copula** — upper tail dependence (do platforms rally together?)
   - **Frank copula** — symmetric without tail dependence
   - **Student-t copula** — symmetric with tail dependence
   - Select via AIC/BIC. Estimate time-varying copula parameters using rolling windows.

4. **Tail dependence coefficients** — Measure λ_L and λ_U (lower and upper tail dependence). If λ_L > λ_U, platforms are more correlated during crashes than rallies — a systemic risk finding.

5. **Lead-lag analysis** — Granger causality tests and cross-correlation at varying lags to determine which platform leads price discovery for different event types.

#### Part B: XGBoost / Gradient Boosting for Cross-Platform Prediction

6. **Feature engineering** — For each platform, at each time step, compute:
   - Recent price returns (1-min, 5-min, 1-hr)
   - Rolling volatility and volume
   - Buy/sell imbalance (taker side ratio)
   - Time-of-day indicators (exploiting the 24× vs 1.37× intraday divergence)
   - Price distance from extremes (0 or 1)
   - Spread proxy (from serial correlation)

7. **Target variable** — Predict the 5-minute-ahead return on platform A using features from platform B (and vice versa). A significant out-of-sample R² indicates cross-platform information leakage.

8. **XGBoost model** — Train gradient-boosted trees with proper temporal cross-validation (expanding window, no look-ahead). Evaluate:
   - Out-of-sample R² and Sharpe ratio of a hypothetical trading strategy
   - Feature importance (SHAP values) to understand what drives cross-platform predictability
   - Comparison with linear baseline (does nonlinear structure matter?)

9. **RL extension** (optional) — Frame the cross-platform arbitrage problem as a Markov Decision Process: state = (prices, volumes, time features on both platforms), action = (buy/sell/hold on each), reward = P&L net of transaction costs. Train a PPO or SAC agent on historical data.

### Expected Contribution

- First copula-based analysis of dependence between centralized and decentralized prediction markets
- Quantification of tail dependence: do these structurally different venues share systemic risk?
- Evidence on price discovery leadership: does the regulated or the decentralized market incorporate information first?
- Practical arbitrage strategy evaluation with realistic transaction costs and latency
- SHAP-based interpretability revealing which microstructural features (volume, time-of-day, buy/sell imbalance) drive cross-platform predictability
- Novel finding if the 24-hour Polymarket leads the US-hours Kalshi during overnight periods (or vice versa during US business hours)

### Key Data Requirements

- Matched-event identification across platforms (market metadata linkage)
- Time-synchronized price series at sub-hourly frequency
- Transaction costs on each platform (Kalshi: per-contract fees; Polymarket: gas + exchange fees)
- Sufficient matched events for statistical power (even a few hundred high-activity matched events with thousands of price observations each would suffice)

---

## Summary Table

| # | Hypothesis | Primary Methods | Key Data Patterns Exploited | Academic Novelty |
|---|---|---|---|---|
| H1 | Microstructure-corrected martingale efficiency | Variance ratio, BDS, Roll's spread | Serial corr −0.34 to −0.47; bid-ask bounce | First CeFi vs DeFi efficiency comparison at scale |
| H2 | EVT tail risk characterization | POT/GPD, GEV, Hill estimator | Kurtosis = 393; extreme price concentration divergence (2.65×) | First EVT analysis of prediction markets; regulatory-tail nexus |
| H3 | Cross-platform dependence and ML arbitrage | Copulas, XGBoost, Granger causality | 24× vs 1.37× intraday patterns; structural divergence on all dimensions | First copula + ML arbitrage study across CeFi/DeFi prediction markets |

---

## Feasibility Notes

- **Computational**: All analyses run on the existing 36GB parquet dataset using DuckDB. EVT and copula fitting require Python (scipy, copulas, arch). XGBoost is standard.
- **Statistical power**: With 476M+ trades, even rare tail events have thousands of observations. The 10 most liquid Kalshi markets alone provide 1.26M price changes.
- **Matched events**: The overlap period (Mar 2023 – Nov 2025) includes several high-profile events traded on both platforms (2024 US presidential election, major sports, crypto/economic events), though exact matching requires metadata work.
- **Publication fit**: H1 targets quantitative finance venues (JFDS, JFE). H2 targets risk/insurance (JBF, IME) or financial ML (ICAIF). H3 targets ML/NeurIPS (Datasets & Benchmarks) given the dataset contribution angle, or ICAIF for the finance-ML intersection.
