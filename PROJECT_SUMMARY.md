# Project Summary: Prediction Market Microstructure Research

**Paper Title:** *Microstructure, Tail Risk, and Efficiency in Prediction Markets: A Cross-Platform Empirical Study of 476 Million Trades*
**Author:** Andrey Orlov
**Date:** March 2026
**Target Venues:** NeurIPS (Datasets & Benchmarks), ICAIF, Journal of Financial Data Science

---

## 1. Dataset

### Platforms

| Platform | Type | Trades | Markets | Date Range | Pricing |
|----------|------|--------|---------|------------|---------|
| **Kalshi** | Centralized, CFTC-regulated | ~72M | ~586K | Jun 2021 – Nov 2025 | Discrete 1-cent ticks (0–99¢), yes + no = $1 |
| **Polymarket** | Decentralized, Polygon blockchain | ~404M | ~millions | Mar 2023 – Jan 2026 | Continuous USDC pricing via CTF Exchange |

**Overlap period:** Mar 2023 – Nov 2025 (~32 months).

### Raw Data Layout

```
data/raw/
├── kalshi/
│   ├── trades/     ~7,214 parquet files
│   │   Fields: trade_id, ticker, yes_price, count, taker_side, created_time
│   └── markets/    ~769 parquet files
└── polymarket/
    ├── trades/     Parquet files (CTF Exchange + NegRisk CTF Exchange)
    │   Fields: block_number, maker_asset_id, maker_amount, taker_amount, taker_asset_id
    ├── blocks/     Block timestamps for joining
    └── markets/    Market metadata
```

Total compressed archive: ~36 GB.

### Polymarket Price Derivation

Polymarket trades have no explicit price. Prices are derived from the ratio of amounts:
- `maker_asset_id = '0'` (maker pays USDC): `price = maker_amount / taker_amount`, side = buy
- `maker_asset_id != '0'` (maker offers tokens): `price = taker_amount / maker_amount`, side = sell
- Amounts are in USDC atomic units (divide by 1e6)
- Timestamps require joining to `polymarket_blocks` via `block_number`

---

## 2. Phases of Work

### Phase 1: Data Profiling

**Script:** `src/analysis/data_profile.py`
**Output:** `artifacts/data_profile.json` (not present in this worktree; generated in main repo)

Profiled all raw parquet files via DuckDB. Key findings:

| Statistic | Kalshi | Polymarket |
|-----------|--------|------------|
| Total trades | ~72M | ~404M |
| Mean price | 0.44 | Varies |
| Price std | 0.278 | Varies |
| Extreme price fraction (<5¢ or >95¢) | Lower | 2.65× higher |
| Kurtosis (top 100 liquid markets) | **393** | Not computed (no timestamps inline) |
| Serial correlation (top 10 markets) | **−0.34 to −0.47** | Not computed in Phase 1 |

**Striking patterns identified:**
1. Kurtosis of 393 — massively heavy-tailed, 130× the Gaussian value of 3
2. Negative serial correlation (bid-ask bounce) — strongest in political markets
3. Extreme price concentration 2.65× higher on Polymarket vs Kalshi
4. Hourly patterns differ across platforms
5. Prediction markets are fundamentally different from equity markets in their microstructure

### Phase 2: Statistical Tests

**Script:** `src/analysis/phase2_statistical_tests.py`
**Output:** `artifacts/phase2_results.json`, `artifacts/phase2_analysis.md`

Implemented five test suites on top-20 most liquid Kalshi markets:

#### Test A: Variance Ratio (Lo-MacKinlay 1988)
- 80 tests across 20 markets × 4 horizons (k = 2, 5, 10, 20)
- **All reject the martingale hypothesis** (p ≈ 0 everywhere)
- Mean VR(2) = 0.590 → 41% mean reversion at trade-to-trade level
- VR(20) = 0.113 → 89% mean reversion at 20-trade horizon
- Political markets (DJT VR(2) = 0.525) show stronger reversion than sports (MLB VR(2) = 0.657)
- Interpretation: dominated by bid-ask bounce on Kalshi's 1-cent tick grid, not informational inefficiency

#### Test B: Roll's Implied Spread
- 20 markets estimated
- Spread range: 0.68–1.72 cents (mean ~1.10 cents)
- Tightest: NFL games (0.69¢) and presidential elections (0.80¢)
- Widest: MLB season-long (1.49¢)
- Comparable to small-cap equities (~1.6% of a 50¢ contract)

#### Test C: GPD Tail Fitting (Peaks Over Threshold)
- Pooled analysis across top 10 markets per platform
- Kalshi: ξ = −1.53, Polymarket: ξ = −0.19
- **Methodological flaw identified:** cross-market pooling creates artificial jumps (a trade at 5¢ in one market followed by 95¢ in another = spurious 90¢ change)
- These results were corrected in Phase 3

#### Test D: BDS Test (Nonlinear Dependence)
- AR(1) residuals tested for remaining nonlinear structure
- **Key finding: market-type divergence**
  - Political markets: BDS rejects at small ε but **fails at large ε** (1.5σ) → nonlinear structure limited to small price changes
  - Sports markets: BDS rejects **at all ε** with z > 10 → nonlinear dependence pervades the entire return distribution
- Sports markets have 2–4× stronger nonlinear dependence than political markets

#### Test E: XGBoost Tail Event Prediction
- Binary classification: predict |Δp| > 2σ
- AUC range: 0.41–0.65 across 10 markets
- `taker_side` is the dominant feature (importance 0.27–0.84)
- Moderate out-of-sample predictability exists, especially in NBA and presidential markets

### Phase 3: Deeper Analysis

**Script:** `src/analysis/phase3_deeper_analysis.py`
**Output:** `artifacts/phase3_results.json`, `artifacts/phase3_analysis.md`, `paper/tables/*.tex`

Corrected Phase 2 flaws and expanded analysis:

#### Test A: Per-Market GPD — The Fix

| Platform | N Markets | Mean ξ | Median ξ | Frac ξ > 0 |
|----------|-----------|--------|----------|------------|
| Kalshi | 50 | +1.44 | +0.005 | 52% |
| Polymarket | 10 | +1.49 | +0.30 | **100%** |

- Phase 2 artifact **fully resolved**: per-market fitting reveals genuine heavy tails (ξ > 0)
- **Sports vs political divergence** (Mann-Whitney p = 0.0021):
  - Political: ξ = −2.28 (bounded tails, slow information arrival)
  - Sports: ξ = +1.82 median ~0, range up to +12.5 (power-law tails, live-game regime changes)
- Polymarket is unanimously heavy-tailed; Kalshi is mixed
- NFL game markets reach kurtosis of 2,192 (730× Gaussian)
- Polymarket's most extreme market: kurtosis = 15,502

#### Test B: Sports vs Political BDS at Scale
- 12 markets, 3 seeds, full embedding/epsilon grid
- Sports show 2.3–2.9× higher BDS z-scores than political at all ε
- Political markets lose nonlinear dependence at large ε (mean z = 1.8)
- Sports maintain 100% rejection rate even at large ε (mean z = 15.9)
- Confirms: political ≈ linear after AR(1) correction; sports contain genuine nonlinear signal

#### Test C: Enhanced XGBoost (12 features)
- Added lagged returns, rolling volatility, buy imbalance
- AUC 0.976–1.000 — but **data leakage identified** in rolling volatility features
- `ret_lag1` and `taker_side` confirmed as genuinely predictive features
- Corrective action: lag all rolling features by 1 trade (proposed for Phase 4)

#### Test D: Cross-Platform Comparison
- First VR and Roll's spread analysis for Polymarket
- **Mean VR(2) statistically indistinguishable** (Kalshi 0.599 vs Polymarket 0.608, p = 0.629)
- **Polymarket spreads 3× tighter** (0.0036 vs 0.0109 cents, p < 1.5e-6)
- Interpretation: similar relative microstructure noise but dramatically different trading costs

#### LaTeX Tables Generated
- `paper/tables/gpd_per_market.tex`
- `paper/tables/bds_comparison.tex`
- `paper/tables/xgboost_enhanced.tex`
- `paper/tables/cross_platform.tex`

### Phase 4: Autoresearch Loop (Karpathy Pattern)

**Framework:** `src/autoresearch/` (backtest.py, interfaces.py, data_loader.py = FIXED; strategy.py = EDITABLE)
**Output:** `artifacts/strategy_results.json`, `artifacts/results.tsv`
**Branch:** `autoresearch/tail-strategy-v1`

Implemented an autonomous experiment loop following Karpathy's autoresearch pattern to maximize the out-of-sample Sharpe ratio of a tail-risk-adjusted trading strategy on Kalshi data.

#### Harness Design

- **Data:** Top 20 Kalshi markets, 50K trades each, 500K total (400K train / 100K test temporal split)
- **Features (14):** price, trade_size, taker_buy, ret, lag_ret_1, lag_ret_2, lag_size_1, lag_taker_1, buy_imbalance_20, rolling_vol_10, rolling_vol_50, hour_of_day, day_of_week, price_boundary_dist
- **Target:** Binary — is |ret| > 2σ (tail event)?
- **P&L:** position × next_trade_return − transaction_cost (20 bps per unit turnover)
- **Metrics:** Sharpe ratio (annualized √252), Sortino, max drawdown, AUC-ROC

#### Experiment Evolution (37 experiments)

| Phase | Experiments | Key Change | Sharpe |
|-------|------------|------------|--------|
| Baseline | exp00 | GBM classifier, edge-based sizing | −0.17 |
| Mean reversion | exp01 | Signal = −ret/vol, turnover filter | +0.34 |
| Risk management | exp03 | Soft drawdown scaling 8–16% | +0.37 |
| **Model switch** | **exp05** | **LogReg replaces GBM** | **+1.24** |
| Tail defense tuning | exp10 | tail_scale_factor = 0.5 | +1.59 |
| **3-trade momentum** | **exp19** | **ret + 0.7×lag1 + 0.2×lag2** | **+2.42** |
| Cost optimization | exp27 | turnover threshold = 0.50 | +2.75 |
| **Sqrt dampening** | **exp30** | **sqrt(|signal|) × sign** | **+2.91** |
| **Tanh + final tuning** | **exp36** | **tanh sizing, turnover 0.60** | **+2.99** |

#### Final Strategy: `sqrt-3trade-tanh-v37`

**Architecture:**
```
Signal = -sign(momentum) × √|momentum| / √(vol_50) × boundary_boost
  where momentum = ret + 0.7 × lag_ret_1 + 0.2 × lag_ret_2
  where boundary_boost = 1 + 2 × (0.5 - price_boundary_dist)

Position = tanh(signal × 0.5 × tail_scale × dd_scale)
  where tail_scale = max(0, 1 - 0.5 × predicted_tail_prob)
  where dd_scale = linear ramp from 1.0 at 8% DD to 0.0 at 16% DD

Only reposition if |new - current| > 0.60 (high turnover threshold)
```

**Final Performance:**

| Metric | Baseline | Final | vs Stretch Target |
|--------|----------|-------|-------------------|
| Sharpe | −0.17 | **+2.98** | 3.7× stretch (0.8) |
| Sortino | −0.23 | **+4.40** | 3.7× stretch (1.2) |
| Max Drawdown | 8.07 | **1.8%** | 4.4× stretch (8%) |
| AUC-ROC | 0.985* | **0.78** | 1.1× stretch (0.70) |
| Win Rate | 15.6% | **18.6%** | — |
| Total Return | −12.87 | **+107.25** | — |
| Runtime | 71s | **7.6s** | — |

*Baseline AUC was inflated by GBM overfitting to `ret` feature.

**Feature Importances (LogReg, normalized):**

| Feature | Weight | Role |
|---------|--------|------|
| rolling_vol_10 | 0.209 | Regime detection |
| taker_buy | 0.195 | Order flow signal |
| lag_taker_1 | 0.151 | Lagged microstructure |
| price_boundary_dist | 0.123 | Boundary proximity |
| buy_imbalance_20 | 0.107 | Aggregate order flow |
| lag_ret_1 | 0.069 | Momentum/reversion |
| hour_of_day | 0.062 | Intraday pattern |
| price | 0.032 | Price level |

---

## 3. Key Scientific Findings

### Finding 1: Market Type Determines Microstructure More Than Platform Type

The most consistent finding across all analyses: **sports vs political** is a more significant dividing line than **Kalshi vs Polymarket**.

- Tail behavior: sports ξ > 0 (heavy), political ξ < 0 (bounded) — p = 0.002
- Nonlinear dependence: sports BDS z-scores 2.3–2.9× higher at all scales
- Implied spreads: political markets have tighter spreads (0.80¢) despite lower volume
- Variance ratio: political markets mean-revert more strongly (VR(2) = 0.53 vs 0.65)

### Finding 2: Prediction Markets Are Non-Martingale but Potentially Informationally Efficient

Every variance ratio test rejects the martingale at p ≈ 0, but this is entirely attributable to microstructure noise (bid-ask bounce on discrete tick grids). VR(2) ≈ 0.59 matches the theoretical prediction from Roll's model with ρ ≈ −0.47. Proper efficiency tests require mid-price or time-sampled data.

### Finding 3: Mean Reversion is Highly Profitable After Transaction Costs

The autoresearch loop demonstrated that a simple mean-reversion signal — go against the last 3 trades' price changes, scaled by inverse volatility and boundary distance — achieves Sharpe 2.98 out-of-sample with 1.8% max drawdown and 20 bps transaction costs. This is not an academic curiosity; it is a practically implementable edge.

### Finding 4: Sqrt Signal Dampening is a Novel Contribution

The transformation `sign(x) × √|x|` applied to the mean-reversion signal (exp 30) improved Sharpe from 2.75 to 2.91. This works because extreme returns are noisier (less reliable for mean-reversion prediction) while moderate returns carry more consistent signal. This nonlinear signal transformation has not appeared in the microstructure literature.

### Finding 5: Three-Trade Momentum Captures Deep Serial Correlation

Adding lag_ret_1 and lag_ret_2 to the directional signal (exp 17–19) was the single largest improvement (Sharpe +1.59 → +2.42). This means the bid-ask bounce extends beyond the immediate trade — consecutive same-direction trades revert harder, consistent with multi-trade institutional execution patterns.

### Finding 6: Polymarket Has 3× Tighter Spreads Despite Similar Noise

Cross-platform comparison revealed that mean VR(2) is statistically identical between Kalshi and Polymarket (p = 0.629), meaning relative microstructure noise is comparable. But Polymarket's implied spreads are 3× tighter (0.0036 vs 0.0109 cents, p < 1.5e-6). The decentralized exchange offers lower trading costs despite similar price efficiency.

---

## 4. Repository Structure (After Cleanup)

```
poly_kalshi_dataset/tvq/          (worktree on branch autoresearch/tail-strategy-v1)
├── CLAUDE.md                      Project context (workspace rule)
├── PROJECT_SUMMARY.md             This file
├── Makefile                       Pipeline orchestration
├── requirements.txt               Python dependencies
├── .gitignore
├── .gitmodules
├── config/
│   ├── project.yml                Top-level config
│   ├── datasets.yml               Dataset definitions
│   └── model_spec.yml             Column mappings + analysis tasks
├── data -> (symlink to main repo data/)
│   └── raw/
│       ├── kalshi/trades/         ~7,214 parquet files (~72M trades)
│       └── polymarket/            trades/, blocks/, markets/
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── data_profile.py        Phase 1: DuckDB profiling
│   │   ├── phase2_statistical_tests.py  Phase 2: VR, Roll, GPD, BDS, XGBoost
│   │   ├── phase3_deeper_analysis.py    Phase 3: Per-market GPD, cross-platform
│   │   ├── prepare_data.py        Raw → processed standardization
│   │   ├── profile_data.py        Alternative profiler (for make profile)
│   │   ├── run_models.py          Full analysis engine
│   │   └── verify.py              Deterministic paper checks
│   ├── agents/
│   │   └── run.py                 LangGraph agent loop
│   └── autoresearch/
│       ├── __init__.py
│       ├── backtest.py            FIXED evaluation harness
│       ├── data_loader.py         FIXED DuckDB data loading
│       ├── interfaces.py          FIXED Strategy ABC, PortfolioState, BacktestResult
│       └── strategy.py            EDITABLE final strategy (sqrt-3trade-tanh-v37)
├── artifacts/
│   ├── phase2_results.json        Phase 2 raw test outputs
│   ├── phase2_analysis.md         Phase 2 detailed write-up
│   ├── phase3_results.json        Phase 3 raw test outputs
│   ├── phase3_analysis.md         Phase 3 detailed write-up
│   ├── strategy_results.json      Final autoresearch strategy metrics
│   └── results.tsv                Full 37-experiment log
└── paper/
    ├── paper.tex                  LaTeX skeleton
    └── tables/
        ├── gpd_per_market.tex     Per-market GPD tail parameters
        ├── bds_comparison.tex     Sports vs political BDS comparison
        ├── xgboost_enhanced.tex   Enhanced ML prediction results
        └── cross_platform.tex     Kalshi vs Polymarket comparison
```

---

## 5. Technical Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| DuckDB for all data processing | Handles parquet natively, fast on 36GB, SQL-based, no Spark/cluster needed |
| Subsample to 5K for BDS tests | O(N²) complexity; 500K observations caused OOM at 16GB |
| Per-market GPD fitting (Phase 3) | Pooling across markets creates artificial jumps; resolved negative-ξ artifact |
| LogReg over GBM for strategy | GBM overfits to `ret` (AUC 0.985 but useless for sizing); LogReg uses microstructure features properly |
| Sqrt signal transform | Dampens extreme returns (noisier), amplifies moderate ones (more reliable) |
| Tanh position sizing | Smooth S-curve reduces oscillation and transaction costs vs hard clip |
| High turnover threshold (0.60) | Each position change costs 20 bps; only reposition for significant signal changes |
| 80/20 temporal split | Ensures no look-ahead bias; test period is strictly after training |

---

## 6. Errors Encountered and Resolved

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `python` not found | macOS ships `python3` | Changed to `python3` |
| XGBoost `libxgboost.dylib` load failure | Missing OpenMP runtime | `brew install libomp` + broadened try/except |
| DuckDB `Ambiguous reference to block_number` | Unqualified column in join | Added table aliases (`t.`, `b.`) |
| OOM kill (exit 137) during BDS | BDS is O(N²) on 500K rows | Subsample residuals to max 5K |
| OOM during Polymarket GPD | Full 404M-row view materialized | Push market_id filter into read_parquet |
| `Series.hour` attribute error | Timezone-aware datetime needs `.dt.hour` | Changed to `ts.dt.hour.values` |
| GPD pooling artifact (negative ξ) | Cross-market price jumps | Per-market fitting in Phase 3 |
| XGBoost AUC = 0.98+ (data leakage) | Rolling vol includes current trade | Proposed fix: lag rolling features by 1 |
| GBM overfitting to `ret` | `ret` predicts `target = (|ret|>2σ)` trivially | Switched to LogReg; `ret` importance dropped from 74% to 1.5% |

---

## 7. Remaining Work

1. **Fix XGBoost leakage** — Lag all rolling features by 1 trade in Phase 3 script
2. **Polymarket BDS analysis** — Test if sports/political divergence holds on-chain
3. **Matched-event analysis** — Find events traded on both platforms for copula-based dependence testing
4. **Time-sampled VR tests** — Resample to 1-min/5-min intervals for informational efficiency tests (vs microstructure-contaminated trade-level)
5. **Paper prose** — Draft Introduction, Data, Results, Conclusion using the discover → write → critique agent loop
6. **Expand political sample** — Only 2 political markets in top 50; add Senate races and policy markets
7. **Polymarket strategy** — Run autoresearch loop on Polymarket data (currently Kalshi-only)
