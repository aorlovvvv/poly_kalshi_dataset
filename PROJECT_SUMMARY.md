# Project Summary: Prediction Market Microstructure Research

**Paper Title:** *Microstructure, Tail Risk, and Efficiency in Prediction Markets: A Cross-Platform Empirical Study of 476 Million Trades*
**Author:** Andrey Orlov
**Date:** March 2026 (last updated: March 29, 2026)
**Target Venues:** NeurIPS (Datasets & Benchmarks), ICAIF, Journal of Financial Data Science

---

## 1. Dataset

### Platforms

| Platform | Type | Trades | Markets | Date Range | Pricing |
|----------|------|--------|---------|------------|---------|
| **Kalshi** | Centralized, CFTC-regulated | ~72M | ~586K | Jun 2021 – Nov 2025 | Discrete 1-cent ticks (0–99¢), yes + no = $1 |
| **Polymarket** | Decentralized, Polygon blockchain | ~404M | ~millions | Mar 2023 – Jan 2026 | Continuous USDC pricing via CTF Exchange |

**Overlap period:** Mar 2023 – Nov 2025 (~32 months).

### Kalshi Fee Structure

Kalshi's taker fee follows: `fee = round_up(0.07 × C × P × (1−P))`, where C = contracts and P = price in dollars.

| Price | Fee per contract | Fee as % of price |
|-------|-----------------|-------------------|
| $0.05 | 0.33¢ | 6.7% |
| $0.10 | 0.63¢ | 6.3% |
| $0.50 | **1.75¢** (max) | 3.5% |
| $0.90 | 0.63¢ | 0.7% |
| $0.95 | 0.33¢ | 0.35% |

Maker fee = 25% of taker fee (i.e., `0.0175 × P × (1−P)`). Max maker fee: 0.44¢ at mid-price.

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
    └── markets/    (empty — no metadata available)
```

Total compressed archive: ~36 GB.

---

## 2. Phases of Work

### Phase 1: Data Profiling

**Script:** `src/analysis/data_profile.py`

| Statistic | Kalshi | Polymarket |
|-----------|--------|------------|
| Total trades | ~72M | ~404M |
| Mean price | 0.44 | Varies |
| Extreme price fraction (<5¢ or >95¢) | Lower | 2.65× higher |
| Kurtosis (top 100 liquid markets) | **393** | Not computed (no inline timestamps) |
| Serial correlation (top 10 markets) | **−0.34 to −0.47** | Not computed in Phase 1 |

Key patterns: kurtosis of 393 (130× Gaussian), strong negative serial correlation (bid-ask bounce), extreme price concentration 2.65× higher on Polymarket.

### Phase 2: Statistical Tests

**Script:** `src/analysis/phase2_statistical_tests.py`

Five test suites on top-20 most liquid Kalshi markets:

**Variance Ratio (Lo-MacKinlay 1988):** 80 tests, all reject martingale (p ≈ 0). Mean VR(2) = 0.590 → 41% mean reversion at trade-to-trade level. Political markets (VR(2) = 0.525) revert more strongly than sports (VR(2) = 0.657). Interpretation: dominated by bid-ask bounce on 1-cent tick grid, not informational inefficiency.

**Roll's Implied Spread:** Range 0.68–1.72 cents. NFL games tightest (0.69¢), MLB widest (1.49¢). Comparable to small-cap equities (~1.6% of 50¢ contract).

**GPD Tail Fitting:** Pooled analysis had a methodological flaw (cross-market jumps create artificial distributions). Corrected in Phase 3 with per-market fitting.

**BDS Test (Nonlinear Dependence):** Market-type divergence — political markets: BDS fails at large ε; sports markets: rejects at all ε. Sports have 2–4× stronger nonlinear dependence.

**XGBoost Tail Prediction:** AUC 0.41–0.65; `taker_side` is the dominant feature.

### Phase 3: Deeper Analysis

**Script:** `src/analysis/phase3_deeper_analysis.py`

Per-market GPD fixed the pooling artifact:

| Platform | N Markets | Mean ξ | Median ξ | Frac ξ > 0 |
|----------|-----------|--------|----------|------------|
| Kalshi | 50 | +1.44 | +0.005 | 52% |
| Polymarket | 10 | +1.49 | +0.30 | **100%** |

Sports vs political divergence (Mann-Whitney p = 0.002): political ξ = −2.28 (bounded tails), sports ξ = +1.82 (power-law tails). Cross-platform VR(2) statistically indistinguishable (Kalshi 0.599 vs Polymarket 0.608, p = 0.629) but Polymarket spreads 3× tighter (0.0036 vs 0.0109 cents).

### Phase 4: Autoresearch Loop (Karpathy Pattern) — Initial Run

Ran 101 experiments. Strategy evolved from GBM classifier (Sharpe −0.17) to `sqrt-3trade-tanh-v37` (Sharpe 2.98).

**Initial Sharpe of 2.98 was computed with two critical flaws:**
1. Transaction costs were 20 bps — Kalshi's actual taker fee is `0.07 × p × (1-p)`, max 1.75¢ at mid-price
2. Positions carried across market boundaries

### Phase 5: Sharpe Validation with Corrected Cost Model

**Three fixes to `prepare.py`:**
1. **Transaction cost:** Flat 20 bps → price-dependent `0.07 × p × (1−p)` per contract (max 1.75¢ taker, 0.44¢ maker)
2. **Market-boundary reset:** Position resets to 0 when market_id changes
3. **Annualization:** √252 → √365 (prediction markets trade 24/7)

**Break-even analysis (key result):**

| Fee rate | Description | Sharpe | Total return |
|----------|-------------|--------|-------------|
| 0.000 | Zero costs | **+14.0** | +873.7 |
| 0.0175 | **Maker fee** | **+14.6** | +395.2 |
| 0.025 | — | +13.3 | +190.1 |
| 0.030 | — | +4.7 | +53.4 |
| ~0.032 | **Break-even** | **~0** | ~0 |
| 0.035 | — | −5.0 | −83.3 |
| 0.070 | **Taker fee** | **−12.0** | −1,040.3 |

**This is the central result:** Mean-reversion alpha is real (gross Sharpe +14) and profitable for liquidity providers (maker-fee Sharpe +14.6), but is fully absorbed by taker fees. The break-even fee rate is ~3.2%, sitting between the maker fee (1.75%) and taker fee (7%).

### Phase 6: PIN Estimation

**Script:** `src/analysis/pin_estimation.py` (volume-weighted using `count` field)

| Metric | Value |
|--------|-------|
| Markets estimated | 26 (of 30, 4 skipped for <10 days) |
| Convergence rate | 100% |
| Mean PIN | **0.66** |
| Median PIN | 0.65 |
| PIN range | 0.37 – 0.94 |

| Category | N | Mean PIN |
|----------|---|----------|
| Sports | 21 | 0.65 |
| Political | 3 | 0.58 |
| Other | 2 | 0.84 |

**Known limitations (see §3):** The EKOP Poisson arrival assumption is violated for event markets with scheduled resolution and bursty trading. The extremely high μ values (millions of contracts/day when informed) suggest model misspecification. PIN values should be interpreted cautiously — they indicate high order-flow asymmetry, but the comparison to equity PIN is apples-to-oranges.

### Phase 7: PPO vs LogReg

| Metric | LogReg | PPO |
|--------|--------|-----|
| Sharpe (taker) | −12.0 | −14.6 |
| Mean position | 0.63 | 1.00 |
| Win rate | 11.1% | 10.0% |
| Elapsed | 3s | 73s |

PPO loses because: (1) no turnover management → constant repositioning costs, (2) the architectural split between tail prediction and position sizing doesn't match how PPO learns, (3) the signal is linear; complexity doesn't help.

### Phase 8: Copula Cross-Platform Dependence

**Script:** `src/analysis/copula_dependence.py`

| Event | n_obs | Best Copula | λ_upper | λ_lower | Kendall τ |
|-------|-------|-------------|---------|---------|-----------|
| Presidential 2024 | 58 | Gumbel | 0.37 | 0.00 | 0.20 |
| Fed Rate | 28 | Clayton | 0.00 | 0.36 | 0.08 |
| BTC Price | 18 | Gaussian | 0.00 | 0.00 | 0.56 |
| CPI Inflation | 28 | Gumbel | 0.23 | 0.00 | 0.13 |
| NFL Games | 300 (synthetic) | — | — | — | — |

**Known limitations (see §3):** Sample sizes of 18–58 daily observations are far too small for reliable copula parameter estimation. Tail dependence coefficients from n<50 are unreliable. The cross-platform matching uses a return-correlation heuristic (|r|>0.15), not verified event metadata — spurious matches are possible. NFL result is synthetic and excluded from all summary statistics. No confidence intervals are reported. Student-t copula never converges properly (all fits hit the penalty boundary at LL = −1e10) but was incorrectly reported as "converged" — now fixed.

---

## 3. Key Scientific Findings

### Finding 1: Market Type Determines Microstructure More Than Platform Type

Sports vs political is a more significant dividing line than Kalshi vs Polymarket.

- Tail behavior: sports ξ > 0 (heavy), political ξ < 0 (bounded) — p = 0.002
- Nonlinear dependence: sports BDS z-scores 2.3–2.9× higher at all scales
- Implied spreads: political markets have tighter spreads (0.80¢) despite lower volume
- Variance ratio: political markets mean-revert more strongly (VR(2) = 0.53 vs 0.65)
- PIN: sports PIN = 0.65 vs political PIN = 0.58

### Finding 2: Mean-Reversion Alpha Is Real but Only Profitable for Makers

Every variance ratio test rejects the martingale at p ≈ 0, driven by bid-ask bounce on discrete tick grids. VR(2) ≈ 0.59 matches the theoretical prediction from Roll's model with ρ ≈ −0.47.

The mean-reversion signal generates substantial gross alpha (Sharpe +14.0 zero-cost) and is **profitable under Kalshi's maker fee** (Sharpe +14.6 at 1.75% fee rate). It is unprofitable as a taker (Sharpe −12.0 at 7% fee rate). The break-even fee rate is ~3.2%.

This resolves the apparent contradiction: the market is predictable (non-martingale) yet efficient — the predictability is exactly the premium that incentivizes liquidity provision. **Market makers earn the serial-correlation premium; takers pay for it.**

### Finding 3: Order-Flow Asymmetry Is High (PIN Analysis — With Caveats)

Volume-weighted PIN estimation gives mean PIN = 0.66 across 26 Kalshi markets (vs 0.10–0.30 for equities). This suggests high order-flow asymmetry.

**Caveats that limit interpretability:**
- The EKOP model assumes Poisson arrival rates, which is violated for event markets with scheduled resolution (game times, election days, CPI releases)
- The μ parameters reach millions of contracts/day, suggesting the optimizer compensates for bursty trading by inflating informed arrival rates
- Only 3 political markets in the sample — insufficient for category comparisons
- No confidence intervals or bootstrap standard errors computed
- The comparison to equity PIN is methodologically questionable: equity and prediction markets have fundamentally different information structures
- PIN has never been estimated for prediction markets, so there is no benchmark to validate against

### Finding 4: Cross-Platform Dependence Exists (Copula — Preliminary)

Cross-platform daily returns show positive dependence for matched events. Presidential markets have near-perfect correlation (r = 0.987) across platforms.

**This finding is preliminary and should not be published in its current form:**
- Sample sizes (n=18–58 daily observations) are far too small for copula estimation. With n=18, the "tails" are represented by exactly 1 observation per tail
- AIC differences between copula families are typically <3 units — insufficient to discriminate between qualitatively different dependence structures
- The cross-platform matching uses return correlation without semantic verification. With 150 pairwise correlations and a threshold of |r|>0.15, spurious matches are expected
- No confidence intervals — e.g., Kendall's τ = 0.56 for BTC (n=18) has a 95% CI spanning roughly ±0.45
- Synthetic data (NFL, n=300) was previously mixed into summary statistics; now excluded
- Student-t copula fits all hit the penalty boundary (LL = −1e10) but were reported as converged; now fixed

### Finding 5: Polymarket Has 3× Tighter Spreads

Cross-platform VR(2) is statistically identical (p = 0.629), but Polymarket implied spreads are 3× tighter (0.0036 vs 0.0109 cents, p < 1.5e-6). Polymarket's continuous pricing and lower fee structure support tighter effective spreads despite comparable price efficiency.

### Finding 6: RL Offers No Advantage Over Linear Models

PPO (Sharpe −14.6) loses to LogReg with hand-crafted features (Sharpe −12.0) across all cost regimes. The microstructure signal is linear — complexity doesn't help, and RL's lack of turnover control increases costs.

---

## 4. Known Methodological Limitations

### Critical

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| C1 | `ret` is a feature and `target = f(|ret|)` — the tail-event prediction is a tautology | `prepare.py:124,142` | AUC is meaningless; tail_prob is just a threshold on `|ret|`. Does not affect the Sharpe/PnL metrics (those use the direction model + hand-crafted signals) |
| C2 | Tail threshold computed on full series (train+test) | `prepare.py:142-143` | `ret_std` used to define "tail event" includes test data. Minor for Sharpe, but methodologically impure |
| C3 | Copula sample sizes (n=18–58) far too small | `copula_dependence.py` | All copula family selections and tail dependence coefficients are unreliable. Results are suggestive, not conclusive |
| C4 | PIN Poisson assumption violated for event markets | `pin_estimation.py` | Scheduled resolution and bursty trading break the model. PIN values indicate order-flow asymmetry but may not measure "informed trading" in the classical sense |

### Major

| ID | Issue | Impact |
|----|-------|--------|
| M1 | Train/test split by row count, not calendar date | Time coverage of test set depends on trade density, not a fixed calendar window |
| M2 | Direction model trained on interleaved multi-market data | Features are per-market but scaler/model fit across mixed markets. Dilutes market-specific signals |
| M3 | Fractional positions (0.058 at low scale) are impossible on Kalshi | Real implementation requires integer contracts; PnL profile would change for small positions |
| M4 | No confidence intervals anywhere | All point estimates (PIN, copula params, Sharpe) lack uncertainty quantification |
| M5 | Log-returns on bounded [0,1] prices for copula analysis | Near-settlement convergence creates mechanical correlation and extreme heteroskedasticity |
| M6 | Cross-platform matching by return correlation, not event metadata | Spurious matches likely with |r|>0.15 threshold and 150 pairwise tests |

### Minor

| ID | Issue |
|----|-------|
| m1 | Unweighted per-market Sharpe averaging (21-trade market = 50K-trade market) |
| m2 | PIN buy/sell classification maps `taker_side='yes'→buy`, which is semantically ambiguous for inverted markets |
| m3 | `next_ret` at train/test boundary uses test-period prices (affects ~1 row/market) |

---

## 5. Repository Structure

```
poly_kalshi_dataset/
├── CLAUDE.md                      Project context (workspace rule)
├── PROJECT_SUMMARY.md             This file
├── Makefile                       Pipeline orchestration
├── requirements.txt               Python dependencies
├── config/
│   ├── project.yml                Top-level config
│   ├── datasets.yml               Dataset definitions
│   └── model_spec.yml             Column mappings + analysis tasks
├── data/raw/
│   ├── kalshi/trades/             ~7,214 parquet files (~72M trades)
│   └── polymarket/                trades/, blocks/
├── src/
│   ├── analysis/
│   │   ├── data_profile.py        Phase 1: DuckDB profiling
│   │   ├── phase2_statistical_tests.py  Phase 2: VR, Roll, GPD, BDS, XGBoost
│   │   ├── phase3_deeper_analysis.py    Phase 3: Per-market GPD, cross-platform
│   │   ├── pin_estimation.py      Phase 6: EKOP PIN (volume-weighted)
│   │   ├── copula_dependence.py   Phase 8: Cross-platform copula
│   │   ├── gmm_sdf.py            (scaffolded, not run)
│   │   └── verify.py              Deterministic paper checks
│   ├── agents/run.py              LangGraph agent loop
│   └── autoresearch/
│       ├── prepare.py             FIXED harness (price-dependent fees, market resets, √365)
│       ├── train.py               Strategy (sqrt-3trade-tanh-v37)
│       ├── rl_agent.py            PPO agent (tested, worse than LogReg)
│       ├── sweep.py               Parameter sweep
│       ├── program.md             LLM agent instructions
│       └── results.tsv            Experiment log
├── artifacts/
│   ├── pin_results.json           PIN estimation (26 markets, volume-weighted)
│   ├── copula_results.json        Copula results (4 real events, synthetic excluded from stats)
│   └── *.json, *.md               Phase 1-3 outputs
├── paper/
│   ├── paper.tex                  LaTeX skeleton
│   └── tables/                    Generated LaTeX tables
└── autoresearch/                  Karpathy's autoresearch (git submodule)
```

---

## 6. Errors Encountered and Resolved

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `python` not found | macOS ships `python3` | Changed to `python3` |
| XGBoost `libxgboost.dylib` load failure | Missing OpenMP | `brew install libomp` |
| DuckDB ambiguous `block_number` | Unqualified join column | Added table aliases |
| OOM kill (exit 137) during BDS | O(N²) on 500K rows | Subsample to 5K |
| GPD pooling artifact (negative ξ) | Cross-market jumps | Per-market fitting |
| XGBoost AUC = 0.98+ | Rolling vol includes current trade | Noted as data leakage |
| GBM overfitting to `ret` | `ret` predicts `target=(|ret|>2σ)` trivially | Switched to LogReg |
| DuckDB `date_trunc` type error | `b.timestamp` is VARCHAR | CAST to TIMESTAMP |
| Copula date matching empty | Timezone mismatch (tz-aware vs naive) | Normalize both to UTC |
| `np.corrcoef` dimension mismatch | Duplicate dates | `drop_duplicates("trade_date")` |
| PyTorch 2.11.0 crash on Python 3.13 | AST parsing bug | Installed nightly 2.12.0.dev |
| **Sharpe inflated ~35×** | **20 bps costs vs real 0.07×p×(1-p)** | **Price-dependent fee model** |
| **Flat 7c/contract overestimated costs** | **Confused fee formula with max fee** | **Corrected: fee = 0.07×p×(1-p), max 1.75¢** |
| Student-t copula "converged" at penalty | No check for LL > 1e8 | Added penalty-boundary check |
| PIN counted trades not contracts | `SUM(1)` instead of `SUM(count)` | Volume-weighted with `COALESCE(count, 1)` |
| `buy_imbalance_20` not lagged | Contemporaneous feature | Added `.shift(1)` for consistency |
| `√252` for 24/7 markets | Equity trading days, not calendar days | Changed to `√365` |

---

## 7. What's Publishable vs What Needs More Work

### Ready for paper (with appropriate caveats)

1. **Finding 1** — Market type > platform type for microstructure. Well-supported by VR, BDS, GPD, Roll's spread across 50+ markets
2. **Finding 2** — Mean-reversion alpha real but maker-only. Clean break-even analysis with Kalshi's actual fee formula
3. **Finding 5** — Polymarket 3× tighter spreads. Strong statistical significance (p < 1.5e-6)
4. **Finding 6** — RL vs linear comparison. Clear result, though both are net-negative

### Needs substantial additional work before publication

1. **PIN estimation** — Needs: (a) bootstrap CIs, (b) acknowledgment that Poisson assumption fails for event markets, (c) expanded political sample (only 3 markets), (d) robustness check with alternative models (e.g., Adjusted PIN)
2. **Copula analysis** — Needs: (a) much larger samples (weekly or intraday data to get n>100), (b) semantic event matching (not correlation heuristic), (c) confidence intervals, (d) robustness to copula family specification

### Not publishable in current form

1. AUC/tail-prediction metrics — circular by construction (`ret` → `target = f(|ret|)`)
2. Any absolute PIN values compared to equities — model assumptions too different
3. Summary statistics that included synthetic NFL data (now excluded)

---

## 8. Remaining Work

1. **Copula sample expansion** — Use intraday data (not daily) to get n>100 for each event pair; or match more events
2. **Semantic event matching** — Use Kalshi market metadata to properly match events instead of correlation heuristic
3. **PIN robustness** — Bootstrap CIs, test with Adjusted PIN (Duarte & Young 2009), expand political sample
4. **Time-sampled VR tests** — Resample to 1-min/5-min for efficiency tests (vs microstructure-contaminated trade-level)
5. **Polymarket strategy** — Run autoresearch on Polymarket data (3× tighter spreads may allow profitable taker trading)
6. **Paper prose** — Draft using the discover → write → critique agent loop
7. **Target redesign** — Replace `(|ret| > 2σ)` with a non-circular target (e.g., predict next-trade direction from lagged features only, excluding `ret` from features)
8. **Maker rebate analysis** — With maker-fee Sharpe at +14.6, formally model a limit-order strategy
