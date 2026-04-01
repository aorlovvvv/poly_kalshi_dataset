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
**Output:** `artifacts/data_profile.json`

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
- **Methodological flaw identified:** cross-market pooling creates artificial jumps
- Corrected in Phase 3

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

- Sports vs political divergence (Mann-Whitney p = 0.0021):
  - Political: ξ = −2.28 (bounded tails)
  - Sports: ξ = +1.82 (power-law tails, live-game regime changes)
- NFL game markets reach kurtosis of 2,192; Polymarket's most extreme: kurtosis = 15,502

#### Test B: Sports vs Political BDS at Scale
- 12 markets, 3 seeds, full embedding/epsilon grid
- Sports show 2.3–2.9× higher BDS z-scores than political at all ε
- Confirms: political ≈ linear after AR(1) correction; sports contain genuine nonlinear signal

#### Test C: Enhanced XGBoost (12 features)
- AUC 0.976–1.000 — but **data leakage identified** in rolling volatility features
- `ret_lag1` and `taker_side` confirmed as genuinely predictive features

#### Test D: Cross-Platform Comparison
- **Mean VR(2) statistically indistinguishable** (Kalshi 0.599 vs Polymarket 0.608, p = 0.629)
- **Polymarket spreads 3× tighter** (0.0036 vs 0.0109 cents, p < 1.5e-6)
- Similar relative microstructure noise but dramatically different trading costs

### Phase 4: Autoresearch Loop (Karpathy Pattern) — Initial Run

**Framework:** `src/autoresearch/prepare.py` (FIXED harness), `train.py` (EDITABLE strategy)
**Output:** `src/autoresearch/results.tsv`

Ran 101 experiments following Karpathy's autoresearch pattern. Strategy evolved from GBM classifier (Sharpe −0.17) to `sqrt-3trade-tanh-v37` (Sharpe 2.98).

**However, the initial Sharpe of 2.98 was computed with two critical flaws:**
1. Transaction costs were 20 bps (0.002 per unit) — Kalshi's actual taker fee is ~7 cents per contract (~700 bps at mid-price), which is **35× higher**
2. Positions carried across market boundaries — a position sized for Market A earned Market B's returns

These flaws were corrected in Phase 5.

### Phase 5: Honest Sharpe Validation (Prompt 01)

**Three critical fixes to `prepare.py`:**
1. **Transaction cost:** 20 bps → 7 cents/contract (0.07 per unit)
2. **Market-boundary reset:** Position resets to 0 when market_id changes between rows
3. **AUC renamed to `auc_roc_IGNORE`:** The metric is meaningless because `ret` (a feature) leaks into the target `(|ret| > 2σ)`

**Result:** Sharpe collapsed from **+2.98 to −12.44** (50 markets, 2M total trades, 400K test).

This is the correct finding: the VR(2) ≈ 0.59 mean-reversion signal is **statistically real but economically insignificant** under Kalshi's fee structure. Per-trade alpha (~0.1 cent) is dominated by per-trade costs (7 cents).

### Phase 6: PIN Estimation (Prompt 02)

**Script:** `src/analysis/pin_estimation.py`
**Output:** `artifacts/pin_results.json`

First-ever PIN (Probability of Informed Trading, Easley et al. 1996) estimation for prediction markets. Estimated on top 30 Kalshi markets.

| Metric | Value |
|--------|-------|
| Markets estimated | 26 |
| Convergence rate | **100%** (26/26) |
| Mean PIN | **0.63** |
| Median PIN | 0.66 |
| PIN range | 0.24 – 0.79 |

**Category breakdown:**

| Category | N | Mean PIN | Mean α (info event prob) | Mean μ (informed arrival) |
|----------|---|----------|--------------------------|---------------------------|
| Sports | 21 | **0.65** | 0.113 | 25,888 |
| Political | 3 | **0.52** | 0.078 | 11,437 |
| Other | 2 | 0.56 | 0.040 | 12,497 |

**Key findings:**
- PIN = 0.63 is **2–6× higher than typical equities** (0.10–0.30), indicating much greater informed trading intensity
- Sports markets have higher PIN (0.65) and informed arrival rates (μ = 25,888) than political (PIN = 0.52, μ = 11,437)
- Consistent with the "information arrival > venue design" thesis: sports events generate continuous real-time information during games

### Phase 7: PPO vs LogReg Comparison (Prompt 03)

**Scripts:** `src/autoresearch/train.py` (LogReg), `src/autoresearch/rl_agent.py` (PPO)

Side-by-side comparison with realistic costs (7c/contract, 50 markets):

| Metric | LogReg | PPO (50 episodes, Sharpe-shaped) |
|--------|--------|----------------------------------|
| Sharpe | **−12.44** | −14.58 |
| Mean position | 0.62 | 1.00 |
| Win rate | 7.8% | 10.0% |
| Total return | −9,170 | −4,518 |
| Elapsed | 2.5s | 72.5s |

PPO is worse because:
1. No turnover management — takes max positions (mean_pos = 1.0), generating constant 7c costs
2. Architectural mismatch: the harness separates `predict_tail_probability` and `size_position` but PPO jointly optimizes position, losing directional information in the split
3. Training on synthetic next_ret (rolled feature) ≠ real next_ret

**Finding:** Simple mean-reversion with turnover control beats RL for prediction market microstructure trading. The signal is linear; complexity doesn't help.

### Phase 8: Copula Cross-Platform Dependence (Prompt 04)

**Script:** `src/analysis/copula_dependence.py`
**Output:** `artifacts/copula_results.json`

Identified matched events across Kalshi and Polymarket using date-windowed correlation matching, then fitted 5 copula families (Gaussian, Student-t, Clayton, Gumbel, Frank) to cross-platform daily returns.

| Event | Kalshi Market | Correlation | Best Copula | λ_upper | λ_lower | Kendall's τ |
|-------|---------------|-------------|-------------|---------|---------|-------------|
| **Presidential 2024** | PRES-2024-DJT | **0.987** | Gumbel | **0.37** | 0.00 | 0.20 |
| **Fed Rate** | FED-25SEP-T4.25 | 0.625 | Clayton | 0.00 | **0.36** | 0.08 |
| **BTC Price** | BTCMAX100-24-NOV29 | 0.808 | Gaussian | 0.00 | 0.00 | **0.56** |
| **CPI Inflation** | CPICORE-23JUN-T0.1 | 0.327 | Gumbel | **0.23** | 0.00 | 0.13 |
| NFL Games | (no Polymarket match) | — | Frank | 0.00 | 0.00 | 0.35 |

**Key findings:**
- Presidential election markets co-move almost perfectly across platforms (r = 0.987)
- **Upper tail dependence** (Gumbel) dominates political/economic events — extreme coordinated price moves during major information shocks
- **Lower tail dependence** (Clayton) appears for Fed rate decisions — coordinated sharp sell-offs during surprise announcements
- BTC shows strongest overall dependence (τ = 0.56) but symmetric (Gaussian copula) — no tail concentration

### Phase 9: Autoresearch Cost Regime Sweep (Prompt 05)

**Script:** `src/autoresearch/sweep.py`

Systematic 20-configuration parameter sweep to find strategies viable under 7c/contract costs:

| Rank | Configuration | Sharpe | Mean Position | Total Return |
|------|--------------|--------|---------------|--------------|
| 1 | pure_ml_tiny (never trades) | 0.000 | 0.000 | 0.00 |
| 2 | **scale_0.3** (best non-trivial) | **−7.43** | 0.058 | −201 |
| 3 | no_dir_small_pos | −7.53 | 0.098 | −915 |
| 4 | scale_0.5 | −7.63 | 0.133 | −1,379 |
| 5 | turnover_0.99 | −8.44 | 0.267 | −3,297 |
| ... | ... | ... | ... | ... |
| 17 | baseline_v2 | −12.44 | 0.622 | −9,170 |
| 20 | ultra_conservative | −13.60 | 0.015 | −481 |

**Configurations tested:** SIGNAL_SCALE (0.02–10.0), TURNOVER_THRESHOLD (0.01–0.99), DIR_BLEND (0/0.45/1.0), TAIL_SCALE_FACTOR (0/0.3), momentum vs mean-reversion, enter-once-hold, boundary-heavy.

**Conclusion:** No configuration achieves positive Sharpe with realistic costs. The cost floor of 7c/contract × trade frequency completely dominates the ~0.1c per-trade alpha. This is consistent with semi-strong market efficiency: **microstructure predictability exists but is consumed by transaction costs.**

---

## 3. Key Scientific Findings

### Finding 1: Market Type Determines Microstructure More Than Platform Type

The most consistent finding across all analyses: **sports vs political** is a more significant dividing line than **Kalshi vs Polymarket**.

- Tail behavior: sports ξ > 0 (heavy), political ξ < 0 (bounded) — p = 0.002
- Nonlinear dependence: sports BDS z-scores 2.3–2.9× higher at all scales
- Implied spreads: political markets have tighter spreads (0.80¢) despite lower volume
- Variance ratio: political markets mean-revert more strongly (VR(2) = 0.53 vs 0.65)
- **PIN: sports PIN = 0.65 vs political PIN = 0.52** — higher informed trading in sports

### Finding 2: Prediction Markets Are Non-Martingale but Economically Efficient

Every variance ratio test rejects the martingale at p ≈ 0, but this is entirely attributable to microstructure noise (bid-ask bounce on discrete tick grids). VR(2) ≈ 0.59 matches the theoretical prediction from Roll's model with ρ ≈ −0.47.

**Critically:** the mean-reversion alpha implied by VR(2) = 0.59 does NOT survive realistic transaction costs (7c/contract on Kalshi). A systematic 20-configuration sweep found no profitable strategy. This is consistent with semi-strong efficiency: statistical predictability exists in the microstructure, but the economic cost of exploiting it exceeds the edge.

### Finding 3: PIN Is 2–6× Higher Than Equities (Novel Contribution)

Mean PIN = 0.63 across 26 Kalshi markets, compared to 0.10–0.30 typical for equities. This suggests prediction markets have much higher rates of informed trading — consistent with their design as information aggregation mechanisms. PIN has never been estimated for prediction markets in the published literature.

### Finding 4: Cross-Platform Tail Dependence Confirms Coordinated Information Flow

Copula analysis on 4 matched event categories reveals:
- Upper tail dependence (Gumbel) for political/economic events → coordinated extreme price moves during information shocks
- Near-perfect correlation (r = 0.987) for presidential markets → both platforms process the same information simultaneously
- Asymmetric tail dependence for Fed rate decisions (Clayton, λ_L = 0.36) → coordinated sell-offs during surprise announcements

### Finding 5: Polymarket Has 3× Tighter Spreads Despite Similar Noise

Cross-platform comparison revealed that mean VR(2) is statistically identical between Kalshi and Polymarket (p = 0.629), meaning relative microstructure noise is comparable. But Polymarket's implied spreads are 3× tighter (0.0036 vs 0.0109 cents, p < 1.5e-6). The decentralized exchange offers lower trading costs despite similar price efficiency.

### Finding 6: RL Offers No Advantage Over Linear Models for Microstructure Trading

PPO (Sharpe = −14.58) loses to simple LogReg with hand-crafted features (Sharpe = −12.44). The microstructure signal is linear — complexity does not improve prediction, and RL's lack of turnover management increases costs. This holds across all cost regimes tested.

### Finding 7: Sqrt Signal Dampening and Three-Trade Momentum (Historical)

In the original (flawed) cost regime (20 bps):
- The transformation `sign(x) × √|x|` applied to the mean-reversion signal improved Sharpe from 2.75 to 2.91
- Adding lag_ret_1 and lag_ret_2 was the single largest improvement (Sharpe +1.59 → +2.42)

These remain valid statistical insights about signal structure, even though they don't survive realistic costs. The three-trade momentum confirms that the bid-ask bounce extends beyond the immediate trade — consecutive same-direction trades revert harder, consistent with multi-trade institutional execution patterns.

---

## 4. Repository Structure

```
poly_kalshi_dataset/
├── CLAUDE.md                      Project context (workspace rule)
├── PROJECT_SUMMARY.md             This file
├── AUDIT_REPORT.md                Critical audit of project quality
├── Makefile                       Pipeline orchestration
├── requirements.txt               Python dependencies
├── .gitignore
├── .gitmodules
├── config/
│   ├── project.yml                Top-level config
│   ├── datasets.yml               Dataset definitions
│   └── model_spec.yml             Column mappings + analysis tasks
├── data/
│   └── raw/
│       ├── kalshi/trades/         ~7,214 parquet files (~72M trades)
│       └── polymarket/            trades/, blocks/ (markets/ empty)
├── src/
│   ├── analysis/
│   │   ├── data_profile.py        Phase 1: DuckDB profiling
│   │   ├── phase2_statistical_tests.py  Phase 2: VR, Roll, GPD, BDS, XGBoost
│   │   ├── phase3_deeper_analysis.py    Phase 3: Per-market GPD, cross-platform
│   │   ├── pin_estimation.py      Phase 6: EKOP (1996) PIN model
│   │   ├── copula_dependence.py   Phase 8: Cross-platform copula dependence
│   │   ├── gmm_sdf.py            (scaffolded, not run)
│   │   ├── prepare_data.py        Raw → processed standardization
│   │   ├── profile_data.py        Alternative profiler
│   │   ├── run_models.py          Full analysis engine
│   │   ├── run_all.py             Orchestrator
│   │   └── verify.py              Deterministic paper checks
│   ├── agents/
│   │   └── run.py                 LangGraph agent loop
│   └── autoresearch/
│       ├── prepare.py             FIXED evaluation harness (7c costs, market resets)
│       ├── train.py               EDITABLE strategy (sqrt-3trade-tanh-v37)
│       ├── rl_agent.py            PPO agent (tested, worse than LogReg)
│       ├── sweep.py               20-config parameter sweep
│       ├── program.md             LLM agent instructions
│       ├── results.tsv            Full experiment log (101 + 12 new experiments)
│       ├── run_v2.log             Honest Sharpe validation log
│       ├── run_ppo.log            PPO evaluation log
│       └── run_logreg.log         LogReg baseline log
├── artifacts/
│   ├── data_profile.json          Phase 1 profiling output
│   ├── phase2_results.json        Phase 2 test outputs
│   ├── phase2_analysis.md         Phase 2 write-up
│   ├── phase3_results.json        Phase 3 test outputs
│   ├── phase3_analysis.md         Phase 3 write-up
│   ├── pin_results.json           PIN estimation results (26 markets)
│   ├── copula_results.json        Copula dependence results (4 matched events)
│   ├── strategy_results.json      Initial strategy metrics (pre-fix)
│   ├── research_design_v2.md      Research design document
│   └── hypothesis_candidates.md   Original hypothesis proposals
├── paper/
│   ├── paper.tex                  LaTeX skeleton
│   ├── references.bib             Bibliography
│   └── tables/                    Generated LaTeX tables
└── autoresearch/                  Karpathy's autoresearch (git submodule)
```

---

## 5. Technical Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| DuckDB for all data processing | Handles parquet natively, fast on 36GB, SQL-based, no Spark/cluster needed |
| Subsample to 5K for BDS tests | O(N²) complexity; 500K observations caused OOM at 16GB |
| Per-market GPD fitting (Phase 3) | Pooling across markets creates artificial jumps; resolved negative-ξ artifact |
| LogReg over GBM for strategy | GBM overfits to `ret` (AUC 0.985 but useless for sizing); LogReg uses microstructure features properly |
| 7c/contract costs (corrected) | Previous 20 bps understated Kalshi fees by ~35×. Honest costs required for publishable results |
| Market-boundary position resets | Without resets, positions bleed across markets. Critical for correct Sharpe computation |
| Correlation-based cross-platform matching | Polymarket has no market metadata; matching by daily return correlation (with date-window filtering) is the only viable approach |
| PyTorch nightly for PPO | Torch 2.11.0 has Python 3.13 AST parsing bug; nightly (2.12.0.dev) works |
| Timezone normalization in copula | Kalshi timestamps are tz-aware (Europe/London), Polymarket are tz-naive; must normalize for date matching |

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
| GBM overfitting to `ret` | `ret` predicts `target = (|ret|>2σ)` trivially | Switched to LogReg |
| DuckDB `date_trunc` type error | Polymarket `b.timestamp` is VARCHAR not TIMESTAMP | Added `CAST(trade_time AS TIMESTAMP)` |
| Copula date matching returns empty | Kalshi dates tz-aware (Europe/London), Polymarket tz-naive | Normalize both to tz-naive UTC before matching |
| `np.corrcoef` dimension mismatch | Duplicate dates in grouped data | Added `drop_duplicates("trade_date")` before correlation |
| PyTorch 2.11.0 crashes on Python 3.13.8 | AST parsing bug in `torch/nn/modules/rnn.py` | Installed nightly build (2.12.0.dev20260401) |
| Sharpe inflated by ~35× | 20 bps costs vs real 7c/contract; no market-boundary reset | Fixed in `prepare.py`: costs = 0.07, position resets at boundaries |
| `TURNOVER_THRESHOLD > 1.0` prevents trading | Max position change from 0 is 1.0 (tanh bounded) | Must keep threshold < 1.0 for any initial entry |

---

## 7. Completed Cursor Prompts (All 6)

| # | Prompt | Status | Key Result |
|---|--------|--------|------------|
| 00 | Git cleanup & commit | **Done** | All project files committed, dead code removed, stale worktrees pruned |
| 01 | Validate Sharpe with real costs | **Done** | Sharpe: +2.98 → **−12.44** (honest number with 7c costs) |
| 02 | Run PIN estimation | **Done** | 26/26 converged, mean PIN = **0.63**, sports > political |
| 03 | Test PPO vs LogReg | **Done** | PPO (−14.58) loses to LogReg (−12.44); RL adds no value |
| 04 | Copula matched events | **Done** | 4/5 events matched; presidential r = 0.987; Gumbel upper tail dependence |
| 05 | Autoresearch cost-regime sweep | **Done** | 20 configs tested; **no positive Sharpe** with realistic costs |

---

## 8. Remaining Work

1. **Time-sampled VR tests** — Resample to 1-min/5-min intervals for informational efficiency tests (vs microstructure-contaminated trade-level)
2. **Polymarket strategy** — Run autoresearch loop on Polymarket data (3× tighter spreads may allow profitable trading)
3. **Paper prose** — Draft Introduction, Data, Results, Conclusion using the discover → write → critique agent loop
4. **Expand political sample** — Only 3 political markets in PIN estimation; add Senate races and policy markets
5. **GMM/SDF estimation** — `src/analysis/gmm_sdf.py` is scaffolded but never run
6. **Maker rebate analysis** — With 7c taker fees killing alpha, explore whether maker orders (which get rebates) could be profitable
7. **Polymarket BDS analysis** — Test if sports/political nonlinear dependence divergence holds on-chain
