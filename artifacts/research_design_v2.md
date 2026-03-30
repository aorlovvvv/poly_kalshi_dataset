# Research Design v2: Information, Tails, and Autonomous Discovery in Prediction Markets
## A Cross-Platform Microstructure Study with Autoresearch Methodology

**Version:** 2.0
**Date:** March 30, 2026
**Dataset:** Kalshi (72M trades) + Polymarket (404M trades) = 476M trades, 36GB
**Target Venues:** NeurIPS 2026 (Datasets & Benchmarks), ICAIF 2026, Journal of Financial Data Science

---

## 1. Unified Thesis

**Information arrival process — not venue design — is the primary determinant of prediction market microstructure.**

This thesis is supported by a striking empirical puzzle: Kalshi (centralized, CFTC-regulated, discrete 1-cent tick) and Polymarket (decentralized, on-chain, continuous pricing) exhibit statistically indistinguishable trade-level efficiency (VR(2) = 0.60 vs 0.61, p = 0.63), yet their absolute trading costs diverge by 3× (Roll's spread 1.09 vs 0.36 cents, p < 10⁻⁶). Meanwhile, *within* each platform, sports markets show 2.3–2.9× stronger nonlinear dependence than political markets (BDS z-scores 15–20 vs 2–7), and their tail indices diverge significantly (p = 0.002). The content of the information arrival process — continuous game state (sports) versus discrete news shocks (politics) — explains more microstructure variation than the venue mechanism itself.

This thesis generates three testable research questions that span the methods of modern financial economics: efficiency testing (martingales, variance ratios), tail risk characterization (EVT, copulas), informed trading (PIN/VPIN), and autonomous strategy discovery via the autoresearch paradigm (RL, gradient boosting, ablation).

---

## 2. Literature Positioning and Gaps

### 2.1 What Exists

The prediction market literature has grown rapidly since 2024, driven by the 2024 US presidential election:

**Microstructure:**
- **Whelan (2025)** provides the first institutional analysis of Kalshi's maker-taker fee structure and order flow, but does not perform statistical efficiency tests or cross-platform comparison.
- **Ng, Peng, Tao & Zhou (2025)** establish that Polymarket leads Kalshi in price discovery for the 2024 election, finding information flows from decentralized to regulated markets. They use Hasbrouck's (1995) information share methodology but do not examine tail risk or market-type heterogeneity.
- **Arbutina, Gombina & Kamal (2025)** reconstruct Polymarket on-chain activity for the 2024 presidential election, documenting whale behavior and price impact. They find concentrated positions but do not apply formal microstructure tests.
- **Becker (2024)** documents wealth transfer patterns in prediction markets, finding systematic losses among retail participants.

**Efficiency:**
- **Tsangarides (2021)** tests martingale efficiency of InTrade political markets using likelihood ratio and Bayes factor analysis. The sample is small (a few hundred observations per market) and pre-dates modern platforms.
- **Reichenbach & Walther (2025)** assess Polymarket accuracy and bias, finding calibration close to rational expectations but with detectable overconfidence at extreme prices. They do not perform trade-level microstructure analysis.

**Pricing Theory:**
- **Ibikunle & Moldovan (2024)** develop a unified kernel approach toward Black-Scholes for prediction markets, proposing continuous-time models for binary contract pricing. This is purely theoretical — no empirical validation on trade data.

**Economic Significance:**
- A recent study documents **$40M in arbitrage profits** from cross-platform prediction market discrepancies, establishing that inefficiencies have material economic significance.

### 2.2 What Does NOT Exist (Gaps This Paper Fills)

| Gap | Status in Literature | Our Contribution |
|-----|---------------------|------------------|
| **EVT for prediction markets** | Zero papers apply Extreme Value Theory to prediction market returns | First GPD/GEV/Hill estimation across 60+ markets on two platforms |
| **PIN/VPIN estimation** | No paper estimates probability of informed trading in prediction markets | First PIN estimates; connects 84% taker-side feature importance to Easley-O'Hara framework |
| **Cross-platform tail comparison** | Ng et al. compare price discovery; no one compares tail structure | First formal test of tail index homogeneity across venue types (p = 0.002 for market type) |
| **Autoresearch as methodology** | Karpathy (2026) introduced autoresearch for LLM training; zero finance applications | First application of autonomous experiment loop to financial strategy optimization |
| **Market-type microstructure decomposition** | All existing studies pool markets or focus on elections only | First evidence that information arrival process (sports vs political) dominates venue design |
| **Large-scale trade-level analysis** | Largest existing study: ~1M trades (Arbutina 2025). Most: <100K | 476M trades — 476× larger than largest existing study |
| **Copula cross-platform dependence** | No formal dependence modeling across prediction market venues | First copula estimation for matched events traded on both platforms |

### 2.3 How We Position

We extend four distinct literatures simultaneously:

1. **Market Microstructure (Roll 1984, Hasbrouck 1995, Easley & O'Hara 1987):** We apply classical microstructure tools — Roll's spread, variance ratios, PIN — to a new asset class (binary prediction contracts) and a new venue type (on-chain decentralized exchange). Our VR(2) ≈ 0.60 finding parallels Roll's original finding of bid-ask bounce in equity markets, but with the novel result that venue design does not affect the magnitude.

2. **Extreme Value Theory (Embrechts et al. 1997, McNeil & Frey 2000, Cont 2001):** We bring EVT to prediction markets for the first time. Our per-market GPD fitting reveals genuine heavy tails (Polymarket: all ξ > 0) — a finding that has no precedent in the prediction market literature and extends the "stylized facts" literature (Cont 2001) to binary contracts.

3. **Informed Trading (Easley, Kiefer, O'Hara & Paperman 1996, Abad & Yagüe 2012):** Our finding that taker-side is the single most important feature (84% importance in Phase 2 XGBoost) directly connects to the PIN literature. We estimate PIN for prediction markets and test whether informed trading intensity differs across venue types and market categories.

4. **Autonomous Research Methodology (Karpathy 2026):** We are the first to apply the autoresearch paradigm to financial strategy optimization. Our 101-experiment loop with disciplined ablation demonstrates that autonomous LLM-guided search discovers non-obvious signal combinations (sqrt-dampened mean reversion, 3-trade momentum, boundary boost) that outperform hand-crafted baselines.

---

## 3. Research Questions

### RQ1: Information Arrival, Efficiency, and Informed Trading

**Question:** How does the information arrival process shape prediction market efficiency, and does informed trading intensity differ between centralized and decentralized venues?

**Theoretical Foundation:** Under the martingale hypothesis (Samuelson 1965), prediction market prices should be unbiased conditional expectations of terminal outcomes. Microstructure frictions (bid-ask bounce, discrete ticks) generate spurious serial correlation that can be decomposed using Roll's (1984) implied spread and Lo-MacKinlay (1988) variance ratios. Easley & O'Hara's (1987) PIN model provides a structural estimate of the fraction of trades initiated by privately informed traders.

**Methods:**
1. **Variance Ratio Tests** (Lo-MacKinlay 1988): VR(q) for q ∈ {2, 5, 10, 20, 50} with heteroskedasticity-robust M2 statistic. Applied to 80+ markets per platform.
2. **Roll's Implied Spread** (Roll 1984): s = 2√(−Cov(Δpₜ, Δpₜ₋₁)) per market. Cross-platform comparison via Mann-Whitney U.
3. **BDS Independence Test** (Brock, Dechert & Scheinkman 1996): Applied to AR(1) residuals at embedding dimensions m ∈ {2,3,4,5} and ε ∈ {0.5, 1.0, 1.5} × σ. Sports vs political decomposition.
4. **PIN Estimation** (Easley, Kiefer, O'Hara & Paperman 1996): Maximum likelihood estimation of (α, δ, μ, εb, εs) from trade arrival counts. VPIN (Easley, López de Prado & O'Hara 2012) as robustness check using volume-bucketed trade classification.

**Hypotheses:**
- H1a: After Roll's spread correction, VR tests accept the martingale hypothesis for political markets but reject for sports markets (due to continuous information arrival during live games).
- H1b: PIN estimates are higher for sports markets than political markets (more rapid information incorporation during game play).
- H1c: PIN does not differ significantly between Kalshi and Polymarket for matched events (venue design does not affect information asymmetry).
- H1d: VPIN predicts short-term volatility spikes (tail events) with AUC > 0.65.

**Novel Contribution:** First PIN/VPIN estimation for prediction markets. First formal test that information arrival process (sports vs political) dominates venue design in determining efficiency.

---

### RQ2: Tail Risk Characterization and Cross-Platform Dependence

**Question:** Do prediction market returns follow Extreme Value Theory distributions, and how are tail risks connected across platforms for matched events?

**Theoretical Foundation:** Extreme Value Theory (Embrechts et al. 1997) provides the asymptotic framework for modeling rare events. The Generalized Pareto Distribution (GPD) characterizes exceedances above a threshold, with shape parameter ξ determining whether tails are bounded (ξ < 0, Weibull), exponential (ξ = 0, Gumbel), or power-law (ξ > 0, Fréchet). For risk management, Expected Shortfall (CVaR) computed from fitted GPD provides coherent risk measures (Artzner et al. 1999). Copulas (Nelsen 2006) separate marginal distributions from dependence structure, enabling us to characterize how tail risks co-move across platforms without assuming joint normality.

**Methods:**
1. **Peaks-Over-Threshold / GPD Fitting** (Embrechts et al. 1997, McNeil & Frey 2000): Per-market GPD(ξ, σ) estimation via MLE with bootstrap 95% CIs. Threshold selection via mean excess plot.
2. **Block Maxima / GEV Fitting** (Coles 2001): GEV domain classification (Weibull/Gumbel/Fréchet) per market.
3. **Hill Estimator** (Hill 1975): Non-parametric tail index estimation with stability analysis across k.
4. **Expected Shortfall / CVaR**: ES_q = E[|R| | |R| > VaR_q] at q ∈ {90, 95, 99}, computed from fitted GPD. Cross-platform comparison.
5. **Copula Estimation for Matched Events**: For events traded on both Kalshi and Polymarket (e.g., 2024 presidential election, major sports events):
   - Estimate marginal distributions (GPD tails + empirical body)
   - Fit parametric copulas: Gaussian, Student-t, Clayton, Gumbel, Frank
   - Select best copula via AIC/BIC
   - Estimate tail dependence coefficients: λ_U (upper) and λ_L (lower)
   - Test: H₀: λ_U = 0 (tail independence) vs H₁: λ_U > 0 (tail dependence)

**Hypotheses:**
- H2a: Polymarket returns are unanimously in the Fréchet domain (ξ > 0), while Kalshi is mixed (confirming Phase 3 finding across larger sample).
- H2b: Sports markets have significantly heavier tails than political markets within each platform (extending p = 0.002 finding).
- H2c: For matched events, cross-platform tail dependence is asymmetric: stronger in the upper tail (large price increases) than the lower tail (large decreases), reflecting coordinated information arrival.
- H2d: Expected Shortfall at 99% is 2–3× higher on Polymarket than Kalshi, with economic significance exceeding $0.05 per contract.

**Novel Contribution:** First EVT analysis of prediction markets. First copula-based cross-platform dependence estimation. First formal evidence that information type (sports vs political) determines tail structure.

---

### RQ3: Autonomous Strategy Discovery via Autoresearch

**Question:** Can an autonomous experiment loop — where an LLM agent iteratively proposes, tests, and selects trading strategy modifications — discover non-obvious microstructure regularities and achieve out-of-sample risk-adjusted returns that exceed hand-crafted baselines?

**Theoretical Foundation:** The autoresearch paradigm (Karpathy 2026) applies the scientific method autonomously: hypothesize → experiment → evaluate → keep/discard. In the original setting (LLM training), 700 experiments over 2 days discovered non-obvious training improvements. We transplant this paradigm to financial strategy optimization, where the search space is trading signals, position sizing rules, and risk management parameters. The fitness metric is out-of-sample Sharpe ratio (or a composite metric incorporating tail risk). This connects to the AutoML literature (Hutter, Kotthoff & Vanschoren 2019) and the broader question of whether autonomous agents can discover financial regularities that human researchers miss.

**Architecture (Karpathy's 3-File Pattern):**

| Component | File | Role | Editable? |
|-----------|------|------|-----------|
| Evaluation Harness | `prepare.py` | Data loading, feature computation, walk-forward backtest, metric calculation | FIXED — never modified by agent |
| Strategy | `train.py` | Signal construction, model training, position sizing | EDITABLE — agent's only lever |
| Instructions | `program.md` | Agent behavior rules, experiment protocol, keep/discard criteria | FIXED |

**Optimization Loop:**
```
LOOP (autonomous, no human intervention):
  1. Agent reads current strategy code + experiment history
  2. Forms hypothesis (e.g., "adding 3-trade momentum will capture short-term trends")
  3. Edits train.py with proposed modification
  4. Git commits the change
  5. Runs 5-minute time-boxed backtest → gets out-of-sample metrics
  6. If Sharpe improved → keep (new baseline)
     If Sharpe equal/worse → git reset --hard (instant revert)
  7. Logs result to results.tsv (commit, sharpe, description, status)
  8. Repeat indefinitely
```

**Fitness Metric Design:**

The primary fitness metric is **daily-sampled Sharpe ratio** (fixing the annualization bug in the current prepare.py):

```
daily_pnl = aggregate per-trade PnL to daily
sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
```

Secondary metrics (logged but not optimized):
- Sortino ratio (downside deviation only)
- Maximum drawdown
- Calmar ratio (return / max drawdown)
- Tail ratio: P(gain > 2σ) / P(loss > 2σ)

**Strategy Components Under Optimization:**

The agent can modify any of these within train.py:

1. **Signal Construction:**
   - Mean reversion signals (current return, lagged returns, VR-based)
   - Momentum signals (3-trade, 5-trade, 10-trade weighted sums)
   - Volatility regime signals (rolling vol, vol-of-vol, GARCH(1,1))
   - Order flow signals (taker-side imbalance, VPIN-inspired toxicity)
   - Temporal signals (hour-of-day, day-of-week, time-to-event)
   - Price boundary signals (distance to 0/1, nonlinear transforms)

2. **Model Architecture:**
   - Logistic regression (current winner: sqrt-3trade-tanh-v37)
   - Gradient boosted trees (sklearn GBM or XGBoost)
   - Neural networks (1-2 hidden layers, ReLU/tanh)
   - PPO reinforcement learning agent (state = features, action = position size)
   - Ensemble methods (stacking, blending)

3. **Position Sizing:**
   - Linear sizing (probability → position)
   - Nonlinear sizing (tanh, sqrt dampening)
   - Kelly criterion (optimal fraction of capital)
   - Risk parity (inverse volatility weighting)
   - RL-learned sizing (PPO action space = continuous [-1, 1])

4. **Risk Management:**
   - Turnover threshold (current: 0.75)
   - Maximum position size
   - Drawdown circuit breaker
   - Tail-risk-conditioned sizing (reduce when VPIN high)

**RL Integration:**

Within the autoresearch loop, the agent can propose replacing the logistic regression model with a PPO (Proximal Policy Optimization) agent:

- **State space:** (price, rolling_vol_10, taker_buy_imbalance, lag_ret_1, lag_ret_2, current_position, unrealized_pnl, drawdown)
- **Action space:** Continuous position size in [-1, 1]
- **Reward:** Per-step PnL minus transaction costs, with Sharpe-ratio-shaped reward scaling
- **Training:** 50 episodes × 10,000 steps per episode, PPO with clipped objective
- **Comparison:** PPO vs LogReg vs XGBoost vs random, all evaluated on same hold-out data

This is how RL enters the autoresearch framework naturally — as one of many strategies the autonomous agent can try. If PPO improves Sharpe, it becomes the new baseline. If not, it gets discarded. The autoresearch loop is agnostic to the model class.

**Hypotheses:**
- H3a: The autoresearch loop discovers strategies with daily Sharpe > 1.0 (after correcting annualization), significantly exceeding a buy-and-hold baseline.
- H3b: The autonomous agent's keep rate is 15–25% (consistent with genuine optimization, not random walk), and the Sharpe trajectory shows monotonic improvement with diminishing returns.
- H3c: The top features discovered by autoresearch (via ablation) are consistent with the microstructure findings of RQ1 (mean reversion from bid-ask bounce, taker-side information).
- H3d: RL (PPO) position sizing outperforms static sizing rules by 0.1–0.3 Sharpe points through adaptive risk management.
- H3e: The autoresearch methodology itself is reproducible: different initial random seeds converge to similar Sharpe levels within 100 experiments.

**Novel Contribution:** First application of the autoresearch paradigm to financial markets. First evidence that autonomous LLM-guided search discovers microstructure regularities (sqrt dampening, boundary boost, direction model blend) not found in existing literature. First comparison of RL vs traditional ML within an autonomous optimization framework.

---

## 4. Methods Coverage Summary

| Requested Method | Where Used | Implementation |
|-----------------|------------|----------------|
| **Martingales** | RQ1 | Lo-MacKinlay VR tests, BDS independence tests |
| **VaR / EVT** | RQ2 | GPD, GEV, Hill estimator, Expected Shortfall (CVaR) |
| **XGBoost** | RQ1 + RQ3 | Feature importance for tail prediction (RQ1); strategy candidate in autoresearch (RQ3) |
| **RL** | RQ3 | PPO agent for position sizing within autoresearch loop |
| **Copulas** | RQ2 | Cross-platform dependence for matched events; tail dependence estimation |
| **Linear Factor Pricing** | RQ1 | PCA factor extraction from cross-section of market returns; test CAPM-like pricing for binary contracts |
| **GMM / SDF** | RQ1 | Binary contracts as Arrow-Debreu securities; estimate stochastic discount factor via Hansen (1982) GMM with Euler equation moment conditions |
| **Volatility Smiles** | RQ2 | Implied volatility surface from binary option prices at different "strikes" (probability levels); test whether BS-implied vol varies with moneyness |
| **Continuous Time Models** | RQ2 (extension) | Jump-diffusion estimation for contract price dynamics; compare Merton (1976) jump-diffusion fit to pure diffusion |
| **VAR / ML** | RQ1 (extension) | VAR(p) across liquid markets for Granger causality; cross-market information flow |
| **PIN / VPIN** | RQ1 | Easley-O'Hara PIN estimation; VPIN for real-time toxicity monitoring |

**Coverage: 11/11 methods** (8 core, 3 extensions). All methods from the requested list are incorporated into the framework.

---

## 5. Data Pipeline

### 5.1 Phase 0: Data Preparation
- Standardize Kalshi (cents → probability) and Polymarket (USDC atomic → USD, block join for timestamps)
- Common schema: (timestamp, market_id, price, quantity, side, platform)
- Filter: remove price ≤ 0 or ≥ 1, remove zero-quantity trades
- Match events across platforms for copula analysis (presidential election, Super Bowl, etc.)
- Output: `data/processed/kalshi_trades.parquet`, `polymarket_trades.parquet`, `matched_events.parquet`

### 5.2 Phase 1: Microstructure Characterization (RQ1)
- Compute per-market: VR(q), Roll's spread, BDS z-scores, PIN estimates
- Cross-platform comparison via Mann-Whitney U tests
- Market-type decomposition (sports, political, economic, other)
- Granger causality via VAR(p) for top 10 liquid markets
- PCA factor extraction from return cross-section
- Output: `artifacts/phase1_efficiency.json`, LaTeX tables

### 5.3 Phase 2: Tail Risk and Dependence (RQ2)
- Per-market GPD/GEV/Hill estimation (50+ Kalshi, 10+ Polymarket)
- Expected Shortfall computation at q ∈ {90, 95, 99}
- Copula estimation for matched events
- Implied volatility surface construction
- Jump-diffusion parameter estimation
- Output: `artifacts/phase2_evt.json`, `artifacts/phase2_copulas.json`, LaTeX tables + figures

### 5.4 Phase 3: Autoresearch (RQ3)
- Fix Sharpe annualization (aggregate to daily)
- Run autonomous experiment loop (target: 200+ experiments)
- Include RL (PPO) as candidate strategy
- Log all experiments to results.tsv
- Ablation analysis of discoveries
- Output: `src/autoresearch/results.tsv`, `artifacts/phase3_autoresearch.json`

### 5.5 Phase 4: Integration
- Generate all paper tables (11+) and figures (8+)
- Compute all LaTeX macros
- Write paper prose (Introduction → Data → Methodology → Results → Discussion → Conclusion)
- Run verify.py for deterministic quality checks

---

## 6. Expected Contributions

### 6.1 Empirical Contributions
1. **Largest prediction market dataset analyzed:** 476M trades across two platforms, 476× larger than existing studies
2. **First EVT analysis:** Per-market GPD reveals Polymarket unanimously heavy-tailed (ξ > 0), sports vs political divergence (p = 0.002)
3. **First PIN estimation:** Connects 84% taker-side feature importance to structural model of informed trading
4. **First copula cross-platform analysis:** Characterizes tail dependence for matched events
5. **Efficiency puzzle:** Same VR(2) but 3× different spreads — venue design affects cost, not efficiency

### 6.2 Methodological Contributions
1. **Autoresearch for finance:** First application of Karpathy's autonomous experiment loop to financial strategy optimization
2. **101+ experiments with disciplined ablation:** Demonstrates that LLM-guided search discovers non-obvious signal combinations
3. **RL vs ML comparison:** Within-framework comparison of PPO, XGBoost, and logistic regression for prediction market execution

### 6.3 Theoretical Contributions
1. **Information arrival > venue design:** Formal evidence (VR tests, BDS, PIN, tail indices) that market content determines microstructure more than venue mechanism
2. **Binary contracts as Arrow-Debreu securities:** GMM estimation of implied state prices from prediction market data
3. **EVT stylized facts for prediction markets:** Extending Cont (2001) to a new asset class

---

## 7. Timeline

| Week | Tasks | Deliverables |
|------|-------|-------------|
| 1 | Fix Sharpe annualization, consolidate codebases, implement PIN | Fixed prepare.py, PIN module |
| 2 | Run autoresearch loop (200 experiments), implement RL (PPO) | results.tsv, PPO agent |
| 3 | Implement copula estimation, matched event identification | Copula results, matched dataset |
| 4 | Run full Phase 1-2 pipeline, generate tables/figures | All artifacts |
| 5 | Write paper (Introduction, Data, Methodology) | paper.tex sections |
| 6 | Write paper (Results, Discussion, Conclusion), iterate | Complete draft |

---

## 8. References to Add

### Microstructure
- Easley, D., Kiefer, N.M., O'Hara, M. & Paperman, J.B. (1996). "Liquidity, Information, and Infrequently Traded Stocks." *Journal of Finance*, 51(4), 1405–1436.
- Easley, D., López de Prado, M.M. & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-frequency World." *Review of Financial Studies*, 25(5), 1457–1493.
- Hasbrouck, J. (1995). "One Security, Many Markets: Determining the Contributions to Price Discovery." *Journal of Finance*, 50(4), 1175–1199.
- Whelan, K. (2025). "The Microstructure of Kalshi." Working paper.

### Extreme Value Theory
- Hill, B.M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution." *Annals of Statistics*, 3(5), 1163–1174.
- Artzner, P., Delbaen, F., Eber, J.M. & Heath, D. (1999). "Coherent Measures of Risk." *Mathematical Finance*, 9(3), 203–228.

### Copulas
- Nelsen, R.B. (2006). *An Introduction to Copulas*. Springer.
- Patton, A.J. (2006). "Modelling Asymmetric Exchange Rate Dependence." *International Economic Review*, 47(2), 527–556.

### Reinforcement Learning
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.

### Autoresearch
- Karpathy, A. (2026). "Autoresearch: Autonomous LLM Research." GitHub repository, 21K stars. https://github.com/karpathy/autoresearch

### Asset Pricing / GMM
- Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029–1054.
- Hansen, L.P. & Singleton, K.J. (1982). "Generalized Instrumental Variables Estimation of Nonlinear Rational Expectations Models." *Econometrica*, 50(5), 1269–1286.

### Prediction Markets (Recent)
- Ng, H., Peng, L., Tao, Y. & Zhou, D. (2025). "Price Discovery and Trading in Modern Prediction Markets." SSRN Working Paper.
- Arbutina, N., Gombina, N. & Kamal, Y. (2025). "The Anatomy of Polymarket." arXiv:2603.03136.
- Reichenbach, F. & Walther, M. (2025). "Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket." SSRN 5910522.

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Sharpe after annualization fix is <1.0 | Report honestly; even Sharpe 0.3–0.5 is interesting for prediction markets where no baseline exists |
| PIN estimation fails (MLE non-convergence) | Use VPIN as alternative; aggregate to coarser time buckets |
| Too few matched events for copula analysis | Focus on 2024 presidential election (highest liquidity on both platforms); supplement with major sports |
| RL (PPO) underperforms logistic regression | Report as finding — simplicity wins. This is actually a publishable result |
| Autoresearch discovers nothing beyond v37 | Run with different seeds, larger feature space; report convergence behavior as methodological finding |

---

## 10. Success Criteria

**Minimum for Submission:**
- Daily Sharpe > 0.5 after annualization fix (or honest report of lower value with analysis of why)
- 200+ autoresearch experiments with monotonic improvement trajectory
- GPD fitted on 60+ markets with bootstrap CIs
- PIN estimated on 20+ markets
- Copula estimated on at least 5 matched events
- All 11 tables and 8+ figures generated
- Complete paper prose

**Target for Top Venue:**
- Daily Sharpe > 1.0 with RL component
- Novel finding: information arrival > venue design, supported by 4+ independent tests
- Clean ablation story: autoresearch discovers signals consistent with microstructure theory
- Copula tail dependence significant for matched events
- PIN estimates consistent with taker-side feature importance

---

*This document supersedes hypothesis_unified_v2.md and research_design_v1.md.*
