# Comprehensive Audit Report: Polymarket vs Kalshi Autoresearch Project

**Date:** 2026-03-30
**Auditor:** Claude (Cowork)
**Scope:** Every file in the repository, assessed against original objectives

---

## OVERALL VERDICT: C+

The project contains genuinely interesting empirical findings and a working autoresearch loop, but suffers from critical infrastructure problems, an incomplete paper, and covers only 3 of 11 requested methods. The strongest work is the Cursor-driven experiment loop (101 iterations, disciplined ablation, Sharpe 2.96 → 3.66). The weakest aspects are the empty paper, the duplicate/conflicting codebases, and an evaluation framework with a misleading Sharpe annualization.

---

## SECTION 1: CRITICAL ISSUES (Blockers)

### 1.1 The Paper Is an Empty Shell

Every substantive section of `paper/paper.tex` is a LaTeX comment placeholder:

- Introduction: `% [To be written by LLM discover/write loop. Key points:]`
- Related Work: `% [Covers: ...]`
- Data Processing: `% [Standardization: ...]`
- RQ1 Results: `% [Phase 2 results: ...]`
- RQ2 Results: `% [Phase 2+3 results: ...]`
- RQ3 Results: `% [Autoresearch results: ...]`
- Discussion: `% [Key findings synthesis: ...]`
- Conclusion: `% [Summary of three RQs ...]`

**Zero prose has been written.** The abstract has macro placeholders but no actual argument. For a NeurIPS/ICAIF submission, this is 0% complete.

### 1.2 No Generated Tables or Figures Exist

`paper/tables/` and `paper/figures/` are both **completely empty**. The paper.tex references 11 tables and 8+ figures — none of which exist on disk. No `paper/results.tex` macro file exists either. The paper cannot compile.

### 1.3 Two Competing, Incompatible Codebases

There are TWO completely separate backtesting systems in `src/autoresearch/`:

| Aspect | Cursor's System (ran) | Cowork's System (never ran) |
|--------|----------------------|----------------------------|
| Harness | `prepare.py` | `backtest.py` |
| Strategy | `train.py` | `strategy.py` |
| Interface | `train(X, y, feature_names)` | `train(features_df, labels, val_df, val_labels) -> dict` |
| Features | 14 (includes `ret`) | 19 (excludes `ret`) |
| Evaluation | `position * next_ret - cost` | temporal splits, tail prediction |
| Split | 80/20 random | train < 2025-06-01 / val / test |
| Costs | 20 bps flat | Kalshi 60bps, Poly 2bps+5bps |
| Results | 101 experiments, Sharpe 3.66 | Never executed |

The `interfaces.py` on disk matches Cursor's interface (3-arg train), but `backtest.py` expects a 4-arg interface. Running `backtest.py` with the on-disk strategy would crash immediately.

**This is a serious project management failure.** Two agents built parallel systems without coordination.

### 1.4 Sharpe Ratio Annualization Is Wrong

The evaluation in `prepare.py` computes:

```python
ANNUALIZATION_FACTOR = np.sqrt(252)
sharpe = mean_pnl / std_pnl * ANNUALIZATION_FACTOR
```

But `pnl` has one entry **per trade** (~100K trades over ~3 months ≈ 1000 trades/day). The sqrt(252) factor assumes one observation per day. With 1000 trades/day, the correct annualization would be sqrt(252,000). The current calculation treats per-trade statistics as if they were daily statistics.

**Impact:** The reported Sharpe of 3.66 is not comparable to standard daily-sampled Sharpe ratios. A rough correction: aggregate to daily PnL first, then compute Sharpe with sqrt(252). The "real" daily Sharpe is likely in the 0.1–0.5 range — still potentially interesting but not the headline-grabbing 3.66.

### 1.5 `data/processed/` Is Empty

The main analysis pipeline (`make profile` → `prepare_data.py` → `make analysis`) has never been run. No processed parquet files exist. The Phase 1 descriptive statistics pipeline is entirely unexecuted.

---

## SECTION 2: METHODOLOGICAL ISSUES

### 2.1 `ret` as a Feature — Nuanced, Not Simply "Leakage"

The on-disk system (prepare.py) includes `ret` (current price change) as a feature. The label is `target = (|ret| > threshold)`. This is data leakage for the AUC metric — you're using current information to predict a current-trade label.

However, the Sharpe metric is computed differently: `pnl = position * next_ret`. So position (derived from current features including `ret`) earns the NEXT return. Using current return as a mean-reversion signal for next-period sizing is legitimate — it's exactly what the VR(2) = 0.59 finding suggests should work.

**Verdict:** The AUC of 0.78-0.81 is meaningless (inflated by `ret` leakage). The Sharpe trajectory is more defensible but still needs the annualization fix above.

### 2.2 Position Sizing Has No Capital Constraint

The backtest sizes positions in [-1, 1] regardless of equity. Early trades risk 77% of initial capital (mean position 0.77). Later, when equity is 100x initial, the same 0.77 position uses <1% of capital. This is not how any real strategy works. A proper backtest would:

1. Size relative to current equity, OR
2. Aggregate to daily returns with a fixed notional

The reported "total return: 212x" (run.log) is an artifact of unlimited additive PnL accumulation.

### 2.3 Only 20 Kalshi Markets, No Polymarket in Autoresearch

The optimization loop uses only 20 Kalshi markets (500K trades). Despite the paper's title being "Cross-Platform," Polymarket was never backtested. The cowork-side `backtest.py` added a `--both` flag but was never run.

### 2.4 GPD Fitting: Phase 2 Artifact Was Correctly Identified and Fixed in Phase 3

This is a positive finding. Phase 2's pooled GPD (ξ = -1.53) was recognized as an artifact of cross-market price jumps. Phase 3 fitted per-market GPDs and found genuinely heavy tails (Polymarket: all 10 markets ξ > 0). The sports vs political divergence (p = 0.002) is a legitimate and publishable finding.

### 2.5 BDS Manual Implementation Has Wrong Formula

The fallback BDS implementation (lines 250-293 of `phase2_statistical_tests.py`) uses a non-standard formula involving log ratios rather than the canonical BDS statistic. In practice, `statsmodels.bds` was likely used, so this only affects environments without statsmodels.

---

## SECTION 3: METHODS COVERAGE

You requested: RL, XGBoost, Martingales, Copulas, VaR/EVT, Linear Factor Pricing, GMM, VAR/ML, Volatility Smiles, Risk Neutral Valuation, Continuous Time Models.

| Method | Status | Where | Assessment |
|--------|--------|-------|------------|
| XGBoost | Partial | Phase 2/3: sklearn GBM, then replaced by LogReg in autoresearch | Not actual XGBoost (import fails). The winning strategy is LogisticRegression. Academically, calling it "gradient boosting" is defensible but the ablation showed LogReg won. |
| Martingale Tests | DONE | Phase 2: Lo-MacKinlay VR tests | Properly implemented. VR(2)=0.59 across 80 markets. Publication-ready finding. |
| VaR / EVT | DONE | Phase 2 (pooled GPD) + Phase 3 (per-market GPD + Hill estimator) | Well-executed after the Phase 3 fix. Per-market GPD with KS goodness-of-fit, Hill stability analysis. Solid. |
| RL | NOT DONE | Mentioned in program.md Phase D. Zero code. | Not a single line of reinforcement learning exists anywhere in the codebase. |
| Copulas | NOT DONE | Phase 3 mentions "copula-based dependence testing" as future work. | No implementation. Would be natural for cross-platform price linkage. |
| Linear Factor Pricing | NOT DONE | Not mentioned anywhere. | Could extract PCA factors from cross-section of market returns. Never attempted. |
| GMM (Generalized Method of Moments) | NOT DONE | program.md mentions "GMM regime detection" — this is Gaussian Mixture Models, not the Hansen (1982) econometric GMM. | The econometric GMM (for estimating SDF parameters or testing moment conditions) is completely absent. |
| VAR / Maximum Likelihood | NOT DONE | Only AR(1) used for BDS residuals. No VAR system. | No vector autoregressive models anywhere. AR(1) is used but that's trivial. |
| Volatility Smiles | COSMETIC | `price_sq` added to backtest.py as "volatility smile proxy" | This is not a volatility smile model. A proper implementation would fit the volatility surface as a function of moneyness, estimate implied volatility across strike prices, or test whether the BS framework applies. |
| Risk Neutral Valuation | NOT DONE | Not mentioned anywhere. | Binary contracts are essentially Arrow-Debreu securities — risk-neutral pricing is a natural theoretical frame. Never explored. |
| Continuous Time Models | NOT DONE | Not mentioned anywhere. | Could estimate jump-diffusion parameters or CIR-type models for contract price dynamics. Never attempted. |

**Score: 3/11 methods implemented (27%).** Of those 3, only Martingale tests and EVT are properly done. The "XGBoost" was replaced by LogisticRegression.

---

## SECTION 4: WHAT ACTUALLY WORKS WELL

### 4.1 The Autoresearch Loop Is the Real Achievement

Cursor executed 101 experiments with disciplined keep/discard logic. The results.tsv shows genuine scientific progress:

- Baseline Sharpe: 2.96
- After 37 experiments: 3.66
- Keep rate: 17/101 (17%) — appropriately conservative
- Discoveries: sqrt dampening, 3-trade momentum, boundary boost, direction model blend, recency training
- Ablation is publication-ready: every change is logged with commit hash, metric, and description

This IS the Karpathy autoresearch pattern working correctly. The progression from GBM → LogReg → hand-crafted signals is a genuine optimization story.

### 4.2 Phase 2/3 Statistical Analysis Is Thorough

The statistical testing pipeline covers real methods:

- **VR tests**: 80 markets, 4 lags each, heteroskedasticity-robust M2 statistic
- **Roll's spread**: First quantitative spread estimates for prediction markets
- **GPD**: Phase 3 per-market fitting with KS tests, Hill estimator, category decomposition
- **BDS**: Sports vs political divergence at z-scores 10-25 is compelling
- **XGBoost**: Despite leakage issues, taker_side dominance (84%) is a genuine and novel finding

### 4.3 Genuinely Novel Findings

Several findings would be of interest to the academic community:

1. **VR(2) identical across platforms (p=0.63) despite 3x spread difference** — challenges the assumption that decentralized markets are more efficient
2. **Sports vs political tail divergence (p=0.002)** — novel structural finding
3. **Taker-side dominance (84% importance)** — links prediction markets to PIN literature
4. **Polymarket spreads 3x tighter than Kalshi (p < 10⁻⁶)** — regulatory implications

### 4.4 Data Scale Is Impressive

476M trades across two platforms, 36GB of raw parquet data. This is one of the largest prediction market datasets analyzed in the literature.

---

## SECTION 5: WHAT'S NEEDED TO BE PUBLICATION-READY

### Must-Fix (Blockers)

1. **Fix Sharpe annualization**: Aggregate PnL to daily before computing Sharpe. Report daily-sampled Sharpe ratio, which is the standard.
2. **Resolve dual codebases**: Pick ONE system (Cursor's, since it has results). Delete or clearly deprecate the other.
3. **Write the paper**: Every section needs actual prose. The abstract promises specific claims — these must be backed by narrative.
4. **Generate all tables and figures**: Run the pipeline end-to-end. The paper references 11 tables and 8+ figures that don't exist.
5. **Generate `results.tex` macros**: All numeric claims must come from macros.
6. **Report AUC honestly**: Acknowledge the `ret` feature issue. Either remove `ret` and re-run, or clearly separate "trading signal" (legitimate) from "prediction accuracy" (leaked).

### Should-Fix (For Academic Rigor)

7. **Add at least 2 more methods from the requested list**: RL and/or Copulas would be most natural.
   - RL: Train a PPO agent for position sizing given (tail_prob, position, vol) state.
   - Copulas: Model cross-platform price dependence for matched events.
8. **Capital-constrained backtest**: Size positions relative to current equity.
9. **Run Polymarket backtest**: The paper claims cross-platform comparison.
10. **Proper XGBoost**: Install xgboost package, use actual XGBClassifier.

### Nice-to-Have (For Impressiveness)

11. **VAR model**: Fit VAR(p) across 5-10 liquid markets, test Granger causality.
12. **GMM**: Estimate Hansen's GMM with Euler equation moment conditions for the prediction market SDF.
13. **Risk-neutral pricing**: Derive implied state prices from contract pairs, test completeness.

---

## SECTION 6: FILE-BY-FILE ASSESSMENT

| File | Quality | Issues |
|------|---------|--------|
| `prepare.py` (Cursor harness) | B+ | Well-structured, DuckDB-native, proper caching. `ret` in features is deliberate for mean-reversion signal. Annualization bug. |
| `train.py` (Cursor strategy) | B+ | Clean implementation. 37 experiments documented. Clever sqrt-dampened mean-reversion. |
| `strategy.py` (Cowork) | B- | Implements Cursor-incompatible interface. Never tested. Dead code. |
| `backtest.py` (Cowork harness) | B | More features (19), correct leakage fix, Polymarket support. But never ran and interface conflicts. |
| `run_loop.py` | B | Good autonomous loop design. Syntax/import pre-checks (added this session). Updated prompt with correct interface. |
| `interfaces.py` | C | Revised to match Cursor's interface, breaking Cowork's backtest.py. |
| `results.tsv` | A | 101 experiments, well-documented, disciplined ablation. The project's best artifact. |
| `prepare.py` evaluate() | B- | Walk-forward is correct. Annualization is wrong. No capital constraint. |
| `phase2_statistical_tests.py` | B+ | Clean OOP design. All 5 tests implemented. BDS fallback formula wrong but statsmodels used in practice. |
| `phase2_synthesis.md` | A- | Excellent analysis. Self-critical about artifacts. Clear research mapping. |
| `phase3_analysis.md` | A | Best document in the project. Properly fixes Phase 2 artifacts. Novel findings. |
| `paper.tex` | D | Structure exists but 0% content. Would not compile (missing tables, figures, macros). |
| `references.bib` | B | 16 references. Covers key papers but missing: Easley & O'Hara (PIN), Hansen (GMM), Black & Scholes. |
| `program.md` | B | Good roadmap. Updated with correct features this session. |
| `validate_backtest.py` | B+ | New file (this session). Good integrity checks. |
| `CLAUDE.md` | A- | Excellent project documentation. Thorough, well-organized. |
| `Makefile` | B | Clean targets. Updated this session. |

---

## SECTION 7: SUMMARY

**Strengths:**
- Impressive dataset (476M trades, 36GB)
- Autoresearch loop works and produced genuine discoveries
- Phase 2/3 statistical analysis is thorough and self-correcting
- Several novel, publishable findings

**Weaknesses:**
- Paper is 0% written (all placeholders)
- No tables or figures generated
- Sharpe annualization error inflates headline result
- Only 3/11 requested methods implemented
- Two competing codebases, neither fully functional
- Polymarket never backtested in autoresearch
- No RL, no Copulas, no GMM, no VAR, no risk-neutral pricing

**Bottom line:** The research pipeline and findings are B+ quality. The paper deliverable is F. The methods coverage is D. To reach the stated target venues (NeurIPS, ICAIF, JFDS), the paper needs to be written, the Sharpe needs to be corrected, and at least 2 more methods from the requested list need to be implemented.
