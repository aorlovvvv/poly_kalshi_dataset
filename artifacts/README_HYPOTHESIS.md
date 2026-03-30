# Hypothesis Documents: Complete Guide

**Status:** Unified hypothesis ready for research implementation
**Primary Document:** `hypothesis_unified_v2.md` (380 lines, fully detailed)

---

## Document Structure

### Core Artifacts

| File | Lines | Purpose | Audience |
|------|-------|---------|----------|
| **hypothesis_unified_v2.md** | 380 | Complete, polished research hypothesis with three integrated phases | Primary: advisors, collaborators, committees |
| hypothesis_draft_v1.md | 99 | Original Claude-generated hypothesis combining EVT + martingale + RL | Reference: evolution of thinking |
| hypothesis_candidates.md | 209 | Data-grounded analysis proposing three independent hypotheses | Reference: data-motivation tracing |
| data_profile.json | — | Raw empirical findings (476M trades, key statistics) | Technical validation |
| research_design_v1.md | 746 | Detailed research design document (earlier iteration) | Reference: implementation precedent |
| SYNTHESIS_NOTES.txt | — | Key design decisions and unification rationale | Implementation guide |

---

## How to Use This Hierarchy

### For Researchers/Advisors
**Start with:** `hypothesis_unified_v2.md`
- Comprehensive, self-contained, publication-quality prose
- Includes abstract, three research questions, integrated methodology, venue fit
- All empirical motivation explicitly tied to data_profile.json findings

### For Co-Authors/Collaborators
**Start with:** `hypothesis_unified_v2.md` + `SYNTHESIS_NOTES.txt`
- Unified document provides the research direction
- Synthesis notes explain the dropped elements (RL, copulas) and why
- Justifies the three-phase sequential structure

### For Technical Implementation
**Reference:** `hypothesis_unified_v2.md` (Sections: "Integrated Methodology", "Phase 2 Analysis Tasks")
- Detailed statistical test specifications with parameter choices
- Feature engineering lists for XGBoost
- Success criteria and evaluation metrics

### For Historical Context
**Reference:** `hypothesis_draft_v1.md` + `hypothesis_candidates.md`
- Shows evolution from high-level ideas → data-grounded hypotheses → unified framework
- Useful for understanding trade-offs (e.g., why RL was dropped, why copulas moved to extensions)

---

## Research Narrative

The unified hypothesis synthesizes two independent analysis threads:

### Thread 1: Claude's Conceptual Hypothesis (draft_v1.md)
- Started with literature gaps: no cross-platform EVT, no prediction market RL, no martingale testing at scale
- Proposed three hypotheses: H1 (EVT + martingale violations + XGBoost), H2 (RL execution), H3 (copulas)
- Recommended combining H1 + H2 as a complete story

### Thread 2: Cursor's Data-Grounded Analysis (hypothesis_candidates.md)
- Profiled 476M+ trades; identified 7 striking empirical patterns
- Proposed three hypotheses grounded in each pattern: H1 (microstructure-corrected martingale + Roll's spread), H2 (EVT + regulatory decomposition), H3 (cross-platform copulas + XGBoost arbitrage)
- Provided detailed methodology specifications for each hypothesis

### Unified Framework (hypothesis_unified_v2.md)
**Synthesis logic:**
- Adopt H1 from Cursor (microstructure-corrected martingale) because it includes Roll's spread model (academically rigorous, directly addresses observed serial correlation)
- Adopt H2 from Cursor (EVT + regulatory-structure decomposition) because it explains the 2.65× extreme-price divergence mechanically
- Adopt Phase 3 from Claude (XGBoost prediction) but drop full RL framework—XGBoost achieves the same "actionable predictability" goal with less engineering overhead
- Drop copulas entirely (requires event matching data engineering not yet done); move to "future extensions"

**Result:** Three sequential research questions (RQ1 → RQ2 → RQ3) that form a cohesive narrative: *Characterize efficiency* → *Explain tail risk* → *Predict extreme events*

---

## Key Empirical Grounding

Every research question is motivated by specific empirical findings from `data_profile.json`:

| RQ | Empirical Trigger | Pattern | Scale |
|----|-------------------|---------|-------|
| RQ1 (Efficiency) | Serial correlation in top 10 Kalshi markets | Lag-1 autocorr: −0.34 to −0.47 (bid-ask bounce) | 287K–1.26M price changes |
| RQ2 (Tail Risk) | Extraordinary kurtosis across 100 liquid markets | Kurtosis = 393 (130× Gaussian); 0.086% of trades ±10pp | 5.6M price changes (Kalshi), 10M (Polymarket) |
| RQ2 (Structure) | Divergent extreme-price clustering | Polymarket 31.7% vs Kalshi 12.0% at ≤5%/≥95% | 2.65× difference |
| RQ3 (Prediction) | Intraday clustering and trade-size skew | Kalshi 24× peak/trough, Polymarket 1.37×; mean/median = 6.33 | 5M+ trades/hour peak (Kalshi) |

---

## Methodology Summary (One-Page)

### Phase 1: Martingale Testing (RQ1)
- **Methods:** Roll's spread, Variance Ratio tests, BDS test
- **Data:** Top 100–1000 Kalshi + Polymarket markets
- **Hypothesis:** Negative serial correlation is purely microstructure noise (bid-ask bounce); efficiency accepted after correction
- **Outcomes:** Variance ratios VR(q) ≈ 1.0; BDS p-value > 0.05

### Phase 2: Extreme Value Theory (RQ2)
- **Methods:** Peaks-Over-Threshold (GPD fitting), Block Maxima (GEV), Hill estimator, regulatory-structure decomposition
- **Data:** Top 100 markets, tail exceedances above 90th percentile
- **Hypothesis:** Polymarket has genuinely heavy tails (Fréchet domain); Kalshi tail bounded by discrete pricing (Weibull domain)
- **Outcomes:** Polymarket ξ ≈ 0.2–0.4; Kalshi ξ < 0; CVaR 2–3× higher on Polymarket

### Phase 3: Tail Risk Prediction (RQ3)
- **Methods:** XGBoost classifier with 12 microstructure features, temporal cross-validation, SHAP feature importance
- **Data:** Top 50 markets per platform; 5-minute return predictions
- **Hypothesis:** Tail events are predictable from lagged microstructure features (time-of-day, volatility, imbalance)
- **Outcomes:** XGBoost AUC-ROC > 0.6; Sharpe ratio improvement +0.1 to +0.3 from position-sizing strategy

---

## Publication Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Phase 1: Data prep + Efficiency testing | 2 weeks | `phase1_efficiency_report.md` + tables |
| Phase 2: EVT + Regulatory decomposition | 2 weeks | `phase2_evt_report.md` + figures (Hill plot, GPD QQ, ES curves) |
| Phase 3: XGBoost + Backtesting | 2 weeks | `phase3_prediction_report.md` + figures (SHAP, calibration, equity curve) |
| Paper writing & integration | 2 weeks | Draft submitted to ICAIF |
| Revision & resubmission | 4 weeks | Final version |

**Target:** ICAIF 2026 (deadline typically April/May; decision by August)

---

## Venue Fit

### Primary: ICAIF 2026
- Emphasize: Machine learning (XGBoost, SHAP) applied to novel asset class (prediction markets)
- Frame: "Tail Risk Prediction in Prediction Markets"
- Length: 12–15 pages (including tables/figures)

### Secondary: NeurIPS Datasets & Benchmarks 2026
- Emphasize: Novel dataset (476M trades, first cross-platform comparison), data profiling, benchmark statistics
- Frame: "The Prediction Market Microstructure Dataset: 476M Trades Across Kalshi and Polymarket"
- Length: 8–12 pages

### Tertiary: Journal of Financial Data Science
- Emphasize: Econometric rigor (variance ratios, BDS, EVT), regulatory implications
- Frame: "Martingale Efficiency and Tail Risk in Prediction Markets: A 476M Trade Empirical Study"
- Length: 20+ pages (journal standard)

---

## Success Metrics

| Phase | Success Criterion | Acceptable Range |
|-------|-------------------|------------------|
| **Phase 1** | Roll spread estimates align with known fees | 0.6–1.5 cents (Kalshi); 0.1–0.3 cents (Polymarket) |
| **Phase 1** | Efficiency: Variance ratios consistent with VR(q) ≈ 1.0 | p-value > 0.05 for adjusted prices |
| **Phase 2** | GPD fit quality: Kolmogorov-Smirnov test | p-value > 0.10 (good fit) |
| **Phase 2** | CVaR differential between platforms | 2–4× (Polymarket > Kalshi) |
| **Phase 3** | XGBoost out-of-sample AUC-PR | > 0.15 (AUC-PR for imbalanced classes) |
| **Phase 3** | Backtest Sharpe improvement | > 0.10 over naive buy-and-hold |
| **Overall** | Paper draft quality | Submittal-ready by end of 6 weeks |

---

## Quick Reference: Key Decisions

**Q: Why drop Reinforcement Learning?**
A: XGBoost achieves the goal (actionable tail-risk prediction) faster, is more interpretable (SHAP), and requires less data engineering. RL remains a strong extension if Phase 3 shows Sharpe improvement > 0.2.

**Q: Why drop Copulas / Cross-Platform Arbitrage?**
A: Requires high-confidence matching of Kalshi and Polymarket contracts for the same underlying events. This data engineering (fuzzy matching, manual curation) is substantial and risks errors. Defer as a separate paper if a matched-event dataset becomes available.

**Q: Why three phases instead of one paper?**
A: Each phase stands alone (efficiency, tail risk, prediction) but builds logically: (1) establish baseline efficiency, (2) characterize tail behavior, (3) show it's economically exploitable. Modular structure allows incremental publishing if Phase 1–2 alone merit a submission.

**Q: Isn't Kalshi's tick-size explanation for the 2.65× extreme-price divergence too hand-wavy?**
A: Not if we test it mechanistically in Phase 2 (Section "Regulatory-Structure Decomposition"): fit GPD to raw prices, then to prices rounded to finer grids; show that finer grid → GPD parameters move toward Polymarket values. Quantifies the tick-size contribution.

---

## Next Steps for Advisors / Collaborators

1. **Read:** `hypothesis_unified_v2.md` (full document, 30–45 min)
2. **Discuss:** 
   - Are the three research questions focused and achievable?
   - Is the methodology (variance ratios, EVT, XGBoost) appropriate and rigorous?
   - Which venue (ICAIF / NeurIPS / JFDS) seems best aligned?
   - Are there missing competitors/references in the literature?
3. **Approve:** methodology, timeline, and success criteria
4. **Assign:** data prep tasks (standardization, cleaning, feature engineering)
5. **Schedule:** weekly progress sync to track Phase 1 → Phase 2 → Phase 3 completion

---

**Document Created:** 2026-03-29  
**Status:** Ready for advisor/committee review  
**Questions?** See `SYNTHESIS_NOTES.txt` for design rationale
