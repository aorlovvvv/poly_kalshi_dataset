# Research Hypothesis Unification: Complete Index

**Project:** Microstructure, Tail Risk, and Efficiency in Prediction Markets  
**Date:** 2026-03-29  
**Status:** COMPLETE - Ready for advisor review  
**Target Venue:** ICAIF 2026 (International Conference on AI Finance)

---

## Quick Navigation

**I'm a(n):**
- [Advisor/Reviewer](#for-advisorsreviewers) — Start here
- [Technical Implementer](#for-technical-implementers) — Start here  
- [Research Collaborator](#for-research-collaborators) — Start here
- [Executive/Project Manager](#for-executive-summary) — Start here

---

## For Advisors/Reviewers

**Time: 45 minutes**

1. Read: `hypothesis_unified_v2.md` (full document)
   - Sections: Abstract → Research Questions → Methodology → Why Strongest
   
2. Skim: `SYNTHESIS_NOTES.txt` (5 min)
   - Explains why RL and copulas were dropped
   - Design decisions justified
   
3. Review: `hypothesis_unified_v2.md` sections:
   - Phase 2 Analysis Tasks (detailed specs)
   - Success Criteria (quantified metrics)

**Discussion Points:**
- Are the three RQs focused and achievable?
- Is the methodology (variance ratios, EVT, XGBoost) appropriate?
- Which venue (ICAIF / NeurIPS / JFDS) fits best?
- Are references/competitors missing?

---

## For Technical Implementers

**Time: 30 minutes to get started**

1. Read: `hypothesis_unified_v2.md` sections:
   - "Integrated Methodology" (plan overview)
   - "Phase 2 Analysis Tasks" (detailed todo items, parameter specs)
   
2. Reference: `data_profile.json`
   - Empirical baseline (476M trades)
   - Validation targets
   
3. Read: `QUICK_START.txt`
   - Success metrics per phase
   - Implementation timeline (6-8 weeks)

**Then:** Create implementation tickets for Phase 1-3

---

## For Research Collaborators

**Time: 1 hour to understand the full project**

1. Read: `README_HYPOTHESIS.md` (complete guide)
   - Document hierarchy
   - Research narrative
   - Methodology summary
   
2. Read: `hypothesis_unified_v2.md` (full document)
   
3. Discuss: Which phase will you own?
   - Phase 1 (Martingale Efficiency)
   - Phase 2 (EVT + Regulatory Decomposition)
   - Phase 3 (XGBoost Tail Prediction)

---

## For Executive Summary

**Time: 10 minutes**

Read these sections from `hypothesis_unified_v2.md`:
- **Title** → Paper name
- **Abstract** → What we're studying, why, and key findings expected
- **Research Questions** → The three RQs (what we're asking)
- **Why This Combination Is Strongest** → Why this approach works

---

## Complete File Listing

| File | Lines | Purpose | Format |
|------|-------|---------|--------|
| **00_INDEX.md** | — | This file; navigation guide | Markdown |
| **hypothesis_unified_v2.md** | 380 | PRIMARY: Complete research hypothesis | Markdown |
| **QUICK_START.txt** | — | One-page visual summary | Plain text |
| **README_HYPOTHESIS.md** | — | Full navigation + methodology guide | Markdown |
| **SYNTHESIS_NOTES.txt** | — | Design decisions explained | Plain text |
| hypothesis_draft_v1.md | 99 | Reference: Original Claude proposal | Markdown |
| hypothesis_candidates.md | 209 | Reference: Cursor's data-grounded analysis | Markdown |
| data_profile.json | — | Reference: Empirical findings (476M trades) | JSON |
| research_design_v1.md | 746 | Reference: Earlier design iteration | Markdown |

---

## Document Relationships

```
                    data_profile.json
                    (empirical facts)
                           ↓
          ┌────────────────┴────────────────┐
          ↓                                  ↓
hypothesis_draft_v1.md         hypothesis_candidates.md
(conceptual ideas)            (data-grounded analysis)
          ↓                                  ↓
          └────────────────┬────────────────┘
                           ↓
            hypothesis_unified_v2.md
            (SYNTHESIS - Primary Document)
                           ↓
            ┌──────────────┬──────────────┐
            ↓              ↓              ↓
        SYNTHESIS      README_HYP    QUICK_START
        _NOTES.txt     OTHESIS.md       .txt
```

---

## The Three Research Phases

### Phase 1: Martingale Efficiency (RQ1)

**Question:** After correcting for microstructure noise, are prediction markets efficient?

**Empirical Trigger:** Observed lag-1 serial correlation of −0.34 to −0.47 (bid-ask bounce)

**Methods:** Roll's spread model, Variance Ratio tests, BDS test

**Duration:** 2 weeks

**Success:** VR p-values > 0.05; Roll spreads = 0.6–1.5 cents (Kalshi)

**Output:** `phase1_efficiency_report.md` + efficiency tables

---

### Phase 2: Extreme Value Theory (RQ2)

**Question:** How do tail indices differ by platform, and why?

**Empirical Trigger:** Kurtosis 393 (130× Gaussian); 2.65× divergence in extreme-price trading

**Methods:** Generalized Pareto Distribution, GEV, Hill estimator, regulatory-structure decomposition

**Duration:** 2 weeks

**Success:** Polymarket ξ ≈ 0.2–0.4 (heavy); Kalshi ξ < 0 (bounded); CVaR 2–3× difference

**Output:** `phase2_evt_report.md` + Hill plots, QQ plots, ES curves

---

### Phase 3: Tail Risk Prediction (RQ3)

**Question:** Can microstructure features predict extreme returns?

**Empirical Trigger:** 24× vs 1.37× intraday clustering; 6.33 mean/median trade-size ratio

**Methods:** XGBoost classifier, SHAP feature importance, backtested trading strategy

**Duration:** 2 weeks

**Success:** XGBoost AUC-ROC > 0.6; Sharpe improvement > 0.1

**Output:** `phase3_prediction_report.md` + SHAP plots, calibration curves, equity curves

---

## Why This Combination Is Strongest

1. **Coherent Narrative:** Characterize efficiency → Explain tail risk → Demonstrate actionable prediction
2. **Data-Grounded:** Every RQ motivated by empirical pattern from 476M trades
3. **Methodologically Rigorous:** Combines classical econometrics (variance ratios), modern statistics (EVT), and machine learning (XGBoost)
4. **Implementable:** No event matching required; all analyses run on existing parquet dataset
5. **Publication-Ready:** Suitable for ICAIF, NeurIPS, or JFDS with minimal scope adjustment

---

## Dropped Elements (and Why)

### Reinforcement Learning (from draft_v1.md)
- **Original Role:** H2 — Learn profitable execution strategies
- **Why Dropped:** XGBoost achieves same goal (actionable prediction) 2-3x faster with better interpretability (SHAP)
- **Future:** Viable extension if Phase 3 Sharpe > 0.2

### Copulas & Event Matching (from hypothesis_candidates.md)
- **Original Role:** H3 — Cross-platform dependence and arbitrage
- **Why Dropped:** Requires high-confidence market matching (Kalshi tickets ↔ Polymarket outcomes). Data engineering (fuzzy matching, manual curation) is substantial and error-prone
- **Future:** Viable standalone paper if matched-event dataset becomes available

---

## Success Criteria Summary

| Phase | Criterion | Target |
|-------|-----------|--------|
| 1 | Roll spread estimates | 0.6–1.5 cents (Kalshi), 0.1–0.3 (Polymarket) |
| 1 | Variance ratio p-value | > 0.05 (efficiency accepted) |
| 2 | GPD shape (Polymarket) | ξ ≈ 0.2–0.4 (heavy) |
| 2 | GPD shape (Kalshi) | ξ < 0 (bounded) |
| 2 | CVaR ratio | 2–3× (Polymarket > Kalshi) |
| 3 | XGBoost AUC-ROC | > 0.6 |
| 3 | XGBoost AUC-PR | > 0.15 |
| 3 | Backtest Sharpe | > 0.1 improvement |
| Overall | Paper draft | Submittal-ready |

---

## Publication Timeline

| Stage | Duration | Output | Deadline |
|-------|----------|--------|----------|
| Phase 1 | 2 weeks | Efficiency report + tables | Week 2 |
| Phase 2 | 2 weeks | EVT report + figures | Week 4 |
| Phase 3 | 2 weeks | Prediction report + strategy | Week 6 |
| Paper + Integration | 2 weeks | Draft submitted to ICAIF | Week 8 |
| Revision Cycle | 4 weeks | Final version | Week 12 |

**Target Submission:** ICAIF 2026 (April/May deadline)

---

## Venue Fit

### Primary: ICAIF 2026
- **Frame:** "Tail Risk Prediction in Prediction Markets using Microstructure ML"
- **Length:** 12–15 pages
- **Emphasis:** Machine learning (XGBoost, SHAP) applied to novel asset class

### Secondary: NeurIPS Datasets & Benchmarks 2026
- **Frame:** "The Prediction Market Microstructure Dataset: 476M Trades Across Kalshi and Polymarket"
- **Length:** 8–12 pages
- **Emphasis:** Novel dataset, benchmarking, empirical characterization

### Tertiary: Journal of Financial Data Science
- **Frame:** "Martingale Efficiency and Tail Risk in Prediction Markets"
- **Length:** 20+ pages
- **Emphasis:** Econometric rigor, regulatory implications

---

## Key Decisions & FAQ

**Q: Why three phases instead of one hypothesis?**  
A: Logical progression (characterize → explain → predict) creates a complete story. Each phase stands alone but builds on the previous. Modular structure allows submission of Phase 1–2 if Phase 3 underperforms.

**Q: Why drop Reinforcement Learning?**  
A: XGBoost achieves same goal (actionable prediction) in 1/3 the time, is more interpretable (SHAP values), requires less data engineering. RL remains viable future work.

**Q: Why drop Copulas?**  
A: Requires high-confidence event matching. Data engineering (fuzzy matching, manual curation) is substantial and error-prone. Defer as separate paper if matched-event dataset emerges.

**Q: What makes this unified version stronger?**  
A: Combines best of both: Claude's XGBoost + Cursor's empirical grounding + Roll's spread model + regulatory-structure decomposition. Creates one coherent narrative instead of three competing ideas.

---

## Getting Started: Next Steps

1. **Advisor Review**
   - Share `hypothesis_unified_v2.md` (30–45 min read)
   - Schedule 30-min discussion
   - Approve methodology, timeline, venue fit

2. **Approve & Assign**
   - Create implementation tickets for Phase 1–3
   - Assign team members to each phase
   - Establish weekly sync cadence

3. **Phase 1 Execution**
   - Standardize data (Kalshi cents → USDC, join Polymarket timestamps)
   - Run Roll's spread estimation
   - Run variance ratio tests
   - Write Phase 1 report

4. **Repeat for Phase 2, Phase 3**

---

## File Locations

**All files located in:**
```
/sessions/wizardly-loving-darwin/mnt/poly_kalshi_dataset/artifacts/
```

**Key files:**
- `hypothesis_unified_v2.md` — PRIMARY DOCUMENT
- `QUICK_START.txt` — One-page visual summary
- `README_HYPOTHESIS.md` — Full navigation guide
- `SYNTHESIS_NOTES.txt` — Design decisions explained

---

## Questions?

Refer to:
- `README_HYPOTHESIS.md` for full methodology details
- `SYNTHESIS_NOTES.txt` for design rationale
- `QUICK_START.txt` for quick reference
- `hypothesis_unified_v2.md` Section "FAQ" for common questions

---

**Created:** 2026-03-29  
**Status:** Ready for implementation  
**Audience:** Advisors, collaborators, implementers, committees  
**Next Action:** Share `hypothesis_unified_v2.md` with advisor; schedule review discussion
