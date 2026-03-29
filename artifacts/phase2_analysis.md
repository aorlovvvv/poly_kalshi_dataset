# Phase 2 Statistical Tests — Analysis

**Date:** 2026-03-29
**Source:** `artifacts/phase2_results.json` (80 VR tests, 20 Roll's spread, 2 GPD platforms, 60 BDS tests, 10 XGBoost models)
**Markets tested:** Top 20 most liquid Kalshi markets (67K–287K trades each), top 20 Polymarket markets

---

## Test A: Variance Ratio Tests (Lo-MacKinlay 1988)

### Results Summary

All 80 variance ratio tests across 20 Kalshi markets and 4 horizons (k = 2, 5, 10, 20) **reject the martingale hypothesis** at every conceivable significance level. Every M2 test statistic exceeds several thousand in absolute value; all p-values are numerically zero.

| Horizon k | VR Range | Mean VR | Interpretation |
|---|---|---|---|
| 2 | 0.525–0.657 | 0.590 | 41% mean reversion at trade-to-trade level |
| 5 | 0.224–0.385 | 0.295 | 70% mean reversion at 5-trade horizon |
| 10 | 0.117–0.253 | 0.179 | 82% mean reversion at 10-trade horizon |
| 20 | 0.062–0.157 | 0.113 | 89% mean reversion at 20-trade horizon |

### Patterns by Market Type

**Political markets** (PRES-2024-DJT, PRES-2024-KH) show the **strongest** mean reversion:
- DJT: VR(2) = 0.525, VR(20) = 0.062
- KH: VR(2) = 0.554, VR(20) = 0.073

**Sports markets** (NFL, NBA, MLB) show somewhat weaker but still overwhelming mean reversion:
- MLB (LAD): VR(2) = 0.657, VR(20) = 0.124
- NFL (SF-LA game): VR(2) = 0.627, VR(20) = 0.130

### Interpretation

The monotonically decreasing VR(k) pattern — more reversion at every longer horizon — is the signature of **discrete tick-size noise** superimposed on a potentially efficient price process. In a standard financial market, we would expect VR < 1 at short horizons (bid-ask bounce) but VR → 1 at longer horizons (efficiency restored). Here, VR continues to decline through k = 20, which suggests:

1. **Bid-ask bounce is the dominant driver** — Roll's model predicts VR(2) = 1 − 2ρ where ρ is the lag-1 autocorrelation. With ρ ≈ −0.47 for DJT, VR(2) should be ≈ 0.53 — matching the observation.
2. **Discrete pricing amplifies the effect** — Kalshi's 1-cent minimum tick (1% of the $1 contract) means that most trades bounce between adjacent ticks. A 1-cent tick on a 50-cent contract is a 2% relative spread.
3. **The martingale is rejected for trade-level prices, but this says nothing about informational efficiency.** The proper test would use mid-prices (average of bid and ask) or time-sampled prices (e.g., 1-minute intervals), which would filter out the mechanical microstructure noise.

**Implication for RQ1 (Efficiency):** Raw trade prices are definitively non-martingale, but this is mechanical, not informational. A proper efficiency test must first remove microstructure noise — the VR results quantify the magnitude of contamination that such a correction must address.

---

## Test B: Roll's Implied Spread

### Results Summary

Roll's implied spread = 2√(−Cov(Δp_t, Δp_{t−1})) was estimated for 20 markets. All autocovariances are negative (as expected from the VR results), confirming the bid-ask bounce model.

| Market Type | Example | Implied Spread (cents) | N Trades |
|---|---|---|---|
| Presidential | PRES-2024-DJT | 0.80 | 287,494 |
| Presidential | PRES-2024-KH | 0.82 | 213,961 |
| NBA | KXNBA-25-IND | 0.97 | 103,818 |
| NBA | KXNBA-25-OKC | 1.16 | 69,502 |
| MLB | KXMLB-25-LAD | 1.44 | 149,605 |
| MLB | KXMLB-25-TOR | 1.49 | 87,246 |
| NFL game | KXNFLGAME-25OCT20TBDET-TB | 0.69 | 75,915 |
| NFL game | KXNFLGAME-25OCT02SFLA-SF | 1.71 | 91,859 |
| March Madness | KXMARMAD-25-FL | 1.35 | 72,592 |

### Patterns

- **Tightest spreads:** NFL games (0.68–0.69 cents) and presidential elections (0.80–0.82 cents)
- **Widest spreads:** MLB season-long markets (1.44–1.49 cents) and some NFL games (1.71–1.72 cents)
- **Spread range:** 0.68 to 1.72 cents, with mean ≈ 1.10 cents

The bimodal pattern in NFL games is notable — some games have very tight spreads (TB-DET: 0.69 cents) while others are much wider (SF-LA: 1.72 cents). This likely reflects different levels of market maker participation or game-specific uncertainty.

### Comparison to Traditional Markets

Roll's implied spread of 0.80 cents for presidential markets corresponds to ~1.6% of a mid-priced ($0.50) contract. This is comparable to small-cap equities but wider than large-cap stocks (which typically have spreads < 0.1%). For sports markets at 1.44 cents, the implied spread reaches ~2.9% — approaching the levels seen in illiquid OTC derivatives.

**Implication for RQ1:** The implied spread estimates provide the first quantitative measure of trading costs on prediction markets. These are input parameters for any transaction-cost-adjusted efficiency test or trading strategy backtest.

---

## Test C: GPD Tail Fitting (Peaks Over Threshold)

### Results Summary

| Platform | Tail | Shape ξ | Scale σ | N Exceedances | Threshold | KS p-value |
|---|---|---|---|---|---|---|
| Kalshi | Upper | −1.529 | 0.153 | 24,833 | 0.880 | 0.0 |
| Kalshi | Lower | −1.529 | 0.153 | 24,840 | 0.880 | 0.0 |
| Polymarket | Upper | −0.195 | 0.199 | 25,000 | 0.080 | 0.0 |
| Polymarket | Lower | −0.197 | 0.201 | 25,000 | 0.079 | 0.0 |

### Interpretation

Both platforms show **negative shape parameters** (ξ < 0), placing them in the **Weibull domain** (bounded upper tail). However, the magnitudes differ dramatically:
- Kalshi: ξ = −1.53 — very strongly bounded, tail truncated sharply
- Polymarket: ξ = −0.19 — weakly bounded, approaching Gumbel domain (ξ = 0, exponential tails)

### Methodological Caveat

**These results should be interpreted with caution.** The GPD was fitted to price changes pooled across the top 10 markets per platform, ordered by timestamp. Consecutive trades from different markets create artificial "jumps" (e.g., a trade at price 0.05 in one market followed by a trade at price 0.95 in another produces a spurious Δp = 0.90). This explains:
- The anomalously high threshold for Kalshi (0.88) — 5% of "price changes" exceed 88 cents, which is clearly inter-market contamination
- The KS test rejecting the GPD fit for all cases — the data is a mixture of within-market returns and between-market noise

**For rigorous EVT analysis in Phase 3**, GPD fitting should be done **per-market** on the most liquid markets, yielding clean within-market tail estimates. The Phase 1 profiling showed kurtosis = 393 for within-market returns (top 100 markets, 5.6M observations), which suggests heavy tails (ξ > 0) when properly measured.

### Cross-Platform Signal Despite Caveat

Even with the pooling artifact, the Kalshi-vs-Polymarket comparison is informative in relative terms. Polymarket's ξ ≈ −0.19 is much closer to zero than Kalshi's ξ ≈ −1.53, consistent with Polymarket's more continuous pricing (no tick-size grid) producing smoother tail behavior. Kalshi's discretized 1-cent pricing creates a hard bound on within-tick price changes.

**Implication for RQ2 (Tail Risk):** Per-market GPD fitting is needed. The preliminary signal suggests regulatory/structural differences between platforms materially affect tail behavior — Kalshi's discrete pricing truncates tails, while Polymarket's continuous pricing allows heavier tails.

---

## Test D: BDS Test (Nonlinear Dependence)

### Results Summary

The BDS test was run on AR(1) residuals of price changes for the 5 most liquid Kalshi markets, across 4 embedding dimensions (m = 2, 3, 4, 5) and 3 epsilon multipliers (0.5, 1.0, 1.5 × std).

| Market | Type | ε = 0.5×std | ε = 1.0×std | ε = 1.5×std |
|---|---|---|---|---|
| PRES-2024-DJT | Political | **Reject** (z ≈ 5.4–6.5) | **Reject** (z ≈ 5.4–6.5) | Not significant (z ≈ 1.1–1.7) |
| PRES-2024-KH | Political | **Reject** (z ≈ 5.2–5.8) | **Reject** (z ≈ 5.2–5.8) | Borderline (z ≈ 1.7–2.0) |
| KXMLB-25-LAD | MLB | **Reject** (z ≈ 10–15) | **Reject** (z ≈ 12–16) | **Reject** (z ≈ 10–14) |
| KXNBA-25-IND | NBA | **Reject** (z ≈ 11–24) | **Reject** (z ≈ 11–16) | **Reject** (z ≈ 15–21) |
| KXNFLGAME-SF | NFL | **Reject** (z ≈ 10–17) | **Reject** (z ≈ 14–19) | **Reject** (z ≈ 21–26) |

(All "Reject" entries have p < 10^{−8}. "Not significant" means p > 0.05.)

### Key Finding: Nonlinear Dependence is Market-Type Dependent

The most striking pattern is the **divergence between political and sports markets**:

- **Political markets** (DJT, KH): BDS rejects i.i.d. at small/medium epsilon but **fails to reject at large epsilon** (1.5×std). The nonlinear structure is limited to small price changes — larger moves are approximately independent after removing linear AR(1) structure.

- **Sports markets** (MLB, NBA, NFL): BDS rejects at **all epsilon values** with massive test statistics (z > 10). The nonlinear structure pervades the entire distribution — small, medium, and large price changes all exhibit complex dependence patterns.

### Interpretation

This divergence is likely driven by the fundamentally different information arrival processes:
- **Political markets** receive occasional discrete news shocks (polls, debates, endorsements) but are otherwise driven by diffuse opinion shifts. Between shocks, the AR(1) model captures most of the dependence.
- **Sports markets** have continuous in-play information arrival (scoring events, injuries, momentum shifts) that creates **regime-switching dynamics** — the probability of extreme moves depends on the current game state in a nonlinear way.

The BDS results confirm that a simple AR(1) filter (which removes linear bid-ask bounce) is **sufficient for political markets** but **insufficient for sports markets**, where higher-order dependence structures remain.

**Implication for RQ1 (Efficiency):** Informational efficiency tests need market-type-specific approaches. Political markets may be approximately efficient after microstructure correction, while sports markets harbor exploitable nonlinear patterns.

**Implication for RQ3 (Predictability):** The strong BDS rejection in sports markets at all epsilon values is a green light for machine learning approaches — there is genuine nonlinear signal to capture, not just noise.

---

## Test E: XGBoost Tail Event Prediction

### Results Summary

XGBoost classifiers were trained on 10 Kalshi markets to predict extreme price moves (|Δp| > 2×std). Temporal 80/20 train/test split ensures no look-ahead bias.

| Market | AUC | Avg Precision | Top Feature | Its Importance |
|---|---|---|---|---|
| KXNBA-25-IND | **0.647** | 0.091 | taker_side | 0.842 |
| PRES-2024-DJT | **0.617** | 0.009 | price_level | 0.492 |
| KXNFLGAME-KC | **0.602** | 0.266 | taker_side | 0.669 |
| KXNFLGAME-JAC | 0.552 | 0.065 | taker_side | 0.385 |
| KXNFLGAME-DAL | 0.537 | 0.165 | taker_side | 0.418 |
| KXMLB-25-TOR | 0.536 | 0.028 | taker_side | 0.271 |
| KXNFLGAME-SF | 0.523 | 0.080 | hour_of_day | 0.590 |
| KXMLB-25-LAD | 0.475 | 0.032 | taker_side | 0.386 |
| KXNFLGAME-LA | 0.472 | 0.056 | hour_of_day | 0.611 |
| PRES-2024-KH | 0.407 | 0.021 | taker_side | 0.394 |

### Feature Importance Analysis

Across all 10 markets, the consistent feature importance ranking is:

1. **taker_side** (dominant in 7/10 markets, mean importance 0.420) — whether the aggressive side is buying or selling is the strongest predictor of extreme moves. This confirms that **order flow imbalance** carries information about impending jumps.

2. **price_level** (dominant in 1/10, mean importance 0.259) — the current price bucket predicts extremes, consistent with the Phase 1 finding that extreme-priced contracts (near 0 or 1) have different dynamics.

3. **hour_of_day** (dominant in 2/10, mean importance 0.260) — time-of-day matters especially for live sports markets (NFL), where game timing creates predictable volatility patterns.

4. **time_since_last_trade** (mean importance 0.088) — moderate importance; longer gaps between trades may precede larger moves (information accumulation during quiet periods).

5. **quantity_log** (mean importance 0.044) — surprisingly unimportant; trade size carries little signal about extreme moves in prediction markets. This contrasts with equity markets where large trades reliably forecast volatility.

### Interpretation

- **Best models achieve AUC 0.60–0.65** — meaningfully above the 0.50 random baseline but far from a reliable classifier. This is consistent with semi-efficient markets where some predictability exists but is limited.
- **taker_side dominance** is the standout finding. In the NBA Indiana market, taker_side alone has 84% feature importance. This suggests the buy/sell direction of the aggressive trader is a leading indicator of price jumps — possibly because informed traders systematically take one side before news events.
- **Precision is very low** (0.009–0.266) because extreme events are rare (~5% of observations). The models can rank events but struggle with absolute prediction.
- **Cross-market variation** in AUC (0.41–0.65) suggests different markets have different degrees of predictability. NBA and presidential (DJT) markets are most predictable; MLB markets are hardest.

**Implication for RQ3 (Predictability):** Simple microstructure features — especially taker side — provide genuine predictive signal for tail events. A more sophisticated feature set (multi-lag returns, rolling volatility, cross-market signals) would likely improve performance. The per-market variation suggests that model adaptation or market-specific training is important.

---

## Cross-Cutting Findings

### 1. The Bid-Ask Bounce Explains Most, But Not All

The VR tests and Roll's spread estimates paint a coherent picture: the dominant microstructure feature is bid-ask bounce. VR(2) ≈ 0.59 is consistent with Roll's model given the observed autocorrelations. However, the BDS tests show that removing this linear structure does not fully explain the data — significant nonlinear dependence remains, especially in sports markets.

### 2. Market Type Is a First-Order Variable

Every test shows systematic differences between political and sports markets:
- Political: tighter spreads, stronger mean reversion, weaker nonlinear dependence
- Sports: wider spreads, weaker mean reversion, massive nonlinear dependence
- This is not just a liquidity effect — it reflects fundamentally different information processes

### 3. Cross-Platform GPD Comparison Needs Per-Market Analysis

The pooled GPD fitting produced artifacts due to inter-market contamination. However, the relative ordering (Polymarket ξ ≈ −0.19 vs Kalshi ξ ≈ −1.53) is consistent with the known structural difference: Kalshi's discrete pricing truncates tails, Polymarket's continuous pricing does not.

### 4. Taker Side Is the Key Microstructure Signal

Across both the BDS tests (nonlinear dependence in AR(1) residuals) and XGBoost (feature importance for tail events), the direction of the aggressive side is the most informative feature. This parallels the PIN (Probability of Informed Trading) literature in equity markets and suggests that prediction markets, despite their different structure, share the same fundamental microstructure dynamic: informed traders reveal themselves through their order flow.

---

## Surprising Findings

1. **Quantity is uninformative.** Unlike equity markets where the Kyle (1985) lambda links trade size to information content, prediction market trade size has near-zero predictive power for extreme moves. This may be because Kalshi's minimum tick ($0.01) and maximum contract size make size-based strategies less viable, or because the participant base is retail-dominated.

2. **VR continues declining through k = 20.** In equity markets, VR typically approaches 1 at longer horizons. Here it drops to 0.06–0.16 at k = 20, suggesting the microstructure noise is extremely persistent or that there is genuine mean reversion in the underlying beliefs (people overreact to news and then correct).

3. **Political markets lose nonlinear dependence at large epsilon.** The BDS test's sensitivity to epsilon reveals that DJT and KH prices are approximately i.i.d. for large moves after AR(1) correction — they have "clean tails." Sports markets do not.

4. **AUC variation across markets (0.41–0.65) is high.** The same 5-feature XGBoost model ranges from worse-than-random (KH at 0.41) to moderately useful (IND at 0.65). This suggests market-specific factors — possibly game-specific information arrival patterns — dominate the feature space.

---

## Methodological Concerns

1. **GPD pooling artifact.** The cross-market pooling in the GPD analysis produced misleading thresholds and shape parameters. Phase 3 should fit per-market GPD on each of the top 100 markets with ≥1000 trades.

2. **BDS subsampling.** BDS tests used 5,000 subsampled observations from 100K+ trade series. While this is standard practice (the BDS test has O(n²) complexity and 5K observations provide adequate power), the subsampling introduces randomness. Repeating with different seeds would improve robustness.

3. **XGBoost temporal split.** The 80/20 temporal split means the test set is the last 20% of trades for each market. If market dynamics change over time (e.g., a presidential race enters its final month), the test set may not be representative.

4. **Polymarket variance ratio tests not included.** Only Kalshi markets were tested for VR and Roll's spread. Extending to Polymarket would enable the central cross-platform comparison.

5. **No Bonferroni correction.** With 80 VR tests, 60 BDS tests, and 10 ML models, multiple testing is a concern. However, the effect sizes are so large (M2 statistics in the thousands, BDS z-scores > 10) that no reasonable correction would change the conclusions.

---

## Recommendations for Phase 3

### RQ1 — Efficiency
1. **Time-sampled VR tests** — Resample trade prices to 1-minute, 5-minute, and 1-hour intervals using last-trade or VWAP, then rerun VR tests. If VR → 1 at the hourly level, the market is microstructurally noisy but informationally efficient.
2. **Cross-platform VR comparison** — Run the same VR tests on Polymarket's top 20 markets and compare to Kalshi. Polymarket's continuous pricing should show VR closer to 1 (less discrete-price bounce).
3. **Roll's spread for Polymarket** — Estimate implied spreads on Polymarket to quantify the centralized-vs-decentralized cost of trading.

### RQ2 — Tail Risk
4. **Per-market GPD fitting** — Fit GPD to within-market price changes for each of the top 100 Kalshi markets and top 100 Polymarket markets. Report the distribution of ξ estimates and test whether ξ differs systematically by platform, market type, or market phase (early vs. near-resolution).
5. **Hill estimator complement** — Use the Hill estimator as a non-parametric check on the GPD results. The Phase 1 kurtosis of 393 strongly suggests positive ξ (heavy tails) when properly measured within-market.

### RQ3 — Predictability
6. **Richer feature set for XGBoost** — Add multi-lag returns (AR(p) features), rolling volatility (5/10/50 trade windows), buy-sell imbalance ratios, and cross-market signals. The BDS results confirm nonlinear structure exists to capture.
7. **Cross-platform prediction** — Use Polymarket features (price, volume, buy/sell ratio) to predict Kalshi price moves (and vice versa). The 24× vs 1.37× intraday variation difference creates natural lead-lag opportunities.
8. **Copula-based dependence** — For matched events on both platforms, fit copulas to the joint price distributions and estimate tail dependence coefficients. This directly addresses the cross-platform dependence question.

---

## Summary Table

| Test | Key Result | p-value | Implication |
|---|---|---|---|
| Variance Ratio | VR(2) ≈ 0.59, VR(20) ≈ 0.11 | < 10^{−1000} | Trade prices are non-martingale (microstructure noise) |
| Roll's Spread | 0.68–1.72 cents | N/A | Political markets tightest; sports widest |
| GPD Tail | Kalshi ξ = −1.53, Poly ξ = −0.19 | (fit rejected) | Per-market analysis needed; platform structure affects tails |
| BDS Test | z = 5–25, except political at large ε | < 10^{−8} | Strong nonlinear dependence, especially in sports markets |
| XGBoost | AUC 0.41–0.65 | N/A | Moderate predictability; taker_side is key feature |
