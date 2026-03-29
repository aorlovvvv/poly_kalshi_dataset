# Autoresearch: Tail-Risk Trading Strategy

## Goal
Maximize out-of-sample Sharpe ratio of a tail-risk-adjusted strategy
on 476M+ prediction market trades.

## Key Insight
Prediction markets exhibit:
- Strong mean reversion (VR ≈ 0.59) due to bid-ask bounce
- Heavy tails (GPD ξ > 0) — extreme moves are more frequent than Gaussian
- Nonlinear dependence (BDS test rejects iid) — past patterns predict future
- Sports markets have heavier tails and stronger dependence than political

## Strategy Design Space
1. **Tail prediction model**: predict P(|Δp| > 2σ) using lagged features
2. **Position sizing**: convert tail probability to a [-1, 1] position
3. **Key features**: lag_ret_1 (mean reversion signal), taker_buy (order flow),
   rolling_vol (regime detection), price_boundary_dist (resolution proximity)

## What Works (from Phase 2-3 analysis)
- `lag_ret_1` is the strongest predictor (mean reversion after trades)
- `taker_side` contains microstructure information
- Rolling volatility captures regime changes
- Sports markets have more predictable tail events

## What To Try
- Kelly criterion sizing: edge / variance
- Volatility targeting
- XGBoost or LightGBM (better than sklearn GBM)
- Feature interactions (ret × vol, taker × price_boundary)
- Regime detection (high vol vs low vol states)
- Mean-reversion signal: go against large lag_ret when vol is low
