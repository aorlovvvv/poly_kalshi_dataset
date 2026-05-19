"""
PIN (Probability of Informed Trading) and VPIN Estimation for Prediction Markets.

Implements:
1. Classical PIN model (Easley, Kiefer, O'Hara & Paperman 1996)
   - MLE estimation of (alpha, delta, mu, epsilon_b, epsilon_s) from
     daily buy/sell trade counts
   - PIN = alpha * mu / (alpha * mu + epsilon_b + epsilon_s)

2. VPIN (Volume-Synchronized PIN, Easley, López de Prado & O'Hara 2012)
   - Volume-bucketed trade classification
   - Rolling VPIN as real-time toxicity metric

Usage:
    python -m src.analysis.pin_estimation [--markets N] [--output PATH]
"""

import json
import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
KALSHI_GLOB = str(ROOT / "data" / "raw" / "kalshi" / "trades" / "*.parquet")


# ---------------------------------------------------------------------------
# PIN Model (Easley, Kiefer, O'Hara & Paperman 1996)
# ---------------------------------------------------------------------------

@dataclass
class PINResult:
    """Result of PIN estimation for a single market."""
    market_id: str
    alpha: float       # Probability of information event
    delta: float       # Probability that news is bad (given info event)
    mu: float          # Informed arrival rate
    epsilon_b: float   # Uninformed buy arrival rate
    epsilon_s: float   # Uninformed sell arrival rate
    pin: float         # Probability of Informed Trading
    n_days: int        # Number of trading days used
    log_likelihood: float
    converged: bool
    category: str = ""


def _pin_log_likelihood(params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
    """
    Negative log-likelihood for the PIN model.

    The PIN model assumes each day is independently drawn:
    - With probability (1-alpha): no information event
      Buys ~ Poisson(epsilon_b), Sells ~ Poisson(epsilon_s)
    - With probability alpha*delta: bad news
      Buys ~ Poisson(epsilon_b), Sells ~ Poisson(epsilon_s + mu)
    - With probability alpha*(1-delta): good news
      Buys ~ Poisson(epsilon_b + mu), Sells ~ Poisson(epsilon_s)

    Parameters: [alpha, delta, mu, epsilon_b, epsilon_s]
    """
    alpha, delta, mu, eb, es = params

    # Enforce bounds
    if not (0 < alpha < 1 and 0 < delta < 1 and mu > 0 and eb > 0 and es > 0):
        return 1e10

    total_ll = 0.0
    for B, S in zip(buys, sells):
        # Use log-sum-exp trick for numerical stability
        # log P(B,S | no info) = B*log(eb) + S*log(es) - eb - es - log(B!) - log(S!)
        log_base = -gammaln(B + 1) - gammaln(S + 1)

        # No information event
        log_p_no_info = np.log(1 - alpha) + B * np.log(eb) + S * np.log(es) - eb - es

        # Bad news: informed sellers enter
        log_p_bad = (np.log(alpha) + np.log(delta) +
                     B * np.log(eb) + S * np.log(es + mu) - eb - (es + mu))

        # Good news: informed buyers enter
        log_p_good = (np.log(alpha) + np.log(1 - delta) +
                      B * np.log(eb + mu) + S * np.log(es) - (eb + mu) - es)

        # Log-sum-exp
        max_log = max(log_p_no_info, log_p_bad, log_p_good)
        day_ll = max_log + np.log(
            np.exp(log_p_no_info - max_log) +
            np.exp(log_p_bad - max_log) +
            np.exp(log_p_good - max_log)
        )

        total_ll += day_ll + log_base

    return -total_ll  # Minimize negative log-likelihood


def estimate_pin(market_id: str, buys: np.ndarray, sells: np.ndarray,
                 category: str = "") -> PINResult:
    """
    Estimate PIN for a single market using MLE.

    Args:
        market_id: Market identifier
        buys: Array of daily buy counts
        sells: Array of daily sell counts
        category: Market category (political, sports, etc.)

    Returns:
        PINResult with estimated parameters
    """
    n_days = len(buys)
    mean_b = np.mean(buys) + 1  # Add 1 to avoid zero
    mean_s = np.mean(sells) + 1

    # Multiple starting points for robustness
    best_result = None
    best_ll = np.inf

    starting_points = [
        [0.3, 0.5, max(mean_b, mean_s) * 0.3, mean_b * 0.7, mean_s * 0.7],
        [0.5, 0.5, max(mean_b, mean_s) * 0.5, mean_b * 0.5, mean_s * 0.5],
        [0.1, 0.5, max(mean_b, mean_s) * 0.1, mean_b * 0.9, mean_s * 0.9],
        [0.4, 0.3, max(mean_b, mean_s) * 0.4, mean_b * 0.6, mean_s * 0.6],
        [0.2, 0.7, max(mean_b, mean_s) * 0.2, mean_b * 0.8, mean_s * 0.8],
    ]

    bounds = [(0.01, 0.99), (0.01, 0.99), (0.1, None), (0.1, None), (0.1, None)]

    for x0 in starting_points:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    _pin_log_likelihood, x0, args=(buys, sells),
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 5000, "ftol": 1e-12}
                )

            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None or not np.isfinite(best_ll):
        return PINResult(
            market_id=market_id, alpha=0, delta=0, mu=0,
            epsilon_b=0, epsilon_s=0, pin=0, n_days=n_days,
            log_likelihood=0, converged=False, category=category
        )

    alpha, delta, mu, eb, es = best_result.x
    pin = alpha * mu / (alpha * mu + eb + es)

    return PINResult(
        market_id=market_id,
        alpha=round(float(alpha), 6),
        delta=round(float(delta), 6),
        mu=round(float(mu), 4),
        epsilon_b=round(float(eb), 4),
        epsilon_s=round(float(es), 4),
        pin=round(float(pin), 6),
        n_days=n_days,
        log_likelihood=round(float(-best_ll), 4),
        converged=best_result.success,
        category=category
    )


# ---------------------------------------------------------------------------
# VPIN (Easley, López de Prado & O'Hara 2012)
# ---------------------------------------------------------------------------

def compute_vpin(prices: np.ndarray, volumes: np.ndarray, sides: np.ndarray,
                 bucket_size: int = 50, n_buckets: int = 50) -> np.ndarray:
    """
    Compute VPIN (Volume-Synchronized PIN) as a rolling toxicity metric.

    Args:
        prices: Trade prices
        volumes: Trade volumes (quantity)
        sides: Trade side (1 = buy, 0 = sell)
        bucket_size: Number of trades per volume bucket
        n_buckets: Number of buckets in rolling VPIN window

    Returns:
        Array of VPIN values (one per bucket after warmup)
    """
    n_trades = len(prices)
    if n_trades < bucket_size * n_buckets:
        return np.array([])

    # Create volume buckets
    n_full_buckets = n_trades // bucket_size
    buy_volumes = np.zeros(n_full_buckets)
    sell_volumes = np.zeros(n_full_buckets)

    for i in range(n_full_buckets):
        start = i * bucket_size
        end = start + bucket_size
        bucket_sides = sides[start:end]
        bucket_vols = volumes[start:end]

        buy_volumes[i] = np.sum(bucket_vols[bucket_sides == 1])
        sell_volumes[i] = np.sum(bucket_vols[bucket_sides == 0])

    # Rolling VPIN
    total_bucket_vol = buy_volumes + sell_volumes
    order_imbalance = np.abs(buy_volumes - sell_volumes)

    vpin_values = []
    for i in range(n_buckets, n_full_buckets):
        window_imbalance = order_imbalance[i - n_buckets:i]
        window_volume = total_bucket_vol[i - n_buckets:i]
        total_vol = np.sum(window_volume)

        if total_vol > 0:
            vpin = np.sum(window_imbalance) / total_vol
        else:
            vpin = 0.0

        vpin_values.append(vpin)

    return np.array(vpin_values)


# ---------------------------------------------------------------------------
# Data loading and estimation pipeline
# ---------------------------------------------------------------------------

def load_market_daily_counts(market_id: str, con: duckdb.DuckDBPyConnection) -> tuple:
    """Load daily buy and sell counts for a Kalshi market."""
    df = con.execute(f"""
        SELECT
            DATE_TRUNC('day', created_time) AS trade_date,
            SUM(CASE WHEN taker_side = 'yes' THEN COALESCE(count, 1) ELSE 0 END) AS buys,
            SUM(CASE WHEN taker_side = 'no' THEN COALESCE(count, 1) ELSE 0 END) AS sells
        FROM read_parquet('{KALSHI_GLOB}', union_by_name=true)
        WHERE ticker = ?
          AND created_time IS NOT NULL
        GROUP BY trade_date
        ORDER BY trade_date
    """, [market_id]).df()

    return df["buys"].values.astype(float), df["sells"].values.astype(float)


def categorize_market(ticker: str) -> str:
    """Heuristic categorization of Kalshi market by ticker prefix."""
    t = ticker.upper()
    if any(x in t for x in ["PRES", "SENATE", "HOUSE", "GOV", "PARTY", "ELECT"]):
        return "political"
    if any(x in t for x in ["NFL", "NBA", "MLB", "NHL", "NCAA", "WNBA", "MLS",
                              "GAME", "MASTERS", "SUPER"]):
        return "sports"
    if any(x in t for x in ["BTC", "ETH", "CRYPTO", "SPX", "NASDAQ", "RATE",
                              "CPI", "GDP", "FED"]):
        return "economic"
    return "other"


def run_pin_estimation(n_markets: int = 30, output_path: str = None):
    """Run PIN estimation on top Kalshi markets."""
    con = duckdb.connect()

    # Get top markets by trade count
    markets = con.execute(f"""
        SELECT ticker, COUNT(*) AS n
        FROM read_parquet('{KALSHI_GLOB}', union_by_name=true)
        WHERE created_time IS NOT NULL
        GROUP BY ticker
        HAVING COUNT(*) >= 500
        ORDER BY n DESC
        LIMIT {n_markets}
    """).fetchall()

    print(f"Estimating PIN for {len(markets)} markets...")

    results = []
    for i, (market_id, n_trades) in enumerate(markets):
        category = categorize_market(market_id)
        buys, sells = load_market_daily_counts(market_id, con)

        if len(buys) < 10:
            print(f"  [{i+1}/{len(markets)}] {market_id}: too few days ({len(buys)}), skipping")
            continue

        pin_result = estimate_pin(market_id, buys, sells, category=category)
        results.append(pin_result)

        status = "OK" if pin_result.converged else "FAIL"
        print(f"  [{i+1}/{len(markets)}] {market_id} ({category}): "
              f"PIN={pin_result.pin:.4f}, alpha={pin_result.alpha:.3f}, "
              f"mu={pin_result.mu:.1f}, days={pin_result.n_days} [{status}]")

    con.close()

    # Summary statistics
    converged = [r for r in results if r.converged]
    print(f"\n--- PIN Estimation Summary ---")
    print(f"Markets estimated: {len(results)}")
    print(f"Converged: {len(converged)}")

    if converged:
        pins = [r.pin for r in converged]
        print(f"Mean PIN: {np.mean(pins):.4f}")
        print(f"Median PIN: {np.median(pins):.4f}")
        print(f"Range: [{min(pins):.4f}, {max(pins):.4f}]")

        # By category
        categories = set(r.category for r in converged)
        for cat in sorted(categories):
            cat_pins = [r.pin for r in converged if r.category == cat]
            if cat_pins:
                print(f"  {cat}: mean PIN = {np.mean(cat_pins):.4f} "
                      f"(n={len(cat_pins)})")

    # Save results
    if output_path is None:
        output_path = str(ROOT / "artifacts" / "pin_results.json")

    output = {
        "n_markets": len(results),
        "n_converged": len(converged),
        "summary": {
            "mean_pin": round(float(np.mean([r.pin for r in converged])), 6) if converged else 0,
            "median_pin": round(float(np.median([r.pin for r in converged])), 6) if converged else 0,
            "std_pin": round(float(np.std([r.pin for r in converged])), 6) if converged else 0,
        },
        "by_category": {},
        "markets": [asdict(r) for r in results]
    }

    for cat in sorted(set(r.category for r in converged)):
        cat_results = [r for r in converged if r.category == cat]
        cat_pins = [r.pin for r in cat_results]
        output["by_category"][cat] = {
            "n": len(cat_results),
            "mean_pin": round(float(np.mean(cat_pins)), 6),
            "median_pin": round(float(np.median(cat_pins)), 6),
            "mean_alpha": round(float(np.mean([r.alpha for r in cat_results])), 6),
            "mean_mu": round(float(np.mean([r.mu for r in cat_results])), 4),
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIN/VPIN estimation for prediction markets")
    parser.add_argument("--markets", type=int, default=30, help="Number of markets to estimate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_pin_estimation(n_markets=args.markets, output_path=args.output)
