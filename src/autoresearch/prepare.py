"""
One-time data preparation and fixed evaluation harness for autoresearch.

Downloads nothing — data is local Kalshi parquet files. Caches processed
feature matrices to ~/.cache/autoresearch_pm/ for fast repeated evaluation.

Usage:
    python prepare.py              # prepare + cache data
    python prepare.py --force      # rebuild cache from scratch

Exports used by train.py:
    Strategy          — abstract base class
    PortfolioState    — portfolio snapshot dataclass
    FEATURE_COLUMNS   — list of 14 feature names
    load_data()       — returns (train_df, test_df, feature_names)
    evaluate()        — walk-forward backtest, returns metrics dict

DO NOT MODIFY THIS FILE. It is the fixed evaluation harness.
"""

import os
import sys
import time
import math
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# Kalshi taker fee: 7% of expected earnings per contract.
#   fee = 0.07 * price * (1 - price)
# This is parabolic: max 1.75¢ at p=0.50, drops to 0.63¢ at p=0.10 or p=0.90,
# and to 0.33¢ at p=0.05 or p=0.95. Source: kalshi.com/docs/kalshi-fee-schedule.pdf
# Maker fee is 25% of taker fee (0.0175 * p * (1-p)).
# We use the taker fee (conservative). The cost function is applied per-trade
# in _backtest_stream using the current price, not a flat per-unit charge.
TAKER_FEE_RATE = 0.07  # 7% of expected earnings (Kalshi taker fee)
ANNUALIZATION_FACTOR = np.sqrt(365)  # prediction markets trade 24/7/365

MAX_MARKETS = 50
MAX_TRADES_PER_MARKET = 50_000
MAX_TRADES_TOTAL = 2_000_000
TAIL_THRESHOLD_SIGMA = 2.0
TRAIN_SPLIT = 0.8

FEATURE_COLUMNS = [
    "price", "trade_size", "taker_buy",
    "ret", "lag_ret_1", "lag_ret_2",
    "lag_size_1", "lag_taker_1",
    "buy_imbalance_20", "rolling_vol_10", "rolling_vol_50",
    "hour_of_day", "day_of_week", "price_boundary_dist",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
KALSHI_GLOB = str(ROOT / "data" / "raw" / "kalshi" / "trades" / "*.parquet")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_pm")

# ---------------------------------------------------------------------------
# Interfaces (fixed — strategies must implement these)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Snapshot of portfolio at a given point in time."""
    position: float = 0.0
    equity: float = 1.0
    peak_equity: float = 1.0
    drawdown: float = 0.0
    trade_count: int = 0
    total_cost: float = 0.0


class Strategy(ABC):
    """Abstract base class that all strategies must implement."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this strategy version."""
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict P(tail event) for each row. Returns 1-D array in [0, 1]."""
        ...

    @abstractmethod
    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        """Decide position in [-1, 1] given predicted tail probability."""
        ...

    @abstractmethod
    def get_feature_importances(self) -> dict[str, float]:
        """Return feature name -> importance mapping."""
        ...


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for a single market's trade series."""
    df = df.copy()
    ts = pd.to_datetime(df["ts"], utc=True)

    df["ret"] = df["price"].diff()
    df["next_ret"] = df["ret"].shift(-1)
    df["lag_ret_1"] = df["ret"].shift(1)
    df["lag_ret_2"] = df["ret"].shift(2)
    df["lag_size_1"] = df["trade_size"].shift(1)
    df["lag_taker_1"] = df["taker_buy"].shift(1)
    df["buy_imbalance_20"] = df["taker_buy"].shift(1).rolling(20, min_periods=1).mean()

    abs_ret = df["ret"].abs()
    df["rolling_vol_10"] = abs_ret.shift(1).rolling(10, min_periods=1).std()
    df["rolling_vol_50"] = abs_ret.shift(1).rolling(50, min_periods=1).std()

    df["hour_of_day"] = ts.dt.hour.values
    df["day_of_week"] = ts.dt.dayofweek.values
    df["price_boundary_dist"] = df["price"].apply(
        lambda p: min(p, 1.0 - p) if 0 < p < 1 else 0.0
    )

    ret_std = df["ret"].std()
    threshold = TAIL_THRESHOLD_SIGMA * ret_std if ret_std > 0 else 0.01
    df["target"] = (df["ret"].abs() > threshold).astype(int)

    return df


# ---------------------------------------------------------------------------
# Data loading and caching
# ---------------------------------------------------------------------------

def _load_from_parquet() -> pd.DataFrame:
    """Load and featurize Kalshi trade data from raw parquet files."""
    con = duckdb.connect()
    print(f"Loading top {MAX_MARKETS} Kalshi markets "
          f"(max {MAX_TRADES_PER_MARKET // 1000}k trades each)...")

    markets = con.execute(f"""
        SELECT ticker AS market_id, COUNT(*) AS n
        FROM read_parquet('{KALSHI_GLOB}', union_by_name=true)
        WHERE created_time IS NOT NULL AND yes_price >= 0 AND yes_price <= 99
        GROUP BY ticker
        HAVING COUNT(*) >= 1000
        ORDER BY n DESC
        LIMIT {MAX_MARKETS}
    """).fetchall()

    print(f"Found {len(markets)} markets")
    all_dfs: list[pd.DataFrame] = []

    for market_id, n_trades in markets:
        df = con.execute(f"""
            SELECT
                ticker AS market_id,
                yes_price::DOUBLE / 100.0 AS price,
                count::DOUBLE AS trade_size,
                CASE WHEN taker_side = 'yes' THEN 1.0 ELSE 0.0 END AS taker_buy,
                created_time AS ts
            FROM read_parquet('{KALSHI_GLOB}', union_by_name=true)
            WHERE ticker = ? AND created_time IS NOT NULL
                  AND yes_price >= 0 AND yes_price <= 99
            ORDER BY created_time
            LIMIT {MAX_TRADES_PER_MARKET}
        """, [market_id]).df()

        if len(df) < 100:
            continue

        df = _add_features(df)
        all_dfs.append(df)

        if sum(len(d) for d in all_dfs) >= MAX_TRADES_TOTAL:
            break

    con.close()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLUMNS + ["target", "next_ret"])
    combined = combined.sort_values("ts").reset_index(drop=True)
    return combined


def prepare_data(force=False):
    """Load raw data, engineer features, cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "kalshi_features.parquet")

    if os.path.exists(cache_path) and not force:
        print(f"Data: already cached at {cache_path}")
        return

    t0 = time.time()
    combined = _load_from_parquet()
    combined.to_parquet(cache_path, index=False)
    elapsed = time.time() - t0
    print(f"Data: {len(combined):,} rows cached to {cache_path} ({elapsed:.1f}s)")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached feature data and split into train/test.

    Returns (train_df, test_df, feature_names).
    """
    cache_path = os.path.join(CACHE_DIR, "kalshi_features.parquet")
    if not os.path.exists(cache_path):
        print("Cache not found, building from raw parquet...")
        prepare_data()

    combined = pd.read_parquet(cache_path)
    split_idx = int(len(combined) * TRAIN_SPLIT)
    train_df = combined.iloc[:split_idx].reset_index(drop=True)
    test_df = combined.iloc[split_idx:].reset_index(drop=True)

    print(f"Data: {len(train_df):,} train, {len(test_df):,} test, "
          f"{combined['market_id'].nunique()} markets")

    return train_df, test_df, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def _kalshi_taker_fee(price: float, trade_delta: float) -> float:
    """Kalshi taker fee: 7% of expected earnings = 0.07 * p * (1-p) per contract.
    We scale by |trade_delta| since position is in [-1, 1] (1.0 = 1 contract)."""
    p = np.clip(price, 0.01, 0.99)
    fee_per_contract = TAKER_FEE_RATE * p * (1.0 - p)
    return abs(trade_delta) * fee_per_contract


def _backtest_stream(strategy: Strategy, tail_probs: np.ndarray,
                     next_rets: np.ndarray, market_ids: np.ndarray,
                     prices: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """
    Walk-forward backtest that resets position at market boundaries.

    Uses Kalshi's actual price-dependent taker fee: 0.07 * p * (1-p).
    Position resets to 0 when market_id changes between consecutive rows.

    Returns (pnl_array, positions_list).
    """
    n = len(tail_probs)
    state = PortfolioState()
    pnl_series = np.zeros(n)
    positions: list[float] = []
    prev_market = None

    for i in range(n):
        cur_market = market_ids[i]

        if cur_market != prev_market and prev_market is not None:
            close_cost = _kalshi_taker_fee(prices[i], state.position)
            state.equity -= close_cost
            state.total_cost += close_cost
            state.position = 0.0
        prev_market = cur_market

        if np.isnan(next_rets[i]):
            positions.append(state.position)
            continue

        desired = np.clip(strategy.size_position(tail_probs[i], state), -1.0, 1.0)
        trade_delta = abs(desired - state.position)
        cost = _kalshi_taker_fee(prices[i], trade_delta)

        pnl = desired * next_rets[i] - cost
        state.position = desired
        state.equity += pnl
        state.total_cost += cost
        state.trade_count += 1
        state.peak_equity = max(state.peak_equity, state.equity)
        state.drawdown = (1.0 - state.equity / state.peak_equity
                          if state.peak_equity > 0 else 0.0)

        pnl_series[i] = pnl
        positions.append(desired)

    return pnl_series, positions


def _compute_sharpe_sortino(pnl_arr: np.ndarray, dates: np.ndarray) -> dict:
    """Compute daily-aggregated Sharpe and Sortino from trade-level PnL."""
    pnl_df = pd.DataFrame({"date": dates[:len(pnl_arr)], "pnl": pnl_arr})
    daily_pnl = pnl_df.groupby("date")["pnl"].sum().values
    daily_pnl = daily_pnl[~np.isnan(daily_pnl)]

    if len(daily_pnl) > 1:
        mean_daily = np.mean(daily_pnl)
        std_daily = np.std(daily_pnl, ddof=1)
        sharpe = (mean_daily / std_daily * ANNUALIZATION_FACTOR) if std_daily > 0 else 0.0

        downside_daily = daily_pnl[daily_pnl < 0]
        downside_std = np.std(downside_daily, ddof=1) if len(downside_daily) > 1 else std_daily
        sortino = (mean_daily / downside_std * ANNUALIZATION_FACTOR) if downside_std > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    return {"sharpe": sharpe, "sortino": sortino, "n_days": len(daily_pnl)}


def evaluate(strategy: Strategy, X_test: np.ndarray, y_test: np.ndarray,
             next_rets: np.ndarray, test_df: pd.DataFrame) -> dict:
    """
    Walk-forward backtest with transaction costs and market-aware positioning.

    Position resets to zero when the market_id changes between consecutive
    rows in the time-sorted stream. This prevents the cross-market blurring
    problem where a position sized for one market earns another market's return.

    The primary metric is sharpe_ratio (daily-aggregated, annualized √365).

    NOTE on auc_roc: This metric is MEANINGLESS for this setup because the
    target = (|ret| > 2σ) and 'ret' is a feature. The AUC measures trivial
    circularity, not predictive skill. It is retained only for backward
    compatibility with results.tsv. Ignore it.
    """
    tail_probs = np.clip(strategy.predict_tail_probability(X_test), 0.0, 1.0)

    # AUC — retained for backward compat but MEANINGLESS (see docstring)
    try:
        auc = roc_auc_score(y_test, tail_probs)
    except Exception:
        auc = 0.5

    market_ids = test_df["market_id"].values
    prices = test_df["price"].values

    pnl_arr, positions = _backtest_stream(
        strategy, tail_probs, next_rets, market_ids, prices
    )

    pnl_clean = pnl_arr[~np.isnan(pnl_arr)]

    # Daily-aggregated Sharpe / Sortino
    timestamps = pd.to_datetime(test_df["ts"], utc=True)
    dates = timestamps.dt.date.values
    metrics = _compute_sharpe_sortino(pnl_arr, dates)
    sharpe = metrics["sharpe"]
    sortino = metrics["sortino"]
    n_days = metrics["n_days"]

    # Max drawdown
    max_dd = 0.0
    running_equity = 1.0
    peak = 1.0
    for p in pnl_clean:
        running_equity += p
        peak = max(peak, running_equity)
        dd = 1.0 - running_equity / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_return = running_equity - 1.0
    wins = np.sum(pnl_clean > 0)
    win_rate = wins / len(pnl_clean) if len(pnl_clean) > 0 else 0.0

    # Per-market Sharpe breakdown (average of individual market Sharpes)
    per_market_sharpes = []
    for mid in test_df["market_id"].unique():
        mask = test_df["market_id"].values == mid
        if mask.sum() < 20:
            continue
        m_pnl = pnl_arr[mask]
        m_dates = dates[mask]
        m_metrics = _compute_sharpe_sortino(m_pnl, m_dates)
        if m_metrics["n_days"] > 1:
            per_market_sharpes.append(m_metrics["sharpe"])

    avg_per_market_sharpe = (
        round(float(np.mean(per_market_sharpes)), 6)
        if per_market_sharpes else 0.0
    )

    avg_price = float(np.mean(prices[~np.isnan(prices)])) if len(prices) > 0 else 0.5
    avg_fee = TAKER_FEE_RATE * avg_price * (1.0 - avg_price)

    return {
        "sharpe_ratio": round(float(sharpe), 6),
        "sortino_ratio": round(float(sortino), 6),
        "total_return": round(float(total_return), 6),
        "max_drawdown": round(float(max_dd), 6),
        "auc_roc_IGNORE": round(float(auc), 4),
        "win_rate": round(float(win_rate), 4),
        "n_trades": len(pnl_clean),
        "n_days": n_days,
        "n_markets": test_df["market_id"].nunique(),
        "mean_position": round(float(np.mean(np.abs(positions))), 4),
        "avg_fee_per_contract": round(avg_fee, 6),
        "avg_per_market_sharpe": avg_per_market_sharpe,
        "per_market_sharpes": {
            mid: round(float(s), 4)
            for mid, s in zip(
                [m for m in test_df["market_id"].unique()
                 if (test_df["market_id"].values == m).sum() >= 20],
                per_market_sharpes
            )
        } if per_market_sharpes else {},
        "feature_importances": strategy.get_feature_importances(),
    }


# ---------------------------------------------------------------------------
# Main — run once to prepare data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for prediction market autoresearch")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild cache even if it exists")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    prepare_data(force=args.force)
    print()

    train_df, test_df, features = load_data()
    print(f"\nFeatures ({len(features)}): {features}")
    print(f"Target rate (train): {train_df['target'].mean():.4f}")
    print(f"Target rate (test):  {test_df['target'].mean():.4f}")
    print()
    print("Done! Ready to train.")
