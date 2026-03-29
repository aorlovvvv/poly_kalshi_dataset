"""
FIXED — DO NOT EDIT.

Loads Kalshi trade data via DuckDB into feature matrices for backtesting.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
KALSHI_GLOB = (ROOT / "data" / "raw" / "kalshi" / "trades" / "*.parquet").as_posix()

MAX_MARKETS = 20
MAX_TRADES_PER_MARKET = 50_000
MAX_TRADES_TOTAL = 500_000

FEATURE_COLUMNS = [
    "price", "trade_size", "taker_buy",
    "ret", "lag_ret_1", "lag_ret_2",
    "lag_size_1", "lag_taker_1",
    "buy_imbalance_20", "rolling_vol_10", "rolling_vol_50",
    "hour_of_day", "day_of_week", "price_boundary_dist",
]

TAIL_THRESHOLD_SIGMA = 2.0


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and featurise Kalshi trade data.

    Returns:
        (train_df, test_df, feature_names)
        Each DataFrame has columns = FEATURE_COLUMNS + ['target', 'market_id', 'ts', 'next_ret']
    """
    con = duckdb.connect()
    log.info("Loading top %d Kalshi markets (max %dk trades each)...",
             MAX_MARKETS, MAX_TRADES_PER_MARKET // 1000)

    markets = con.execute(f"""
        SELECT ticker AS market_id, COUNT(*) AS n
        FROM read_parquet('{KALSHI_GLOB}', union_by_name=true)
        WHERE created_time IS NOT NULL AND yes_price >= 0 AND yes_price <= 99
        GROUP BY ticker
        HAVING COUNT(*) >= 1000
        ORDER BY n DESC
        LIMIT {MAX_MARKETS}
    """).fetchall()

    log.info("Found %d markets", len(markets))
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

    split_idx = int(len(combined) * 0.8)
    train_df = combined.iloc[:split_idx].reset_index(drop=True)
    test_df = combined.iloc[split_idx:].reset_index(drop=True)

    log.info("Train: %d rows, Test: %d rows, Markets: %d",
             len(train_df), len(test_df), len(all_dfs))

    return train_df, test_df, FEATURE_COLUMNS


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

    df["buy_imbalance_20"] = (
        df["taker_buy"].rolling(20, min_periods=1).mean()
    )

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
