"""
Data profiling script for prediction market research.
Profiles all raw datasets to support hypothesis development.
Outputs: artifacts/data_profile.json + printed summary.

Usage: python -m src.analysis.data_profile
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()
    out: dict[str, Any] = {}

    # Register raw views
    globs = {
        "kalshi_trades": "data/raw/kalshi/trades/*.parquet",
        "kalshi_markets": "data/raw/kalshi/markets/*.parquet",
        "polymarket_trades": "data/raw/polymarket/trades/*.parquet",
        "polymarket_blocks": "data/raw/polymarket/blocks/*.parquet",
    }
    for name, g in globs.items():
        path = (ROOT / g).as_posix()
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}', union_by_name=true)")

    # ── 1. Row counts (fast via parquet metadata) ──
    print("=" * 60)
    print("1. ROW COUNTS")
    print("=" * 60)
    counts = {}
    for name, g in globs.items():
        path = (ROOT / g).as_posix()
        n = con.execute(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_file_metadata('{path}')").fetchone()[0]
        counts[name] = int(n)
        print(f"  {name}: {n:,}")
    out["row_counts"] = counts

    # ── 2. Kalshi trades: price distribution ──
    print("\n" + "=" * 60)
    print("2. KALSHI TRADES: PRICE DISTRIBUTION (yes_price, cents 0-99)")
    print("=" * 60)
    r = con.execute("""
        SELECT
            MIN(yes_price) AS min_price,
            MAX(yes_price) AS max_price,
            AVG(yes_price) AS mean_price,
            MEDIAN(yes_price) AS median_price,
            STDDEV(yes_price) AS std_price,
            approx_quantile(yes_price, 0.01) AS p01,
            approx_quantile(yes_price, 0.05) AS p05,
            approx_quantile(yes_price, 0.25) AS p25,
            approx_quantile(yes_price, 0.75) AS p75,
            approx_quantile(yes_price, 0.95) AS p95,
            approx_quantile(yes_price, 0.99) AS p99
        FROM kalshi_trades
        WHERE yes_price >= 0 AND yes_price <= 99
    """).fetchone()
    cols = ["min", "max", "mean", "median", "std", "p01", "p05", "p25", "p75", "p95", "p99"]
    kalshi_price = {c: float(v) if v is not None else None for c, v in zip(cols, r)}
    out["kalshi_price_distribution"] = kalshi_price
    for k, v in kalshi_price.items():
        print(f"  {k}: {v}")

    # ── 3. Kalshi trades: date range ──
    print("\n" + "=" * 60)
    print("3. KALSHI TRADES: DATE RANGE")
    print("=" * 60)
    r = con.execute("SELECT MIN(created_time) AS earliest, MAX(created_time) AS latest FROM kalshi_trades WHERE created_time IS NOT NULL").fetchone()
    out["kalshi_date_range"] = {"earliest": str(r[0]), "latest": str(r[1])}
    print(f"  Earliest: {r[0]}")
    print(f"  Latest:   {r[1]}")

    # ── 4. Kalshi: distinct markets ──
    print("\n" + "=" * 60)
    print("4. KALSHI: DISTINCT MARKETS")
    print("=" * 60)
    n_markets = con.execute("SELECT COUNT(DISTINCT ticker) FROM kalshi_trades").fetchone()[0]
    out["kalshi_distinct_markets"] = int(n_markets)
    print(f"  Distinct tickers: {n_markets:,}")

    # ── 5. Kalshi: taker side distribution ──
    print("\n" + "=" * 60)
    print("5. KALSHI: TAKER SIDE DISTRIBUTION")
    print("=" * 60)
    rows = con.execute("SELECT taker_side, COUNT(*) AS n FROM kalshi_trades GROUP BY taker_side ORDER BY taker_side").fetchall()
    side_dist = {str(r[0]): int(r[1]) for r in rows}
    out["kalshi_taker_side"] = side_dist
    for k, v in side_dist.items():
        print(f"  {k}: {v:,}")

    # ── 6. Kalshi: trade size distribution ──
    print("\n" + "=" * 60)
    print("6. KALSHI: TRADE SIZE DISTRIBUTION (count column)")
    print("=" * 60)
    r = con.execute("""
        SELECT
            MIN(count) AS min_size, MAX(count) AS max_size,
            AVG(count) AS mean_size, MEDIAN(count) AS median_size,
            STDDEV(count) AS std_size,
            approx_quantile(count, 0.95) AS p95,
            approx_quantile(count, 0.99) AS p99,
            AVG(count) / MEDIAN(count) AS mean_median_ratio
        FROM kalshi_trades WHERE count > 0
    """).fetchone()
    size_cols = ["min", "max", "mean", "median", "std", "p95", "p99", "mean_median_ratio"]
    kalshi_size = {c: float(v) if v is not None else None for c, v in zip(size_cols, r)}
    out["kalshi_trade_size"] = kalshi_size
    for k, v in kalshi_size.items():
        print(f"  {k}: {v}")

    # ── 7. Polymarket: sample rows ──
    print("\n" + "=" * 60)
    print("7. POLYMARKET TRADES: SAMPLE (5 rows)")
    print("=" * 60)
    sample = con.execute("SELECT * FROM polymarket_trades LIMIT 5").fetchdf()
    print(sample.to_string())
    out["polymarket_sample_columns"] = list(sample.columns)

    # ── 8. Polymarket: date range (join to blocks) ──
    print("\n" + "=" * 60)
    print("8. POLYMARKET: DATE RANGE (via block join)")
    print("=" * 60)
    r = con.execute("""
        SELECT MIN(try_cast(b.timestamp AS TIMESTAMP)) AS earliest,
               MAX(try_cast(b.timestamp AS TIMESTAMP)) AS latest
        FROM polymarket_trades t
        INNER JOIN polymarket_blocks b ON t.block_number = b.block_number
    """).fetchone()
    out["polymarket_date_range"] = {"earliest": str(r[0]), "latest": str(r[1])}
    print(f"  Earliest: {r[0]}")
    print(f"  Latest:   {r[1]}")

    # ── 9. Polymarket: buy vs sell ratio ──
    print("\n" + "=" * 60)
    print("9. POLYMARKET: BUY vs SELL (maker_asset_id='0' vs not)")
    print("=" * 60)
    rows = con.execute("""
        SELECT
            CASE WHEN maker_asset_id = '0' THEN 'buy' ELSE 'sell' END AS side,
            COUNT(*) AS n
        FROM polymarket_trades
        GROUP BY side ORDER BY side
    """).fetchall()
    poly_side = {str(r[0]): int(r[1]) for r in rows}
    out["polymarket_buy_sell"] = poly_side
    for k, v in poly_side.items():
        print(f"  {k}: {v:,}")

    # ── 10. Polymarket: price distribution (sampled) ──
    print("\n" + "=" * 60)
    print("10. POLYMARKET: PRICE DISTRIBUTION (sampled)")
    print("=" * 60)
    r = con.execute("""
        WITH priced AS (
            SELECT
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount::DOUBLE / taker_amount::DOUBLE
                     ELSE taker_amount::DOUBLE / maker_amount::DOUBLE
                END AS price
            FROM polymarket_trades
            WHERE maker_amount > 0 AND taker_amount > 0
            USING SAMPLE 1000000
        )
        SELECT
            MIN(price) AS min_price, MAX(price) AS max_price,
            AVG(price) AS mean_price, MEDIAN(price) AS median_price,
            STDDEV(price) AS std_price,
            approx_quantile(price, 0.01) AS p01,
            approx_quantile(price, 0.05) AS p05,
            approx_quantile(price, 0.25) AS p25,
            approx_quantile(price, 0.75) AS p75,
            approx_quantile(price, 0.95) AS p95,
            approx_quantile(price, 0.99) AS p99
        FROM priced
        WHERE price >= 0 AND price <= 1
    """).fetchone()
    poly_price = {c: float(v) if v is not None else None for c, v in zip(cols, r)}
    out["polymarket_price_distribution"] = poly_price
    for k, v in poly_price.items():
        print(f"  {k}: {v}")

    # ── 11. Hourly distribution ──
    print("\n" + "=" * 60)
    print("11. HOURLY DISTRIBUTION")
    print("=" * 60)
    print("  Kalshi:")
    k_hourly = con.execute("""
        SELECT EXTRACT(HOUR FROM created_time) AS hr, COUNT(*) AS n
        FROM kalshi_trades WHERE created_time IS NOT NULL
        GROUP BY hr ORDER BY hr
    """).fetchall()
    out["kalshi_hourly"] = {int(r[0]): int(r[1]) for r in k_hourly}
    for r in k_hourly:
        print(f"    Hour {int(r[0]):2d}: {int(r[1]):>10,}")

    print("  Polymarket:")
    p_hourly = con.execute("""
        SELECT EXTRACT(HOUR FROM try_cast(b.timestamp AS TIMESTAMP)) AS hr, COUNT(*) AS n
        FROM polymarket_trades t
        INNER JOIN polymarket_blocks b ON t.block_number = b.block_number
        GROUP BY hr ORDER BY hr
    """).fetchall()
    out["polymarket_hourly"] = {int(r[0]): int(r[1]) for r in p_hourly}
    for r in p_hourly:
        print(f"    Hour {int(r[0]):2d}: {int(r[1]):>10,}")

    # ── 12. Kalshi: markets by trade count ──
    print("\n" + "=" * 60)
    print("12. KALSHI: MARKETS BY TRADE COUNT THRESHOLDS")
    print("=" * 60)
    r = con.execute("""
        WITH market_counts AS (
            SELECT ticker, COUNT(*) AS n FROM kalshi_trades GROUP BY ticker
        )
        SELECT
            COUNT(*) AS total_markets,
            SUM(CASE WHEN n >= 100 THEN 1 ELSE 0 END) AS gte_100,
            SUM(CASE WHEN n >= 1000 THEN 1 ELSE 0 END) AS gte_1000,
            SUM(CASE WHEN n >= 10000 THEN 1 ELSE 0 END) AS gte_10000,
            SUM(CASE WHEN n >= 100000 THEN 1 ELSE 0 END) AS gte_100000
        FROM market_counts
    """).fetchone()
    market_thresholds = {
        "total": int(r[0]), "gte_100": int(r[1]), "gte_1000": int(r[2]),
        "gte_10000": int(r[3]), "gte_100000": int(r[4])
    }
    out["kalshi_market_thresholds"] = market_thresholds
    for k, v in market_thresholds.items():
        print(f"  {k}: {v:,}")

    # ── 13. Extreme price fractions ──
    print("\n" + "=" * 60)
    print("13. EXTREME PRICE FRACTIONS")
    print("=" * 60)
    r = con.execute("""
        SELECT
            AVG(CASE WHEN yes_price <= 5 OR yes_price >= 95 THEN 1.0 ELSE 0.0 END) AS frac_extreme_5,
            AVG(CASE WHEN yes_price <= 10 OR yes_price >= 90 THEN 1.0 ELSE 0.0 END) AS frac_extreme_10,
            AVG(CASE WHEN yes_price > 40 AND yes_price < 60 THEN 1.0 ELSE 0.0 END) AS frac_tossup,
            AVG(CASE WHEN yes_price > 25 AND yes_price < 75 THEN 1.0 ELSE 0.0 END) AS frac_uncertain
        FROM kalshi_trades
        WHERE yes_price >= 0 AND yes_price <= 99
    """).fetchone()
    extreme = {
        "kalshi_frac_extreme_5pct": float(r[0]),
        "kalshi_frac_extreme_10pct": float(r[1]),
        "kalshi_frac_tossup": float(r[2]),
        "kalshi_frac_uncertain": float(r[3]),
    }

    r2 = con.execute("""
        WITH priced AS (
            SELECT
                CASE WHEN maker_asset_id = '0'
                     THEN maker_amount::DOUBLE / taker_amount::DOUBLE
                     ELSE taker_amount::DOUBLE / maker_amount::DOUBLE
                END AS price
            FROM polymarket_trades
            WHERE maker_amount > 0 AND taker_amount > 0
            USING SAMPLE 5000000
        )
        SELECT
            AVG(CASE WHEN price <= 0.05 OR price >= 0.95 THEN 1.0 ELSE 0.0 END) AS frac_extreme_5,
            AVG(CASE WHEN price <= 0.10 OR price >= 0.90 THEN 1.0 ELSE 0.0 END) AS frac_extreme_10,
            AVG(CASE WHEN price > 0.40 AND price < 0.60 THEN 1.0 ELSE 0.0 END) AS frac_tossup,
            AVG(CASE WHEN price > 0.25 AND price < 0.75 THEN 1.0 ELSE 0.0 END) AS frac_uncertain
        FROM priced
        WHERE price >= 0 AND price <= 1
    """).fetchone()
    extreme["polymarket_frac_extreme_5pct"] = float(r2[0])
    extreme["polymarket_frac_extreme_10pct"] = float(r2[1])
    extreme["polymarket_frac_tossup"] = float(r2[2])
    extreme["polymarket_frac_uncertain"] = float(r2[3])
    out["extreme_price_fractions"] = extreme
    for k, v in extreme.items():
        print(f"  {k}: {v:.4f}")

    # ── 14. Martingale-relevant: serial correlation of price changes ──
    print("\n" + "=" * 60)
    print("14. KALSHI: SERIAL CORRELATION OF PRICE CHANGES (top 10 liquid markets)")
    print("=" * 60)
    autocorr = con.execute("""
        WITH top_markets AS (
            SELECT ticker, COUNT(*) AS n FROM kalshi_trades GROUP BY ticker ORDER BY n DESC LIMIT 10
        ),
        ordered AS (
            SELECT t.ticker, t.yes_price, t.created_time,
                   LAG(t.yes_price) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_price
            FROM kalshi_trades t
            INNER JOIN top_markets m ON t.ticker = m.ticker
            ORDER BY t.ticker, t.created_time
        ),
        changes AS (
            SELECT ticker, (yes_price - prev_price) AS dp,
                   LAG(yes_price - prev_price) OVER (PARTITION BY ticker ORDER BY created_time) AS prev_dp
            FROM ordered
            WHERE prev_price IS NOT NULL
        )
        SELECT ticker,
               CORR(dp, prev_dp) AS lag1_autocorr,
               COUNT(*) AS n_changes
        FROM changes
        WHERE prev_dp IS NOT NULL
        GROUP BY ticker
        ORDER BY n_changes DESC
    """).fetchall()
    out["kalshi_serial_correlation"] = [
        {"ticker": r[0], "lag1_autocorr": float(r[1]) if r[1] is not None else None, "n_changes": int(r[2])}
        for r in autocorr
    ]
    for r in autocorr:
        print(f"  {r[0]}: lag1_autocorr={r[1]:.6f}, n={int(r[2]):,}")

    # ── 15. Tail risk: extreme returns distribution ──
    print("\n" + "=" * 60)
    print("15. KALSHI: PRICE JUMP DISTRIBUTION (for EVT analysis)")
    print("=" * 60)
    jumps = con.execute("""
        WITH top_markets AS (
            SELECT ticker FROM kalshi_trades GROUP BY ticker HAVING COUNT(*) >= 1000 ORDER BY COUNT(*) DESC LIMIT 100
        ),
        ordered AS (
            SELECT t.ticker, t.yes_price::DOUBLE / 100.0 AS price, t.created_time,
                   LAG(t.yes_price::DOUBLE / 100.0) OVER (PARTITION BY t.ticker ORDER BY t.created_time) AS prev_price
            FROM kalshi_trades t
            INNER JOIN top_markets m ON t.ticker = m.ticker
        ),
        returns AS (
            SELECT ticker, (price - prev_price) AS ret
            FROM ordered WHERE prev_price IS NOT NULL AND prev_price > 0
        )
        SELECT
            COUNT(*) AS n,
            AVG(ret) AS mean_ret,
            STDDEV(ret) AS std_ret,
            approx_quantile(ret, 0.01) AS p01,
            approx_quantile(ret, 0.05) AS p05,
            approx_quantile(ret, 0.95) AS p95,
            approx_quantile(ret, 0.99) AS p99,
            AVG(CASE WHEN ABS(ret) > 0.10 THEN 1.0 ELSE 0.0 END) AS frac_large_jumps,
            KURTOSIS(ret) AS kurtosis,
            SKEWNESS(ret) AS skewness
        FROM returns
    """).fetchone()
    tail_cols = ["n", "mean_ret", "std_ret", "p01", "p05", "p95", "p99", "frac_large_jumps", "kurtosis", "skewness"]
    tail_risk = {c: float(v) if v is not None else None for c, v in zip(tail_cols, jumps)}
    out["kalshi_price_jumps"] = tail_risk
    for k, v in tail_risk.items():
        print(f"  {k}: {v}")

    # ── Save results ──
    out_path = ROOT / "artifacts" / "data_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Profile complete in {elapsed:.1f}s. Saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
