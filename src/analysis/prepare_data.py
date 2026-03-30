"""
Data preparation: reads raw parquet files, cleans and standardizes them,
writes to data/processed/ so that all downstream analysis reads clean data
with no joins or transformations needed.

Outputs:
  data/processed/kalshi_trades.parquet    — standardized trades
  data/processed/polymarket_trades.parquet — standardized trades (with timestamps from block join)
  data/processed/kalshi_markets.parquet   — market snapshots with resolution info
  data/processed/trades_all.parquet       — combined standardized trades from both platforms

Data semantics applied:
  Kalshi:
    price = yes_price / 100  (cents → probability [0,1])
    quantity = count          (number of contracts, each worth $1 at resolution)
    notional = price * count  (expected value in dollars)
    side = taker_side         ('yes' or 'no')

  Polymarket:
    Trades have two directions:
      maker_asset_id='0': maker pays USDC for outcome tokens
        price = maker_amount / taker_amount
        quantity_usdc = maker_amount / 1e6
        quantity_tokens = taker_amount / 1e6
        side = 'buy'
      maker_asset_id!='0': maker offers tokens for USDC
        price = taker_amount / maker_amount
        quantity_usdc = taker_amount / 1e6
        quantity_tokens = maker_amount / 1e6
        side = 'sell'

    Timestamps: joined from polymarket_blocks on block_number.
    market_id: the non-'0' asset_id (identifies the conditional token / market).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def abs_glob(g: str) -> str:
    p = Path(g)
    return g if p.is_absolute() else (ROOT / g).as_posix()


def main() -> None:
    cfg = load_yaml(ROOT / "config" / "project.yml")
    ds_cfg = load_yaml(ROOT / cfg["datasets_config"])

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Register raw views
    for name, meta in ds_cfg["datasets"].items():
        g = abs_glob(meta["glob"])
        con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM read_parquet('{g}', union_by_name=true)")

    # ── Kalshi trades ──
    print("Preparing kalshi_trades...")
    con.execute(f"""
        COPY (
            SELECT
                trade_id,
                ticker AS market_id,
                yes_price::DOUBLE / 100.0 AS price,
                (100 - yes_price)::DOUBLE / 100.0 AS price_no,
                count::DOUBLE AS quantity,
                (yes_price::DOUBLE / 100.0) * count::DOUBLE AS notional,
                taker_side AS side,
                created_time AS ts,
                _fetched_at
            FROM raw_kalshi_trades
            WHERE created_time IS NOT NULL
                AND yes_price >= 0 AND yes_price <= 99
        ) TO '{(out / "kalshi_trades.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM '{(out / 'kalshi_trades.parquet').as_posix()}'").fetchone()[0]
    print(f"  kalshi_trades: {n:,} rows")

    # ── Polymarket trades (with block join for timestamps) ──
    print("Preparing polymarket_trades (joining blocks for timestamps)...")
    con.execute(f"""
        COPY (
            SELECT
                t.block_number,
                t.transaction_hash,
                t.log_index,
                CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                     ELSE t.maker_asset_id END AS market_id,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount::DOUBLE / t.taker_amount::DOUBLE
                     ELSE t.taker_amount::DOUBLE / t.maker_amount::DOUBLE
                END AS price,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.maker_amount::DOUBLE / 1e6
                     ELSE t.taker_amount::DOUBLE / 1e6
                END AS notional,
                CASE WHEN t.maker_asset_id = '0'
                     THEN t.taker_amount::DOUBLE / 1e6
                     ELSE t.maker_amount::DOUBLE / 1e6
                END AS quantity_tokens,
                CASE WHEN t.maker_asset_id = '0' THEN 'buy' ELSE 'sell' END AS side,
                t._contract AS contract_type,
                t.fee::DOUBLE / 1e6 AS fee_usdc,
                try_cast(b.timestamp AS TIMESTAMP) AS ts,
                t.maker,
                t.taker
            FROM raw_polymarket_trades t
            INNER JOIN raw_polymarket_blocks b ON t.block_number = b.block_number
            WHERE t.maker_amount > 0 AND t.taker_amount > 0
        ) TO '{(out / "polymarket_trades.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM '{(out / 'polymarket_trades.parquet').as_posix()}'").fetchone()[0]
    print(f"  polymarket_trades: {n:,} rows")

    # ── Kalshi markets ──
    print("Preparing kalshi_markets...")
    con.execute(f"""
        COPY (
            SELECT
                ticker AS market_id,
                event_ticker,
                title,
                status,
                result,
                yes_bid::DOUBLE / 100.0 AS yes_bid,
                yes_ask::DOUBLE / 100.0 AS yes_ask,
                no_bid::DOUBLE / 100.0 AS no_bid,
                no_ask::DOUBLE / 100.0 AS no_ask,
                last_price::DOUBLE / 100.0 AS last_price,
                (yes_ask::DOUBLE - yes_bid::DOUBLE) / 100.0 AS spread,
                volume,
                open_interest,
                created_time,
                open_time,
                close_time,
                _fetched_at
            FROM raw_kalshi_markets
        ) TO '{(out / "kalshi_markets.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM '{(out / 'kalshi_markets.parquet').as_posix()}'").fetchone()[0]
    print(f"  kalshi_markets: {n:,} rows")

    # ── Combined trades_all ──
    print("Building trades_all...")
    con.execute(f"""
        COPY (
            SELECT 'kalshi' AS platform, market_id, price, quantity AS quantity, notional, side, ts
            FROM '{(out / "kalshi_trades.parquet").as_posix()}'
            UNION ALL
            SELECT 'polymarket' AS platform, market_id, price, quantity_tokens AS quantity, notional, side, ts
            FROM '{(out / "polymarket_trades.parquet").as_posix()}'
        ) TO '{(out / "trades_all.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM '{(out / 'trades_all.parquet').as_posix()}'").fetchone()[0]
    print(f"  trades_all: {n:,} rows")

    print("\nData preparation complete. All files in data/processed/")


if __name__ == "__main__":
    main()
