"""
Analysis engine: reads model_spec.yml + datasets.yml, builds standardized trade views,
runs a comprehensive set of analyses, and outputs tables/figures/macros.

Data semantics (verified):
  Kalshi trades: yes_price is cents (0-99), yes_price+no_price=100 always.
      count = # contracts. created_time = timestamp. ticker = market ID.
      taker_side = 'yes' or 'no'.

  Polymarket trades: timestamp column is ALL NULL.
      block_number must be joined to polymarket_blocks for timestamps.
      Two trade directions:
        maker_asset_id='0': maker pays USDC → price = maker_amount/taker_amount
        maker_asset_id!='0' (taker_asset_id='0'): taker pays USDC → price = taker_amount/maker_amount
      Both produce prices in [0,1]. taker_asset_id or maker_asset_id (the non-'0' one) is the token/market ID.
      Amounts are in USDC atomic units (6 decimals).

  Polymarket blocks: block_number → timestamp (ISO string).
  Kalshi markets: snapshot data with bid/ask/volume/result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def abs_glob(g: str) -> str:
    p = Path(g)
    return g if p.is_absolute() else (ROOT / g).as_posix()


def write_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(
        index=False, escape=True, caption=caption, label=label,
        longtable=False, float_format="%.4f",
    )
    path.write_text(tex + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_yaml(ROOT / "config" / "project.yml")
    ds_cfg = load_yaml(ROOT / cfg["datasets_config"])
    spec = load_yaml(ROOT / "config" / "model_spec.yml")

    max_rows = cfg.get("analysis", {}).get("max_rows_per_dataset")
    limit_sql = f"LIMIT {int(max_rows)}" if max_rows else ""

    con = duckdb.connect()

    for name, meta in ds_cfg["datasets"].items():
        g = abs_glob(meta["glob"])
        con.execute(f"CREATE OR REPLACE VIEW ds_{name} AS SELECT * FROM read_parquet('{g}', union_by_name=true)")

    # ── Build trades_all with correct per-platform semantics ──

    kalshi_sql = f"""
        SELECT
            'kalshi' AS platform,
            created_time AS ts,
            ticker AS market_id,
            yes_price::DOUBLE / 100.0 AS price,
            count::DOUBLE AS quantity,
            (yes_price::DOUBLE / 100.0) * count::DOUBLE AS notional,
            taker_side
        FROM ds_kalshi_trades
        WHERE created_time IS NOT NULL
        {limit_sql}
    """

    # Polymarket: handle both trade directions
    polymarket_sql = f"""
        SELECT
            'polymarket' AS platform,
            try_cast(b.timestamp AS TIMESTAMP) AS ts,
            CASE WHEN t.maker_asset_id = '0' THEN t.taker_asset_id
                 ELSE t.maker_asset_id END AS market_id,
            CASE WHEN t.maker_asset_id = '0'
                 THEN t.maker_amount::DOUBLE / t.taker_amount::DOUBLE
                 ELSE t.taker_amount::DOUBLE / t.maker_amount::DOUBLE
            END AS price,
            CASE WHEN t.maker_asset_id = '0'
                 THEN t.taker_amount::DOUBLE / 1e6
                 ELSE t.maker_amount::DOUBLE / 1e6
            END AS quantity,
            CASE WHEN t.maker_asset_id = '0'
                 THEN t.maker_amount::DOUBLE / 1e6
                 ELSE t.taker_amount::DOUBLE / 1e6
            END AS notional,
            CASE WHEN t.maker_asset_id = '0' THEN 'buy' ELSE 'sell' END AS taker_side
        FROM ds_polymarket_trades t
        INNER JOIN ds_polymarket_blocks b ON t.block_number = b.block_number
        WHERE t.maker_amount > 0 AND t.taker_amount > 0
        {limit_sql}
    """

    con.execute(f"CREATE OR REPLACE TEMP VIEW trades_all AS {kalshi_sql} UNION ALL {polymarket_sql}")

    tasks = spec.get("analysis", {}).get("tasks", [])

    out_tables = ROOT / "paper" / "tables"
    out_figs = ROOT / "paper" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Collect all computed findings as text for the discover node
    findings: list[str] = []

    # ── trade_summary ──
    if "trade_summary" in tasks:
        df = con.execute("""
            SELECT
                platform,
                COUNT(*) AS n_trades,
                COUNT(DISTINCT market_id) AS n_markets,
                MIN(ts) AS earliest,
                MAX(ts) AS latest,
                AVG(price) AS avg_price,
                MEDIAN(price) AS median_price,
                STDDEV(price) AS std_price,
                AVG(quantity) AS avg_qty,
                SUM(notional) AS total_notional
            FROM trades_all
            WHERE ts IS NOT NULL AND price IS NOT NULL
                AND price >= 0 AND price <= 1
            GROUP BY platform
            ORDER BY platform
        """).fetchdf()
        write_table(df, out_tables / "trade_summary.tex",
                    caption="Trade activity summary by platform",
                    label="tab:trade_summary")
        findings.append(f"TRADE SUMMARY:\n{df.to_string(index=False)}")
        print(f"  trade_summary done")

    # ── price_distribution ──
    if "price_distribution" in tasks:
        df = con.execute("""
            SELECT
                platform,
                approx_quantile(price, 0.01) AS p01,
                approx_quantile(price, 0.05) AS p05,
                approx_quantile(price, 0.25) AS p25,
                approx_quantile(price, 0.50) AS p50,
                approx_quantile(price, 0.75) AS p75,
                approx_quantile(price, 0.95) AS p95,
                approx_quantile(price, 0.99) AS p99,
                STDDEV(price) AS std
            FROM trades_all
            WHERE price IS NOT NULL AND price >= 0 AND price <= 1
            GROUP BY platform
            ORDER BY platform
        """).fetchdf()
        write_table(df, out_tables / "price_distribution.tex",
                    caption="Price distribution quantiles by platform (probability scale)",
                    label="tab:price_distribution")
        findings.append(f"PRICE DISTRIBUTION:\n{df.to_string(index=False)}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, platform in zip(axes, ["kalshi", "polymarket"]):
            prices = con.execute(f"""
                SELECT price FROM trades_all
                WHERE platform = '{platform}' AND price >= 0 AND price <= 1
                USING SAMPLE 500000
            """).fetchdf()["price"].values
            ax.hist(prices, bins=100, alpha=0.8, edgecolor="none", density=True)
            ax.set_title(platform.capitalize())
            ax.set_xlabel("Price (implied probability)")
            ax.set_ylabel("Density")
        plt.suptitle("Price Distribution by Platform")
        plt.tight_layout()
        plt.savefig(out_figs / "price_distribution.png", dpi=200)
        plt.close()
        print(f"  price_distribution done")

    # ── volume_by_day ──
    if "volume_by_day" in tasks:
        df = con.execute("""
            SELECT
                platform,
                date_trunc('day', ts) AS day,
                COUNT(*) AS n_trades,
                SUM(notional) AS total_notional
            FROM trades_all
            WHERE ts IS NOT NULL
            GROUP BY platform, day
            ORDER BY day, platform
        """).fetchdf()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for platform, g in df.groupby("platform"):
            ax1.plot(g["day"], g["n_trades"], label=platform, alpha=0.7, linewidth=0.5)
            ax2.plot(g["day"], g["total_notional"], label=platform, alpha=0.7, linewidth=0.5)
        ax1.set_ylabel("Daily trade count")
        ax1.legend()
        ax1.set_title("Daily Trading Activity")
        ax2.set_ylabel("Daily notional volume")
        ax2.set_xlabel("Date")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "volume_by_day.png", dpi=200)
        plt.close()

        # Compute growth stats for findings
        monthly = con.execute("""
            SELECT platform, date_trunc('month', ts) AS month, COUNT(*) AS n, SUM(notional) AS vol
            FROM trades_all WHERE ts IS NOT NULL
            GROUP BY platform, month ORDER BY platform, month
        """).fetchdf()
        findings.append(f"MONTHLY VOLUME (last 6 months per platform):\n{monthly.groupby('platform').tail(6).to_string(index=False)}")
        print(f"  volume_by_day done")

    # ── trade_size_distribution ──
    if "trade_size_distribution" in tasks:
        df = con.execute("""
            SELECT
                platform,
                approx_quantile(quantity, 0.25) AS q25,
                approx_quantile(quantity, 0.50) AS q50,
                approx_quantile(quantity, 0.75) AS q75,
                approx_quantile(quantity, 0.90) AS q90,
                approx_quantile(quantity, 0.99) AS q99,
                AVG(quantity) AS mean,
                STDDEV(quantity) AS std,
                AVG(quantity) / MEDIAN(quantity) AS mean_median_ratio
            FROM trades_all
            WHERE quantity > 0
            GROUP BY platform ORDER BY platform
        """).fetchdf()
        write_table(df, out_tables / "trade_size_distribution.tex",
                    caption="Trade size distribution by platform",
                    label="tab:trade_size")
        findings.append(f"TRADE SIZE DISTRIBUTION:\n{df.to_string(index=False)}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, platform in zip(axes, ["kalshi", "polymarket"]):
            qtys = con.execute(f"""
                SELECT quantity FROM trades_all
                WHERE platform = '{platform}' AND quantity > 0
                USING SAMPLE 500000
            """).fetchdf()["quantity"].values
            cap = np.percentile(qtys, 99)
            ax.hist(qtys[qtys < cap], bins=80, alpha=0.8, edgecolor="none", log=True)
            ax.set_title(f"{platform.capitalize()}")
            ax.set_xlabel("Trade size")
            ax.set_ylabel("Count (log)")
        plt.suptitle("Trade Size Distribution (clipped at p99)")
        plt.tight_layout()
        plt.savefig(out_figs / "trade_size_distribution.png", dpi=200)
        plt.close()
        print(f"  trade_size_distribution done")

    # ── hourly_pattern ──
    if "hourly_pattern" in tasks:
        df = con.execute("""
            SELECT
                platform,
                EXTRACT(HOUR FROM ts) AS hour_utc,
                COUNT(*) AS n_trades,
                SUM(notional) AS total_notional
            FROM trades_all
            WHERE ts IS NOT NULL
            GROUP BY platform, hour_utc
            ORDER BY platform, hour_utc
        """).fetchdf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for platform, g in df.groupby("platform"):
            ax1.plot(g["hour_utc"], g["n_trades"], marker="o", label=platform, alpha=0.8)
            ax2.plot(g["hour_utc"], g["total_notional"], marker="o", label=platform, alpha=0.8)
        ax1.set_xlabel("Hour (UTC)")
        ax1.set_ylabel("Total trade count")
        ax1.set_title("Trade Count by Hour")
        ax1.set_xticks(range(0, 24, 2))
        ax1.legend()
        ax2.set_xlabel("Hour (UTC)")
        ax2.set_ylabel("Total notional")
        ax2.set_title("Notional Volume by Hour")
        ax2.set_xticks(range(0, 24, 2))
        ax2.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "hourly_pattern.png", dpi=200)
        plt.close()

        peak_hours = df.loc[df.groupby("platform")["n_trades"].idxmax()][["platform", "hour_utc", "n_trades"]]
        findings.append(f"HOURLY PATTERN (peak hours):\n{peak_hours.to_string(index=False)}")
        print(f"  hourly_pattern done")

    # ── price_concentration ──
    if "price_concentration" in tasks:
        df = con.execute("""
            SELECT
                platform,
                COUNT(*) AS total,
                AVG(CASE WHEN price <= 0.05 OR price >= 0.95 THEN 1.0 ELSE 0.0 END) AS frac_extreme,
                AVG(CASE WHEN price <= 0.10 OR price >= 0.90 THEN 1.0 ELSE 0.0 END) AS frac_confident,
                AVG(CASE WHEN price > 0.40 AND price < 0.60 THEN 1.0 ELSE 0.0 END) AS frac_tossup,
                AVG(CASE WHEN price > 0.25 AND price < 0.75 THEN 1.0 ELSE 0.0 END) AS frac_uncertain
            FROM trades_all
            WHERE price >= 0 AND price <= 1
            GROUP BY platform ORDER BY platform
        """).fetchdf()
        write_table(df, out_tables / "price_concentration.tex",
                    caption="Price concentration: fraction of trades in probability ranges",
                    label="tab:price_concentration")
        findings.append(f"PRICE CONCENTRATION:\n{df.to_string(index=False)}")
        print(f"  price_concentration done")

    # ── taker_side_analysis (Kalshi-specific: buy/sell imbalance) ──
    if "taker_side_analysis" in tasks:
        df = con.execute("""
            SELECT
                platform,
                taker_side,
                COUNT(*) AS n_trades,
                AVG(price) AS avg_price,
                MEDIAN(price) AS median_price,
                SUM(notional) AS total_notional
            FROM trades_all
            WHERE taker_side IS NOT NULL
            GROUP BY platform, taker_side
            ORDER BY platform, taker_side
        """).fetchdf()
        write_table(df, out_tables / "taker_side_analysis.tex",
                    caption="Trade direction analysis by platform",
                    label="tab:taker_side")
        findings.append(f"TAKER SIDE ANALYSIS:\n{df.to_string(index=False)}")
        print(f"  taker_side_analysis done")

    # ── Write results.tex macros ──
    results_path = ROOT / "paper" / "results.tex"
    existing = results_path.read_text(encoding="utf-8") if results_path.exists() else "% Auto-generated.\n"

    # Strip any previous analysis macros (keep profiler macros)
    lines_keep = [l for l in existing.strip().split("\n") if "% Analysis" not in l and "\\NStandardized" not in l
                  and "\\NKalshi" not in l and "\\NPolymarket" not in l]
    lines = lines_keep + ["% Analysis macros"]

    stats = con.execute("""
        SELECT platform, COUNT(*) AS n, COUNT(DISTINCT market_id) AS nm
        FROM trades_all WHERE price >= 0 AND price <= 1
        GROUP BY platform ORDER BY platform
    """).fetchall()
    total = sum(r[1] for r in stats)
    lines.append(f"\\newcommand{{\\NStandardizedTrades}}{{{total}}}")
    for row in stats:
        key = row[0].capitalize()
        lines.append(f"\\newcommand{{\\N{key}Trades}}{{{row[1]}}}")
        lines.append(f"\\newcommand{{\\N{key}Markets}}{{{row[2]}}}")

    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Write findings digest for the discover node ──
    findings_path = ROOT / "artifacts" / "findings.md"
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    findings_path.write_text(
        "# Analysis Findings\n\nAuto-generated by `make analysis`.\n\n"
        + "\n\n---\n\n".join(findings) + "\n",
        encoding="utf-8",
    )

    print(f"\nDone. {len(tasks)} tasks, findings written to artifacts/findings.md")


if __name__ == "__main__":
    main()
