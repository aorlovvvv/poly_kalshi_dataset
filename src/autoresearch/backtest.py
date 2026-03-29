"""
FIXED — DO NOT EDIT.

Evaluation harness: loads data, trains strategy, runs walk-forward backtest,
computes Sharpe ratio and other metrics. Saves results to artifacts/.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data_loader import load_data, FEATURE_COLUMNS
from .interfaces import BacktestResult, PortfolioState
from .strategy import create_strategy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

TRANSACTION_COST_BPS = 20          # 20 bps round-trip cost
ANNUALISATION_FACTOR = np.sqrt(252)


def run_backtest() -> BacktestResult:
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    train_df, test_df, feature_names = load_data()

    X_train = train_df[feature_names].values.astype(np.float64)
    y_train = train_df["target"].values.astype(np.int32)
    X_test = test_df[feature_names].values.astype(np.float64)
    y_test = test_df["target"].values.astype(np.int32)
    next_rets = test_df["next_ret"].values.astype(np.float64)

    # ── Train ─────────────────────────────────────────────────────────────
    strategy = create_strategy()
    log.info("Training strategy: %s", strategy.name())
    strategy.train(X_train, y_train, feature_names)

    # ── Predict ───────────────────────────────────────────────────────────
    tail_probs = strategy.predict_tail_probability(X_test)
    tail_probs = np.clip(tail_probs, 0.0, 1.0)

    # AUC
    try:
        auc = roc_auc_score(y_test, tail_probs)
    except Exception:
        auc = 0.5

    # ── Walk-forward P&L ──────────────────────────────────────────────────
    cost_per_unit = TRANSACTION_COST_BPS / 10_000
    state = PortfolioState()
    pnl_series: list[float] = []
    positions: list[float] = []

    for i in range(len(X_test)):
        if np.isnan(next_rets[i]):
            pnl_series.append(0.0)
            positions.append(state.position)
            continue

        desired = strategy.size_position(tail_probs[i], state)
        desired = np.clip(desired, -1.0, 1.0)
        trade_delta = abs(desired - state.position)
        cost = trade_delta * cost_per_unit

        pnl = desired * next_rets[i] - cost
        state.position = desired
        state.equity += pnl
        state.total_cost += cost
        state.trade_count += 1
        state.peak_equity = max(state.peak_equity, state.equity)
        state.drawdown = 1.0 - state.equity / state.peak_equity if state.peak_equity > 0 else 0.0

        pnl_series.append(pnl)
        positions.append(desired)

    pnl_arr = np.array(pnl_series)
    valid = ~np.isnan(pnl_arr)
    pnl_clean = pnl_arr[valid]

    # ── Metrics ───────────────────────────────────────────────────────────
    mean_pnl = np.mean(pnl_clean) if len(pnl_clean) > 0 else 0.0
    std_pnl = np.std(pnl_clean) if len(pnl_clean) > 0 else 1.0
    sharpe = (mean_pnl / std_pnl * ANNUALISATION_FACTOR) if std_pnl > 0 else 0.0

    downside = pnl_clean[pnl_clean < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1.0
    sortino = (mean_pnl / downside_std * ANNUALISATION_FACTOR) if downside_std > 0 else 0.0

    total_ret = state.equity - 1.0
    max_dd = 0.0
    running_equity = 1.0
    peak = 1.0
    for p in pnl_clean:
        running_equity += p
        peak = max(peak, running_equity)
        dd = 1.0 - running_equity / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    wins = np.sum(pnl_clean > 0)
    win_rate = wins / len(pnl_clean) if len(pnl_clean) > 0 else 0.0

    # Aggregate to daily returns for reporting
    test_ts = pd.to_datetime(test_df["ts"], utc=True)
    daily_df = pd.DataFrame({"pnl": pnl_series, "date": test_ts.dt.date.values})
    daily_rets = daily_df.groupby("date")["pnl"].sum().tolist()

    result = BacktestResult(
        strategy_name=strategy.name(),
        sharpe_ratio=round(float(sharpe), 4),
        sortino_ratio=round(float(sortino), 4),
        total_return=round(float(total_ret), 6),
        max_drawdown=round(float(max_dd), 6),
        auc_roc=round(float(auc), 4),
        n_trades=len(pnl_clean),
        n_markets=test_df["market_id"].nunique(),
        win_rate=round(float(win_rate), 4),
        avg_pnl_per_trade=round(float(mean_pnl), 8),
        feature_importances=strategy.get_feature_importances(),
        daily_returns=daily_rets,
        metadata={
            "elapsed_seconds": round(time.time() - t0, 1),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "mean_position": float(np.mean(np.abs(positions))),
        },
    )
    return result


def main():
    result = run_backtest()

    # Print summary
    print("\n" + "=" * 70)
    print(f"STRATEGY: {result.strategy_name}")
    print("=" * 70)
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:+.4f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:+.4f}")
    print(f"  Total Return:    {result.total_return:+.6f}")
    print(f"  Max Drawdown:    {result.max_drawdown:.6f}")
    print(f"  AUC-ROC:         {result.auc_roc:.4f}")
    print(f"  Win Rate:        {result.win_rate:.4f}")
    print(f"  N Trades:        {result.n_trades:,}")
    print(f"  Avg PnL/Trade:   {result.avg_pnl_per_trade:+.8f}")
    print(f"  Total Cost:      {result.metadata.get('mean_position', 0):.4f} avg|pos|")
    print(f"  Elapsed:         {result.metadata['elapsed_seconds']:.1f}s")
    print("=" * 70)
    print("Top Features:")
    for name, imp in sorted(result.feature_importances.items(),
                            key=lambda x: -x[1])[:5]:
        print(f"  {name:25s} {imp:.4f}")
    print("=" * 70)

    # Save
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "strategy_results.json"
    out_data = {
        "strategy_name": result.strategy_name,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "auc_roc": result.auc_roc,
        "n_trades": result.n_trades,
        "n_markets": result.n_markets,
        "win_rate": result.win_rate,
        "avg_pnl_per_trade": result.avg_pnl_per_trade,
        "feature_importances": result.feature_importances,
        "metadata": result.metadata,
    }
    out_path.write_text(json.dumps(out_data, indent=2, default=str))
    log.info("Results saved to %s", out_path)

    return result


if __name__ == "__main__":
    main()
