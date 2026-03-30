"""
Phase 2: Advanced Statistical Testing for Prediction Markets Research Paper

Implements the following tests on ~36GB of prediction market trade data:
  A. Variance Ratio Tests (Lo-MacKinlay 1988) — test martingale hypothesis
  B. Roll's Implied Spread — estimate bid-ask spreads from negative autocorrelation
  C. GPD Tail Fitting (Peaks Over Threshold) — characterize extreme price movements
  D. BDS Test — test for nonlinear dependence in residuals
  E. XGBoost Tail Event Prediction — predict extreme price moves (machine learning)

Data sources:
  - Raw: data/raw/kalshi/trades/*.parquet, data/raw/polymarket/trades/*.parquet
  - Processed: data/processed/kalshi_trades.parquet, polymarket_trades.parquet
  - Output: artifacts/phase2_results.json

Uses DuckDB for all heavy lifting (joins, aggregations, filtering).
Only pulls small DataFrames into Python for statistical tests.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.stattools import bds
    HAS_STATSMODELS_BDS = True
except ImportError:
    HAS_STATSMODELS_BDS = False

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress scientific notation in output and numpy warnings
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class VarianceRatioTest:
    """
    Lo-MacKinlay (1988) heteroskedasticity-robust variance ratio test.

    Tests the null hypothesis that price changes follow a martingale (VR(k) = 1).
    Returns variance ratio, test statistic, and p-value.
    """

    @staticmethod
    def compute_vr_m2(returns: np.ndarray, k: int) -> tuple[float, float, float]:
        """
        Compute variance ratio VR(k) and heteroskedasticity-robust test statistic M2.

        Args:
            returns: 1D array of price changes
            k: lag for k-period returns

        Returns:
            (vr, m2_stat, p_value)
        """
        n = len(returns)
        if n < k + 1:
            return np.nan, np.nan, np.nan

        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)
        if var_1 == 0:
            return np.nan, np.nan, np.nan

        # k-period returns (non-overlapping for variance ratio)
        ret_k = np.array([np.sum(returns[i : i + k]) for i in range(0, n - k + 1)])
        var_k = np.var(ret_k, ddof=1)

        # Variance ratio
        vr = var_k / (k * var_1)

        # Heteroskedasticity-robust variance (M2 test statistic)
        # delta_j = 2(k-j)/k * (sum of squared returns products / sum of squared returns)^2
        squared_returns = returns**2
        sum_sq = np.sum(squared_returns)

        if sum_sq == 0:
            return vr, np.nan, np.nan

        numerator = 0.0
        for j in range(1, k):
            weight = 2.0 * (k - j) / k
            # Autocovariance of squared returns
            delta_j = np.sum(
                squared_returns[:-j] * squared_returns[j:]
            ) / (sum_sq**2)
            numerator += weight**2 * delta_j

        if numerator <= 0:
            return vr, np.nan, np.nan

        # M2 = sqrt(n) * (VR - 1) / sqrt(numerator)
        m2 = (vr - 1) * np.sqrt(n) / np.sqrt(numerator)

        # Two-tailed p-value from standard normal
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(m2)))

        return vr, m2, p_value


class RollsSpread:
    """
    Roll (1984) implied spread estimator.

    Estimates bid-ask spread from negative serial autocorrelation of price changes.
    s = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
    """

    @staticmethod
    def compute_spread(prices: np.ndarray) -> tuple[float, float]:
        """
        Compute Roll's implied spread.

        Args:
            prices: 1D array of prices (not necessarily ordered)

        Returns:
            (implied_spread, autocovariance)
        """
        if len(prices) < 2:
            return np.nan, np.nan

        # Price changes
        deltas = np.diff(prices)

        # Autocovariance at lag 1
        mean_delta = np.mean(deltas)
        autocovar = np.mean(
            (deltas[:-1] - mean_delta) * (deltas[1:] - mean_delta)
        )

        # Implied spread: if autocovar >= 0 (shouldn't happen in theory), spread = 0
        if autocovar >= 0:
            spread = 0.0
        else:
            spread = 2.0 * np.sqrt(-autocovar)

        return spread, autocovar


class GPDTailFitting:
    """
    Peaks-over-Threshold (POT) method for tail risk estimation.

    Fits Generalized Pareto Distribution to exceedances above threshold.
    """

    @staticmethod
    def fit_gpd_tail(
        data: np.ndarray, threshold_percentile: float = 95
    ) -> dict[str, Any]:
        """
        Fit GPD to upper tail exceedances.

        Args:
            data: 1D array (can be price changes or their absolute values)
            threshold_percentile: percentile for threshold (default 95)

        Returns:
            dict with threshold, shape_xi, scale_sigma, n_exceedances, ks_stat, ks_pvalue
        """
        # Compute threshold
        threshold = np.percentile(data, threshold_percentile)
        exceedances = data[data > threshold] - threshold

        if len(exceedances) < 5:
            return {
                "threshold": threshold,
                "shape_xi": np.nan,
                "scale_sigma": np.nan,
                "n_exceedances": len(exceedances),
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
            }

        # Fit GPD: scipy.stats.genpareto.fit returns (shape, loc, scale)
        # loc should be 0 for exceedances, shape is xi, scale is sigma
        try:
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        except Exception:
            return {
                "threshold": threshold,
                "shape_xi": np.nan,
                "scale_sigma": np.nan,
                "n_exceedances": len(exceedances),
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
            }

        # KS test: compare empirical CDF to fitted GPD CDF
        empirical_cdf = np.arange(1, len(exceedances) + 1) / len(exceedances)
        fitted_cdf = stats.genpareto.cdf(
            np.sort(exceedances), shape, loc=0, scale=scale
        )
        ks_stat = np.max(np.abs(empirical_cdf - fitted_cdf))
        ks_pvalue = stats.kstest(exceedances, lambda x: stats.genpareto.cdf(x, shape, loc=0, scale=scale))[
            1
        ]

        return {
            "threshold": threshold,
            "shape_xi": float(shape),
            "scale_sigma": float(scale),
            "n_exceedances": int(len(exceedances)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }


class BDSTest:
    """
    BDS (Brock, Dechert, Scheinkman) test for nonlinear dependence.

    Tests if residuals are independently and identically distributed (null)
    vs. showing nonlinear structure (alternative).
    """

    @staticmethod
    def bds_test_manual(
        residuals: np.ndarray, m: int = 2, eps: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Manual implementation of BDS test (fallback if statsmodels not available).

        Implements the standard BDS statistic:
            W_m = sqrt(n) * (C_m(eps) - C_1(eps)^m) / sigma_m
        where C_m is the correlation integral at embedding dimension m,
        and sigma_m is the standard deviation under the null of i.i.d.

        Reference: Brock, Dechert, Scheinkman & LeBaron (1996).

        For large n, uses vectorized pairwise distance computation.
        For very large n (>5000), subsamples to keep runtime reasonable.

        Args:
            residuals: 1D array of residuals
            m: embedding dimension
            eps: distance threshold (if None, use 0.5 * std)

        Returns:
            (bds_statistic, p_value)
        """
        n = len(residuals)
        if n < 2 * m:
            return np.nan, np.nan
        if eps is None:
            eps = 0.5 * np.std(residuals)
        if eps <= 0:
            return np.nan, np.nan

        # Subsample if too large (O(n^2) pairwise distances)
        max_n = 5000
        if n > max_n:
            idx = np.random.choice(n, max_n, replace=False)
            idx.sort()
            residuals = residuals[idx]
            n = max_n

        # Correlation integral: fraction of pairs within eps
        def correlation_integral(dim):
            n_emb = n - dim + 1
            if n_emb < 2:
                return 0.0
            count = 0
            total = n_emb * (n_emb - 1) // 2
            for i in range(n_emb):
                for j in range(i + 1, n_emb):
                    match = True
                    for k in range(dim):
                        if abs(residuals[i + k] - residuals[j + k]) >= eps:
                            match = False
                            break
                    if match:
                        count += 1
            return count / total if total > 0 else 0.0

        c_m = correlation_integral(m)
        c_1 = correlation_integral(1)

        if c_1 <= 0 or c_1 >= 1:
            return np.nan, np.nan

        # BDS statistic: W = sqrt(n) * (C_m - C_1^m) / sigma
        # Under H0 (iid), sigma^2 is estimated from the data.
        # Simplified variance estimate (Brock et al. 1996, Theorem 1):
        # sigma^2 ≈ 4 * K^m * (sum of products involving C_1 and K)
        # where K = proportion of triples within eps.
        # For the fallback, use the simplified asymptotic formula:
        # sigma ≈ C_1^m * sqrt(K_factor) where K_factor accounts for
        # dependence in the indicator functions.
        #
        # Conservative approximation: sigma ≈ sqrt(m) * C_1^(m-1) * sqrt(C_1*(1-C_1)) * 2/sqrt(n)
        # This gives a rough z-score. For precise values, use statsmodels.
        bds_diff = c_m - c_1**m
        # Approximate standard error (conservative)
        se = max(1e-10, np.sqrt(4.0 * (c_1 * (1.0 - c_1))**m / n))
        bds_stat = bds_diff / se

        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(bds_stat)))

        return float(bds_stat), float(p_value)

    @staticmethod
    def compute_bds(
        residuals: np.ndarray,
        m_values: list[int] = None,
        epsilon_multipliers: list[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Compute BDS test across multiple embedding dimensions and epsilon values.

        Args:
            residuals: 1D array of residuals (preferably AR(1) residuals)
            m_values: embedding dimensions to test
            epsilon_multipliers: epsilon as multiple of std(residuals)

        Returns:
            list of dicts with m, epsilon_mult, bds_stat, p_value
        """
        if m_values is None:
            m_values = [2, 3, 4, 5]
        if epsilon_multipliers is None:
            epsilon_multipliers = [0.5, 1.0, 1.5]

        results = []
        std_res = np.std(residuals)

        for m in m_values:
            for eps_mult in epsilon_multipliers:
                eps = eps_mult * std_res

                # Try statsmodels first
                if HAS_STATSMODELS_BDS:
                    try:
                        bds_stat, p_value = bds(residuals, m, eps)
                    except Exception:
                        bds_stat, p_value = BDSTest.bds_test_manual(
                            residuals, m, eps
                        )
                else:
                    bds_stat, p_value = BDSTest.bds_test_manual(
                        residuals, m, eps
                    )

                results.append(
                    {
                        "m": m,
                        "epsilon_mult": eps_mult,
                        "bds_statistic": float(bds_stat) if not np.isnan(bds_stat) else None,
                        "p_value": float(p_value) if not np.isnan(p_value) else None,
                    }
                )

        return results


class TailEventPrediction:
    """
    Machine learning model to predict extreme price movements.

    Features: hour_of_day, price_level (bucketed), trade_size_log, taker_side_binary,
    time_since_last_trade_log
    Target: 1 if |price_change| > 2*std, 0 otherwise
    """

    @staticmethod
    def create_features(
        df: pd.DataFrame, threshold_std_mult: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix and target for tail event prediction.

        Args:
            df: DataFrame with columns [ts, price, quantity, taker_side]
            threshold_std_mult: threshold = threshold_std_mult * std(price_change)

        Returns:
            (X, y) where X is feature matrix (n, 5) and y is binary target
        """
        df = df.sort_values("ts").reset_index(drop=True)

        # Compute price changes
        prices = df["price"].values
        price_changes = np.diff(np.concatenate(([prices[0]], prices)))
        price_changes_abs = np.abs(price_changes)

        # Target: extreme price move
        threshold = threshold_std_mult * np.std(price_changes_abs)
        y = (price_changes_abs > threshold).astype(int)

        # Features
        n = len(df)
        X = np.zeros((n, 5))

        # Feature 1: hour of day
        ts = pd.to_datetime(df["ts"])
        X[:, 0] = ts.hour.values

        # Feature 2: price level (bucketed into 10 buckets)
        price_buckets = pd.qcut(df["price"], q=10, labels=False, duplicates="drop")
        X[:, 1] = price_buckets.values

        # Feature 3: log(quantity + 1)
        X[:, 2] = np.log1p(df["quantity"].values)

        # Feature 4: taker_side binary (yes=1, no=0 or buy=1, sell=0)
        side_values = df["taker_side"].values if "taker_side" in df.columns else df["side"].values
        X[:, 3] = (side_values == "yes").astype(int) | (side_values == "buy").astype(int)

        # Feature 5: log(time since last trade in seconds + 1)
        time_diffs = np.diff(np.concatenate(([0.0], ts.values.astype("int64") / 1e9)))
        X[:, 4] = np.log1p(np.maximum(time_diffs, 0))

        return X, y

    @staticmethod
    def train_and_evaluate(
        market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Train classifier on a single market's trades.

        Args:
            market_data: dict with keys ['ticker', 'df'] where df has [ts, price, quantity, taker_side/side]

        Returns:
            dict with auc, precision, recall, top_5_features, n_samples
        """
        ticker = market_data["ticker"]
        df = market_data["df"]

        if len(df) < 100:
            return {
                "ticker": ticker,
                "auc": None,
                "precision": None,
                "recall": None,
                "top_5_features": None,
                "n_samples": len(df),
                "error": "too_few_samples",
            }

        try:
            X, y = TailEventPrediction.create_features(df)

            # Need at least one positive example
            if np.sum(y) == 0 or np.sum(y) == len(y):
                return {
                    "ticker": ticker,
                    "auc": None,
                    "precision": None,
                    "recall": None,
                    "top_5_features": None,
                    "n_samples": len(df),
                    "error": "no_positive_examples",
                }

            # Temporal split: 80% train, 20% test
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train classifier (try XGBoost, fallback to GradientBoosting)
            if HAS_XGBOOST:
                clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                )
            else:
                clf = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )

            clf.fit(X_train_scaled, y_train)

            # Predict on test set
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
            y_pred = clf.predict(X_test_scaled)

            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            # Use average precision for single precision value
            avg_precision = auc(recall, precision)

            # Feature importance
            feature_names = [
                "hour_of_day",
                "price_level",
                "quantity_log",
                "taker_side",
                "time_since_last_log",
            ]
            importances = clf.feature_importances_
            top_features = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )[:5]

            return {
                "ticker": ticker,
                "auc": float(auc_score),
                "precision": float(avg_precision),
                "recall": float(np.mean(recall[:-1])),  # Approximate
                "top_5_features": [{"name": f, "importance": float(i)} for f, i in top_features],
                "n_samples": len(df),
                "error": None,
            }

        except Exception as e:
            return {
                "ticker": ticker,
                "auc": None,
                "precision": None,
                "recall": None,
                "top_5_features": None,
                "n_samples": len(df),
                "error": str(e),
            }


class Phase2StatisticalTests:
    """
    Orchestrator for all Phase 2 statistical tests.
    """

    def __init__(self):
        self.con = duckdb.connect()
        self.results = {
            "variance_ratio_tests": [],
            "rolls_spread_tests": [],
            "gpd_tail_fitting": {},
            "bds_tests": [],
            "tail_event_prediction": [],
            "metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_top_markets": 20,
                "sampling_note": "Markets sampled down to 500K trades max if larger",
            },
        }

    def run_all_tests(self) -> None:
        """Execute all statistical tests."""
        logger.info("Starting Phase 2 Statistical Tests")

        # Register data views
        self._register_views()

        # Get top 20 most liquid markets for each platform
        logger.info("Identifying top 20 most liquid markets per platform...")
        top_kalshi = self._get_top_markets("kalshi", 20)
        top_polymarket = self._get_top_markets("polymarket", 20)

        logger.info(f"Found {len(top_kalshi)} Kalshi markets, {len(top_polymarket)} Polymarket markets")

        # Test A: Variance Ratio
        logger.info("Running Variance Ratio Tests (Lo-MacKinlay)...")
        for ticker in top_kalshi:
            self._run_variance_ratio_test_market(ticker)

        # Test B: Roll's Spread
        logger.info("Running Roll's Implied Spread Tests...")
        for ticker in top_kalshi:
            self._run_rolls_spread_market(ticker)

        # Test C: GPD Tail Fitting
        logger.info("Running GPD Tail Fitting...")
        self._run_gpd_tail_fitting("kalshi", top_kalshi[:10])
        self._run_gpd_tail_fitting("polymarket", top_polymarket[:10])

        # Test D: BDS Test
        logger.info("Running BDS Tests...")
        for ticker in top_kalshi[:5]:
            self._run_bds_test_market(ticker)

        # Test E: XGBoost Tail Prediction
        logger.info("Running Tail Event Prediction...")
        for ticker in top_kalshi[:10]:
            self._run_tail_event_prediction_market(ticker)

        # Save results
        self._save_results()
        logger.info(f"All tests complete. Results saved to {ARTIFACTS / 'phase2_results.json'}")

    def _register_views(self) -> None:
        """Register parquet files as DuckDB views."""
        # Check for processed data first
        if (DATA_PROCESSED / "kalshi_trades.parquet").exists():
            logger.info("Using processed data from data/processed/")
            self.con.execute(f"""
                CREATE OR REPLACE VIEW kalshi_trades AS
                SELECT * FROM read_parquet('{(DATA_PROCESSED / "kalshi_trades.parquet").as_posix()}')
            """)
            self.con.execute(f"""
                CREATE OR REPLACE VIEW polymarket_trades AS
                SELECT * FROM read_parquet('{(DATA_PROCESSED / "polymarket_trades.parquet").as_posix()}')
            """)
        else:
            # Fall back to raw data
            logger.info("Processed data not found. Using raw data from data/raw/")
            self.con.execute(f"""
                CREATE OR REPLACE VIEW kalshi_trades AS
                SELECT
                    trade_id, ticker AS market_id,
                    yes_price::DOUBLE / 100.0 AS price,
                    count::DOUBLE AS quantity,
                    taker_side,
                    created_time AS ts
                FROM read_parquet('{(DATA_RAW / "kalshi" / "trades" / "*.parquet").as_posix()}',
                                 union_by_name=true)
                WHERE created_time IS NOT NULL AND yes_price >= 0 AND yes_price <= 99
            """)
            self.con.execute(f"""
                CREATE OR REPLACE VIEW polymarket_trades AS
                SELECT
                    block_number,
                    CASE WHEN maker_asset_id = '0' THEN taker_asset_id ELSE maker_asset_id END AS market_id,
                    CASE WHEN maker_asset_id = '0'
                         THEN maker_amount::DOUBLE / taker_amount::DOUBLE
                         ELSE taker_amount::DOUBLE / maker_amount::DOUBLE
                    END AS price,
                    CASE WHEN maker_asset_id = '0'
                         THEN taker_amount::DOUBLE / 1e6
                         ELSE maker_amount::DOUBLE / 1e6
                    END AS quantity,
                    CASE WHEN maker_asset_id = '0' THEN 'buy' ELSE 'sell' END AS side,
                    try_cast(b.timestamp AS TIMESTAMP) AS ts
                FROM read_parquet('{(DATA_RAW / "polymarket" / "trades" / "*.parquet").as_posix()}',
                                 union_by_name=true) t
                INNER JOIN read_parquet('{(DATA_RAW / "polymarket" / "blocks" / "*.parquet").as_posix()}',
                                       union_by_name=true) b
                ON t.block_number = b.block_number
                WHERE maker_amount > 0 AND taker_amount > 0
            """)

    def _get_top_markets(self, platform: str, n: int = 20) -> list[str]:
        """Get top N markets by trade count."""
        if platform == "kalshi":
            query = """
                SELECT market_id, COUNT(*) as trade_count
                FROM kalshi_trades
                GROUP BY market_id
                ORDER BY trade_count DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT market_id, COUNT(*) as trade_count
                FROM polymarket_trades
                GROUP BY market_id
                ORDER BY trade_count DESC
                LIMIT ?
            """

        result = self.con.execute(query, [n]).fetchall()
        return [row[0] for row in result]

    def _run_variance_ratio_test_market(self, ticker: str) -> None:
        """Run VR test for a single market."""
        # Fetch price time series
        df = self.con.execute(f"""
            SELECT price, ts
            FROM kalshi_trades
            WHERE market_id = ?
            ORDER BY ts
            LIMIT 500000
        """, [ticker]).df()

        if len(df) < 20:
            logger.warning(f"Not enough trades for {ticker}")
            return

        prices = df["price"].values
        returns = np.diff(prices)

        # Test for k = 2, 5, 10, 20
        for k in [2, 5, 10, 20]:
            if len(returns) < k + 1:
                continue

            vr, m2, p_value = VarianceRatioTest.compute_vr_m2(returns, k)

            self.results["variance_ratio_tests"].append(
                {
                    "ticker": ticker,
                    "k": k,
                    "vr": float(vr) if not np.isnan(vr) else None,
                    "m2_statistic": float(m2) if not np.isnan(m2) else None,
                    "p_value": float(p_value) if not np.isnan(p_value) else None,
                    "n_observations": len(returns),
                }
            )

    def _run_rolls_spread_market(self, ticker: str) -> None:
        """Run Roll's spread test for a single market."""
        # Fetch prices
        df = self.con.execute(f"""
            SELECT price, ts
            FROM kalshi_trades
            WHERE market_id = ?
            ORDER BY ts
            LIMIT 500000
        """, [ticker]).df()

        if len(df) < 10:
            logger.warning(f"Not enough trades for {ticker}")
            return

        prices = df["price"].values
        spread, autocovar = RollsSpread.compute_spread(prices)

        self.results["rolls_spread_tests"].append(
            {
                "ticker": ticker,
                "implied_spread": float(spread) if not np.isnan(spread) else None,
                "autocovariance": float(autocovar) if not np.isnan(autocovar) else None,
                "n_trades": len(prices),
            }
        )

    def _run_gpd_tail_fitting(
        self, platform: str, markets: list[str]
    ) -> None:
        """Fit GPD to tail of price changes for a platform."""
        # Fetch price changes
        if platform == "kalshi":
            view = "kalshi_trades"
        else:
            view = "polymarket_trades"

        market_list = ",".join([f"'{m}'" for m in markets[:10]])
        df = self.con.execute(f"""
            SELECT market_id, price, ts
            FROM {view}
            WHERE market_id IN ({market_list})
            ORDER BY ts
            LIMIT 500000
        """).df()

        if len(df) < 50:
            logger.warning(f"Not enough data for {platform}")
            return

        prices = df["price"].values
        price_changes = np.diff(prices)

        # Fit to upper tail (positive changes)
        upper_result = GPDTailFitting.fit_gpd_tail(
            price_changes, threshold_percentile=95
        )

        # Fit to lower tail (negative changes)
        lower_result = GPDTailFitting.fit_gpd_tail(
            -price_changes, threshold_percentile=95
        )

        self.results["gpd_tail_fitting"][platform] = {
            "upper_tail": {**upper_result, "tail": "upper"},
            "lower_tail": {**lower_result, "tail": "lower"},
            "n_price_changes": len(price_changes),
        }

    def _run_bds_test_market(self, ticker: str) -> None:
        """Run BDS test on AR(1) residuals for a single market."""
        # Fetch price time series
        df = self.con.execute(f"""
            SELECT price, ts
            FROM kalshi_trades
            WHERE market_id = ?
            ORDER BY ts
            LIMIT 500000
        """, [ticker]).df()

        if len(df) < 50:
            logger.warning(f"Not enough trades for BDS test on {ticker}")
            return

        prices = df["price"].values
        returns = np.diff(prices)

        # Fit AR(1) and get residuals
        if len(returns) < 2:
            return

        mean_ret = np.mean(returns)
        # Simple AR(1): r_t = mean + rho * (r_{t-1} - mean) + epsilon_t
        x = returns[:-1] - mean_ret
        y = returns[1:] - mean_ret

        if np.var(x) > 0:
            rho = np.cov(x, y)[0, 1] / np.var(x)
            residuals = y - rho * x
        else:
            residuals = returns[1:] - mean_ret

        # Run BDS test
        bds_results = BDSTest.compute_bds(residuals)

        for result in bds_results:
            self.results["bds_tests"].append(
                {"ticker": ticker, **result, "n_residuals": len(residuals)}
            )

    def _run_tail_event_prediction_market(self, ticker: str) -> None:
        """Run ML tail prediction for a single market."""
        # Fetch trades
        df = self.con.execute(f"""
            SELECT price, quantity, taker_side AS taker_side_col, ts
            FROM kalshi_trades
            WHERE market_id = ?
            ORDER BY ts
            LIMIT 500000
        """, [ticker]).df()

        if len(df) < 100:
            logger.warning(f"Not enough trades for prediction on {ticker}")
            return

        # Rename column for consistency
        df = df.rename(columns={"taker_side_col": "taker_side"})

        result = TailEventPrediction.train_and_evaluate(
            {"ticker": ticker, "df": df}
        )
        self.results["tail_event_prediction"].append(result)

    def _save_results(self) -> None:
        """Save results to JSON."""
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        output_path = ARTIFACTS / "phase2_results.json"

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("PHASE 2 STATISTICAL TESTS SUMMARY")
        print("=" * 80)
        print(f"Variance Ratio Tests: {len(self.results['variance_ratio_tests'])} results")
        print(f"Roll's Spread Tests: {len(self.results['rolls_spread_tests'])} results")
        print(f"GPD Tail Fitting: {len(self.results['gpd_tail_fitting'])} platforms")
        print(f"BDS Tests: {len(self.results['bds_tests'])} results")
        print(f"Tail Event Prediction: {len(self.results['tail_event_prediction'])} markets")
        print("=" * 80)


def main() -> None:
    """Entry point."""
    try:
        tester = Phase2StatisticalTests()
        tester.run_all_tests()
        logger.info("Phase 2 tests completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Phase 2 tests failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
