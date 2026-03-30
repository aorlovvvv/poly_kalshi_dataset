"""
Autoresearch strategy script. CPU-only, single-file.
This is the ONLY file the agent modifies.

Usage: python train.py
       python train.py > run.log 2>&1

Current strategy: sqrt mean-reversion with 3-trade momentum, boundary
weighting, tanh sizing, and optimized turnover management.

Key findings from 37 experiments:
  1. Mean reversion is the dominant alpha (VR ~ 0.59): go against recent returns
  2. Three-trade momentum: ret + 0.7*lag1 + 0.2*lag2 captures deeper serial correlation
  3. Sqrt signal dampening: extreme returns are noisier, moderate ones more reliable
  4. Boundary boost: prices near 0/1 boundaries revert harder
  5. LogReg (not GBM): doesn't overfit to ret, uses microstructure features properly
  6. Tanh sizing: smooth position transitions reduce cost drag
  7. High turnover threshold (0.60): only reposition for meaningful signal changes
  8. Minimal tail defense (0.5): mean reversion works even during moderate tail events
"""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prepare import (Strategy, PortfolioState, load_data, evaluate,
                     FEATURE_COLUMNS)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Position sizing / risk
TURNOVER_THRESHOLD = 0.75   # ignore position changes smaller than this
SIGNAL_SCALE = 0.70         # multiplier on raw signal before tanh
TAIL_SCALE_FACTOR = 0.3     # how much to reduce position for tail events

# Signal construction
LAG1_WEIGHT = 0.7           # weight on lag_ret_1 in combined momentum
LAG2_WEIGHT = 0.2           # weight on lag_ret_2 in combined momentum
BOUNDARY_COEFF = 2.0        # boost for prices near 0/1 boundaries
DIR_BLEND = 0.2             # weight on ML direction model (vs hand-crafted)

# Model
MODEL_C = 0.01              # LogisticRegression regularization
MODEL_MAX_ITER = 200
DIR_MODEL_C = 0.1           # regularization for direction model

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class TailRiskStrategy(Strategy):

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=MODEL_C, max_iter=MODEL_MAX_ITER, random_state=42,
        )
        self.dir_scaler = StandardScaler()
        self.dir_model = LogisticRegression(
            C=DIR_MODEL_C, max_iter=MODEL_MAX_ITER, random_state=42,
        )
        self._feature_names: list[str] = []
        self._importances: dict[str, float] = {}
        self._direction_signals: list[float] = []
        self._call_idx: int = 0
        self._direction_target: np.ndarray | None = None

    def name(self) -> str:
        return "sqrt-3trade-tanh-v37"

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

        if self._direction_target is not None:
            n = len(X_train)
            half = 3 * n // 4
            X_dir = self.dir_scaler.fit_transform(X_train[half:])
            self.dir_model.fit(X_dir, self._direction_target[half:])

        coefs = np.abs(self.model.coef_[0])
        total = coefs.sum()
        self._importances = {
            name: round(float(v / total), 6) if total > 0 else 0.0
            for name, v in zip(feature_names, coefs)
        }

    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        ret_idx = self._feature_names.index("ret")
        lag1_idx = self._feature_names.index("lag_ret_1")
        lag2_idx = self._feature_names.index("lag_ret_2")
        vol_idx = self._feature_names.index("rolling_vol_50")
        bd_idx = self._feature_names.index("price_boundary_dist")

        rets = X[:, ret_idx]
        lag1 = np.nan_to_num(X[:, lag1_idx], 0.0)
        lag2 = np.nan_to_num(X[:, lag2_idx], 0.0)
        vols = X[:, vol_idx]
        bds = X[:, bd_idx]
        vols_safe = np.where(vols > 1e-6, vols, 1e-6)

        combined_ret = rets + LAG1_WEIGHT * lag1 + LAG2_WEIGHT * lag2
        boundary_boost = 1.0 + BOUNDARY_COEFF * (0.5 - np.clip(bds, 0, 0.5))

        sqrt_signal = -np.sign(combined_ret) * np.sqrt(np.abs(combined_ret))
        sqrt_vol = np.sqrt(vols_safe)
        hand_crafted = sqrt_signal / sqrt_vol * boundary_boost

        if self._direction_target is not None:
            X_dir = self.dir_scaler.transform(X)
            dir_prob = self.dir_model.predict_proba(X_dir)[:, 1]
            dir_signal = (dir_prob - 0.5) * 2.0  # map [0,1] -> [-1,1]
            blended = (1 - DIR_BLEND) * hand_crafted + DIR_BLEND * dir_signal
        else:
            blended = hand_crafted

        self._direction_signals = blended.tolist()
        self._call_idx = 0

        return probs

    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        if self._call_idx < len(self._direction_signals):
            raw_signal = self._direction_signals[self._call_idx]
            self._call_idx += 1
        else:
            return 0.0

        tail_scale = max(0.0, 1.0 - TAIL_SCALE_FACTOR * tail_prob)
        desired = float(np.tanh(raw_signal * SIGNAL_SCALE * tail_scale))

        delta = abs(desired - state.position)
        is_entry = abs(state.position) < 0.05
        threshold = 0.20 if is_entry else TURNOVER_THRESHOLD
        if delta < threshold:
            return state.position

        return desired

    def get_feature_importances(self) -> dict[str, float]:
        return self._importances


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

t_start = time.time()

# Load data
train_df, test_df, feature_names = load_data()
X_train = train_df[feature_names].values.astype(np.float64)
y_train = train_df["target"].values.astype(np.int32)
X_test = test_df[feature_names].values.astype(np.float64)
y_test = test_df["target"].values.astype(np.int32)
next_rets = test_df["next_ret"].values.astype(np.float64)

# Train
strategy = TailRiskStrategy()
strategy._direction_target = (train_df["next_ret"] > 0).astype(int).values
print(f"Strategy: {strategy.name()}")
strategy.train(X_train, y_train, feature_names)

# Evaluate
results = evaluate(strategy, X_test, y_test, next_rets, test_df)

t_end = time.time()

# ---------------------------------------------------------------------------
# Output (fixed format — grep-friendly, matches Karpathy's convention)
# ---------------------------------------------------------------------------

print()
print("---")
print(f"sharpe_ratio:     {results['sharpe_ratio']:.6f}")
print(f"sortino_ratio:    {results['sortino_ratio']:.6f}")
print(f"total_return:     {results['total_return']:.6f}")
print(f"max_drawdown:     {results['max_drawdown']:.6f}")
print(f"auc_roc:          {results['auc_roc']:.4f}")
print(f"win_rate:         {results['win_rate']:.4f}")
print(f"n_trades:         {results['n_trades']}")
print(f"n_markets:        {results['n_markets']}")
print(f"mean_position:    {results['mean_position']:.4f}")
print(f"elapsed_seconds:  {t_end - t_start:.1f}")

print()
print("Top features:")
for name, imp in sorted(results["feature_importances"].items(),
                        key=lambda x: -x[1])[:5]:
    print(f"  {name:25s} {imp:.4f}")
