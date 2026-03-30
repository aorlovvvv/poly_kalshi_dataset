"""
EDITABLE — This is the ONLY file the autoresearch loop modifies.

Exp 5: LogisticRegression (fast) + boundary-weighted mean-reversion.
Hypothesis: GBM takes 60s but real edge is from mean-reversion signal.
LogReg trains in <1s with same AUC (ret is linearly predictive of target).
Add price_boundary_dist weighting: mean-revert more aggressively near
extremes (0/1) where bounce-back is strongest.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .interfaces import PortfolioState, Strategy

DD_SOFT_START = 0.08
DD_HARD_STOP = 0.16
TURNOVER_THRESHOLD = 0.15
SIGNAL_SCALE = 0.4
TAIL_SCALE_FACTOR = 0.5


class TailRiskStrategy(Strategy):

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=1.0, max_iter=500, random_state=42,
        )
        self._feature_names: list[str] = []
        self._importances: dict[str, float] = {}
        self._direction_signals: list[float] = []
        self._call_idx: int = 0

    def name(self) -> str:
        return "logreg-minimal-tail-def-v10"

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
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

        combined_ret = rets + 0.7 * lag1 + 0.2 * lag2
        boundary_boost = 1.0 + 2.0 * (0.5 - np.clip(bds, 0, 0.5))

        self._direction_signals = (-combined_ret / vols_safe * boundary_boost).tolist()
        self._call_idx = 0

        return probs

    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        if self._call_idx < len(self._direction_signals):
            raw_signal = self._direction_signals[self._call_idx]
            self._call_idx += 1
        else:
            return 0.0

        dd_scale = 1.0
        if state.drawdown > DD_HARD_STOP:
            dd_scale = 0.0
        elif state.drawdown > DD_SOFT_START:
            dd_scale = (DD_HARD_STOP - state.drawdown) / (DD_HARD_STOP - DD_SOFT_START)

        tail_scale = max(0.0, 1.0 - TAIL_SCALE_FACTOR * tail_prob)
        desired = float(np.clip(raw_signal * SIGNAL_SCALE * tail_scale * dd_scale, -1.0, 1.0))

        if abs(desired - state.position) < TURNOVER_THRESHOLD:
            return state.position

        return desired

    def get_feature_importances(self) -> dict[str, float]:
        return self._importances


def create_strategy() -> Strategy:
    return TailRiskStrategy()
