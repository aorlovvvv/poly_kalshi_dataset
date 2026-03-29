"""
EDITABLE — This is the ONLY file the autoresearch loop modifies.

Exp 1: Mean-reversion sizing with directional signal + turnover filter.
Hypothesis: VR ≈ 0.59 means strong mean reversion. Going against current
ret should be profitable, scaled by inverse volatility. Reduce position
during detected tail events (uncertain direction). Turnover filter to
cut transaction costs.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .interfaces import PortfolioState, Strategy


class TailRiskStrategy(Strategy):

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=42,
        )
        self._feature_names: list[str] = []
        self._importances: dict[str, float] = {}
        self._direction_signals: list[float] = []
        self._vol_signals: list[float] = []
        self._call_idx: int = 0

    def name(self) -> str:
        return "mean-revert-v1"

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        self._feature_names = list(feature_names)
        self.model.fit(X_train, y_train)
        imp = self.model.feature_importances_
        self._importances = {
            name: round(float(v), 6)
            for name, v in zip(feature_names, imp)
        }

    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(X)[:, 1]

        ret_idx = self._feature_names.index("ret")
        vol_idx = self._feature_names.index("rolling_vol_50")

        rets = X[:, ret_idx]
        vols = X[:, vol_idx]
        vols_safe = np.where(vols > 1e-6, vols, 1e-6)

        self._direction_signals = (-rets / vols_safe).tolist()
        self._vol_signals = vols.tolist()
        self._call_idx = 0

        return probs

    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        if self._call_idx < len(self._direction_signals):
            raw_signal = self._direction_signals[self._call_idx]
            self._call_idx += 1
        else:
            return 0.0

        tail_scale = max(0.0, 1.0 - 2.0 * tail_prob)

        desired = float(np.clip(raw_signal * 0.3 * tail_scale, -1.0, 1.0))

        if abs(desired - state.position) < 0.15:
            return state.position

        return desired

    def get_feature_importances(self) -> dict[str, float]:
        return self._importances


def create_strategy() -> Strategy:
    return TailRiskStrategy()
