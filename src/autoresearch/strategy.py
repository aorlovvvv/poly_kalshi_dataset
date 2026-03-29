"""
EDITABLE — This is the ONLY file the autoresearch loop modifies.

Baseline strategy: GradientBoosting for tail prediction + simple threshold sizing.
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

    def name(self) -> str:
        return "baseline-gbm-v1"

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
        return self.model.predict_proba(X)[:, 1]

    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        edge = tail_prob - 0.5
        position = np.clip(edge * 2.0, -1.0, 1.0)
        return float(position)

    def get_feature_importances(self) -> dict[str, float]:
        return self._importances


def create_strategy() -> Strategy:
    return TailRiskStrategy()
