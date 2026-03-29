"""
FIXED — DO NOT EDIT.

Abstract interfaces for the autoresearch trading strategy harness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PortfolioState:
    """Snapshot of the portfolio at a given point in time."""
    position: float = 0.0          # current position [-1, 1]
    cash: float = 1.0              # cash balance (starts at 1.0)
    equity: float = 1.0            # total portfolio value
    peak_equity: float = 1.0       # high-water mark
    drawdown: float = 0.0          # current drawdown from peak
    trade_count: int = 0           # total trades executed
    total_cost: float = 0.0        # cumulative transaction costs paid


@dataclass
class BacktestResult:
    """Results from running a strategy through the backtester."""
    strategy_name: str = ""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    auc_roc: float = 0.5
    n_trades: int = 0
    n_markets: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    feature_importances: dict[str, float] = field(default_factory=dict)
    daily_returns: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class that all strategies must implement."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this strategy version."""
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str]) -> None:
        """Fit the model on training data.

        Args:
            X_train: feature matrix (n_samples, n_features)
            y_train: binary target (1 = tail event, 0 = normal)
            feature_names: list of column names matching X_train columns
        """
        ...

    @abstractmethod
    def predict_tail_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of a tail event for each row.

        Args:
            X: feature matrix (n_samples, n_features)

        Returns:
            1-D array of probabilities in [0, 1]
        """
        ...

    @abstractmethod
    def size_position(self, tail_prob: float,
                      state: PortfolioState) -> float:
        """Decide position size given predicted tail probability.

        Args:
            tail_prob: predicted P(tail event) for next trade
            state: current portfolio snapshot

        Returns:
            desired position in [-1, 1].
            Positive = long (expect price to rise or mean-revert up)
            Negative = short (expect price to drop or mean-revert down)
        """
        ...

    @abstractmethod
    def get_feature_importances(self) -> dict[str, float]:
        """Return feature name → importance mapping."""
        ...
