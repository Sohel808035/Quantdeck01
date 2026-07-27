"""
execution_layer/backtesting/portfolio.py
─────────────────────────────────────────
Portfolio State Tracker Module.
Tracks daily holdings, exposure, and cash positions throughout the backtest.
"""

from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import numpy as np

from execution_layer.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Tracks daily portfolio state across the backtest:
      - Holdings per asset (in currency)
      - Gross exposure (sum of absolute position values)
      - Net exposure (long minus short)
      - Cash buffer
      - Daily weight snapshot
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def build_exposure_panel(
        self,
        holding_weights: pd.DataFrame,
        equity_curve: pd.Series,
    ) -> pd.DataFrame:
        """
        Computes daily currency holdings for each asset based on portfolio equity.

        Returns:
            DataFrame of holdings in currency (shape: dates x tickers).
        """
        equity_aligned = equity_curve.reindex(holding_weights.index).ffill().fillna(self.config.initial_capital)
        holdings = holding_weights.multiply(equity_aligned, axis=0)
        return holdings

    def daily_gross_exposure(self, holding_weights: pd.DataFrame) -> pd.Series:
        """Total gross exposure (sum of absolute weights) per day."""
        return holding_weights.abs().sum(axis=1)

    def daily_net_exposure(self, holding_weights: pd.DataFrame) -> pd.Series:
        """Net long-short exposure per day (positive = net long)."""
        return holding_weights.sum(axis=1)

    def concentration_hhi(self, holding_weights: pd.DataFrame) -> pd.Series:
        """
        Daily Herfindahl-Hirschman Index (HHI) = sum(w_i^2).
        HHI = 1 means perfect concentration; HHI = 1/N means equally distributed.
        """
        normed = holding_weights.div(holding_weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        return (normed ** 2).sum(axis=1)

    def effective_n_stocks(self, holding_weights: pd.DataFrame) -> pd.Series:
        """Effective number of positions = 1 / HHI."""
        hhi = self.concentration_hhi(holding_weights)
        return (1.0 / hhi.replace(0, np.nan)).fillna(0)
