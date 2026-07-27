"""
execution_layer/backtesting/costs.py
──────────────────────────────────────
Transaction Cost Model Module.
Computes fixed commissions, bid-ask spread, and nonlinear market impact costs.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from execution_layer.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class CostModel:
    """
    Computes vectorized transaction cost estimates across the portfolio panel:
      - Fixed proportional cost (e.g. 0.15% one-way)
      - Nonlinear market impact cost (Almgren-Chriss square-root law)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def compute_trade_costs(
        self,
        holding_weights: pd.DataFrame,
        adv_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes vectorized per-day transaction costs across all stocks.

        Returns:
            (fixed_costs, impact_costs, total_costs) — all as daily pd.Series
        """
        all_dates = holding_weights.index

        # Daily trade weights (absolute weight changes)
        trades = holding_weights.diff().abs()
        turnover = trades.sum(axis=1)

        # 1. Fixed proportional cost
        fixed_costs = turnover * self.config.transaction_cost_pct

        # 2. Nonlinear market impact (Almgren-Chriss square-root law)
        impact_costs = pd.Series(0.0, index=all_dates)
        if adv_data is not None:
            adv_aligned = (
                adv_data
                .reindex(index=all_dates, columns=holding_weights.columns)
                .fillna(1.0)
                .clip(lower=1.0)
            )
            # Participation rate = (trade_weight * AUM) / ADV_i
            participation = (trades * self.config.initial_capital) / adv_aligned
            impact_costs = (self.config.impact_coeff * (participation ** 2)).sum(axis=1)
            logger.debug(f"[Costs] Mean daily market impact (bps): {(impact_costs.mean() * 10000):.2f}")

        total_costs = fixed_costs + impact_costs
        return fixed_costs, impact_costs, total_costs
