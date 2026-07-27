"""
execution_layer/backtesting/execution.py
─────────────────────────────────────────
Execution Module.
Orchestrates trade flow: applies weights to returns, deducts costs, and builds gross/net return series.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from execution_layer.backtesting.config import BacktestConfig
from execution_layer.backtesting.costs import CostModel

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Connects signal weights to PnL generation:
      - Aligns returns to holding weights
      - Computes gross and net portfolio return series
      - Builds equity curve from initial capital
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.cost_model = CostModel(config=self.config)

    def run(
        self,
        holding_weights: pd.DataFrame,
        stock_returns: pd.DataFrame,
        adv_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Executes the strategy signal over the full history.

        Returns:
            (gross_ret, net_ret, equity_curve, turnover, fixed_costs, impact_costs)
        """
        # Align returns to holding weight columns
        aligned_returns = (
            stock_returns
            .reindex(columns=holding_weights.columns)
            .fillna(0.0)
            .reindex(index=holding_weights.index)
            .fillna(0.0)
        )

        gross_ret = (holding_weights * aligned_returns).sum(axis=1)

        fixed_costs, impact_costs, total_costs = self.cost_model.compute_trade_costs(
            holding_weights=holding_weights,
            adv_data=adv_data,
        )

        trades = holding_weights.diff().abs()
        turnover = trades.sum(axis=1)

        net_ret = gross_ret - total_costs

        equity_curve = self.config.initial_capital * (1.0 + net_ret).cumprod()

        logger.debug(
            f"[Execution] Net return: mean={net_ret.mean():.5f}, "
            f"total fixed costs={fixed_costs.sum():.2f}, "
            f"total impact={impact_costs.sum():.2f}"
        )

        return gross_ret, net_ret, equity_curve, turnover, fixed_costs, impact_costs
