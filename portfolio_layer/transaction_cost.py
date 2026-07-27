"""
portfolio_layer/transaction_cost.py
───────────────────────────────────
QuantSphereX Transaction Cost & Market Impact Model.
Computes fixed commissions, slippage, and Almgren-Chriss square-root market impact costs.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TransactionCostEngine:
    """Computes realistic execution friction, commission, and market impact cost penalty."""

    def __init__(
        self,
        commission_bps: float = 10.0,       # 10 bps fixed broker fee
        slippage_bps: float = 5.0,          # 5 bps bid-ask slippage
        impact_coef: float = 0.10,          # Square-root impact coefficient
    ):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.impact_coef = impact_coef

    def estimate_trade_costs(
        self,
        trades_currency: pd.Series,
        adv_data: Optional[pd.Series] = None,
        daily_vol: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Estimates trade execution cost in currency for each trade.

        Cost = Fixed Commission + Fixed Slippage + Square-Root Market Impact
        Impact Cost = Impact_Coef * Daily_Vol * sqrt(Trade_Value / ADV) * Trade_Value
        """
        trade_abs = trades_currency.abs()
        fixed_cost = trade_abs * ((self.commission_bps + self.slippage_bps) / 10_000.0)

        impact_cost = pd.Series(0.0, index=trades_currency.index)
        if adv_data is not None and daily_vol is not None:
            adv_sub = adv_data.reindex(trades_currency.index).fillna(1e8)
            vol_sub = daily_vol.reindex(trades_currency.index).fillna(0.015)

            participation = np.sqrt((trade_abs / adv_sub.replace(0, np.nan)).fillna(0))
            impact_cost = self.impact_coef * vol_sub * participation * trade_abs

        return fixed_cost + impact_cost

    def net_weights_after_costs(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float = 1e7,
        adv_data: Optional[pd.Series] = None,
        daily_vol: Optional[pd.Series] = None,
    ) -> tuple[pd.Series, float]:
        """
        Deducts estimated transaction costs from rebalanced portfolio weights.

        Returns:
            (net_target_weights, total_cost_deducted_in_currency)
        """
        all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index)))
        curr = current_weights.reindex(all_tickers).fillna(0.0)
        targ = target_weights.reindex(all_tickers).fillna(0.0)

        trade_weights = targ - curr
        trades_curr = trade_weights * portfolio_value

        costs_curr = self.estimate_trade_costs(trades_curr, adv_data, daily_vol)
        total_cost = float(costs_curr.sum())

        # Net weights after paying costs out of portfolio equity
        remaining_equity = portfolio_value - total_cost
        if remaining_equity <= 0:
            return targ, total_cost

        net_weights = (targ * portfolio_value) / remaining_equity
        net_weights = net_weights / net_weights.sum()

        return net_weights, total_cost
