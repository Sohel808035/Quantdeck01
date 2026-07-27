"""
execution_layer/simulator/cost_model.py
────────────────────────────────────────
Execution Cost Model for the QuantSphereX Simulator.
Computes bid-ask spread half-costs, stochastic/volume-dependent slippage, and commissions.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from execution_layer.simulator.config import ExecutionSimulatorConfig
from execution_layer.simulator.order import OrderSide

logger = logging.getLogger(__name__)


class ExecutionCostModel:
    """Calculates transaction friction: spreads, slippage, and commissions."""

    def __init__(self, config: Optional[ExecutionSimulatorConfig] = None):
        self.config = config or ExecutionSimulatorConfig()

    def compute_fill_price(
        self,
        base_price: float,
        side: OrderSide,
        order_quantity: float,
        available_volume: float = 1e6,
        volatility: float = 0.015,
    ) -> Tuple[float, float, float]:
        """
        Computes effective fill price, slippage, and spread cost.

        Buy Fill Price  = Base_Price * (1 + half_spread_bps) * (1 + slippage_bps)
        Sell Fill Price = Base_Price * (1 - half_spread_bps) * (1 - slippage_bps)

        Returns:
            (effective_fill_price, total_slippage_in_price, half_spread_cost_in_price)
        """
        half_spread_pct = (self.config.bid_ask_spread_bps / 2.0) / 10_000.0

        # Market impact & liquidity-based slippage scaling
        # Slippage grows with order size relative to available candle volume
        participation = order_quantity / max(1.0, available_volume)
        volume_impact = 0.10 * np.sqrt(max(0.0, participation))  # Square-root law
        vol_impact = 0.5 * volatility

        total_slippage_pct = (self.config.slippage_bps / 10_000.0) + volume_impact + vol_impact

        half_spread_price = base_price * half_spread_pct
        slippage_price = base_price * total_slippage_pct

        if side == OrderSide.BUY:
            fill_price = base_price + half_spread_price + slippage_price
        else:
            fill_price = base_price - half_spread_price - slippage_price

        return float(fill_price), float(slippage_price), float(half_spread_price)

    def compute_commission(
        self,
        trade_value_currency: float,
    ) -> float:
        """
        Computes tiered commission fee:
        Fee = Fixed_Per_Order + (Trade_Value * commission_bps)
        """
        bps_fee = trade_value_currency * (self.config.commission_bps / 10_000.0)
        return float(self.config.fixed_commission_per_order + bps_fee)
