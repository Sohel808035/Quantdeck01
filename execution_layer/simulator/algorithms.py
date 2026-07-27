"""
execution_layer/simulator/algorithms.py
────────────────────────────────────────
Algorithmic Execution Slicing Engines: VWAP and TWAP.
Slices large parent orders into smaller child orders over time.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import pandas as pd
import numpy as np

from execution_layer.simulator.config import ExecutionSimulatorConfig
from execution_layer.simulator.order import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class AlgorithmicExecutionEngine:
    """Generates child order slices for VWAP and TWAP parent orders."""

    def __init__(self, config: Optional[ExecutionSimulatorConfig] = None):
        self.config = config or ExecutionSimulatorConfig()

    def generate_vwap_slices(
        self,
        parent_order: Order,
        volume_profile: Optional[pd.Series] = None,
        n_slices: Optional[int] = None,
    ) -> List[Order]:
        """
        Slices parent order into N child orders proportional to intraday volume profile.
        Child Quantity_i = Parent_Quantity * (Volume_i / Total_Volume)
        """
        n = n_slices or self.config.vwap_slices
        total_qty = parent_order.quantity

        if volume_profile is not None and len(volume_profile) >= n:
            vol_subset = volume_profile.tail(n)
            profile_weights = (vol_subset / vol_subset.sum()).values
        else:
            # Default U-shaped volume profile (higher at open & close, lower at midday)
            x = np.linspace(-1.5, 1.5, n)
            profile_weights = x**2 + 0.5
            profile_weights = profile_weights / np.sum(profile_weights)

        child_orders = []
        for i, weight in enumerate(profile_weights):
            child_qty = total_qty * weight
            child = Order(
                ticker=parent_order.ticker,
                side=parent_order.side,
                order_type=OrderType.MARKET,
                quantity=child_qty,
                limit_price=parent_order.limit_price,
                algorithm_params={"slice_idx": i, "total_slices": n, "profile_weight": weight},
            )
            child_orders.append(child)

        return child_orders

    def generate_twap_slices(
        self,
        parent_order: Order,
        n_slices: Optional[int] = None,
    ) -> List[Order]:
        """
        Slices parent order into N equal-sized child orders across uniform time intervals.
        Child Quantity_i = Parent_Quantity / N
        """
        n = n_slices or self.config.twap_slices
        total_qty = parent_order.quantity
        child_qty = total_qty / float(n)

        child_orders = []
        for i in range(n):
            child = Order(
                ticker=parent_order.ticker,
                side=parent_order.side,
                order_type=OrderType.MARKET,
                quantity=child_qty,
                limit_price=parent_order.limit_price,
                algorithm_params={"slice_idx": i, "total_slices": n},
            )
            child_orders.append(child)

        return child_orders
