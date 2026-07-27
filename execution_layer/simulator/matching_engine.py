"""
execution_layer/simulator/matching_engine.py
─────────────────────────────────────────────
Matching Engine for Market and Limit Orders.
Simulates market depth, order book matching, latency delays, and partial fills.
"""

from __future__ import annotations
import logging
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from execution_layer.simulator.config import ExecutionSimulatorConfig
from execution_layer.simulator.order import (
    Order, OrderType, OrderSide, OrderStatus, Fill
)
from execution_layer.simulator.cost_model import ExecutionCostModel

logger = logging.getLogger(__name__)


class MatchingEngine:
    """Simulates market depth matching for Market and Limit orders."""

    def __init__(
        self,
        config: Optional[ExecutionSimulatorConfig] = None,
        cost_model: Optional[ExecutionCostModel] = None,
    ):
        self.config = config or ExecutionSimulatorConfig()
        self.cost_model = cost_model or ExecutionCostModel(config=self.config)

    def process_order(
        self,
        order: Order,
        candle_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Order:
        """
        Processes an order against a market candle DataFrame.
        candle_df must contain ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        if order.is_complete:
            return order

        if candle_df.empty:
            order.status = OrderStatus.REJECTED
            return order

        latest_candle = candle_df.iloc[-1]
        open_px = float(latest_candle.get("Open", latest_candle.get("Close", 100.0)))
        high_px = float(latest_candle.get("High", open_px))
        low_px  = float(latest_candle.get("Low", open_px))
        close_px = float(latest_candle.get("Close", open_px))
        volume  = float(latest_candle.get("Volume", 1e6))

        fill_time = timestamp if timestamp is not None else pd.Timestamp.now()
        # Add latency delay (latency_ms converted to Timedelta)
        fill_time = fill_time + pd.Timedelta(milliseconds=self.config.latency_ms)

        if order.order_type == OrderType.MARKET:
            self._fill_market_order(order, close_px, volume, fill_time)
        elif order.order_type == OrderType.LIMIT:
            self._fill_limit_order(order, open_px, high_px, low_px, close_px, volume, fill_time)

        return order

    def _fill_market_order(
        self,
        order: Order,
        base_price: float,
        volume: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Immediate market order fill with liquidity participation caps."""
        rem_qty = order.remaining_quantity

        # Partial fill check based on max volume participation
        if self.config.enable_partial_fills:
            max_fillable = volume * self.config.max_volume_participation
            fill_qty = min(rem_qty, max_fillable)
        else:
            fill_qty = rem_qty

        if fill_qty <= 0:
            return

        fill_px, slip, spread = self.cost_model.compute_fill_price(
            base_price=base_price,
            side=order.side,
            order_quantity=fill_qty,
            available_volume=volume,
        )

        trade_val = fill_px * fill_qty
        comm = self.cost_model.compute_commission(trade_val)

        fill = Fill(
            timestamp=timestamp,
            quantity=fill_qty,
            price=fill_px,
            commission=comm,
            slippage=slip * fill_qty,
            bid_ask_spread=spread * fill_qty,
        )
        order.add_fill(fill)

    def _fill_limit_order(
        self,
        order: Order,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
        volume: float,
        timestamp: pd.Timestamp,
    ) -> None:
        """Limit order matching: BUY limit triggers if Low <= Limit; SELL limit triggers if High >= Limit."""
        if order.limit_price is None:
            order.status = OrderStatus.REJECTED
            return

        limit_px = order.limit_price
        can_fill = False

        if order.side == OrderSide.BUY and low_px <= limit_px:
            can_fill = True
            base_price = min(limit_px, open_px)  # Price improvement
        elif order.side == OrderSide.SELL and high_px >= limit_px:
            can_fill = True
            base_price = max(limit_px, open_px)

        if not can_fill:
            return

        rem_qty = order.remaining_quantity

        if self.config.enable_partial_fills:
            max_fillable = volume * self.config.max_volume_participation
            fill_qty = min(rem_qty, max_fillable)
        else:
            fill_qty = rem_qty

        if fill_qty <= 0:
            return

        # Limit orders earn passive spread benefit (no taker penalty)
        fill_px, slip, spread = self.cost_model.compute_fill_price(
            base_price=base_price,
            side=order.side,
            order_quantity=fill_qty,
            available_volume=volume,
        )
        fill_px = limit_px  # Limit order fills at or better than limit_px

        trade_val = fill_px * fill_qty
        comm = self.cost_model.compute_commission(trade_val)

        fill = Fill(
            timestamp=timestamp,
            quantity=fill_qty,
            price=fill_px,
            commission=comm,
            slippage=0.0,  # Passive limit order
            bid_ask_spread=0.0,
        )
        order.add_fill(fill)
