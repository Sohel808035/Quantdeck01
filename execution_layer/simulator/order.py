"""
execution_layer/simulator/order.py
───────────────────────────────────
Data Transfer Objects (DTOs) for Orders, Fills, and Execution Reports.
"""

from __future__ import annotations
import uuid
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    VWAP = "VWAP"
    TWAP = "TWAP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Fill:
    """Represents a single trade fill chunk."""
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    bid_ask_spread: float = 0.0


@dataclass
class Order:
    """Represents a trading order in the simulator."""
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    algorithm_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    def add_fill(self, fill: Fill) -> None:
        """Applies a fill event to the order, updating average fill price and status."""
        self.fills.append(fill)
        new_qty = self.filled_quantity + fill.quantity
        if new_qty > 0:
            self.avg_fill_price = (
                (self.avg_fill_price * self.filled_quantity) + (fill.price * fill.quantity)
            ) / new_qty
        self.filled_quantity = new_qty
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class ExecutionReport:
    """Comprehensive execution summary report for an order or batch."""
    order_id: str
    ticker: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    fill_rate_pct: float
    arrival_price: float
    avg_fill_price: float
    implementation_shortfall_bps: float
    total_commission: float
    total_slippage: float
    status: str
    latency_ms: float
    fills_count: int
    execution_time_seconds: float
