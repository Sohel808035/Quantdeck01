"""
execution_layer/simulator/report.py
───────────────────────────────────
Execution Reporting and Implementation Shortfall Analytics.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from execution_layer.simulator.order import Order, OrderSide, ExecutionReport

logger = logging.getLogger(__name__)


class ExecutionReportGenerator:
    """Generates execution reports, implementation shortfall metrics, and fill analytics."""

    def build_report(
        self,
        order: Order,
        arrival_price: float,
        latency_ms: float = 50.0,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> ExecutionReport:
        """
        Builds a detailed ExecutionReport for a completed order.

        Implementation Shortfall (bps):
          Buy:  (Avg_Fill_Price - Arrival_Price) / Arrival_Price * 10,000
          Sell: (Arrival_Price - Avg_Fill_Price) / Arrival_Price * 10,000
        """
        fill_rate = (order.filled_quantity / order.quantity) * 100.0 if order.quantity > 0 else 0.0

        if arrival_price > 0 and order.filled_quantity > 0:
            if order.side == OrderSide.BUY:
                shortfall_bps = ((order.avg_fill_price - arrival_price) / arrival_price) * 10_000.0
            else:
                shortfall_bps = ((arrival_price - order.avg_fill_price) / arrival_price) * 10_000.0
        else:
            shortfall_bps = 0.0

        exec_time = 0.0
        if start_time is not None and end_time is not None:
            exec_time = float((end_time - start_time).total_seconds())

        return ExecutionReport(
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            fill_rate_pct=round(fill_rate, 2),
            arrival_price=round(arrival_price, 4),
            avg_fill_price=round(order.avg_fill_price, 4),
            implementation_shortfall_bps=round(shortfall_bps, 2),
            total_commission=round(order.total_commission, 2),
            total_slippage=round(order.total_slippage, 4),
            status=order.status.value,
            latency_ms=latency_ms,
            fills_count=len(order.fills),
            execution_time_seconds=round(exec_time, 4),
        )

    def summary_table(self, reports: List[ExecutionReport]) -> pd.DataFrame:
        """Converts a list of ExecutionReport objects into a summary DataFrame."""
        if not reports:
            return pd.DataFrame()

        data = [r.__dict__ for r in reports]
        df = pd.DataFrame(data)
        return df
