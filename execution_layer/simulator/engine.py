"""
execution_layer/simulator/engine.py
───────────────────────────────────
Master Execution Simulator API Orchestrator.
Exposes simple entry points for executing Market, Limit, VWAP, and TWAP orders
against market candle panels independently from backtesting.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

from execution_layer.simulator.config import ExecutionSimulatorConfig
from execution_layer.simulator.order import (
    Order, OrderType, OrderSide, OrderStatus, ExecutionReport
)
from execution_layer.simulator.matching_engine import MatchingEngine
from execution_layer.simulator.algorithms import AlgorithmicExecutionEngine
from execution_layer.simulator.cost_model import ExecutionCostModel
from execution_layer.simulator.report import ExecutionReportGenerator

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """Professional Independent Execution Simulator Engine."""

    def __init__(self, config: Optional[ExecutionSimulatorConfig] = None):
        self.config = config or ExecutionSimulatorConfig()
        self.cost_model = ExecutionCostModel(config=self.config)
        self.matching_engine = MatchingEngine(config=self.config, cost_model=self.cost_model)
        self.algo_engine = AlgorithmicExecutionEngine(config=self.config)
        self.report_generator = ExecutionReportGenerator()

    def execute_market_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        candle_df: pd.DataFrame,
        arrival_price: Optional[float] = None,
    ) -> Tuple[Order, ExecutionReport]:
        """
        Executes an immediate Market Order against market candle data.

        Args:
            ticker:        Stock ticker symbol.
            side:          'BUY' or 'SELL'.
            quantity:      Shares/contracts to trade.
            candle_df:     OHLCV candle DataFrame.
            arrival_price: Decision arrival price (defaults to last candle Close).

        Returns:
            Tuple of (Order, ExecutionReport).
        """
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order = Order(
            ticker=ticker,
            side=side_enum,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        if arrival_price is None:
            arrival_price = float(candle_df.iloc[-1]["Close"]) if not candle_df.empty else 100.0

        t_start = pd.Timestamp.now()
        self.matching_engine.process_order(order, candle_df, timestamp=t_start)
        t_end = pd.Timestamp.now()

        report = self.report_generator.build_report(
            order=order,
            arrival_price=arrival_price,
            latency_ms=self.config.latency_ms,
            start_time=t_start,
            end_time=t_end,
        )

        return order, report

    def execute_limit_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        limit_price: float,
        candle_df: pd.DataFrame,
        arrival_price: Optional[float] = None,
    ) -> Tuple[Order, ExecutionReport]:
        """
        Executes a Limit Order against market candle data.
        """
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order = Order(
            ticker=ticker,
            side=side_enum,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
        )

        if arrival_price is None:
            arrival_price = float(candle_df.iloc[-1]["Close"]) if not candle_df.empty else limit_price

        t_start = pd.Timestamp.now()
        self.matching_engine.process_order(order, candle_df, timestamp=t_start)
        t_end = pd.Timestamp.now()

        report = self.report_generator.build_report(
            order=order,
            arrival_price=arrival_price,
            latency_ms=self.config.latency_ms,
            start_time=t_start,
            end_time=t_end,
        )

        return order, report

    def execute_vwap_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        candle_df: pd.DataFrame,
        n_slices: int = 10,
        arrival_price: Optional[float] = None,
    ) -> Tuple[Order, ExecutionReport]:
        """
        Executes a VWAP algorithmic order sliced across intraday volume profile.
        """
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        parent_order = Order(
            ticker=ticker,
            side=side_enum,
            order_type=OrderType.VWAP,
            quantity=quantity,
        )

        if arrival_price is None:
            arrival_price = float(candle_df.iloc[-1]["Close"]) if not candle_df.empty else 100.0

        vol_profile = candle_df["Volume"] if "Volume" in candle_df.columns else None
        slices = self.algo_engine.generate_vwap_slices(parent_order, volume_profile=vol_profile, n_slices=n_slices)

        t_start = pd.Timestamp.now()
        for child in slices:
            self.matching_engine.process_order(child, candle_df, timestamp=t_start)
            for fill in child.fills:
                parent_order.add_fill(fill)

        t_end = pd.Timestamp.now()

        report = self.report_generator.build_report(
            order=parent_order,
            arrival_price=arrival_price,
            latency_ms=self.config.latency_ms,
            start_time=t_start,
            end_time=t_end,
        )

        return parent_order, report

    def execute_twap_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        candle_df: pd.DataFrame,
        n_slices: int = 10,
        arrival_price: Optional[float] = None,
    ) -> Tuple[Order, ExecutionReport]:
        """
        Executes a TWAP algorithmic order sliced equally over uniform time intervals.
        """
        side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        parent_order = Order(
            ticker=ticker,
            side=side_enum,
            order_type=OrderType.TWAP,
            quantity=quantity,
        )

        if arrival_price is None:
            arrival_price = float(candle_df.iloc[-1]["Close"]) if not candle_df.empty else 100.0

        slices = self.algo_engine.generate_twap_slices(parent_order, n_slices=n_slices)

        t_start = pd.Timestamp.now()
        for child in slices:
            self.matching_engine.process_order(child, candle_df, timestamp=t_start)
            for fill in child.fills:
                parent_order.add_fill(fill)

        t_end = pd.Timestamp.now()

        report = self.report_generator.build_report(
            order=parent_order,
            arrival_price=arrival_price,
            latency_ms=self.config.latency_ms,
            start_time=t_start,
            end_time=t_end,
        )

        return parent_order, report
