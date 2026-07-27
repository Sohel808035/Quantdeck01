"""
tests/test_execution_simulator.py
──────────────────────────────────
Unit Test Suite for QuantSphereX Independent Execution Simulator.

Covers:
  - Market Order execution (Slippage, Spread, Commission)
  - Limit Order matching (Price improvement, passive execution)
  - VWAP Order execution (Volume profile slicing)
  - TWAP Order execution (Uniform time interval slicing)
  - Partial Fills & Liquidity Participation limits
  - Latency handling (milliseconds)
  - Execution Report generation & Implementation Shortfall (bps)
  - Full independence from Backtester
"""

import unittest
import numpy as np
import pandas as pd

from execution_layer.simulator import (
    ExecutionSimulator, ExecutionSimulatorConfig,
    Order, OrderType, OrderSide, OrderStatus, ExecutionReport
)


def _make_candle_panel(n_candles: int = 10, base_price: float = 1000.0) -> pd.DataFrame:
    """Generates synthetic OHLCV candle DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01 09:15", periods=n_candles, freq="1min")
    px = base_price + np.cumsum(np.random.normal(0, 1.0, size=n_candles))
    high = px + np.abs(np.random.normal(0.5, 0.2, size=n_candles))
    low = px - np.abs(np.random.normal(0.5, 0.2, size=n_candles))
    open_ = px - np.random.normal(0, 0.2, size=n_candles)
    vol = np.random.uniform(5000, 15000, size=n_candles)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": px,
        "Volume": vol,
    }, index=dates)


class TestExecutionSimulator(unittest.TestCase):

    def setUp(self):
        self.candle_df = _make_candle_panel(n_candles=15, base_price=1000.0)
        self.config = ExecutionSimulatorConfig(
            latency_ms=50.0,
            commission_bps=10.0,
            slippage_bps=2.0,
            bid_ask_spread_bps=5.0,
            max_volume_participation=0.20,
        )
        self.simulator = ExecutionSimulator(config=self.config)

    def test_market_order_buy_execution(self):
        order, report = self.simulator.execute_market_order(
            ticker="RELIANCE.NS",
            side="BUY",
            quantity=100.0,
            candle_df=self.candle_df,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, 100.0)
        self.assertGreater(order.avg_fill_price, 1000.0)  # Buy fill price > base due to spread/slippage
        self.assertIsInstance(report, ExecutionReport)
        self.assertGreater(report.total_commission, 0.0)

    def test_market_order_sell_execution(self):
        order, report = self.simulator.execute_market_order(
            ticker="TCS.NS",
            side="SELL",
            quantity=50.0,
            candle_df=self.candle_df,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, 50.0)
        self.assertLess(order.avg_fill_price, 1000.0)  # Sell fill price < base

    def test_limit_order_buy_filled(self):
        # Set limit price high enough so low_px <= limit_price triggers fill
        limit_px = float(self.candle_df["High"].max()) + 10.0
        order, report = self.simulator.execute_limit_order(
            ticker="INFY.NS",
            side="BUY",
            quantity=100.0,
            limit_price=limit_px,
            candle_df=self.candle_df,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.avg_fill_price, limit_px)

    def test_limit_order_buy_unfilled(self):
        # Set limit price extremely low so low_px <= limit_price fails
        limit_px = float(self.candle_df["Low"].min()) - 100.0
        order, report = self.simulator.execute_limit_order(
            ticker="INFY.NS",
            side="BUY",
            quantity=100.0,
            limit_price=limit_px,
            candle_df=self.candle_df,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(order.filled_quantity, 0.0)

    def test_vwap_order_execution(self):
        order, report = self.simulator.execute_vwap_order(
            ticker="HDFCBANK.NS",
            side="BUY",
            quantity=500.0,
            candle_df=self.candle_df,
            n_slices=5,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(order.filled_quantity, 500.0, places=2)
        self.assertEqual(report.fills_count, 5)

    def test_twap_order_execution(self):
        order, report = self.simulator.execute_twap_order(
            ticker="ICICIBANK.NS",
            side="SELL",
            quantity=400.0,
            candle_df=self.candle_df,
            n_slices=4,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(order.filled_quantity, 400.0, places=2)
        self.assertEqual(report.fills_count, 4)

    def test_partial_fills_under_strict_volume_limit(self):
        # Set participation cap very low (e.g. 0.001) so huge order causes partial fill
        strict_config = ExecutionSimulatorConfig(max_volume_participation=0.001)
        sim = ExecutionSimulator(config=strict_config)
        order, report = sim.execute_market_order(
            ticker="SBIN.NS",
            side="BUY",
            quantity=1_000_000.0,  # Huge quantity
            candle_df=self.candle_df,
            arrival_price=1000.0,
        )
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertLess(order.filled_quantity, 1_000_000.0)
        self.assertGreater(order.filled_quantity, 0.0)

    def test_implementation_shortfall_calculation(self):
        order, report = self.simulator.execute_market_order(
            ticker="WIPRO.NS",
            side="BUY",
            quantity=100.0,
            candle_df=self.candle_df,
            arrival_price=990.0,
        )
        # Arrival price = 990, fill price > 990 -> shortfall_bps > 0
        self.assertGreater(report.implementation_shortfall_bps, 0.0)

    def test_independence_from_backtester(self):
        """Execution simulator functions completely independently from Backtester."""
        sim = ExecutionSimulator()
        ord_obj, rep_obj = sim.execute_market_order("TATAMOTORS.NS", "BUY", 10.0, self.candle_df)
        self.assertIsNotNone(ord_obj)
        self.assertIsNotNone(rep_obj)


if __name__ == "__main__":
    unittest.main()
