"""
QuantSphereX Execution Layer Domain Package.
Contains Backtester (legacy), Modular BacktestEngine, Stress Testing, and Execution Simulator.
"""

# Legacy (preserved 100% for backward compatibility)
from execution_layer.backtester import Backtester
from execution_layer.stress_tester import run_stress_tests

# Modular Backtesting Engine
from execution_layer.backtesting import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)

# Independent Execution Simulator
from execution_layer.simulator import (
    ExecutionSimulator,
    ExecutionSimulatorConfig,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionReport,
)

__all__ = [
    # Legacy
    "Backtester",
    "run_stress_tests",
    # Modular Backtesting
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Simulator
    "ExecutionSimulator",
    "ExecutionSimulatorConfig",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "ExecutionReport",
]
