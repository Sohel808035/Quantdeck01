"""
QuantSphereX Modular Backtesting Subpackage.
Separated into: Signals, Execution, Portfolio, Costs, Metrics, Factor Attribution, and Reports.
Backward-compatible with legacy Backtester.
"""

from execution_layer.backtesting.config import BacktestConfig
from execution_layer.backtesting.engine import BacktestEngine, BacktestResult
from execution_layer.backtesting.signals import SignalPipeline
from execution_layer.backtesting.execution import ExecutionEngine
from execution_layer.backtesting.portfolio import PortfolioTracker
from execution_layer.backtesting.costs import CostModel
from execution_layer.backtesting.metrics import MetricsEngine
from execution_layer.backtesting.factor_attribution import FactorAttributionEngine
from execution_layer.backtesting.reports import ReportGenerator

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "SignalPipeline",
    "ExecutionEngine",
    "PortfolioTracker",
    "CostModel",
    "MetricsEngine",
    "FactorAttributionEngine",
    "ReportGenerator",
]
