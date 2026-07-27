"""
execution_layer/backtesting/engine.py
──────────────────────────────────────
Master BacktestEngine Orchestrator.
Wires all modular components: Signals → Execution → Portfolio → Metrics → Reports.
Backward-compatible with existing Backtester API.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from execution_layer.backtesting.config import BacktestConfig
from execution_layer.backtesting.signals import SignalPipeline
from execution_layer.backtesting.execution import ExecutionEngine
from execution_layer.backtesting.portfolio import PortfolioTracker
from execution_layer.backtesting.metrics import MetricsEngine
from execution_layer.backtesting.factor_attribution import FactorAttributionEngine
from execution_layer.backtesting.reports import ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Full backtest result container."""
    metrics: Dict[str, Any] = field(default_factory=dict)
    rolling_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_attribution: Dict[str, float] = field(default_factory=dict)
    tearsheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    holding_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_exposure: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """
    Modular, institutional-grade backtesting engine for QuantSphereX.
    Wires: Signals → Execution → Portfolio → Metrics → Factor Attribution → Reports.

    Usage:
        engine = BacktestEngine(config=BacktestConfig(initial_capital=1e7))
        result = engine.run(weights_schedule, stock_returns, regime_exposure)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.signal_pipeline = SignalPipeline(config=self.config)
        self.execution_engine = ExecutionEngine(config=self.config)
        self.portfolio_tracker = PortfolioTracker(config=self.config)
        self.metrics_engine = MetricsEngine(config=self.config)
        self.factor_engine = FactorAttributionEngine()
        self.report_generator = ReportGenerator()

    def run(
        self,
        weights_schedule: pd.DataFrame,
        stock_returns: pd.DataFrame,
        regime_exposure: Optional[pd.Series] = None,
        adv_data: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        benchmark_ret: Optional[pd.Series] = None,
        export_dir: Optional[str] = None,
    ) -> BacktestResult:
        """
        Executes the full modular backtest pipeline.

        Args:
            weights_schedule: Periodic (e.g. monthly) weight DataFrame.
            stock_returns:    Daily stock return panel.
            regime_exposure:  Optional daily regime exposure scalars.
            adv_data:         Optional daily ADV (for market impact).
            factor_returns:   Optional daily factor return DataFrame for attribution.
            benchmark_ret:    Optional daily benchmark return Series.
            export_dir:       Optional path to export CSV/JSON reports.

        Returns:
            BacktestResult with all components populated.
        """
        all_dates = stock_returns.index
        logger.info(f"[BacktestEngine] Running from {all_dates[0].date()} to {all_dates[-1].date()}")

        # ── 1. Signal Pipeline ──────────────────────────────────────────────────
        holding_weights = self.signal_pipeline.build_holding_weights(
            weights_schedule=weights_schedule,
            all_dates=all_dates,
            regime_exposure=regime_exposure,
        )

        # Align returns for vol targeting
        aligned_returns = (
            stock_returns
            .reindex(columns=holding_weights.columns)
            .fillna(0.0)
            .reindex(index=all_dates)
            .fillna(0.0)
        )
        holding_weights = self.signal_pipeline.apply_vol_targeting(holding_weights, aligned_returns)

        # ── 2. Execution Engine ─────────────────────────────────────────────────
        gross_ret, net_ret, equity_curve, turnover, fixed_costs, impact_costs = (
            self.execution_engine.run(
                holding_weights=holding_weights,
                stock_returns=stock_returns,
                adv_data=adv_data,
            )
        )

        # ── 3. Portfolio Tracker ────────────────────────────────────────────────
        portfolio_exposure = self.portfolio_tracker.build_exposure_panel(holding_weights, equity_curve)

        # ── 4. Metrics ──────────────────────────────────────────────────────────
        metrics = self.metrics_engine.full_period_metrics(
            equity_curve=equity_curve,
            daily_ret=net_ret,
            turnover=turnover,
            fixed_costs=fixed_costs,
            impact_costs=impact_costs,
            benchmark_ret=benchmark_ret,
        )

        rolling_metrics = self.metrics_engine.rolling_metrics(net_ret)

        # ── 5. Factor Attribution ───────────────────────────────────────────────
        factor_attr = {}
        if factor_returns is not None:
            factor_attr = self.factor_engine.compute_factor_regression(net_ret, factor_returns)

        # ── 6. Report ───────────────────────────────────────────────────────────
        tearsheet = self.report_generator.build_tearsheet(metrics, rolling_metrics, factor_attr)
        self.report_generator.print_summary(metrics)

        if export_dir:
            self.report_generator.export_csv(metrics, output_dir=export_dir)
            self.report_generator.export_json(metrics, output_path=f"{export_dir}/backtest_summary.json")

        return BacktestResult(
            metrics=metrics,
            rolling_metrics=rolling_metrics,
            factor_attribution=factor_attr,
            tearsheet=tearsheet,
            holding_weights=holding_weights,
            portfolio_exposure=portfolio_exposure,
        )
