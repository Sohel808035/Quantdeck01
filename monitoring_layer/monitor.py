"""
monitoring_layer/monitor.py
─────────────────────────────
Master MonitoringLayer Orchestrator.
Single entry point wiring all monitoring components.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, Alert
from monitoring_layer.data_quality import DataQualityMonitor
from monitoring_layer.drift import DriftMonitor
from monitoring_layer.system_health import SystemHealthMonitor
from monitoring_layer.strategy_monitor import StrategyMonitor
from monitoring_layer.logger import StructuredLogger
from monitoring_layer.dashboard import MonitoringDashboard

logger = logging.getLogger(__name__)


class MonitoringLayer:
    """
    QuantSphereX Monitoring Layer — single entry point for all system diagnostics.

    Components:
      - DataQualityMonitor  : Data schema, missing, staleness, outliers
      - DriftMonitor        : Feature and prediction distributional shift
      - SystemHealthMonitor : CPU, memory, latency, errors
      - StrategyMonitor     : Rolling Sharpe, Rolling IC, drawdown
      - AlertEngine         : Alert generation, deduplication, dispatch
      - StructuredLogger    : JSON-formatted rotating log
      - MonitoringDashboard : Rich terminal dashboard

    Usage:
        ml = MonitoringLayer()
        ml.check_data_quality(df, "prices")
        ml.check_system_health()
        ml.render_dashboard()
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.alert_engine = AlertEngine(config=self.config)
        self.data_quality = DataQualityMonitor(self.config, self.alert_engine)
        self.drift = DriftMonitor(self.config, self.alert_engine)
        self.system = SystemHealthMonitor(self.config, self.alert_engine)
        self.strategy = StrategyMonitor(self.config, self.alert_engine)
        self.slog = StructuredLogger(
            name=f"{self.config.service_name}.monitoring",
            config=self.config,
        )
        self.dashboard = MonitoringDashboard(
            service_name=f"{self.config.service_name} Monitoring Layer"
        )
        self._last_health: Dict[str, Any] = {}
        self._last_dq: Dict[str, Any] = {}
        self._last_drift: Dict[str, Any] = {}
        self._last_strategy: Dict[str, Any] = {}

    # ── Data Quality ─────────────────────────────────────────────────────────

    def check_data_quality(self, df: pd.DataFrame, feed_name: str = "feed") -> Dict[str, Any]:
        """Runs all data quality checks on the given DataFrame."""
        with self.system.track_latency(f"data_quality.{feed_name}"):
            report = self.data_quality.check(df, feed_name)
        self._last_dq = report
        self.slog.info("data_quality_check", feed=feed_name, passed=report.get("passed"), failed=report.get("failed_checks"))
        return report

    # ── Feature & Prediction Drift ────────────────────────────────────────────

    def check_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Checks for feature distributional drift between reference and current windows."""
        with self.system.track_latency("drift.feature"):
            report = self.drift.check_feature_drift(reference_df, current_df, feature_cols)
        self._last_drift = report
        self.slog.info("feature_drift_check", drifted_features=report.get("drifted_features"), drift_rate=report.get("drift_rate"))
        return report

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Checks for prediction score distributional drift."""
        with self.system.track_latency(f"drift.prediction.{model_name}"):
            report = self.drift.check_prediction_drift(reference_predictions, current_predictions, model_name)
        self.slog.info("prediction_drift_check", model=model_name, psi=report.get("psi"), drifted=report.get("drifted"))
        return report

    # ── System Health ─────────────────────────────────────────────────────────

    def check_system_health(self) -> Dict[str, Any]:
        """Checks current CPU, memory, and returns system health report."""
        report = self.system.check_cpu_memory()
        report["latency_summary"] = self.system.latency_summary()
        report["error_summary"] = self.system.error_summary()
        self._last_health = report
        self.slog.info("system_health_check", cpu_pct=report.get("cpu_pct"), memory_pct=report.get("memory_pct"))
        return report

    # ── Strategy Monitor ──────────────────────────────────────────────────────

    def check_rolling_sharpe(
        self, daily_returns: pd.Series, window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Checks rolling Sharpe ratio and fires alert if below threshold."""
        report = self.strategy.check_rolling_sharpe(daily_returns, window)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("rolling_sharpe_check", latest=report.get("latest_sharpe"), breach=report.get("breach"))
        return report

    def check_rolling_ic(
        self,
        alpha_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Checks rolling Information Coefficient and fires alert if below threshold."""
        report = self.strategy.check_rolling_ic(alpha_scores, forward_returns, window)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("rolling_ic_check", latest=report.get("latest_ic"), breach=report.get("breach"))
        return report

    def check_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Checks current drawdown and fires alert if breach threshold exceeded."""
        report = self.strategy.check_drawdown(equity_curve)
        self._last_strategy.update({k: v for k, v in report.items() if not isinstance(v, pd.Series)})
        self.slog.info("drawdown_check", current=report.get("current_drawdown"), breach=report.get("breach"))
        return report

    # ── Full Health Check ─────────────────────────────────────────────────────

    def full_health_check(
        self,
        data_df: Optional[pd.DataFrame] = None,
        daily_returns: Optional[pd.Series] = None,
        equity_curve: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Runs all monitoring checks in a single call.
        Returns a consolidated health report.
        """
        results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "service": self.config.service_name,
        }

        results["system_health"] = self.check_system_health()

        if data_df is not None:
            results["data_quality"] = self.check_data_quality(data_df)

        if daily_returns is not None:
            results["rolling_sharpe"] = self.check_rolling_sharpe(daily_returns)

        if equity_curve is not None:
            results["drawdown"] = self.check_drawdown(equity_curve)

        alert_summary = self.alert_engine.summary()
        results["alerts_summary"] = alert_summary

        # Determine overall health
        critical_count = alert_summary.get("CRITICAL", 0)
        warning_count = alert_summary.get("WARNING", 0)
        if critical_count > 0:
            results["overall_health"] = "CRITICAL"
        elif warning_count > 0:
            results["overall_health"] = "WARNING"
        else:
            results["overall_health"] = "HEALTHY"

        self.slog.info("full_health_check", overall=results["overall_health"])
        return results

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def render_dashboard(self) -> None:
        """Renders the monitoring dashboard using last-known state of all monitors."""
        self.dashboard.render(
            health_report=self._last_health,
            data_quality_report=self._last_dq,
            drift_report=self._last_drift,
            strategy_report=self._last_strategy,
            recent_alerts=self.alert_engine.recent_alerts(n=10),
        )

    # ── Alert Helpers ─────────────────────────────────────────────────────────

    def recent_alerts(self, n: int = 20) -> List[Alert]:
        return self.alert_engine.recent_alerts(n)

    def alert_summary(self) -> Dict[str, int]:
        return self.alert_engine.summary()
