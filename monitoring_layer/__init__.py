"""
QuantSphereX Monitoring Layer.
Provides unified monitoring for data quality, feature/prediction drift,
system health (CPU/memory/latency/errors), strategy performance (Sharpe/IC/drawdown),
alert generation, structured logging, and terminal dashboard.
"""

from monitoring_layer.config import (
    MonitoringConfig,
    DataQualityConfig,
    DriftConfig,
    SystemHealthConfig,
    StrategyMonitorConfig,
    AlertConfig,
)
from monitoring_layer.alert_engine import AlertEngine, Alert, AlertSeverity
from monitoring_layer.data_quality import DataQualityMonitor
from monitoring_layer.drift import DriftMonitor
from monitoring_layer.system_health import SystemHealthMonitor
from monitoring_layer.strategy_monitor import StrategyMonitor
from monitoring_layer.logger import StructuredLogger, build_monitoring_logger
from monitoring_layer.dashboard import MonitoringDashboard
from monitoring_layer.monitor import MonitoringLayer

__all__ = [
    # Config
    "MonitoringConfig",
    "DataQualityConfig",
    "DriftConfig",
    "SystemHealthConfig",
    "StrategyMonitorConfig",
    "AlertConfig",
    # Alerts
    "AlertEngine",
    "Alert",
    "AlertSeverity",
    # Monitors
    "DataQualityMonitor",
    "DriftMonitor",
    "SystemHealthMonitor",
    "StrategyMonitor",
    # Logging
    "StructuredLogger",
    "build_monitoring_logger",
    # Dashboard
    "MonitoringDashboard",
    # Master
    "MonitoringLayer",
]
