"""
monitoring_layer/config.py
──────────────────────────
Monitoring Layer configuration DTO.
All thresholds, windows, and alert settings live here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DataQualityConfig:
    max_missing_pct: float = 0.05          # Flag if > 5% of values missing
    max_staleness_days: int = 3            # Flag if data not updated for N days
    zscore_outlier_threshold: float = 5.0  # Z-score beyond which a value is an outlier
    min_rows: int = 10                     # Minimum rows expected in any feed


@dataclass
class DriftConfig:
    psi_warning_threshold: float = 0.1    # PSI: 0.1–0.2 = slight shift
    psi_critical_threshold: float = 0.2   # PSI: > 0.2 = significant shift
    ks_pvalue_threshold: float = 0.05     # KS test p-value below = drift detected
    rolling_window_days: int = 63         # Baseline window (quarterly)
    min_samples_for_drift: int = 30       # Minimum observations to run drift test


@dataclass
class SystemHealthConfig:
    cpu_warning_pct: float = 75.0         # CPU % above which warns
    cpu_critical_pct: float = 90.0        # CPU % above which is critical
    memory_warning_pct: float = 80.0      # Memory % above which warns
    memory_critical_pct: float = 95.0     # Memory % above which is critical
    latency_warning_ms: float = 500.0     # Latency (ms) warning threshold
    latency_critical_ms: float = 2000.0   # Latency (ms) critical threshold
    error_rate_warning: float = 0.01      # 1% error rate triggers warning
    error_rate_critical: float = 0.05     # 5% error rate is critical


@dataclass
class StrategyMonitorConfig:
    min_sharpe: float = 0.5               # Rolling Sharpe below = alert
    min_ic: float = 0.02                  # Rolling IC below = alert
    max_drawdown_breach: float = -0.20    # Drawdown beyond -20% = alert
    sharpe_window: int = 63              # Days for rolling Sharpe
    ic_window: int = 21                  # Days for rolling IC


@dataclass
class AlertConfig:
    enable_console: bool = True
    enable_file: bool = True
    log_dir: str = "logs/monitoring"
    max_alerts_per_hour: int = 100        # Rate-limit alerts


@dataclass
class MonitoringConfig:
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    system: SystemHealthConfig = field(default_factory=SystemHealthConfig)
    strategy: StrategyMonitorConfig = field(default_factory=StrategyMonitorConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    environment: str = "production"
    service_name: str = "QuantSphereX"
