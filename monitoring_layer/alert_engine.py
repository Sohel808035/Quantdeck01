"""
monitoring_layer/alert_engine.py
──────────────────────────────────
Alert Engine — Severity classification, deduplication, rate limiting, and dispatch.
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from monitoring_layer.config import MonitoringConfig

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Represents a single monitoring alert."""
    severity: AlertSeverity
    category: str        # e.g. 'DATA_QUALITY', 'DRIFT', 'SYSTEM', 'STRATEGY'
    metric: str          # e.g. 'missing_pct', 'cpu_usage', 'rolling_sharpe'
    value: float         # The actual observed value
    threshold: float     # The breached threshold
    message: str         # Human-readable description
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "metric": self.metric,
            "value": round(self.value, 6),
            "threshold": round(self.threshold, 6),
            "message": self.message,
            "timestamp": self.timestamp,
        }


class AlertEngine:
    """
    Manages alert generation with:
      - Severity levels (INFO / WARNING / CRITICAL)
      - Per-metric deduplication (suppress repeat alerts within cooldown)
      - Hourly rate limiting
      - Pluggable notification handlers
    """

    COOLDOWN_SECONDS = 300  # 5-minute cooldown per metric per severity

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._alert_history: List[Alert] = []
        self._last_fired: Dict[str, float] = {}   # key = "category.metric.severity"
        self._hourly_bucket: deque = deque()       # timestamps within last hour
        self._handlers: List[Callable[[Alert], None]] = []

        if self.config.alerts.enable_console:
            self._handlers.append(self._console_handler)

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Attach a custom notification handler (e.g. Slack webhook, PagerDuty)."""
        self._handlers.append(handler)

    def fire(
        self,
        severity: AlertSeverity,
        category: str,
        metric: str,
        value: float,
        threshold: float,
        message: str,
    ) -> Optional[Alert]:
        """
        Creates and dispatches an alert, subject to rate limiting and deduplication.

        Returns:
            The Alert object if fired, or None if suppressed.
        """
        now = time.time()

        # 1. Cooldown deduplication
        key = f"{category}.{metric}.{severity.value}"
        last = self._last_fired.get(key, 0.0)
        if now - last < self.COOLDOWN_SECONDS:
            return None

        # 2. Hourly rate limiting
        self._hourly_bucket = deque(
            t for t in self._hourly_bucket if now - t < 3600
        )
        if len(self._hourly_bucket) >= self.config.alerts.max_alerts_per_hour:
            logger.warning("[AlertEngine] Hourly rate limit reached — alert suppressed.")
            return None

        alert = Alert(
            severity=severity, category=category, metric=metric,
            value=value, threshold=threshold, message=message, timestamp=now,
        )

        self._last_fired[key] = now
        self._hourly_bucket.append(now)
        self._alert_history.append(alert)

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"[AlertEngine] Handler error: {e}")

        return alert

    def _console_handler(self, alert: Alert) -> None:
        icons = {AlertSeverity.INFO: "ℹ️", AlertSeverity.WARNING: "⚠️", AlertSeverity.CRITICAL: "🚨"}
        icon = icons.get(alert.severity, "")
        logger.warning(
            f"{icon} [{alert.severity.value}] [{alert.category}] "
            f"{alert.metric}={alert.value:.4f} (threshold={alert.threshold}) | {alert.message}"
        )

    def recent_alerts(self, n: int = 20) -> List[Alert]:
        """Returns the N most recent alerts."""
        return list(reversed(self._alert_history[-n:]))

    def summary(self) -> Dict[str, int]:
        """Returns count of alerts by severity."""
        counts: Dict[str, int] = defaultdict(int)
        for a in self._alert_history:
            counts[a.severity.value] += 1
        return dict(counts)

    def clear(self) -> None:
        """Clears all alert history (for testing)."""
        self._alert_history.clear()
        self._last_fired.clear()
        self._hourly_bucket.clear()
