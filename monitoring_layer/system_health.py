"""
monitoring_layer/system_health.py
──────────────────────────────────
System Health Monitor.
Tracks CPU, memory, per-operation latency, error rates, and process-level diagnostics.
"""

from __future__ import annotations
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generator, List, Optional

import numpy as np

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    operation: str
    latency_ms: float
    timestamp: float
    success: bool


class SystemHealthMonitor:
    """
    Monitors compute resource usage and application performance:
      - CPU and memory usage via psutil (optional dependency)
      - Per-operation latency tracking with @contextmanager
      - Error rate tracking across named operations
      - System-level health summary
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self.cfg = self.config.system
        self._latency_log: Deque[LatencyRecord] = deque(maxlen=10_000)
        self._error_counts: Dict[str, int] = {}
        self._call_counts: Dict[str, int] = {}

    # ── CPU & Memory ────────────────────────────────────────────────────────

    def check_cpu_memory(self) -> Dict[str, Any]:
        """Reads current CPU and memory usage. Requires psutil."""
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)

            if cpu_pct >= self.cfg.cpu_critical_pct:
                self.alert_engine.fire(
                    AlertSeverity.CRITICAL, "SYSTEM", "cpu_pct",
                    value=cpu_pct, threshold=self.cfg.cpu_critical_pct,
                    message=f"CPU usage critical: {cpu_pct:.1f}%",
                )
            elif cpu_pct >= self.cfg.cpu_warning_pct:
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "SYSTEM", "cpu_pct",
                    value=cpu_pct, threshold=self.cfg.cpu_warning_pct,
                    message=f"CPU usage elevated: {cpu_pct:.1f}%",
                )

            if mem_pct >= self.cfg.memory_critical_pct:
                self.alert_engine.fire(
                    AlertSeverity.CRITICAL, "SYSTEM", "memory_pct",
                    value=mem_pct, threshold=self.cfg.memory_critical_pct,
                    message=f"Memory usage critical: {mem_pct:.1f}% ({mem_used_gb:.2f}GB / {mem_total_gb:.2f}GB)",
                )
            elif mem_pct >= self.cfg.memory_warning_pct:
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "SYSTEM", "memory_pct",
                    value=mem_pct, threshold=self.cfg.memory_warning_pct,
                    message=f"Memory usage elevated: {mem_pct:.1f}%",
                )

            return {
                "cpu_pct": cpu_pct,
                "memory_pct": mem_pct,
                "memory_used_gb": round(mem_used_gb, 3),
                "memory_total_gb": round(mem_total_gb, 3),
                "cpu_ok": cpu_pct < self.cfg.cpu_warning_pct,
                "memory_ok": mem_pct < self.cfg.memory_warning_pct,
            }
        except ImportError:
            logger.warning("[SystemHealth] psutil not installed — skipping CPU/memory check.")
            return {"cpu_pct": None, "memory_pct": None, "psutil_available": False}

    # ── Latency Tracking ────────────────────────────────────────────────────

    @contextmanager
    def track_latency(self, operation: str) -> Generator[None, None, None]:
        """
        Context manager that measures execution time of a block and fires alerts if slow.

        Usage:
            with monitor.track_latency("data_fetch"):
                data = fetch_data()
        """
        t0 = time.perf_counter()
        success = True
        try:
            yield
        except Exception:
            success = False
            self.record_error(operation)
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            record = LatencyRecord(
                operation=operation,
                latency_ms=latency_ms,
                timestamp=time.time(),
                success=success,
            )
            self._latency_log.append(record)
            self._call_counts[operation] = self._call_counts.get(operation, 0) + 1

            if latency_ms >= self.cfg.latency_critical_ms:
                self.alert_engine.fire(
                    AlertSeverity.CRITICAL, "SYSTEM", f"latency.{operation}",
                    value=latency_ms, threshold=self.cfg.latency_critical_ms,
                    message=f"Critical latency: '{operation}' took {latency_ms:.0f}ms.",
                )
            elif latency_ms >= self.cfg.latency_warning_ms:
                self.alert_engine.fire(
                    AlertSeverity.WARNING, "SYSTEM", f"latency.{operation}",
                    value=latency_ms, threshold=self.cfg.latency_warning_ms,
                    message=f"Slow operation: '{operation}' took {latency_ms:.0f}ms.",
                )

    def record_error(self, operation: str) -> None:
        """Records a failed operation call."""
        self._error_counts[operation] = self._error_counts.get(operation, 0) + 1
        total = self._call_counts.get(operation, 1)
        errors = self._error_counts[operation]
        error_rate = errors / max(total, 1)

        if error_rate >= self.cfg.error_rate_critical:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "SYSTEM", f"error_rate.{operation}",
                value=error_rate, threshold=self.cfg.error_rate_critical,
                message=f"Critical error rate for '{operation}': {error_rate:.1%} ({errors}/{total})",
            )
        elif error_rate >= self.cfg.error_rate_warning:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "SYSTEM", f"error_rate.{operation}",
                value=error_rate, threshold=self.cfg.error_rate_warning,
                message=f"Elevated error rate for '{operation}': {error_rate:.1%}",
            )

    def latency_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Returns latency percentile summary for a specific operation or all operations."""
        records = list(self._latency_log)
        if operation:
            records = [r for r in records if r.operation == operation]
        if not records:
            return {}
        latencies = [r.latency_ms for r in records]
        return {
            "operation": operation or "all",
            "count": len(latencies),
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
        }

    def error_summary(self) -> Dict[str, Dict[str, Any]]:
        """Returns error rates per operation."""
        result = {}
        for op, errors in self._error_counts.items():
            calls = self._call_counts.get(op, errors)
            result[op] = {
                "errors": errors,
                "calls": calls,
                "error_rate": round(errors / max(calls, 1), 4),
            }
        return result
