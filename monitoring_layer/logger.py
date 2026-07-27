"""
monitoring_layer/logger.py
──────────────────────────
Structured JSON Logger.
Provides rotating file handler with JSON-formatted log records for the monitoring layer.
"""

from __future__ import annotations
import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from monitoring_layer.config import MonitoringConfig


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON strings for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_object.update(record.extra)
        return json.dumps(log_object)


def build_monitoring_logger(
    name: str = "quantspherex.monitoring",
    config: Optional[MonitoringConfig] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates and returns a logger with:
      - Rotating JSON file handler (10MB max, 5 backups)
      - Console handler with readable formatting

    Args:
        name:   Logger name (used as namespace).
        config: MonitoringConfig (reads log_dir from alerts config).
        level:  Logging level.

    Returns:
        Configured Logger instance.
    """
    cfg = config or MonitoringConfig()
    log_dir = cfg.alerts.log_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(log_dir, "monitoring.json.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # ── JSON rotating file handler ─────────────────────────────────────
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        # ── Human-readable console handler ─────────────────────────────────
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)

    return logger


class StructuredLogger:
    """
    Thin wrapper around a standard Logger that supports structured key-value context.

    Usage:
        slog = StructuredLogger("monitor.data_quality")
        slog.info("check_complete", feed="prices", passed=True, missing_pct=0.002)
    """

    def __init__(self, name: str, config: Optional[MonitoringConfig] = None):
        self._logger = build_monitoring_logger(name, config)

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        msg = json.dumps({"event": event, **kwargs})
        self._logger.log(level, msg)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, event, **kwargs)
