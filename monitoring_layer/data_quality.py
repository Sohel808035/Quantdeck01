"""
monitoring_layer/data_quality.py
──────────────────────────────────
Data Quality Monitor.
Checks for missing values, staleness, outliers, schema violations, and distributional anomalies.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Runs structural and statistical data quality checks on any DataFrame feed.
    Reports violations and fires alerts via AlertEngine.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self.cfg = self.config.data_quality

    def check(self, df: pd.DataFrame, feed_name: str = "feed") -> Dict[str, Any]:
        """
        Runs all data quality checks on a DataFrame.

        Args:
            df:         Input DataFrame to audit.
            feed_name:  Logical name for this feed (used in alert messages).

        Returns:
            Dict with quality report keyed by check name.
        """
        report: Dict[str, Any] = {"feed": feed_name, "checks": {}, "passed": True}

        if df is None or df.empty:
            self.alert_engine.fire(
                AlertSeverity.CRITICAL, "DATA_QUALITY", "empty_feed",
                value=0.0, threshold=self.cfg.min_rows,
                message=f"[{feed_name}] DataFrame is empty or None.",
            )
            report["passed"] = False
            return report

        # ── 1. Row count ────────────────────────────────────────────────────
        row_check = self._check_row_count(df, feed_name)
        report["checks"]["row_count"] = row_check

        # ── 2. Missing values ───────────────────────────────────────────────
        missing_check = self._check_missing(df, feed_name)
        report["checks"]["missing_values"] = missing_check

        # ── 3. Outliers (Z-score) ───────────────────────────────────────────
        outlier_check = self._check_outliers(df, feed_name)
        report["checks"]["outliers"] = outlier_check

        # ── 4. Staleness (last timestamp) ───────────────────────────────────
        stale_check = self._check_staleness(df, feed_name)
        report["checks"]["staleness"] = stale_check

        # ── 5. Duplicate rows ───────────────────────────────────────────────
        dup_check = self._check_duplicates(df, feed_name)
        report["checks"]["duplicates"] = dup_check

        # ── 6. Constant columns (zero variance) ────────────────────────────
        const_check = self._check_constant_columns(df, feed_name)
        report["checks"]["constant_columns"] = const_check

        # Overall pass/fail
        failed = [k for k, v in report["checks"].items() if not v.get("passed", True)]
        report["passed"] = len(failed) == 0
        report["failed_checks"] = failed
        return report

    def _check_row_count(self, df: pd.DataFrame, name: str) -> Dict:
        n = len(df)
        passed = n >= self.cfg.min_rows
        if not passed:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "DATA_QUALITY", "row_count",
                value=float(n), threshold=float(self.cfg.min_rows),
                message=f"[{name}] Only {n} rows found (min={self.cfg.min_rows}).",
            )
        return {"passed": passed, "n_rows": n}

    def _check_missing(self, df: pd.DataFrame, name: str) -> Dict:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"passed": True, "missing_pct": 0.0}
        missing_pct = float(numeric.isnull().mean().mean())
        passed = missing_pct <= self.cfg.max_missing_pct
        if not passed:
            sev = AlertSeverity.CRITICAL if missing_pct > 0.20 else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "DATA_QUALITY", "missing_pct",
                value=missing_pct, threshold=self.cfg.max_missing_pct,
                message=f"[{name}] Missing data {missing_pct:.1%} exceeds threshold.",
            )
        worst_col = numeric.isnull().mean().idxmax()
        return {"passed": passed, "missing_pct": missing_pct, "worst_column": worst_col}

    def _check_outliers(self, df: pd.DataFrame, name: str) -> Dict:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"passed": True, "outlier_count": 0}
        z = ((numeric - numeric.mean()) / (numeric.std() + 1e-8)).abs()
        outlier_count = int((z > self.cfg.zscore_outlier_threshold).sum().sum())
        total = numeric.shape[0] * numeric.shape[1]
        outlier_pct = outlier_count / max(total, 1)
        passed = outlier_pct < 0.01  # <1% outliers is acceptable
        if not passed:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "DATA_QUALITY", "outlier_pct",
                value=outlier_pct, threshold=0.01,
                message=f"[{name}] {outlier_count} outlier cells ({outlier_pct:.2%}).",
            )
        return {"passed": passed, "outlier_count": outlier_count, "outlier_pct": outlier_pct}

    def _check_staleness(self, df: pd.DataFrame, name: str) -> Dict:
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"passed": True, "stale_days": None}
        last_date = df.index.max()
        now = pd.Timestamp.now(tz=last_date.tzinfo)
        if last_date.tzinfo is None:
            now = pd.Timestamp.now()
        stale_days = (now - last_date).days
        passed = stale_days <= self.cfg.max_staleness_days
        if not passed:
            sev = AlertSeverity.CRITICAL if stale_days > 7 else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "DATA_QUALITY", "staleness_days",
                value=float(stale_days), threshold=float(self.cfg.max_staleness_days),
                message=f"[{name}] Data is {stale_days} days stale (last: {last_date.date()}).",
            )
        return {"passed": passed, "stale_days": stale_days, "last_date": str(last_date.date())}

    def _check_duplicates(self, df: pd.DataFrame, name: str) -> Dict:
        dup_count = int(df.index.duplicated().sum())
        passed = dup_count == 0
        if not passed:
            self.alert_engine.fire(
                AlertSeverity.WARNING, "DATA_QUALITY", "duplicate_rows",
                value=float(dup_count), threshold=0.0,
                message=f"[{name}] {dup_count} duplicate index entries found.",
            )
        return {"passed": passed, "duplicate_count": dup_count}

    def _check_constant_columns(self, df: pd.DataFrame, name: str) -> Dict:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"passed": True, "constant_columns": []}
        const_cols = list(numeric.columns[numeric.std() < 1e-10])
        passed = len(const_cols) == 0
        if not passed:
            self.alert_engine.fire(
                AlertSeverity.INFO, "DATA_QUALITY", "constant_columns",
                value=float(len(const_cols)), threshold=0.0,
                message=f"[{name}] Constant columns detected: {const_cols}.",
            )
        return {"passed": passed, "constant_columns": const_cols}
