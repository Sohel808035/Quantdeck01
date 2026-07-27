"""
tests/test_monitoring_layer.py
────────────────────────────────
Unit Test Suite for QuantSphereX Monitoring Layer.

Covers:
  1.  AlertEngine        - Fire, cooldown, rate limiting, dedup, summary
  2.  DataQualityMonitor - Missing, outlier, staleness, duplicates, constants
  3.  DriftMonitor       - PSI computation, feature drift, prediction drift
  4.  SystemHealthMonitor- CPU/memory checks, latency tracking, error rate
  5.  StrategyMonitor    - Rolling Sharpe, Rolling IC, drawdown checks
  6.  StructuredLogger   - Logger creation, JSON format
  7.  MonitoringDashboard- Plain-text render (no rich required)
  8.  MonitoringLayer    - Full orchestration, full_health_check, alerts
"""

import os
import time
import logging
import tempfile
import unittest
import numpy as np
import pandas as pd

from monitoring_layer.config import MonitoringConfig, AlertConfig
from monitoring_layer.alert_engine import AlertEngine, Alert, AlertSeverity
from monitoring_layer.data_quality import DataQualityMonitor
from monitoring_layer.drift import DriftMonitor, _compute_psi
from monitoring_layer.system_health import SystemHealthMonitor
from monitoring_layer.strategy_monitor import StrategyMonitor
from monitoring_layer.logger import StructuredLogger, build_monitoring_logger
from monitoring_layer.dashboard import MonitoringDashboard
from monitoring_layer.monitor import MonitoringLayer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(log_dir=None):
    cfg = MonitoringConfig()
    if log_dir:
        cfg.alerts.log_dir = log_dir
    return cfg


def _make_returns(n=252, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)


def _make_df(n_rows=100, n_cols=5, seed=42):
    np.random.seed(seed)
    end_date = pd.Timestamp.now()
    dates = pd.date_range(end=end_date, periods=n_rows, freq="B")
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(np.random.randn(n_rows, n_cols), index=dates, columns=cols)


# ─── 1. AlertEngine ──────────────────────────────────────────────────────────

class TestAlertEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine()
        self.engine.clear()

    def test_fire_returns_alert(self):
        a = self.engine.fire(
            AlertSeverity.WARNING, "TEST", "metric", 0.5, 0.3, "test message"
        )
        self.assertIsInstance(a, Alert)
        self.assertEqual(a.severity, AlertSeverity.WARNING)

    def test_cooldown_deduplication(self):
        self.engine.fire(AlertSeverity.WARNING, "TEST", "metric", 0.5, 0.3, "first")
        second = self.engine.fire(AlertSeverity.WARNING, "TEST", "metric", 0.6, 0.3, "second")
        self.assertIsNone(second)  # Suppressed by cooldown

    def test_different_metric_not_suppressed(self):
        self.engine.fire(AlertSeverity.WARNING, "TEST", "metric_a", 0.5, 0.3, "msg a")
        b = self.engine.fire(AlertSeverity.WARNING, "TEST", "metric_b", 0.5, 0.3, "msg b")
        self.assertIsNotNone(b)  # Different metric — should fire

    def test_summary_counts(self):
        self.engine.fire(AlertSeverity.WARNING, "A", "m1", 0.5, 0.3, "w1")
        self.engine.fire(AlertSeverity.CRITICAL, "A", "m2", 0.9, 0.8, "c1")
        s = self.engine.summary()
        self.assertEqual(s.get("WARNING", 0), 1)
        self.assertEqual(s.get("CRITICAL", 0), 1)

    def test_recent_alerts_order(self):
        self.engine.fire(AlertSeverity.INFO, "A", "m1", 0.1, 0.0, "first")
        self.engine.fire(AlertSeverity.INFO, "A", "m2", 0.2, 0.0, "second")
        recent = self.engine.recent_alerts(n=2)
        self.assertEqual(len(recent), 2)

    def test_custom_handler_called(self):
        received = []
        self.engine.register_handler(lambda a: received.append(a))
        self.engine.fire(AlertSeverity.INFO, "A", "m9", 0.1, 0.0, "test")
        self.assertEqual(len(received), 1)

    def test_alert_to_dict(self):
        a = self.engine.fire(AlertSeverity.CRITICAL, "B", "m8", 0.9, 0.5, "critical!")
        d = a.to_dict()
        self.assertIn("severity", d)
        self.assertIn("timestamp", d)
        self.assertEqual(d["severity"], "CRITICAL")


# ─── 2. DataQualityMonitor ────────────────────────────────────────────────────

class TestDataQualityMonitor(unittest.TestCase):
    def setUp(self):
        self.dq = DataQualityMonitor()

    def test_clean_df_passes(self):
        df = _make_df(n_rows=100)
        report = self.dq.check(df, "clean_feed")
        self.assertTrue(report["passed"])

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        report = self.dq.check(df, "empty_feed")
        self.assertFalse(report["passed"])

    def test_missing_values_detected(self):
        df = _make_df(n_rows=100)
        df.iloc[:50, :] = np.nan  # 50% missing
        report = self.dq.check(df, "missing_feed")
        self.assertIn("missing_values", report["checks"])
        self.assertFalse(report["checks"]["missing_values"]["passed"])

    def test_duplicate_index_detected(self):
        df = _make_df(n_rows=50)
        df = pd.concat([df, df])  # Duplicate all rows
        report = self.dq.check(df, "dup_feed")
        self.assertFalse(report["checks"]["duplicates"]["passed"])

    def test_constant_column_detected(self):
        df = _make_df(n_rows=50)
        df["const_col"] = 5.0  # Constant column
        report = self.dq.check(df, "const_feed")
        self.assertFalse(report["checks"]["constant_columns"]["passed"])
        self.assertIn("const_col", report["checks"]["constant_columns"]["constant_columns"])

    def test_outlier_detection(self):
        df = _make_df(n_rows=100)
        df.iloc[0, 0] = 1e9  # Extreme outlier
        report = self.dq.check(df, "outlier_feed")
        self.assertIn("outliers", report["checks"])

    def test_report_has_feed_name(self):
        df = _make_df()
        report = self.dq.check(df, "my_feed")
        self.assertEqual(report["feed"], "my_feed")


# ─── 3. DriftMonitor ─────────────────────────────────────────────────────────

class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.reference = pd.DataFrame(
            np.random.randn(200, 3), columns=["f1", "f2", "f3"],
            index=pd.date_range("2021-01-01", periods=200, freq="B")
        )

    def test_psi_zero_for_identical_distributions(self):
        psi = _compute_psi(np.random.randn(500), np.random.randn(500))
        # PSI for same distribution should be near 0
        self.assertLess(psi, 0.2)

    def test_psi_high_for_shifted_distribution(self):
        ref = np.random.randn(500)
        cur = np.random.randn(500) + 5.0  # Huge shift
        psi = _compute_psi(ref, cur)
        self.assertGreater(psi, 0.2)

    def test_feature_drift_no_drift(self):
        dm = DriftMonitor()
        current = pd.DataFrame(
            np.random.randn(200, 3), columns=["f1", "f2", "f3"],
            index=pd.date_range("2022-01-01", periods=200, freq="B")
        )
        report = dm.check_feature_drift(self.reference, current)
        self.assertIn("features", report)
        self.assertIn("drift_rate", report)

    def test_feature_drift_detects_shift(self):
        dm = DriftMonitor()
        shifted = pd.DataFrame(
            np.random.randn(200, 3) + 10.0, columns=["f1", "f2", "f3"],
            index=pd.date_range("2022-01-01", periods=200, freq="B")
        )
        report = dm.check_feature_drift(self.reference, shifted)
        # With a shift of 10 std devs, at least one feature should drift
        self.assertGreater(len(report["drifted_features"]), 0)

    def test_prediction_drift_no_drift(self):
        dm = DriftMonitor()
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(200)
        report = dm.check_prediction_drift(ref_preds, cur_preds, model_name="xgboost")
        self.assertIn("psi", report)
        self.assertIn("drifted", report)

    def test_prediction_drift_detects_shift(self):
        dm = DriftMonitor()
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(200) + 8.0  # Large shift
        report = dm.check_prediction_drift(ref_preds, cur_preds, model_name="xgboost")
        self.assertTrue(report["drifted"])

    def test_insufficient_data_returns_gracefully(self):
        dm = DriftMonitor()
        ref = np.random.randn(5)  # Too few samples
        cur = np.random.randn(5)
        report = dm.check_prediction_drift(ref, cur)
        self.assertEqual(report.get("status"), "insufficient_data")


# ─── 4. SystemHealthMonitor ───────────────────────────────────────────────────

class TestSystemHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.sm = SystemHealthMonitor()

    def test_cpu_memory_returns_dict(self):
        report = self.sm.check_cpu_memory()
        self.assertIsInstance(report, dict)

    def test_latency_track_records_measurement(self):
        with self.sm.track_latency("test_op"):
            time.sleep(0.01)
        summary = self.sm.latency_summary("test_op")
        self.assertIn("mean_ms", summary)
        self.assertGreater(summary["mean_ms"], 0)

    def test_latency_percentiles_available(self):
        for _ in range(10):
            with self.sm.track_latency("perf_test"):
                time.sleep(0.001)
        summary = self.sm.latency_summary("perf_test")
        self.assertIn("p95_ms", summary)
        self.assertIn("p99_ms", summary)

    def test_error_recording(self):
        self.sm._call_counts["my_op"] = 100
        self.sm.record_error("my_op")
        summary = self.sm.error_summary()
        self.assertIn("my_op", summary)
        self.assertGreater(summary["my_op"]["errors"], 0)

    def test_latency_track_records_exception(self):
        try:
            with self.sm.track_latency("failing_op"):
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        # Error should be recorded
        summary = self.sm.error_summary()
        self.assertIn("failing_op", summary)


# ─── 5. StrategyMonitor ──────────────────────────────────────────────────────

class TestStrategyMonitor(unittest.TestCase):
    def setUp(self):
        self.returns = _make_returns(n=252)
        self.equity = (1_000_000 * (1 + self.returns).cumprod())

    def test_rolling_sharpe_returns_dict(self):
        sm = StrategyMonitor()
        result = sm.check_rolling_sharpe(self.returns, window=63)
        self.assertIn("latest_sharpe", result)
        self.assertIn("breach", result)

    def test_rolling_sharpe_breach_for_low_returns(self):
        sm = StrategyMonitor()
        bad_returns = pd.Series(-0.01, index=self.returns.index)
        result = sm.check_rolling_sharpe(bad_returns, window=63)
        self.assertTrue(result["breach"])

    def test_drawdown_check_returns_metrics(self):
        sm = StrategyMonitor()
        result = sm.check_drawdown(self.equity)
        self.assertIn("current_drawdown", result)
        self.assertIn("max_drawdown", result)
        self.assertLessEqual(result["max_drawdown"], 0.0)

    def test_drawdown_breach_fires_for_deep_dd(self):
        sm = StrategyMonitor()
        crashing = pd.Series(
            [1_000_000 * (0.5 ** (i / 50)) for i in range(100)],
            index=pd.date_range("2022-01-01", periods=100, freq="B")
        )
        result = sm.check_drawdown(crashing)
        self.assertTrue(result["breach"])

    def test_rolling_ic_returns_dict(self):
        sm = StrategyMonitor()
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        tickers = ["A", "B", "C", "D", "E"]
        alpha_scores = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=tickers)
        fwd_rets = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=tickers)
        result = sm.check_rolling_ic(alpha_scores, fwd_rets, window=21)
        self.assertIn("latest_ic", result)

    def test_insufficient_data_handled(self):
        sm = StrategyMonitor()
        short_returns = _make_returns(n=10)
        result = sm.check_rolling_sharpe(short_returns, window=63)
        self.assertIn("status", result)


# ─── 6. StructuredLogger ─────────────────────────────────────────────────────

class TestStructuredLogger(unittest.TestCase):
    def tearDown(self):
        # Close file handlers to release Windows locks
        for name in ["test.logger", "test.monitor", "file.test", "quantspherex.monitoring"]:
            logger = logging.getLogger(name)
            for h in list(logger.handlers):
                h.close()
                logger.removeHandler(h)

    def test_logger_creates_without_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            slog = StructuredLogger("test.logger", config=cfg)
            slog.info("test_event", key1="val1", key2=42)
            # Close handlers inside context to release lock
            for h in list(slog._logger.handlers):
                h.close()
                slog._logger.removeHandler(h)

    def test_build_monitoring_logger_returns_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            lg = build_monitoring_logger("test.monitor", config=cfg)
            self.assertIsInstance(lg, logging.Logger)
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)

    def test_log_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(log_dir=tmpdir)
            slog = StructuredLogger("file.test", config=cfg)
            slog.warning("warn_event", value=99)
            files = os.listdir(tmpdir)
            self.assertTrue(any(".log" in f for f in files))
            for h in list(slog._logger.handlers):
                h.close()
                slog._logger.removeHandler(h)


# ─── 7. MonitoringDashboard ──────────────────────────────────────────────────

class TestMonitoringDashboard(unittest.TestCase):
    def test_plain_render_no_exception(self):
        """Dashboard should render without errors even without 'rich' library."""
        db = MonitoringDashboard.__new__(MonitoringDashboard)
        db.service_name = "Test"
        db._rich = None  # Force plain-text mode
        # Should not raise
        db._render_plain(
            health={"cpu_pct": 45.0, "memory_pct": 60.0},
            dq={"checks": {"missing_values": {"passed": True}}},
            drift={"features": {"f1": {"psi": 0.05, "ks_pvalue": 0.3, "drifted": False}}},
            strategy={"latest_sharpe": 0.85, "breach": False},
            alerts=[],
        )

    def test_dashboard_init(self):
        db = MonitoringDashboard(service_name="TestSvc")
        self.assertEqual(db.service_name, "TestSvc")


# ─── 8. MonitoringLayer (Full Orchestration) ──────────────────────────────────

class TestMonitoringLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = MonitoringConfig()
        self.cfg.alerts.log_dir = self.tmp
        self.cfg.alerts.enable_console = False
        self.ml = MonitoringLayer(config=self.cfg)
        self.returns = _make_returns(n=252)
        self.equity = 1_000_000 * (1 + self.returns).cumprod()
        self.df = _make_df(n_rows=100)

    def tearDown(self):
        if hasattr(self.ml, 'slog'):
            for h in list(self.ml.slog._logger.handlers):
                h.close()
                self.ml.slog._logger.removeHandler(h)

    def test_check_data_quality_passes_clean(self):
        report = self.ml.check_data_quality(self.df, "test_feed")
        self.assertIn("passed", report)

    def test_check_rolling_sharpe(self):
        result = self.ml.check_rolling_sharpe(self.returns, window=63)
        self.assertIn("latest_sharpe", result)

    def test_check_drawdown(self):
        result = self.ml.check_drawdown(self.equity)
        self.assertIn("current_drawdown", result)

    def test_check_feature_drift(self):
        ref = _make_df(n_rows=200)
        cur = _make_df(n_rows=100, seed=99)
        result = self.ml.check_feature_drift(ref, cur)
        self.assertIn("drift_rate", result)

    def test_check_prediction_drift(self):
        ref_preds = np.random.randn(200)
        cur_preds = np.random.randn(100)
        result = self.ml.check_prediction_drift(ref_preds, cur_preds, "test_model")
        self.assertIn("drifted", result)

    def test_full_health_check(self):
        result = self.ml.full_health_check(
            data_df=self.df,
            daily_returns=self.returns,
            equity_curve=self.equity,
        )
        self.assertIn("overall_health", result)
        self.assertIn(result["overall_health"], ["HEALTHY", "WARNING", "CRITICAL"])

    def test_alert_summary_returns_dict(self):
        summary = self.ml.alert_summary()
        self.assertIsInstance(summary, dict)

    def test_recent_alerts_returns_list(self):
        alerts = self.ml.recent_alerts(n=5)
        self.assertIsInstance(alerts, list)

    def test_render_dashboard_no_exception(self):
        self.ml._last_health = {"cpu_pct": 30.0, "memory_pct": 50.0}
        self.ml.dashboard._rich = None  # Force plain-text
        self.ml.render_dashboard()  # Should not raise

    def test_latency_tracked_during_dq_check(self):
        self.ml.check_data_quality(self.df, "latency_test")
        summary = self.ml.system.latency_summary()
        self.assertGreater(summary.get("count", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
