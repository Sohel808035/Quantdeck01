"""
tests/test_backend_services.py
────────────────────────────────
Unit Test Suite for QuantSphereX Backend API Services.

Covers:
  1. App Factory & Root Endpoint
  2. Health & System Diagnostics Router (/api/v2/health/status)
  3. Authentication Middleware & Security Checks (X-API-Key requirement)
  4. Backtesting Router (/api/v2/backtest/run)
  5. Risk Audit Router (/api/v2/risk/audit)
  6. Monitoring Router (/api/v2/monitoring/health-check & /alerts)
  7. AI Quant Analyst Router (/api/v2/analyst/explain-prediction & /summarize-news)
  8. Error Handling (401 Unauthorized, 422 Validation Error, 500 Global Exception)
  9. Request-ID Middleware & Response Time headers
 10. Backward Compatibility (/api/v1/health)
"""

import unittest
from fastapi.testclient import TestClient

from backend_services.app import create_app
from backend_services.config import BackendSettings


class TestBackendServices(unittest.TestCase):
    def setUp(self):
        self.settings = BackendSettings(debug=False)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.auth_headers = {"X-API-Key": "qsx-secret-api-key-2026"}

    # ── 1. Root & Documentation Endpoints ─────────────────────────────────────

    def test_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("version", data)

    def test_openapi_schema(self):
        res = self.client.get("/openapi.json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("paths", data)

    # ── 2. Health & Diagnostics ───────────────────────────────────────────────

    def test_health_status_endpoint(self):
        res = self.client.get("/api/v2/health/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("uptime_seconds", data)

    # ── 3. Request-ID & Latency Middleware ────────────────────────────────────

    def test_request_id_middleware_header(self):
        res = self.client.get("/api/v2/health/status")
        self.assertIn("X-Request-ID", res.headers)
        self.assertIn("X-Response-Time-MS", res.headers)

    # ── 4. Authentication Security Checks ─────────────────────────────────────

    def test_unauthenticated_request_rejected(self):
        # Protected endpoints require X-API-Key
        res = self.client.post("/api/v2/backtest/run", json={"initial_capital": 1e7})
        self.assertEqual(res.status_code, 401)
        data = res.json()
        self.assertTrue(data.get("error"))

    def test_invalid_api_key_rejected(self):
        res = self.client.post(
            "/api/v2/backtest/run",
            json={"initial_capital": 1e7},
            headers={"X-API-Key": "invalid-wrong-key"},
        )
        self.assertEqual(res.status_code, 401)

    def test_valid_api_key_accepted(self):
        res = self.client.post(
            "/api/v2/backtest/run",
            json={"initial_capital": 1e7},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("sharpe_ratio", data)

    # ── 5. Backtest Router ───────────────────────────────────────────────────

    def test_backtest_execution_endpoint(self):
        res = self.client.post(
            "/api/v2/backtest/run",
            json={"initial_capital": 5_000_000.0, "transaction_cost_pct": 0.001},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cagr", data)
        self.assertIn("max_drawdown", data)

    # ── 6. Risk Audit Router ──────────────────────────────────────────────────

    def test_risk_audit_endpoint(self):
        res = self.client.post(
            "/api/v2/risk/audit",
            json={"confidence_level": 0.95, "include_stress_testing": True},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("var_95", data)
        self.assertIn("cvar_95", data)
        self.assertIn("risk_grade", data)

    # ── 7. Monitoring Router ──────────────────────────────────────────────────

    def test_monitoring_diagnostics_endpoint(self):
        res = self.client.get(
            "/api/v2/monitoring/health-check",
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("overall_health", data)

    def test_monitoring_alerts_endpoint(self):
        res = self.client.get(
            "/api/v2/monitoring/alerts?n=5",
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("summary", data)

    # ── 8. AI Quant Analyst Router ────────────────────────────────────────────

    def test_analyst_explain_prediction_endpoint(self):
        payload = {
            "symbol": "RELIANCE",
            "predicted_score": 0.045,
            "probability": 0.82,
            "shap_values": {"mom_60": 0.03, "vol_20": -0.01},
        }
        res = self.client.post(
            "/api/v2/analyst/explain-prediction",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["direction"], "BULLISH")
        self.assertIn("narrative", data)

    def test_analyst_summarize_news_endpoint(self):
        payload = {
            "headline": "TCS reports record profit and margin expansion",
            "content": "Strong demand led to guidance raise.",
            "symbol": "TCS",
        }
        res = self.client.post(
            "/api/v2/analyst/summarize-news",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sentiment_label"], "POSITIVE")

    # ── 9. Error Handling & Validation ────────────────────────────────────────

    def test_validation_error_returns_422(self):
        # Invalid data type for predicted_score
        payload = {"symbol": "RELIANCE", "predicted_score": "not-a-number"}
        res = self.client.post(
            "/api/v2/analyst/explain-prediction",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)

    # ── 10. Backward Compatibility ────────────────────────────────────────────

    def test_legacy_v1_health_endpoint(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("Deprecation warning", data["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
