"""
QuantSphereX Backend Services Package.
"""

from backend_services.config import BackendSettings
from backend_services.errors import QuantBackendError, AuthenticationError, EngineExecutionError
from backend_services.auth import verify_api_key, verify_token_or_key
from backend_services.dependencies import get_settings, get_backtest_engine, get_risk_engine, get_monitoring_layer
from backend_services.app import create_app, app

__all__ = [
    "BackendSettings",
    "QuantBackendError",
    "AuthenticationError",
    "EngineExecutionError",
    "verify_api_key",
    "verify_token_or_key",
    "get_settings",
    "get_backtest_engine",
    "get_risk_engine",
    "get_monitoring_layer",
    "create_app",
    "app",
]
