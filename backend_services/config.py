"""
backend_services/config.py
──────────────────────────
Backend Configuration DTO.
Handles environment variables, API key authentication credentials, CORS origins, and server settings.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BackendSettings:
    """Master configuration for QuantSphereX Backend Services."""
    title: str = "QuantSphereX Institutional Quant API"
    version: str = "2.0.0"
    description: str = (
        "High-performance institutional quant platform backend providing Alpha Signals, "
        "Backtesting, Risk Management, Execution Simulation, System Monitoring, and AI Analyst Services."
    )
    api_prefix: str = "/api/v2"
    debug: bool = False
    environment: str = os.getenv("APP_ENV", "production")

    # Security & Auth
    api_key_header_name: str = "X-API-Key"
    valid_api_keys: List[str] = field(
        default_factory=lambda: [
            os.getenv("QUANTSPHEREX_API_KEY", "qsx-secret-api-key-2026"),
            "qsx-institutional-key-demo",
        ]
    )
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "quantspherex-super-secret-jwt-key")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # CORS
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8000",
            "*",
        ]
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
