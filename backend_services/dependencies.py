"""
backend_services/dependencies.py
─────────────────────────────────
Dependency Injection Container & Service Providers.
Decouples quant layer instances from router endpoint handlers.
"""

from __future__ import annotations
from typing import Generator
from functools import lru_cache

from backend_services.config import BackendSettings
from execution_layer.backtesting import BacktestEngine, BacktestConfig
from execution_layer.simulator import ExecutionSimulator, ExecutionSimulatorConfig
from risk_layer import RiskEngine, RiskConfig
from monitoring_layer import MonitoringLayer, MonitoringConfig
from ai_quant_analyst import AIQuantAnalyst, AIAnalystConfig


# ── Configuration Singleton ──────────────────────────────────────────────────

@lru_cache()
def get_settings() -> BackendSettings:
    """Returns cached BackendSettings instance."""
    return BackendSettings()


# ── Quant Engine Singleton Providers ──────────────────────────────────────────

@lru_cache()
def get_backtest_engine() -> BacktestEngine:
    """Returns singleton BacktestEngine service."""
    return BacktestEngine(config=BacktestConfig())


@lru_cache()
def get_risk_engine() -> RiskEngine:
    """Returns singleton RiskEngine service."""
    return RiskEngine(config=RiskConfig())


@lru_cache()
def get_execution_simulator() -> ExecutionSimulator:
    """Returns singleton ExecutionSimulator service."""
    return ExecutionSimulator(config=ExecutionSimulatorConfig())


@lru_cache()
def get_monitoring_layer() -> MonitoringLayer:
    """Returns singleton MonitoringLayer service."""
    return MonitoringLayer(config=MonitoringConfig())


@lru_cache()
def get_ai_quant_analyst() -> AIQuantAnalyst:
    """Returns singleton AIQuantAnalyst service."""
    return AIQuantAnalyst(config=AIAnalystConfig())
