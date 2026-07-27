"""
backend_services/schemas/requests_responses.py
─────────────────────────────────────────────────
Pydantic Request & Response Data Transfer Objects (DTOs).
Provides OpenAPI schema validation and interactive Swagger documentation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health & Diagnostics ──────────────────────────────────────────────────────

class HealthStatusResponse(BaseModel):
    status: str = Field(..., example="HEALTHY")
    environment: str = Field(..., example="production")
    version: str = Field(..., example="2.0.0")
    timestamp: str = Field(..., example="2026-07-27T23:50:00Z")
    uptime_seconds: float = Field(..., example=3600.0)


# ── Backtest Engine Schemas ───────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    initial_capital: float = Field(10_000_000.0, description="Initial capital in currency (INR)", example=10000000.0)
    transaction_cost_pct: float = Field(0.0015, description="One-way proportional transaction cost", example=0.0015)
    apply_vol_targeting: bool = Field(True, description="Enable 18% vol targeting", example=True)
    signal_lag_days: int = Field(1, description="Execution lag in days", example=1)
    rebalance_frequency: str = Field("monthly", description="'daily', 'weekly', or 'monthly'", example="monthly")


class BacktestResponse(BaseModel):
    cagr: float = Field(..., example=0.185)
    ann_vol: float = Field(..., example=0.142)
    sharpe_ratio: float = Field(..., example=1.30)
    sortino_ratio: float = Field(..., example=1.85)
    max_drawdown: float = Field(..., example=-0.115)
    calmar_ratio: float = Field(..., example=1.60)
    final_equity: float = Field(..., example=11850000.0)
    total_trades_count: int = Field(..., example=250)


# ── Risk Engine Schemas ───────────────────────────────────────────────────────

class RiskAuditRequest(BaseModel):
    confidence_level: float = Field(0.95, example=0.95)
    lookback_days: int = Field(252, example=252)
    include_stress_testing: bool = Field(True, example=True)


class RiskAuditResponse(BaseModel):
    var_95: float = Field(..., example=0.018)
    cvar_95: float = Field(..., example=0.027)
    tail_risk_ratio: float = Field(..., example=1.50)
    top_5_concentration_pct: float = Field(..., example=0.28)
    effective_n_positions: float = Field(..., example=15.4)
    mandate_met: bool = Field(..., example=True)
    risk_grade: str = Field(..., example="LOW")


# ── AI Analyst Schemas ────────────────────────────────────────────────────────

class PredictionExplainRequest(BaseModel):
    symbol: str = Field(..., example="RELIANCE")
    predicted_score: float = Field(..., example=0.045)
    probability: Optional[float] = Field(0.78, example=0.78)
    shap_values: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"mom_60": 0.03, "vol_20": -0.01, "rsi_14": 0.005}
    )


class PredictionExplainResponse(BaseModel):
    symbol: str
    direction: str = Field(..., example="BULLISH")
    stance: str = Field(..., example="Strong Outperform")
    narrative: str
    key_drivers: List[str]
    shap_summary: Optional[str] = None


class NewsSummaryRequest(BaseModel):
    headline: str = Field(..., example="Reliance Industries reports 15% YoY net profit growth")
    content: str = Field(..., example="Strong refining margins and digital expansion drove revenue to record levels.")
    symbol: Optional[str] = Field("RELIANCE", example="RELIANCE")


class NewsSummaryResponse(BaseModel):
    symbol: Optional[str]
    headline: str
    sentiment_score: float = Field(..., example=0.65)
    sentiment_label: str = Field(..., example="POSITIVE")
    market_impact_rating: str = Field(..., example="HIGH BULLISH")
    key_takeaways: List[str]
