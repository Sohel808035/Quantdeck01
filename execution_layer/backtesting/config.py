"""
execution_layer/backtesting/config.py
──────────────────────────────────────
Configuration DTO for the QuantSphereX Modular Backtesting Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    """Master configuration for the modular backtesting engine."""
    initial_capital: float = 10_000_000.0  # ₹1 crore default
    transaction_cost_pct: float = 0.0015   # 0.15% one-way
    impact_coeff: float = 0.10             # Square-root market impact coefficient
    target_vol: float = 0.18              # 18% annualised volatility target
    apply_vol_targeting: bool = True
    signal_lag_days: int = 1               # Execution lag: signals from T fill at T+1
    rebalance_frequency: str = "monthly"   # 'daily', 'weekly', 'monthly'
    rolling_window: int = 63               # 63-day (~quarterly) rolling window for metrics
    risk_free_rate: float = 0.065          # Indian risk-free rate (T-Bill yield, annualised)
    benchmark_col: Optional[str] = None    # Optional benchmark column name in returns df
