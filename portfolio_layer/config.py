"""
portfolio_layer/config.py
─────────────────────────
Configuration settings for the QuantSphereX Portfolio Engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from portfolio_layer.base import PortfolioConstraints


@dataclass
class PortfolioEngineConfig:
    """Master configuration for portfolio optimization and execution."""
    default_optimizer: str = "equal_weight"
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    # Rebalancing
    rebalance_frequency: str = "monthly"   # 'daily', 'weekly', 'monthly'
    rebalance_drift_threshold: float = 0.05  # Trigger rebalance if allocation drifts >5%

    # Transaction Costs & Impact
    commission_bps: float = 10.0           # 10 bps fixed commission
    slippage_bps: float = 5.0              # 5 bps estimated slippage
    use_market_impact_model: bool = True   # Square-root market impact law

    # Position Sizing
    volatility_target: float = 0.15        # 15% annualized volatility target
    fractional_kelly_lambda: float = 0.50  # Half-Kelly sizing
