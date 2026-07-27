"""
risk_layer/stress_testing.py
──────────────────────────────
QuantSphereX Stress Testing Engine.
Replays historical financial crises and computes simulated portfolio PnL impact.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Predefined historical crisis return shock profiles (peak-to-trough market shocks)
HISTORICAL_CRISES = {
    "2008_Global_Financial_Crisis": -0.55,   # GFC 55% drawdown
    "2011_US_Debt_Downgrade":       -0.18,   # 18% drawdown
    "2016_China_Devaluation":       -0.15,   # 15% drawdown
    "2020_COVID_Crash":             -0.38,   # COVID 38% crash
    "2022_Rate_Hike_Bear_Market":   -0.25,   # 25% inflation/rate shock
    "2024_Global_Vol_Spike":        -0.12,   # 12% vol shock
}


class StressTestingEngine:
    """Simulates portfolio impact under historical crisis conditions."""

    def __init__(self, crisis_shocks: Optional[Dict[str, float]] = None):
        self.crisis_shocks = crisis_shocks or HISTORICAL_CRISES

    def run_historical_replay(
        self,
        weights: pd.Series,
        stock_betas: Optional[pd.Series] = None,
        portfolio_value: float = 1e7,
    ) -> Dict[str, float]:
        """
        Replays historical crises against current portfolio weights.

        If stock_betas are provided, asset-level shock = Market_Shock * Stock_Beta.
        Returns dict of crisis name -> estimated portfolio loss in currency.
        """
        if weights.empty:
            return {}

        betas = stock_betas.reindex(weights.index).fillna(1.0) if stock_betas is not None else pd.Series(1.0, index=weights.index)
        port_beta = float((weights * betas).sum())

        results = {}
        for crisis_name, mkt_shock in self.crisis_shocks.items():
            port_return_shock = mkt_shock * port_beta
            loss_currency = float(-port_return_shock * portfolio_value)
            results[crisis_name] = round(loss_currency, 2)

        return results
