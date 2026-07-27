"""
risk_layer/scenario_analysis.py
────────────────────────────────
Macro Scenario Analysis Matrix Engine.
Evaluates portfolio sensitivity under multi-factor macro stress scenarios.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Standard macroeconomic stress scenarios
MACRO_SCENARIOS = {
    "Crude_Oil_Price_Spike_30pct":  {"Oil": 0.30, "Inflation": 0.02, "Market_Shock": -0.12},
    "Interest_Rate_Hike_100bps":   {"Rates": 0.01, "USD_INR": 0.03, "Market_Shock": -0.10},
    "India_VIX_Regime_Spike_50pct": {"VIX": 0.50, "Market_Shock": -0.15},
    "Global_Liquidity_Tightening":  {"USD_INR": 0.05, "Market_Shock": -0.20},
    "Stagflation_Environment":     {"Inflation": 0.04, "Market_Shock": -0.25},
}


class ScenarioAnalysisEngine:
    """Evaluates multi-variable macro scenario shocks on portfolio equity."""

    def __init__(self, scenarios: Optional[Dict[str, Dict[str, float]]] = None):
        self.scenarios = scenarios or MACRO_SCENARIOS

    def run_scenario_matrix(
        self,
        weights: pd.Series,
        stock_betas: Optional[pd.Series] = None,
        portfolio_value: float = 1e7,
    ) -> Dict[str, float]:
        """
        Evaluates portfolio loss in currency for each macro scenario.
        """
        if weights.empty:
            return {}

        betas = stock_betas.reindex(weights.index).fillna(1.0) if stock_betas is not None else pd.Series(1.0, index=weights.index)
        port_beta = float((weights * betas).sum())

        results = {}
        for scenario_name, factors in self.scenarios.items():
            mkt_shock = factors.get("Market_Shock", -0.10)
            port_shock = mkt_shock * port_beta
            loss = float(-port_shock * portfolio_value)
            results[scenario_name] = round(loss, 2)

        return results
