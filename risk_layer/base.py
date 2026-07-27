"""
risk_layer/base.py
──────────────────
Data Transfer Objects (DTOs) and Base Classes for the QuantSphereX Risk Engine.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsReport:
    """Comprehensive institutional risk audit report."""
    portfolio_value: float = 1e7
    # VaR & CVaR
    var_95_historical: float = 0.0
    var_99_historical: float = 0.0
    var_95_parametric: float = 0.0
    var_99_parametric: float = 0.0
    var_95_monte_carlo: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    lvar_95: float = 0.0               # Liquidity-adjusted VaR

    # Concentration & Limits
    hhi_index: float = 0.0             # Herfindahl-Hirschman Index
    effective_n_stocks: float = 0.0    # 1 / sum(w^2)
    top_5_concentration: float = 0.0
    top_10_concentration: float = 0.0
    max_position_weight: float = 0.0
    position_limits_passed: bool = True

    # Tail & Dispersion
    skewness: float = 0.0
    kurtosis: float = 0.0
    evt_tail_index: float = 0.0
    avg_pairwise_correlation: float = 0.0
    pca_top3_variance_pct: float = 0.0

    # Liquidity
    days_to_liquidate_95pct: float = 0.0
    forced_liquidation_cost: float = 0.0

    # Sector & Country
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    country_exposures: Dict[str, float] = field(default_factory=dict)

    # Factor & Stress
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    stress_test_losses: Dict[str, float] = field(default_factory=dict)
    scenario_impacts: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Returns a flat key metrics dictionary for logging."""
        return {
            "VaR_95_Hist": round(self.var_95_historical, 4),
            "VaR_99_Hist": round(self.var_99_historical, 4),
            "CVaR_95": round(self.cvar_95, 4),
            "CVaR_99": round(self.cvar_99, 4),
            "LVaR_95": round(self.lvar_95, 4),
            "HHI": round(self.hhi_index, 4),
            "N_eff": round(self.effective_n_stocks, 2),
            "Top5_Conc": round(self.top_5_concentration, 4),
            "Max_Pos": round(self.max_position_weight, 4),
            "Avg_Corr": round(self.avg_pairwise_correlation, 4),
            "PCA_Top3_Var": round(self.pca_top3_variance_pct, 4),
            "Days_To_Liquidate": round(self.days_to_liquidate_95pct, 2),
            "Limits_Passed": self.position_limits_passed,
        }
