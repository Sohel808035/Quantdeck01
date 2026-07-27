"""
risk_layer/liquidity_risk.py
─────────────────────────────
Liquidity Risk Engine.
Computes Days-To-Liquidate (DTL), liquidation market impact, and Liquidity-Adjusted VaR (LVaR).
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LiquidityRiskEngine:
    """Evaluates portfolio liquidity risk, liquidation time, and LVaR."""

    def __init__(
        self,
        max_adv_participation: float = 0.10,  # Max 10% daily volume participation
        impact_coef: float = 0.10,
    ):
        self.max_adv_participation = max_adv_participation
        self.impact_coef = impact_coef

    def days_to_liquidate(
        self,
        weights: pd.Series,
        adv_data: pd.Series,
        portfolio_value: float = 1e7,
    ) -> pd.Series:
        """
        Computes Days-To-Liquidate for each asset given ADV and participation limit.
        DTL_i = Position_Value_i / (max_participation * ADV_i)
        """
        if weights.empty or adv_data.empty:
            return pd.Series(0.0, index=weights.index)

        adv_sub = adv_data.reindex(weights.index).fillna(1e6)
        position_values = weights * portfolio_value
        daily_liquidable_currency = adv_sub * self.max_adv_participation

        dtl = position_values / daily_liquidable_currency.replace(0, np.nan)
        return dtl.fillna(0.0)

    def portfolio_days_to_liquidate(
        self,
        weights: pd.Series,
        adv_data: pd.Series,
        portfolio_value: float = 1e7,
        target_pct: float = 0.95,
    ) -> float:
        """Returns the number of days required to liquidate target_pct (e.g. 95%) of portfolio."""
        dtl = self.days_to_liquidate(weights, adv_data, portfolio_value)
        if dtl.empty:
            return 0.0
        return float(dtl.quantile(target_pct))

    def liquidity_adjusted_var(
        self,
        base_var_95: float,
        weights: pd.Series,
        adv_data: pd.Series,
        portfolio_value: float = 1e7,
        horizon_days: int = 10,
    ) -> float:
        """
        Liquidity-Adjusted VaR (LVaR = Base_VaR + Liquidation_Cost_Penalty).
        LVaR accounts for bid-ask spread and price impact incurred during liquidation.
        """
        dtl_95 = self.portfolio_days_to_liquidate(weights, adv_data, portfolio_value)
        liquidation_penalty_pct = 0.002 * max(1.0, dtl_95 / horizon_days)
        return float(base_var_95 + liquidation_penalty_pct)
