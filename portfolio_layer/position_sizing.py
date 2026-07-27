"""
portfolio_layer/position_sizing.py
──────────────────────────────────
QuantSphereX Position Sizing Engine.
Handles volatility-targeted position sizing, conviction weighting, and cash buffer management.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PositionSizingEngine:
    """Calculates position sizes based on volatility targets, alpha conviction, and cash buffers."""

    def __init__(
        self,
        volatility_target: float = 0.15,  # 15% annualized target portfolio volatility
        cash_buffer_pct: float = 0.02,     # 2% cash buffer
    ):
        self.volatility_target = volatility_target
        self.cash_buffer_pct = cash_buffer_pct

    def volatility_targeted_weights(
        self,
        weights: pd.Series,
        returns_df: pd.DataFrame,
        window: int = 60,
    ) -> pd.Series:
        """
        Rescales portfolio weights so that estimated annualized portfolio volatility
        equals the target volatility (or caps leverage if volatility is low).
        """
        if weights.empty or returns_df is None or returns_df.empty:
            return weights

        sub_ret = returns_df[[t for t in weights.index if t in returns_df.columns]].tail(window)
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return weights

        cov = sub_ret.cov().values
        w = weights.values
        realized_vol = np.sqrt(np.dot(w, np.dot(cov, w))) * np.sqrt(252)

        if realized_vol > 0:
            scale = self.volatility_target / realized_vol
            # Cap maximum gross leverage at 1.5x
            scale = min(scale, 1.5)
            rescaled_w = weights * scale
        else:
            rescaled_w = weights

        return rescaled_w

    def conviction_weighted_sizing(
        self,
        base_weights: pd.Series,
        alpha_scores: pd.Series,
        alpha_power: float = 2.0,
    ) -> pd.Series:
        """
        Tilts base position weights by alpha conviction scores (high rank = higher position).
        """
        if base_weights.empty or alpha_scores is None or alpha_scores.empty:
            return base_weights

        scores = alpha_scores.reindex(base_weights.index).fillna(0.5)
        # Power scaling to amplify conviction difference
        tilt = (scores ** alpha_power)
        tilted_w = base_weights * tilt

        if tilted_w.sum() > 0:
            return tilted_w / tilted_w.sum()

        return base_weights

    def apply_cash_buffer(self, weights: pd.Series) -> pd.Series:
        """Rescales asset weights to reserve a cash buffer (e.g. 2% cash -> 98% invested)."""
        if weights.empty:
            return weights

        invested_fraction = 1.0 - self.cash_buffer_pct
        if weights.sum() > 0:
            norm_w = weights / weights.sum()
            return norm_w * invested_fraction
        return weights
