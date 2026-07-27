"""
portfolio_layer/plugins/equal_weight.py
─────────────────────────────────────────
Equal Weight Portfolio Plugin (1/N with ADV caps).
"""

from __future__ import annotations
import logging
from typing import Set, Optional
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class EqualWeightPlugin(BasePortfolioPlugin):
    """Equal Weight (1/N) optimizer with ADV liquidity capping."""

    @property
    def name(self) -> str:
        return "equal_weight"

    @property
    def description(self) -> str:
        return "Naive 1/N equal weighting with ADV liquidity constraints and redistribution."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))
        n = len(tickers)
        weights = pd.Series(1.0 / n, index=tickers, name="weight")

        if constraints is None:
            constraints = PortfolioConstraints()

        if adv_data is not None and len(adv_data) > 0:
            adv_subset = adv_data.reindex(tickers).fillna(0)
            max_weights = (adv_subset * constraints.max_adv_pct) / constraints.portfolio_value

            capped_mask = weights > max_weights
            if capped_mask.any():
                weights[capped_mask] = max_weights[capped_mask]
                residual = 1.0 - weights.sum()
                uncapped = ~capped_mask
                if uncapped.any() and residual > 0.001:
                    weights[uncapped] += (residual / uncapped.sum())

        # Normalize sum to 1.0
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights
