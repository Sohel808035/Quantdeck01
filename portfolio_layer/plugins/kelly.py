"""
portfolio_layer/plugins/kelly.py
──────────────────────────────────
Kelly Criterion Portfolio Plugin.
Optimal fractional Kelly sizing w* = (f_frac / lambda) * (mu / sigma^2).
"""

from __future__ import annotations
import logging
from typing import Set, Optional
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class KellyCriterionPlugin(BasePortfolioPlugin):
    """Fractional Kelly Criterion portfolio sizing for optimal long-term growth rate."""

    @property
    def name(self) -> str:
        return "kelly"

    @property
    def description(self) -> str:
        return "Fractional Kelly Criterion optimal sizing balancing expected return and variance."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        kelly_fraction: float = 0.50,  # Half-Kelly for safety
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))

        if returns_df is None or returns_df.empty:
            logger.warning("[Kelly] No returns_df provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return pd.Series(1.0 / len(tickers), index=tickers)

        cov = sub_ret.cov().values
        # Means: use alpha scores if available, else historical mean returns
        if alpha_scores is not None:
            raw_scores = alpha_scores.reindex(sub_ret.columns).fillna(0.5).values
            # Scale alpha score around 0 expected return
            mu = (raw_scores - 0.5) * 0.001  # Daily expected return delta
        else:
            mu = sub_ret.mean().values

        # Full Kelly formula: w = Σ^(-1) * μ
        try:
            cov_inv = np.linalg.pinv(cov)
            w_raw = np.dot(cov_inv, mu)
        except Exception:
            w_raw = np.ones(len(tickers)) / len(tickers)

        # Apply Fractional Kelly safety factor (Half-Kelly default)
        w_kelly = w_raw * kelly_fraction

        # Long-only constraint (clip negative bets)
        w_kelly = np.maximum(0, w_kelly)

        if np.sum(w_kelly) > 0:
            weights = pd.Series(w_kelly / np.sum(w_kelly), index=sub_ret.columns)
        else:
            weights = pd.Series(1.0 / len(tickers), index=tickers)

        weights = weights.reindex(tickers).fillna(1.0 / len(tickers))
        return weights / weights.sum()
