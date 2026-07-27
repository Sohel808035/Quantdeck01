"""
portfolio_layer/plugins/min_variance.py
─────────────────────────────────────────
Minimum Variance Portfolio Plugin.
Finds weights w minimizing w^T Σ w subject to sum(w) = 1 and w >= 0.
"""

from __future__ import annotations
import logging
from typing import Set, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class MinimumVariancePlugin(BasePortfolioPlugin):
    """Minimum Variance Portfolio optimizer minimizing portfolio volatility."""

    @property
    def name(self) -> str:
        return "min_variance"

    @property
    def description(self) -> str:
        return "Minimum Variance Portfolio minimizing total portfolio variance w^T Σ w."

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

        if returns_df is None or returns_df.empty:
            logger.warning("[Min Variance] No returns_df provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return pd.Series(1.0 / len(tickers), index=tickers)

        cov = sub_ret.cov().values
        n = len(sub_ret.columns)

        def _obj(w):
            return np.dot(w, np.dot(cov, w))

        w0 = np.ones(n) / n
        bounds = tuple((0.0, 0.20) for _ in range(n))  # Cap single asset at 20%
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(_obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            w_opt = res.x
        else:
            w_opt = w0

        weights = pd.Series(w_opt, index=sub_ret.columns)
        weights = weights.reindex(tickers).fillna(1.0 / len(tickers))
        return weights / weights.sum()
