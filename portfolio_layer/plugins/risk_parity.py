"""
portfolio_layer/plugins/risk_parity.py
────────────────────────────────────────
Risk Parity / Equal Risk Contribution (ERC) Portfolio Plugin.
"""

from __future__ import annotations
import logging
from typing import Set, Optional
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class RiskParityPlugin(BasePortfolioPlugin):
    """Equal Risk Contribution (ERC) optimizer via Inverse Volatility / Numerical Risk Parity."""

    @property
    def name(self) -> str:
        return "risk_parity"

    @property
    def description(self) -> str:
        return "Equal Risk Contribution (ERC) where each asset contributes equally to portfolio risk."

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
            # Fallback to equal weight if no historical returns supplied
            logger.warning("[Risk Parity] No returns_df provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna(how="all")
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return pd.Series(1.0 / len(tickers), index=tickers)

        cov = sub_ret.cov().values
        n = len(sub_ret.columns)

        # Naive inverse-volatility as initial seed
        vols = np.sqrt(np.diag(cov))
        vols[vols == 0] = 1e-4
        inv_vol_w = (1.0 / vols) / np.sum(1.0 / vols)

        # Cyclical / Iterative Risk Parity optimization
        w = inv_vol_w.copy()
        for _ in range(20):
            port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            if port_vol == 0:
                break
            marginal_risk = np.dot(cov, w) / port_vol
            risk_contrib = w * marginal_risk
            target_risk = port_vol / n
            w = w * (target_risk / (risk_contrib + 1e-8))
            w = w / np.sum(w)

        weights = pd.Series(w, index=sub_ret.columns)

        # Ensure all selected tickers exist in output
        weights = weights.reindex(tickers).fillna(1.0 / len(tickers))
        weights = weights / weights.sum()
        return weights
