"""
portfolio_layer/plugins/black_litterman.py
────────────────────────────────────────────
Black-Litterman Portfolio Plugin.
Blends market equilibrium returns with model alpha views.
"""

from __future__ import annotations
import logging
from typing import Set, Optional
import pandas as pd
import numpy as np

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class BlackLittermanPlugin(BasePortfolioPlugin):
    """Black-Litterman model blending market equilibrium returns and model alpha views."""

    @property
    def name(self) -> str:
        return "black_litterman"

    @property
    def description(self) -> str:
        return "Black-Litterman portfolio optimization combining market equilibrium returns and ML alpha views."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))

        if returns_df is None or returns_df.empty:
            logger.warning("[Black-Litterman] No returns_df provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return pd.Series(1.0 / len(tickers), index=tickers)

        cov = sub_ret.cov().values
        n = len(sub_ret.columns)

        # 1. Market equilibrium returns (assume benchmark is equal weight prior)
        w_prior = np.ones(n) / n
        pi = risk_aversion * np.dot(cov, w_prior)

        # 2. Incorporate ML Alpha Scores as Views
        if alpha_scores is not None:
            views_raw = alpha_scores.reindex(sub_ret.columns).fillna(0.5).values
            # Scale views around 0 mean
            Q = (views_raw - 0.5) * 0.10  # 10% expected return spread for top alpha
            P = np.eye(n)
            Omega = np.diag(np.diag(tau * cov))  # View uncertainty matrix

            # Black-Litterman posterior mean calculation
            tau_cov_inv = np.linalg.inv(tau * cov)
            omega_inv = np.linalg.inv(Omega)

            post_cov = np.linalg.inv(tau_cov_inv + np.dot(P.T, np.dot(omega_inv, P)))
            post_er = np.dot(post_cov, np.dot(tau_cov_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q)))
        else:
            post_er = pi

        # 3. Compute optimal weights w = (1 / delta) * Σ^(-1) * post_er
        cov_inv = np.linalg.inv(cov)
        w_opt = (1.0 / risk_aversion) * np.dot(cov_inv, post_er)

        # Long-only projection (clip negative weights)
        w_opt = np.maximum(0, w_opt)
        if np.sum(w_opt) > 0:
            w_opt = w_opt / np.sum(w_opt)
        else:
            w_opt = w_prior

        weights = pd.Series(w_opt, index=sub_ret.columns)
        weights = weights.reindex(tickers).fillna(1.0 / len(tickers))
        return weights / weights.sum()
