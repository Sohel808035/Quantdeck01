"""
execution_layer/backtesting/factor_attribution.py
──────────────────────────────────────────────────
Factor Attribution Module.
Decomposes portfolio returns into systematic factor contributions (Fama-French style).
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class FactorAttributionEngine:
    """
    Decomposes total portfolio returns into factor components via linear regression:
      - Market Beta (systematic exposure)
      - Factor alphas (alpha net of market)
      - Residual (stock-specific idiosyncratic)
    """

    def compute_factor_regression(
        self,
        portfolio_ret: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        OLS regression of portfolio returns against factor return series.
        Factor returns should be daily (same frequency as portfolio_ret).

        Returns:
            Dict of factor_name -> factor_loading (beta), plus alpha and R².
        """
        if portfolio_ret.empty or factor_returns.empty:
            return {}

        common_idx = portfolio_ret.index.intersection(factor_returns.index)
        if len(common_idx) < 30:
            return {}

        y = portfolio_ret.reindex(common_idx).fillna(0).values
        X_raw = factor_returns.reindex(common_idx).fillna(0).values
        # Add intercept column
        X = np.column_stack([np.ones(len(y)), X_raw])

        try:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return {}

        alpha_daily = betas[0]
        factor_betas = betas[1:]

        y_hat = X @ betas
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        result = {"alpha_daily": round(float(alpha_daily), 6), "r_squared": round(float(r2), 4)}
        for i, col in enumerate(factor_returns.columns):
            result[f"beta_{col}"] = round(float(factor_betas[i]), 4)

        result["alpha_annualised"] = round(float(alpha_daily * 252), 4)
        return result

    def return_attribution(
        self,
        portfolio_ret: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes daily factor contribution to portfolio returns.

        Returns:
            DataFrame with columns: [factor_contrib, residual] indexed by date.
        """
        regression = self.compute_factor_regression(portfolio_ret, factor_returns)
        if not regression:
            return pd.DataFrame()

        common_idx = portfolio_ret.index.intersection(factor_returns.index)
        y = portfolio_ret.reindex(common_idx).fillna(0)

        factor_contributions = pd.DataFrame(index=common_idx)
        for col in factor_returns.columns:
            beta_key = f"beta_{col}"
            beta = regression.get(beta_key, 0.0)
            factor_contributions[col] = beta * factor_returns[col].reindex(common_idx).fillna(0)

        factor_contributions["total_factor"] = factor_contributions.sum(axis=1)
        factor_contributions["residual"] = y - regression.get("alpha_daily", 0.0) - factor_contributions["total_factor"]
        factor_contributions["portfolio_ret"] = y

        return factor_contributions

    def information_coefficient_decay(
        self,
        alpha_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        horizons: list[int] = [1, 5, 10, 21],
    ) -> pd.DataFrame:
        """
        Computes IC decay — Spearman rank correlation between alpha scores
        and forward returns at multiple horizons.
        """
        results = []
        for h in horizons:
            fwd_h = forward_returns.shift(-h)
            common_dates = alpha_scores.index.intersection(fwd_h.index)
            ics = []
            for d in common_dates:
                try:
                    a = alpha_scores.loc[d].dropna()
                    f = fwd_h.loc[d].reindex(a.index).dropna()
                    common = a.index.intersection(f.index)
                    if len(common) > 5:
                        ic, _ = spearmanr(a[common], f[common])
                        ics.append(ic)
                except Exception:
                    pass
            mean_ic = float(np.nanmean(ics)) if ics else 0.0
            results.append({"horizon": h, "mean_ic": round(mean_ic, 4), "n_obs": len(ics)})

        return pd.DataFrame(results)
