"""
risk_layer/factor_risk.py
───────────────────────────
Factor Risk Engine.
Computes multi-factor risk decomposition and Marginal Contribution to Risk (MCR).
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorRiskEngine:
    """Decomposes portfolio risk into systematic factor exposures and stock-specific risk."""

    def compute_factor_exposures(
        self,
        weights: pd.Series,
        factor_beta_matrix: pd.DataFrame,
    ) -> pd.Series:
        """
        Computes portfolio-level factor exposures.
        factor_beta_matrix shape: (tickers x factors)
        """
        if weights.empty or factor_beta_matrix.empty:
            return pd.Series(dtype=float)

        common_tickers = list(set(weights.index) & set(factor_beta_matrix.index))
        if not common_tickers:
            return pd.Series(dtype=float)

        w = weights.reindex(common_tickers).fillna(0)
        w = w / w.sum()

        betas = factor_beta_matrix.reindex(common_tickers).fillna(0)
        port_factor_betas = (betas.T * w).sum(axis=1)

        return port_factor_betas.rename("factor_exposure")

    def marginal_contribution_to_risk(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes Marginal Contribution to Risk (MCR) and Percent Contribution to Risk (PCR).
        MCR_i = (Σ w)_i / σ_p
        PCR_i = w_i * MCR_i / σ_p
        """
        common = list(set(weights.index) & set(cov_matrix.index))
        if not common or len(common) < 2:
            return pd.DataFrame()

        w = weights.reindex(common).fillna(0).values
        w = w / np.sum(w)
        cov = cov_matrix.loc[common, common].values

        port_var = float(np.dot(w, np.dot(cov, w)))
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-4

        mcr = np.dot(cov, w) / port_vol
        pcr = (w * mcr) / port_vol

        return pd.DataFrame({
            "weight": w,
            "mcr": mcr,
            "pcr": pcr,
        }, index=common).sort_values("pcr", ascending=False)
