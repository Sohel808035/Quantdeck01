"""
risk_layer/var_cvar.py
───────────────────────
Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR / Expected Shortfall) Engine.
Computes Historical, Parametric (Gaussian / Cornish-Fisher), and Monte Carlo VaR/CVaR.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger(__name__)


class VaRCVaREngine:
    """Calculates Value at Risk and Conditional Value at Risk across multiple methodologies."""

    def __init__(self, confidence_levels: Optional[list[float]] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]

    def historical_var_cvar(
        self,
        returns_series: pd.Series,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Historical Simulation VaR and CVaR.
        VaR is the -1 * alpha quantile of historical returns.
        CVaR is the mean of returns falling below -VaR.
        """
        s = returns_series.dropna()
        if len(s) == 0:
            return 0.0, 0.0

        alpha = 1.0 - confidence
        var = -float(np.percentile(s, alpha * 100))

        tail_returns = s[s <= -var]
        cvar = -float(tail_returns.mean()) if len(tail_returns) > 0 else var

        return max(0.0, var), max(0.0, cvar)

    def parametric_var_cvar(
        self,
        returns_series: pd.Series,
        confidence: float = 0.95,
        use_cornish_fisher: bool = True,
    ) -> Tuple[float, float]:
        """
        Parametric Gaussian or Cornish-Fisher VaR & CVaR.
        Cornish-Fisher expansion adjusts for skewness and kurtosis.
        """
        s = returns_series.dropna()
        if len(s) == 0:
            return 0.0, 0.0

        mu = s.mean()
        sigma = s.std()
        if sigma == 0:
            return 0.0, 0.0

        z = norm.ppf(confidence)

        if use_cornish_fisher and len(s) > 30:
            sk = skew(s)
            kt = kurtosis(s)  # Excess kurtosis
            # Cornish-Fisher z-score adjustment
            z = z + (z**2 - 1) * sk / 6.0 + (z**3 - 3 * z) * kt / 24.0 - (2 * z**3 - 5 * z) * (sk**2) / 36.0

        var = -(mu - z * sigma)
        # Closed-form Gaussian CVaR = -mu + sigma * phi(z) / (1 - alpha)
        alpha = 1.0 - confidence
        cvar = -mu + sigma * norm.pdf(z) / alpha

        return max(0.0, float(var)), max(0.0, float(cvar))

    def monte_carlo_var_cvar(
        self,
        returns_series: pd.Series,
        confidence: float = 0.95,
        n_simulations: int = 10_000,
    ) -> Tuple[float, float]:
        """Monte Carlo simulation VaR & CVaR drawn from fitted return distribution."""
        s = returns_series.dropna()
        if len(s) == 0:
            return 0.0, 0.0

        mu = s.mean()
        sigma = s.std()
        if sigma == 0:
            return 0.0, 0.0

        np.random.seed(42)
        sim_returns = np.random.normal(mu, sigma, n_simulations)
        sim_series = pd.Series(sim_returns)

        return self.historical_var_cvar(sim_series, confidence=confidence)
