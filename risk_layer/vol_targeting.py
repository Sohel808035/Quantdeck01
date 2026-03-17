"""
risk_layer/vol_targeting.py  (NEW)
────────────────────────────────────
Volatility Targeting — scales total portfolio exposure so that the
realised annualised portfolio volatility stays near the target.

Mechanism:
  1. Compute rolling realised vol of the PORTFOLIO using its own daily returns.
  2. Scale factor = target_vol / realised_vol
  3. Cap scale factor at 1.0 (never lever up) and floor at 0.2 (never go to near-zero).
  4. Multiply all weight allocations by this scalar.

This is applied AFTER regime exposure so both overlays compound correctly.
"""

from __future__ import annotations

import logging

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


def compute_vol_target_scalar(
    portfolio_returns: pd.Series,
    target_vol:   float = 0.18,
    lookback:     int   = 60,
    min_scale:    float = 0.20,
    max_scale:    float = 1.00,
) -> pd.Series:
    """
    Given a daily portfolio return series, computes a daily exposure scalar.

    Args:
        portfolio_returns : daily net returns of the unscaled portfolio
        target_vol        : desired annualised volatility (default 18%)
        lookback          : rolling window for vol estimation (default 60 days)
        min_scale         : floor — never below 20% exposure
        max_scale         : cap   — never lever above 100%

    Returns a daily Series aligned to portfolio_returns.index.
    """
    realised_vol = portfolio_returns.rolling(lookback, min_periods=20).std() * np.sqrt(252)
    scalar = (target_vol / realised_vol).clip(lower=min_scale, upper=max_scale)
    # First lookback days: no history, use 1.0 (no scaling)
    scalar = scalar.fillna(1.0)
    scalar.name = "vol_target_scalar"
    return scalar
