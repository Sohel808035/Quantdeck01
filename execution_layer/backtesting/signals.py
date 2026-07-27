"""
execution_layer/backtesting/signals.py
──────────────────────────────────────
Signal Pipeline Module.
Handles weight schedule expansion, signal lag application, and regime scaling.
"""

from __future__ import annotations
import logging
from typing import Optional
import pandas as pd
import numpy as np

from execution_layer.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class SignalPipeline:
    """
    Transforms raw weight schedule into daily holding signals with:
      - Temporal alignment across all trading dates
      - Configurable execution lag (default: T+1)
      - Regime exposure scaling
      - Volatility-targeted leverage scaling
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def build_holding_weights(
        self,
        weights_schedule: pd.DataFrame,
        all_dates: pd.DatetimeIndex,
        regime_exposure: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Step 1: Expand monthly/periodic weights into daily weight panel.
        Step 2: Apply signal lag (default T+1 — no look-ahead bias).
        Step 3: Apply regime exposure scaling.

        Args:
            weights_schedule: Weight DataFrame indexed by rebalance dates.
            all_dates:        Full trading calendar (DatetimeIndex).
            regime_exposure:  Optional daily Series of regime exposure scalars [0, 1].

        Returns:
            Daily holding weights DataFrame aligned to all_dates.
        """
        # Expand & forward-fill weights over all trading dates
        daily_weights = (
            weights_schedule
            .reindex(all_dates)
            .ffill()
            .fillna(0.0)
        )

        # Apply regime exposure scaling
        if regime_exposure is not None:
            regime_aligned = regime_exposure.reindex(all_dates).ffill().fillna(1.0)
            daily_weights = daily_weights.multiply(regime_aligned, axis=0)
            logger.debug(f"[Signals] Regime scaling applied. Mean exposure: {regime_aligned.mean():.3f}")

        # Apply signal lag: weights set at end of day T → applied at day T+1
        holding_weights = daily_weights.shift(self.config.signal_lag_days).fillna(0.0)

        return holding_weights

    def apply_vol_targeting(
        self,
        holding_weights: pd.DataFrame,
        aligned_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Applies volatility targeting by scaling holding weights so that
        realised portfolio volatility approximates the target vol.
        """
        if not self.config.apply_vol_targeting:
            return holding_weights

        from risk_layer.vol_targeting import compute_vol_target_scalar
        gross_ret = (holding_weights * aligned_returns).sum(axis=1)
        scalar = compute_vol_target_scalar(
            gross_ret,
            target_vol=self.config.target_vol,
            lookback=60,
        )
        holding_weights = holding_weights.multiply(scalar, axis=0)
        logger.debug(f"[Signals] Vol targeting applied. Mean scalar: {scalar.mean():.3f}")
        return holding_weights
