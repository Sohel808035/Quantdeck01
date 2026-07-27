"""
portfolio_layer/rebalancing.py
───────────────────────────────
QuantSphereX Rebalancing Engine.
Handles calendar schedules, drift-based triggers, and turnover penalty dampeners.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RebalancingEngine:
    """Manages portfolio rebalance timing, drift monitoring, and no-trade zones."""

    def __init__(
        self,
        turnover_threshold: float = 0.005,  # 50 bps no-trade zone
        drift_trigger_pct: float = 0.05,    # Rebalance if weight drifts > 5%
    ):
        self.turnover_threshold = turnover_threshold
        self.drift_trigger_pct = drift_trigger_pct

    def apply_turnover_penalty(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        threshold: Optional[float] = None,
    ) -> pd.Series:
        """
        No-Trade Zone (Turnover Dampener).
        If the required position change for a stock is < threshold (50 bps),
        keep the current weight to avoid expensive micro-trades.
        """
        if threshold is None:
            threshold = self.turnover_threshold

        if current_weights.empty:
            return target_weights

        all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index)))
        curr = current_weights.reindex(all_tickers).fillna(0.0)
        targ = target_weights.reindex(all_tickers).fillna(0.0)

        diff = (targ - curr).abs()

        final_weights = curr.copy()
        trade_mask = diff > threshold
        final_weights[trade_mask] = targ[trade_mask]

        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()

        return final_weights

    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> bool:
        """
        Drift-based rebalance trigger.
        Returns True if maximum allocation drift across all stocks > drift_trigger_pct.
        """
        if current_weights.empty or target_weights.empty:
            return True

        all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index)))
        curr = current_weights.reindex(all_tickers).fillna(0.0)
        targ = target_weights.reindex(all_tickers).fillna(0.0)

        max_drift = (targ - curr).abs().max()
        return float(max_drift) > self.drift_trigger_pct

    def compute_turnover(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
    ) -> float:
        """Computes 1-way portfolio turnover ratio."""
        all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index)))
        curr = current_weights.reindex(all_tickers).fillna(0.0)
        targ = target_weights.reindex(all_tickers).fillna(0.0)
        return float((targ - curr).abs().sum() / 2.0)
