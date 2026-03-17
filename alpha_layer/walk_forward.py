"""
alpha_layer/walk_forward.py  (v2 — Monthly rebalance cadence)
──────────────────────────────────────────────────────────────
Expanding-window walk-forward engine that retrains the model monthly.

Design
──────
  • Initial training period : first N years (configurable, default 5)
  • Retraining cadence      : monthly (first business day of each month)
  • Window type             : EXPANDING (keeps all past data)
  • No randomisation        : All splits are deterministic calendar-based

Why monthly retraining?
  Weekly would be computationally expensive with 200 stocks.
  Yearly is too infrequent — model drifts as market regimes shift.
  Monthly balances freshness vs. stability.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """
    V3 Institutional Walk-forward:
    Generates (train_slice, predict_date) pairs with mandatory PURGING and EMBARGO.
    
    Why?
    For a 60-day target (H=60), the label at Date T is not known until T+60.
    If we predict on Date T_p, any training sample from T_p - 60 onwards 
    is 'contaminated' because its target return overlaps with the future.
    """

    def __init__(self, train_years: int = 3, rebalance_months: int = 6, horizon_days: int = 60, embargo_days: int = 10):
        self.train_years = train_years
        self.rebalance_months = rebalance_months
        self.horizon_days = horizon_days
        self.embargo_days = embargo_days

    def generate_splits(
        self,
        panel: pd.DataFrame,
        target_col: str = "target_fwd60",
    ) -> Generator[Tuple[pd.DataFrame, pd.Timestamp], None, None]:
        """
        Institutional Walk-forward (Step 5):
        - Train window = 3 years (Sliding)
        - Test/Rebalance window = 6 months
        """
        dates = panel.index.get_level_values("Date").unique().sort_values()

        # Start after first 3 years
        start_date = dates[0] + pd.DateOffset(years=self.train_years)
        rebalance_dates = pd.date_range(start=start_date, end=dates[-1], freq=f"{self.rebalance_months}MS")
        rebalance_dates = [dates[dates >= d][0] for d in rebalance_dates if (dates >= d).any()]

        total_purge_days = self.horizon_days + self.embargo_days

        for pred_date in rebalance_dates:
            # ── Sliding Window Filter ────────────────────────────────────────
            # End of training: pred_date - purge gap
            train_end = pred_date - pd.Timedelta(days=total_purge_days)
            # Start of training: train_end - 3 years
            train_start = train_end - pd.DateOffset(years=self.train_years)
            
            train_mask = (
                (panel.index.get_level_values("Date") >= train_start) &
                (panel.index.get_level_values("Date") <= train_end) &
                (panel[target_col].notna())
            )
            train_slice = panel.loc[train_mask]

            if len(train_slice) < 500: # Threshold for institutional training
                logger.warning(f"  Insufficient data for {pred_date.date()}, skipping.")
                continue

            yield train_slice, pred_date
