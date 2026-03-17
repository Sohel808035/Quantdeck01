"""
alpha_layer/targets.py  (v2)
─────────────────────────────
Forward return target: Price[t+60] / Price[t] - 1

The negative shift pulls the future price back to today's row without
leaking — because all features are lagged by 1 day, the model only trains
on (yesterday's features → today+60's return) which is perfectly causal.
"""

from __future__ import annotations

import logging
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

TARGET_COL = "target_fwd60"


def add_forward_return(df: pd.DataFrame, price_col: str = "Close", horizon: int = 60) -> pd.DataFrame:
    """
    Adds a forward return column to a per-stock DataFrame.
    Input: flat DataFrame for a single ticker, indexed by Date.
    """
    result = df.copy()
    future_price = result[price_col].shift(-horizon)
    result[TARGET_COL] = future_price / result[price_col] - 1
    return result


def build_target_panel(
    stock_panel: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 60,
) -> pd.DataFrame:
    """
    V4 Institutional Upgrade (Step 1):
    1. Computes forward returns per ticker.
    2. Converts forward returns into Daily Cross-Sectional Percentile Ranks.
    
    Returns panel with TARGET_COL in [0.0, 1.0] range.
    """
    logger.info(f"Building {horizon}-day cross-sectional rank targets...")
    
    # 1. Compute Raw Forward returns
    panel = stock_panel.groupby(level=1, group_keys=False).apply(
        lambda df: add_forward_return(df, price_col=price_col, horizon=horizon)
    )
    
    raw_target = panel[TARGET_COL]
    
    # 2. Cross-Sectional Percentile Rank
    # We rank all stocks available on each date by their future return.
    ranks = raw_target.groupby(level=0).rank(pct=True)
    
    panel[TARGET_COL] = ranks
    
    n_valid = panel[TARGET_COL].notna().sum()
    logger.info(f"  Rank target created. Valid rows: {n_valid:,} (Range: [{ranks.min():.2f}, {ranks.max():.2f}])")
    return panel
