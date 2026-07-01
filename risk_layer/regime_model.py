"""
risk_layer/regime_model.py  (v2)
──────────────────────────────────
NIFTY 200-DMA Regime Filter

Rule:
  If NIFTY Close < 200-day SMA:
      Reduce total gross exposure to 50% (half to cash).
  Else:
      Full 100% exposure.

This keeps the portfolio defensive during confirmed bear markets.
"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

def compute_regime_exposure(
    nifty_df: pd.DataFrame, 
    vix_df: Optional[pd.DataFrame] = None,
    price_col: str = "Close"
) -> pd.Series:
    """
    V4 Institutional Regime Model:
    1. Trend Filter: If NIFTY < 200DMA -> 50% exposure.
    2. Crash Guard: If NIFTY daily ret < -2% -> scale exposure by 0.5 for following day.
    3. Vol Scaling: If India VIX > 25 -> scale exposure by 0.7.
    
    Returns a daily Series of exposure scalars [0.0 to 1.0].
    """
    px = nifty_df[price_col]
    dma200 = px.rolling(200, min_periods=100).mean()
    
    # 1. Base Trend Exposure
    exposure = (px >= dma200).map({True: 1.0, False: 0.6})
    
    # 2. Crash Guard (§V)
    daily_ret = px.pct_change()
    crash_mask = daily_ret < -0.025 # Tighter crash guard
    # Apply crash guard to NEXT day
    exposure[crash_mask.shift(1).fillna(False)] *= 0.5
    
    # 3. Volatility Scaling (Step 4 mandate)
    if vix_df is not None:
        vix_series = vix_df[price_col].reindex(exposure.index).ffill()
        
        def _get_vix_exposure(v):
            if v < 22: return 1.0
            if v <= 30: return 0.8
            return 0.6 # floor at 0.6 instead of 0.4
            
        vol_scalar = vix_series.apply(_get_vix_exposure)
        exposure *= vol_scalar

    # SMOOTHING & HYSTERESIS (Turnover Control)
    # 1. EMA Smoothing
    exposure = exposure.ewm(span=5).mean()
    
    # 2. Hysteresis (Threshold = 0.10)
    current_val = exposure.iloc[0]
    final_out = [current_val]
    for val in exposure.iloc[1:]:
        if abs(val - current_val) > 0.10:
            current_val = val
        final_out.append(current_val)
    
    exposure = pd.Series(final_out, index=exposure.index)
    logger.info(f"  Regime Scaling: Mean = {exposure.mean():.3f}")

    exposure.name = "regime_exposure"
    return exposure
