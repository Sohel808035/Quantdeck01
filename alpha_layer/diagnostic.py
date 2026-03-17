"""
alpha_layer/diagnostic.py
─────────────────────────
Institutional auditing tools:
1. IC Decay (1D to 90D).
2. T-Stat significance for features.
3. Turnover attribution.
"""

from __future__ import annotations
import logging
from typing import Dict, List
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

logger = logging.getLogger(__name__)

def compute_ic_decay(
    predictions: pd.DataFrame, 
    stock_panel: pd.DataFrame, 
    price_col: str = "Close",
    horizons: List[int] = [1, 5, 10, 21, 63, 126]
) -> Dict[int, float]:
    """
    Computes the Information Coefficient (Spearman Rank Corr) for 
    multiple forward horizons using the SAME prediction scores.
    
    This identifies the 'Alpha Half-Life'.
    """
    decay_stats = {}
    
    # ── 1. Wide Price Frame ────────────────────────────────────────────────
    prices = stock_panel[price_col].unstack(level=1).sort_index()
    
    # ── 2. Walk through horizons ──────────────────────────────────────────
    for h in horizons:
        # Calculate forward returns for this horizon
        fwd_ret = prices.shift(-h) / prices - 1
        fwd_ret_long = fwd_ret.stack().dropna()
        
        # Align predictions to these returns
        # predictions index: [Date], values: Ticker-Series of scores
        # We need long format
        pred_long = predictions.stack()
        
        common_idx = pred_long.index.intersection(fwd_ret_long.index)
        if len(common_idx) > 100:
            y_p = pred_long.loc[common_idx]
            y_t = fwd_ret_long.loc[common_idx]
            ic = y_p.corr(y_t, method="spearman")
            decay_stats[h] = ic
            logger.info(f"[Diagnostic] IC Decay @ {h}D: {ic:.4f}")
            
    return decay_stats

def compute_feature_contribution(
    model, 
    X: pd.DataFrame, 
    y: pd.Series
) -> pd.Series:
    """
    Calculates the relative contribution (feature importance) to the model.
    """
    import xgboost as xgb
    if hasattr(model, "get_booster"):
        importance = model.get_booster().get_score(importance_type="gain")
        return pd.Series(importance).sort_values(ascending=False)
    return pd.Series(dtype=float)
