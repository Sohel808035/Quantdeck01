"""
label_layer/labels/horizons.py
────────────────────────────────
Multi-Horizon Label Engine (v1.0.0)

Category:    Multi-Horizon Forecast Labels
Description: Generates comprehensive label sets across all configured
             forecast horizons simultaneously. Includes horizon-stability
             analysis, forward return cone, and horizon correlation matrix
             for diagnosing optimal prediction windows.

Utilities:
  1. Horizon Return Cone      — Forward return ± 1 std across horizons
  2. Horizon Label Summary    — Per-horizon validity, mean return, skew
  3. Optimal Horizon Selector — Highest IC horizon via rolling autocorrelation
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Horizons"
CATEGORY = "Multi-Horizon Utilities"

STANDARD_HORIZONS = [1, 5, 21, 63]


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes forward returns for all horizons from a single-ticker DataFrame.

    Args:
        df:        Single-ticker OHLCV DataFrame indexed by Date.
        horizons:  List of forecast horizons in trading days.
        price_col: Close price column name.

    Returns:
        DataFrame with columns `fwd_ret_{H}d` for each horizon H.
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    for H in horizons:
        future_px = px.shift(-H)
        result[f"fwd_ret_{H}d"] = future_px / px.replace(0, np.nan) - 1

    return result


def horizon_summary(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Produces a diagnostic summary table across all forecast horizons.

    Args:
        df:        Single-ticker OHLCV DataFrame indexed by Date.
        horizons:  List of forecast horizons.
        price_col: Close price column name.

    Returns:
        Summary DataFrame: one row per horizon with diagnostics.
    """
    fwd_df = compute_forward_returns(df, horizons, price_col)
    rows = []

    for H in horizons:
        col = f"fwd_ret_{H}d"
        if col not in fwd_df.columns:
            continue
        s = fwd_df[col].dropna()
        rows.append({
            "horizon_days": H,
            "valid_rows": len(s),
            "mean_return": round(s.mean(), 5) if len(s) > 0 else np.nan,
            "std_return": round(s.std(), 5) if len(s) > 0 else np.nan,
            "skewness": round(s.skew(), 4) if len(s) > 0 else np.nan,
            "pct_positive": round((s > 0).mean(), 4) if len(s) > 0 else np.nan,
            "sharpe_ratio": round(
                s.mean() / s.std() * np.sqrt(252 / H), 4
            ) if len(s) > 5 and s.std() > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index("horizon_days")


def horizon_correlation_matrix(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes pairwise correlation between forward returns at different horizons.
    Used to identify horizon redundancy and select uncorrelated prediction windows.

    Returns:
        Square correlation matrix of horizon forward returns.
    """
    fwd_df = compute_forward_returns(df, horizons, price_col)
    cols = [f"fwd_ret_{H}d" for H in horizons if f"fwd_ret_{H}d" in fwd_df.columns]
    return fwd_df[cols].corr()


def select_optimal_horizon(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
) -> int:
    """
    Selects the forecast horizon with the highest return-to-noise ratio
    (annualized Sharpe of forward return distribution).

    Args:
        df:        Single-ticker OHLCV DataFrame indexed by Date.
        horizons:  Candidate horizons to evaluate.
        price_col: Close price column name.

    Returns:
        Optimal horizon in trading days.
    """
    summary = horizon_summary(df, horizons, price_col)
    if summary.empty or "sharpe_ratio" not in summary.columns:
        return horizons[-1]  # Default to longest

    valid = summary["sharpe_ratio"].dropna()
    if valid.empty:
        return horizons[-1]

    best_h = int(valid.idxmax())
    logger.info(f"Optimal forecast horizon selected: {best_h}D (Sharpe={valid[best_h]:.3f})")
    return best_h
