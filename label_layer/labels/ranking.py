"""
label_layer/labels/ranking.py
──────────────────────────────
Ranking Label Engine (v1.0.0)

Category:    Cross-Sectional Ordinal Labels
Description: Generates cross-sectional ordinal ranking labels from forward
             returns. Ranking is done across all available stocks on each
             date — the core target used in institutional long/short equity.

             These labels power the existing XGBoost pipeline and are
             the highest-IC target type in the QuantSphereX engine.

Label Families:
  1. Percentile Rank       — [0.0, 1.0] rank on each date (existing pipeline)
  2. Z-Score Rank          — Standardized cross-sectional rank
  3. Decile Rank           — Integer 1–10 decile bucket
  4. Information Ratio     — Forward return / cross-sectional std (IC-oriented)
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Ranking"
CATEGORY = "Cross-Sectional Ordinal Labels"


def _cs_rank_pct(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank. Output in [0.0, 1.0]."""
    return series.rank(pct=True)


def _cs_zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalization."""
    mu = series.mean()
    sd = series.std()
    return (series - mu) / (sd + 1e-8)


def _cs_decile(series: pd.Series) -> pd.Series:
    """Cross-sectional decile bucket (1 = worst, 10 = best)."""
    try:
        return pd.qcut(series.rank(method="first"), q=10, labels=False).astype(float) + 1
    except Exception:
        return pd.Series(np.nan, index=series.index)


def compute_single_ticker(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes raw forward returns for a single ticker.
    Cross-sectional ranking is performed in compute_panel() across all tickers.

    Args:
        df:        Single-ticker DataFrame indexed by Date.
        horizons:  List of forecast horizons in trading days.
        price_col: Close price column name.

    Returns:
        DataFrame with raw forward return columns for cross-sectional ranking.
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    for H in horizons:
        future_px = px.shift(-H)
        result[f"_raw_fwd_{H}d"] = future_px / px.replace(0, np.nan) - 1

    return result


def compute_panel(
    raw_panel: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """
    Applies cross-sectional ranking transformations to a multi-ticker panel.

    Input must have a (Date, Ticker) MultiIndex with raw forward return
    columns from compute_single_ticker().

    Args:
        raw_panel:  Panel DataFrame with (Date, Ticker) MultiIndex.
        horizons:   List of forecast horizons in trading days.

    Returns:
        Panel with cross-sectional ranking label columns (same MultiIndex).
    """
    result = raw_panel.copy()

    for H in horizons:
        raw_col = f"_raw_fwd_{H}d"
        if raw_col not in raw_panel.columns:
            continue

        # Group by Date for cross-sectional operations
        grp = raw_panel[raw_col].groupby(level=0)

        result[f"rank_pct_fwd_{H}d"]    = grp.transform(_cs_rank_pct)
        result[f"rank_zscore_fwd_{H}d"] = grp.transform(_cs_zscore)
        result[f"rank_decile_fwd_{H}d"] = grp.transform(_cs_decile)

        # Information-ratio style label: return / cross-sectional std
        cs_std = grp.transform("std")
        result[f"rank_ir_fwd_{H}d"] = raw_panel[raw_col] / (cs_std + 1e-8)

        # Drop the raw helper column from output
        result.drop(columns=[raw_col], inplace=True, errors="ignore")

    return result


def validate(result: pd.DataFrame, horizons: List[int]) -> dict:
    """Validates ranking label output for expected columns and value ranges."""
    report = {"label_engine": NAME, "version": VERSION, "passed": True, "warnings": []}

    for H in horizons:
        pct_col = f"rank_pct_fwd_{H}d"
        if pct_col not in result.columns:
            report["warnings"].append(f"Missing required ranking column: {pct_col}")
            report["passed"] = False
        else:
            valid = result[pct_col].dropna()
            if len(valid) > 0:
                if valid.min() < 0 or valid.max() > 1:
                    report["warnings"].append(
                        f"Percentile rank '{pct_col}' out of [0,1] bounds"
                    )
                    report["passed"] = False

    return report
