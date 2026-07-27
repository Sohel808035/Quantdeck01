"""
label_layer/labels/classification.py
──────────────────────────────────────
Classification Label Engine (v1.0.0)

Category:    Discrete Supervised Labels
Description: Generates discrete class labels from forward returns.
             Supports binary, tertile, and quintile classification
             using both fixed thresholds and cross-sectional quantiles.

Label Families:
  1. Binary Classification    — 1 (up ≥ threshold) / 0 (down < threshold)
  2. Tertile Classification   — 2 (top) / 1 (mid) / 0 (bottom) by CS quantile
  3. Quintile Classification  — 0-4 cross-sectional quantile buckets
  4. Excess Return Binary     — 1 if beats benchmark, 0 otherwise
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Classification"
CATEGORY = "Discrete Labels"

# Class label constants
CLASS_UP   = 1
CLASS_DOWN = 0
CLASS_TOP  = 2
CLASS_MID  = 1
CLASS_BOT  = 0


def _to_tertile(series: pd.Series, low_cut: float = 0.333, high_cut: float = 0.667) -> pd.Series:
    """Converts a return series to tertile labels using cross-sectional quantile cuts."""
    result = pd.Series(CLASS_MID, index=series.index, dtype=float)
    lo = series.quantile(low_cut)
    hi = series.quantile(high_cut)
    result[series <= lo] = CLASS_BOT
    result[series >= hi] = CLASS_TOP
    result[series.isna()] = np.nan
    return result


def _to_quintile(series: pd.Series, n: int = 5) -> pd.Series:
    """Converts a return series to n-tile bucket labels (0 to n-1)."""
    labels = pd.qcut(series.rank(method="first"), q=n, labels=False)
    return labels.astype(float)


def compute(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
    benchmark_ret: Optional[pd.Series] = None,
    binary_threshold: float = 0.0,
    tertile_thresholds: Optional[List[float]] = None,
    quintile_count: int = 5,
) -> pd.DataFrame:
    """
    Computes discrete classification labels for every requested horizon.

    Args:
        df:                  Single-ticker OHLCV DataFrame indexed by Date.
        horizons:            List of forecast horizons in trading days.
        price_col:           Close price column name.
        benchmark_ret:       Benchmark daily return Series for excess-return labels.
        binary_threshold:    Return threshold for binary classification (default: 0%).
        tertile_thresholds:  [low_quantile, high_quantile] cutpoints (default: 33/67%).
        quintile_count:      Number of quantile buckets for quintile labels.

    Returns:
        DataFrame of classification label columns indexed by Date.
    """
    if tertile_thresholds is None:
        tertile_thresholds = [0.333, 0.667]

    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    for H in horizons:
        future_px = px.shift(-H)
        fwd_ret = future_px / px.replace(0, np.nan) - 1
        label_base = f"fwd_{H}d"

        # ── 1. Binary Label ───────────────────────────────────────────────────
        binary = pd.Series(np.nan, index=df.index)
        binary[fwd_ret.notna()] = (fwd_ret[fwd_ret.notna()] > binary_threshold).astype(float)
        result[f"binary_{label_base}"] = binary

        # ── 2. Tertile Label ──────────────────────────────────────────────────
        valid_mask = fwd_ret.notna()
        tertile = pd.Series(np.nan, index=df.index)
        if valid_mask.sum() > 10:
            tertile[valid_mask] = _to_tertile(
                fwd_ret[valid_mask],
                low_cut=tertile_thresholds[0],
                high_cut=tertile_thresholds[1],
            )
        result[f"tertile_{label_base}"] = tertile

        # ── 3. Quintile Label ─────────────────────────────────────────────────
        quintile = pd.Series(np.nan, index=df.index)
        if valid_mask.sum() > quintile_count * 2:
            try:
                quintile[valid_mask] = _to_quintile(fwd_ret[valid_mask], n=quintile_count)
            except Exception:
                pass
        result[f"quintile_{label_base}"] = quintile

        # ── 4. Excess Return Binary Label ─────────────────────────────────────
        if benchmark_ret is not None:
            bm = benchmark_ret.reindex(df.index).fillna(0)
            bm_compound = (1 + bm).rolling(H).apply(np.prod, raw=True).shift(-(H - 1))
            excess = fwd_ret - (bm_compound - 1)
            exc_binary = pd.Series(np.nan, index=df.index)
            exc_binary[excess.notna()] = (excess[excess.notna()] > 0).astype(float)
            result[f"excess_binary_{label_base}"] = exc_binary

    return result


def validate(result: pd.DataFrame, horizons: List[int]) -> dict:
    """Validates classification label output for expected columns and value ranges."""
    report = {"label_engine": NAME, "version": VERSION, "passed": True, "warnings": []}

    for H in horizons:
        binary_col = f"binary_fwd_{H}d"
        if binary_col not in result.columns:
            report["warnings"].append(f"Missing required column: {binary_col}")
            report["passed"] = False
        else:
            valid = result[binary_col].dropna()
            if not set(valid.unique()).issubset({0.0, 1.0}):
                report["warnings"].append(f"Binary label '{binary_col}' contains non-binary values")
                report["passed"] = False

            # Class balance check
            if len(valid) > 0:
                pos_rate = valid.mean()
                if pos_rate < 0.2 or pos_rate > 0.8:
                    report["warnings"].append(
                        f"Class imbalance in '{binary_col}': {pos_rate:.1%} positive"
                    )

    return report
