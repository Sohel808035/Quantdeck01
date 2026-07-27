"""
feature_layer/factors/growth.py
─────────────────────────────────
Growth Factor Engine (v1.0.0)

Category:    Fundamental Growth
Description: Quantifies business growth rates via revenue acceleration,
             earnings growth persistence, expansion signals, and analyst
             revision proxy signals.

Alpha Families:
  1. Price Acceleration    — Second derivative of price momentum (rate of change)
  2. Volume Growth         — Expanding volume signals institutional accumulation
  3. Revenue Proxy Growth  — Price × Volume as revenue analog
  4. Expansion Velocity    — Breakout of all-time highs, relative market share gain
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Growth"
CATEGORY = "Fundamental Growth"

OUTPUT_COLUMNS = [
    "price_acceleration",
    "volume_growth_1m",
    "volume_growth_3m",
    "revenue_proxy_growth",
    "expansion_breakout",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """
    Computes Growth factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Unused for Growth factors.
        price_col:    Column name for close price.
        volume_col:   Column name for volume.

    Returns:
        DataFrame with Growth feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    # ── 1. Price Acceleration (2nd derivative of momentum) ────────────────────
    ret_1m = px / px.shift(21) - 1
    ret_3m = px / px.shift(63) - 1
    result["price_acceleration"] = ret_1m - (ret_3m / 3)  # Current growth vs trend

    # ── 2. Volume Growth ──────────────────────────────────────────────────────
    if volume_col in df.columns:
        vol = df[volume_col].replace(0, np.nan)
        vol_sma_1m = vol.rolling(21).mean()
        vol_sma_3m = vol.rolling(63).mean()
        vol_sma_base = vol.rolling(126).mean()
        result["volume_growth_1m"] = vol_sma_1m / vol_sma_base.replace(0, np.nan) - 1
        result["volume_growth_3m"] = vol_sma_3m / vol_sma_base.replace(0, np.nan) - 1
    else:
        result["volume_growth_1m"] = np.nan
        result["volume_growth_3m"] = np.nan

    # ── 3. Revenue Proxy Growth (Price × Volume turnover) ────────────────────
    if volume_col in df.columns:
        turnover = px * df[volume_col].replace(0, np.nan)
        turnover_ma = turnover.rolling(21).mean()
        turnover_base = turnover.rolling(126).mean()
        result["revenue_proxy_growth"] = turnover_ma / turnover_base.replace(0, np.nan) - 1
    else:
        result["revenue_proxy_growth"] = np.nan

    # ── 4. Expansion Breakout (distance from 52W high) ───────────────────────
    high_52w = px.rolling(252).max()
    result["expansion_breakout"] = px / high_52w.replace(0, np.nan) - 1  # 0 = at ATH

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Growth factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False
        else:
            nan_pct = result[col].isna().mean()
            if nan_pct > 0.75:
                report["warnings"].append(f"High NaN in '{col}': {nan_pct:.1%}")

    return report


def benchmark(df: pd.DataFrame, context_ret: Optional[pd.Series] = None) -> dict:
    """Benchmarks compute execution time for this factor engine."""
    import time
    t0 = time.perf_counter()
    result = compute(df, context_ret)
    elapsed = time.perf_counter() - t0
    valid = validate(result)
    return {
        "factor": NAME,
        "version": VERSION,
        "rows": len(result),
        "columns": len(result.columns),
        "execution_seconds": round(elapsed, 4),
        "validation": valid,
    }
