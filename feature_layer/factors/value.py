"""
feature_layer/factors/value.py
────────────────────────────────
Value Factor Engine (v1.0.0)

Category:    Fundamental Valuation
Description: Computes price-based relative valuation signals including
             P/E ratio proxies, Price-to-Book analogs, intrinsic value
             distance, and relative valuation vs. sector.

Alpha Families:
  1. Relative Price Ratios  — P/E proxy, P/B proxy via historical cost basis
  2. Valuation Distance     — Price vs. 52W high/low anchors (Kahneman anchoring)
  3. Earnings Yield         — Inverse P/E, Earnings Yield spread
  4. Mean Reversion Value   — Distance from long-term intrinsic price trend
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Value"
CATEGORY = "Fundamental Valuation"

OUTPUT_COLUMNS = [
    "pe_proxy",
    "pb_proxy",
    "earnings_yield",
    "distance_52w_high",
    "distance_52w_low",
    "price_to_trend",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes Value factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Unused for Value factors.
        price_col:    Column name for close price.

    Returns:
        DataFrame with Value feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    # ── 1. P/E Proxy (from fundamentals or rolling earnings proxy) ───────────
    if "PE_Ratio" in df.columns:
        result["pe_proxy"] = df["PE_Ratio"].replace(0, np.nan)
        result["earnings_yield"] = 1.0 / result["pe_proxy"]
    else:
        # Proxy: inverse of price acceleration (rising fast = expensive)
        price_accel = px / px.rolling(252).mean()
        result["pe_proxy"] = price_accel
        result["earnings_yield"] = 1.0 / price_accel.replace(0, np.nan)

    # ── 2. P/B Proxy ─────────────────────────────────────────────────────────
    if "PB_Ratio" in df.columns:
        result["pb_proxy"] = df["PB_Ratio"].replace(0, np.nan)
    else:
        # Proxy: price relative to 3Y moving average (cost basis analog)
        rolling_3y = px.rolling(756).mean()  # ~3 years
        result["pb_proxy"] = (px / rolling_3y.replace(0, np.nan))

    # ── 3. 52-Week Anchor Valuations ──────────────────────────────────────────
    high_52w = px.rolling(252).max()
    low_52w  = px.rolling(252).min()
    result["distance_52w_high"] = (px / high_52w.replace(0, np.nan)) - 1   # Negative = discount
    result["distance_52w_low"]  = (px / low_52w.replace(0, np.nan)) - 1    # Positive = premium over floor

    # ── 4. Price-to-Long-Term Trend (HP Filter Proxy) ─────────────────────────
    trend = px.rolling(504).mean()  # 2-year rolling trend
    result["price_to_trend"] = (px - trend) / trend.replace(0, np.nan)

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Value factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False
        else:
            nan_pct = result[col].isna().mean()
            if nan_pct > 0.5:
                report["warnings"].append(f"High NaN ratio in '{col}': {nan_pct:.1%}")

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
