"""
feature_layer/factors/alternative.py
───────────────────────────────────────
Alternative Data Factor Engine (v1.0.0)

Category:    Alternative & Proprietary Data
Description: Computes alternative data signals including calendar effects,
             options-market-implied sentiment proxies, technical pattern
             recognition, and structural price anomalies.

Alpha Families:
  1. Calendar Effects      — Day-of-week, month-of-year, pre-earnings drift
  2. Technical Patterns    — Candlestick pattern recognition (Doji, Hammer)
  3. Gap Analysis          — Overnight gap frequency and magnitude
  4. Seasonal Strength     — Month-end / window-dressing signal
  5. Relative Strength     — Cross-sectional rank-based signal
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Alternative"
CATEGORY = "Alternative Data"

OUTPUT_COLUMNS = [
    "weekday_effect",
    "month_end_effect",
    "overnight_gap",
    "gap_frequency",
    "doji_signal",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
    open_col: str = "Open",
) -> pd.DataFrame:
    """
    Computes Alternative Data factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Unused for Alternative factors.
        price_col:    Column name for close price.
        open_col:     Column name for open price.

    Returns:
        DataFrame with Alternative Data feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    # Ensure Date index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # ── 1. Calendar Effects ───────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        # Monday = 0 premium (weekend news absorption)
        result["weekday_effect"] = df.index.dayofweek.astype(float)
        # Month-end window dressing (last 3 trading days of month)
        result["month_end_effect"] = (df.index.day >= 28).astype(float)
    else:
        result["weekday_effect"]  = np.nan
        result["month_end_effect"] = np.nan

    # ── 2. Overnight Gap ─────────────────────────────────────────────────────
    if open_col in df.columns:
        prev_close = px.shift(1)
        gap_pct = (df[open_col] - prev_close) / prev_close.replace(0, np.nan)
        result["overnight_gap"] = gap_pct

        # Gap frequency: fraction of days in last 21 with |gap| > 0.5%
        result["gap_frequency"] = (
            gap_pct.abs() > 0.005
        ).rolling(21).mean()
    else:
        result["overnight_gap"]  = np.nan
        result["gap_frequency"]  = np.nan

    # ── 3. Doji Candlestick Signal ────────────────────────────────────────────
    # Doji: body is < 5% of High-Low range (indecision candle)
    if all(c in df.columns for c in [open_col, "High", "Low"]):
        body  = (px - df[open_col]).abs()
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        is_doji = (body / hl_range < 0.05).astype(float)
        result["doji_signal"] = is_doji.rolling(5).sum() / 5  # 5-day doji density
    else:
        result["doji_signal"] = np.nan

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Alternative Data factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False

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
