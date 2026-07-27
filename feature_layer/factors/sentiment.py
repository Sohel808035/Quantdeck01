"""
feature_layer/factors/sentiment.py
────────────────────────────────────
Sentiment Factor Engine (v1.0.0)

Category:    Market Sentiment & Behavioral
Description: Captures market microstructure sentiment proxies including
             insider-behaviour approximations, contrarian signals,
             herding behaviour, and supply/demand imbalances.

Alpha Families:
  1. Price Accumulation    — On-Balance Volume (OBV) trend
  2. Demand Pressure       — Close vs. High-Low range position
  3. Short Interest Proxy  — High gap-down frequency as squeeze proxy
  4. Contrarian Signal     — RSI-based overbought/oversold
  5. Herding Signal        — Rolling autocorrelation of cross-sectional returns
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Sentiment"
CATEGORY = "Market Sentiment & Behavioral"

OUTPUT_COLUMNS = [
    "obv_trend",
    "williams_pct_r",
    "rsi_14",
    "money_flow_index",
    "close_location_value",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """
    Computes Sentiment factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Unused for Sentiment factors.
        price_col:    Column name for close price.
        volume_col:   Column name for volume.

    Returns:
        DataFrame with Sentiment feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()

    # ── 1. On-Balance Volume Trend (OBV) ─────────────────────────────────────
    if volume_col in df.columns:
        vol = df[volume_col].replace(0, np.nan).fillna(0)
        ret_sign = np.sign(px.pct_change())
        obv = (vol * ret_sign).cumsum()
        obv_sma = obv.rolling(21).mean()
        result["obv_trend"] = (obv - obv_sma) / obv_sma.abs().replace(0, np.nan)
    else:
        result["obv_trend"] = np.nan

    # ── 2. Williams %R (14-day) ───────────────────────────────────────────────
    if "High" in df.columns and "Low" in df.columns:
        high14 = df["High"].rolling(14).max()
        low14  = df["Low"].rolling(14).min()
        result["williams_pct_r"] = (high14 - px) / (high14 - low14).replace(0, np.nan) * -100
    else:
        result["williams_pct_r"] = np.nan

    # ── 3. RSI (14-day) ───────────────────────────────────────────────────────
    delta = px.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    # ── 4. Money Flow Index (MFI, 14-day) ────────────────────────────────────
    if all(c in df.columns for c in ["High", "Low", volume_col]):
        typical_price = (df["High"] + df["Low"] + px) / 3
        raw_mf = typical_price * df[volume_col].replace(0, np.nan)
        pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
        neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
        pos_sum = pos_mf.rolling(14).sum()
        neg_sum = neg_mf.rolling(14).sum()
        mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
        result["money_flow_index"] = 100 - (100 / (1 + mfi_ratio))
    else:
        result["money_flow_index"] = np.nan

    # ── 5. Close Location Value (CLV) ────────────────────────────────────────
    # CLV = ((Close - Low) - (High - Close)) / (High - Low)
    # Positive = closes near high (bullish), Negative = closes near low (bearish)
    if "High" in df.columns and "Low" in df.columns:
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        clv = ((px - df["Low"]) - (df["High"] - px)) / hl_range
        result["close_location_value"] = clv.rolling(21).mean()
    else:
        result["close_location_value"] = np.nan

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Sentiment factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in ["rsi_14"]:
        if col in result.columns:
            out_of_range = ((result[col] < 0) | (result[col] > 100)).sum()
            if out_of_range > 0:
                report["warnings"].append(f"RSI out of [0,100] range: {out_of_range} rows")
                report["passed"] = False

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
