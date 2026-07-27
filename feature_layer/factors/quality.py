"""
feature_layer/factors/quality.py
──────────────────────────────────
Quality Factor Engine (v1.0.0)

Category:    Fundamental Quality
Description: Computes profitability, capital efficiency, and balance sheet
             strength signals using Return on Equity (ROE), Return on Assets (ROA),
             Earnings Stability, Gross Margin, and Accruals ratio.

Alpha Families:
  1. Profitability     — ROE, ROA
  2. Earnings Quality  — Accruals ratio (lower = higher quality earnings)
  3. Growth Stability  — Earnings volatility, earnings growth consistency
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Quality"
CATEGORY = "Fundamental Quality"

OUTPUT_COLUMNS = [
    "roe",
    "roa",
    "earnings_growth",
    "earnings_stability",
    "gross_margin",
    "accruals_ratio",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes Quality factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
                      Optionally contains ROE, ROA, Earnings_Growth fundamental fields.
        context_ret:  Unused for Quality factors (included for interface consistency).
        price_col:    Column name for close price.

    Returns:
        DataFrame with Quality feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    # ── 1. Profitability Factors (from fundamental ingestor or NaN) ───────────
    for source_col, target_col in [("ROE", "roe"), ("ROA", "roa"), ("Earnings_Growth", "earnings_growth")]:
        if source_col in df.columns:
            result[target_col] = df[source_col].replace(0, np.nan)
        else:
            result[target_col] = np.nan

    # ── 2. Earnings Stability (rolling std of returns as profitability proxy) ─
    # Low return volatility ≈ consistent business model
    roll_std = daily_ret.rolling(126).std()  # 6-month earnings proxy
    # Invert so low volatility = high quality score
    result["earnings_stability"] = -roll_std

    # ── 3. Gross Margin Proxy (High/Low range relative to Close) ─────────────
    # Higher H-L spread relative to price signals higher pricing power uncertainty
    if "High" in df.columns and "Low" in df.columns:
        hl_range = (df["High"] - df["Low"]) / px.replace(0, np.nan)
        result["gross_margin"] = -hl_range.rolling(21).mean()  # Tighter range = stable pricing
    else:
        result["gross_margin"] = np.nan

    # ── 4. Accruals Ratio Proxy ───────────────────────────────────────────────
    # Use return autocorrelation as accruals proxy (low autocorr = cash-driven earnings)
    result["accruals_ratio"] = (
        daily_ret.rolling(21).apply(lambda x: x.autocorr() if len(x.dropna()) > 5 else np.nan, raw=False)
    )

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Quality factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    core_cols = ["roe", "roa", "earnings_growth"]
    for col in core_cols:
        if col in result.columns:
            non_null = result[col].notna().mean()
            if non_null < 0.1:
                report["warnings"].append(
                    f"Column '{col}' is mostly NaN ({non_null:.1%} valid). "
                    f"Fundamental data pipeline may not be connected."
                )

    computed_cols = ["earnings_stability", "gross_margin", "accruals_ratio"]
    for col in computed_cols:
        if col not in result.columns:
            report["warnings"].append(f"Missing computed column: {col}")
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
