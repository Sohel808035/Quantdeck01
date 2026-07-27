"""
feature_layer/factors/liquidity.py
────────────────────────────────────
Liquidity Factor Engine (v1.0.0)

Category:    Market Liquidity & Microstructure
Description: Measures market microstructure quality, trading friction,
             and institutional accessibility using volume, turnover,
             bid-ask spread proxies, and Amihud illiquidity ratio.

Alpha Families:
  1. Amihud Illiquidity    — Price impact per unit of trading volume
  2. Turnover Ratio        — Volume / shares outstanding proxy
  3. Effective Spread      — High-Low intraday range as bid-ask proxy
  4. Volume Consistency    — Coefficient of variation of volume
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Liquidity"
CATEGORY = "Market Microstructure"

OUTPUT_COLUMNS = [
    "amihud_illiquidity",
    "turnover_ratio",
    "hl_spread_proxy",
    "volume_consistency",
    "liquidity_score",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """
    Computes Liquidity factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Unused for Liquidity factors.
        price_col:    Column name for close price.
        volume_col:   Column name for volume.

    Returns:
        DataFrame with Liquidity feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    if volume_col in df.columns:
        vol = df[volume_col].replace(0, np.nan)

        # ── 1. Amihud Illiquidity Ratio ───────────────────────────────────────
        # Amihud (2002): |Return| / Dollar Volume  — higher = less liquid
        dollar_vol = px * vol
        daily_amihud = daily_ret.abs() / dollar_vol.replace(0, np.nan)
        result["amihud_illiquidity"] = daily_amihud.rolling(21).mean() * 1e6  # Scaled

        # ── 2. Turnover Ratio (Volume / 30D Average Volume) ───────────────────
        avg_vol = vol.rolling(63).mean()
        result["turnover_ratio"] = vol / avg_vol.replace(0, np.nan)

        # ── 4. Volume Consistency (Coefficient of Variation, lower = more stable) ─
        vol_mean = vol.rolling(21).mean()
        vol_std  = vol.rolling(21).std()
        result["volume_consistency"] = -(vol_std / vol_mean.replace(0, np.nan))  # Negate: stable = good

    else:
        result["amihud_illiquidity"] = np.nan
        result["turnover_ratio"]     = np.nan
        result["volume_consistency"] = np.nan

    # ── 3. High-Low Spread Proxy (Corwin & Schultz, 2012) ────────────────────
    if "High" in df.columns and "Low" in df.columns:
        hl_ratio = np.log(df["High"] / df["Low"].replace(0, np.nan))
        result["hl_spread_proxy"] = hl_ratio.rolling(21).mean()
    else:
        result["hl_spread_proxy"] = np.nan

    # ── 5. Composite Liquidity Score ─────────────────────────────────────────
    # Z-score and aggregate (lower spread & amihud + higher volume = liquid)
    valid_cols = [c for c in ["amihud_illiquidity", "hl_spread_proxy", "turnover_ratio"]
                  if c in result.columns and result[c].notna().any()]
    if len(valid_cols) >= 2:
        z_parts = []
        for c in valid_cols:
            s = result[c]
            mu = s.rolling(252).mean()
            sd = s.rolling(252).std().replace(0, np.nan)
            z = (s - mu) / sd
            if c in ["amihud_illiquidity", "hl_spread_proxy"]:
                z = -z  # Invert: high illiquidity = low liquidity score
            z_parts.append(z)
        result["liquidity_score"] = pd.concat(z_parts, axis=1).mean(axis=1)
    else:
        result["liquidity_score"] = np.nan

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Liquidity factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    critical_cols = ["amihud_illiquidity", "liquidity_score"]
    for col in critical_cols:
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
