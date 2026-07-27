"""
feature_layer/factors/macro.py
────────────────────────────────
Macro Factor Engine (v1.0.0)

Category:    Macroeconomic Sensitivity
Description: Measures systematic macroeconomic sensitivity by computing
             beta exposures to benchmark indices, volatility regimes,
             and long-term trend state signals.

Alpha Families:
  1. Market Beta         — Rolling 60D CAPM beta vs. Nifty 50 / benchmark
  2. Volatility Regime   — India VIX-based regime signal (high/low vol)
  3. Market Breadth      — Rolling correlation with market return
  4. Regime Indicator    — 200-DMA trend filter derived from market benchmark
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Macro"
CATEGORY = "Macroeconomic Sensitivity"

OUTPUT_COLUMNS = [
    "market_beta",
    "macro_correlation",
    "vix_regime",
    "above_200dma",
    "macro_regime_score",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes Macro factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Market benchmark return Series (Nifty 50). Needed for beta.
        price_col:    Column name for close price.

    Returns:
        DataFrame with Macro feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    # ── 1. Market Beta ────────────────────────────────────────────────────────
    if context_ret is not None:
        mkt = context_ret.reindex(daily_ret.index).fillna(0)
        cov_ym = daily_ret.rolling(60).cov(mkt)
        var_m  = mkt.rolling(60).var()
        result["market_beta"]      = (cov_ym / var_m.replace(0, np.nan))
        result["macro_correlation"] = daily_ret.rolling(60).corr(mkt)
    else:
        result["market_beta"]      = np.nan
        result["macro_correlation"] = np.nan

    # ── 2. Volatility Regime (VIX proxy from context or local realized vol) ───
    if "VIX" in df.columns:
        vix = df["VIX"]
        vix_threshold = 20.0  # Standard Nifty VIX threshold
        result["vix_regime"] = (vix > vix_threshold).astype(float)
    else:
        # Proxy: if 20D realized vol > 1.5x of 1Y realized vol → high vol regime
        vol_20d  = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_base = daily_ret.rolling(252).std() * np.sqrt(252)
        result["vix_regime"] = (vol_20d > vol_base * 1.5).astype(float)

    # ── 3. Trend Filter (200 DMA Above/Below) ────────────────────────────────
    ma_200 = px.rolling(200).mean()
    result["above_200dma"] = (px > ma_200).astype(float)

    # ── 4. Macro Regime Score (composite) ────────────────────────────────────
    # 1 = Bull (above 200DMA, low vol), -1 = Bear, 0 = Neutral
    regime = pd.Series(0.0, index=result.index)
    if "above_200dma" in result.columns:
        regime += result["above_200dma"] * 0.5
    if "vix_regime" in result.columns:
        regime -= result["vix_regime"] * 0.5  # High VIX = bearish
    result["macro_regime_score"] = regime

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Macro factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    critical = ["above_200dma", "vix_regime", "macro_regime_score"]
    for col in critical:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False

    if "market_beta" in result.columns and result["market_beta"].notna().mean() < 0.1:
        report["warnings"].append(
            "market_beta is mostly NaN. Pass market benchmark returns via context_ret."
        )

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
