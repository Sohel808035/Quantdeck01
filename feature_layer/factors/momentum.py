"""
feature_layer/factors/momentum.py
──────────────────────────────────
Momentum Factor Engine (v1.0.0)

Category:    Momentum & Trend Following
Description: Computes multi-horizon price momentum, residual beta-adjusted
             momentum, mean-reversion indicators, and breakout strength signals.

Alpha Families:
  1. Raw Price Momentum  — 1M / 3M / 6M / 12M returns
  2. Residual Momentum   — Beta-neutralized sector residual momentum
  3. Mean Reversion      — 5D / 10D short-term bounce signals
  4. Trend Persistence   — Rolling Sharpe, Bollinger distance, Breakout Intensity
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Momentum"
CATEGORY = "Momentum & Trend"

OUTPUT_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "residual_momentum",
    "return_5d",
    "return_10d",
    "bollinger_distance",
    "rolling_sharpe",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes all Momentum factor columns for a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Optional sector or market return Series for beta adjustment.
        price_col:    Column name for close price. Defaults to 'Close'.

    Returns:
        DataFrame with Momentum feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    # ── 1. Raw Price Momentum ─────────────────────────────────────────────────
    result["return_1m"]  = px / px.shift(21) - 1
    result["return_3m"]  = px / px.shift(63) - 1
    result["return_6m"]  = px / px.shift(126) - 1
    result["return_12m"] = px / px.shift(252) - 1

    # ── 2. Residual Momentum (Beta-Neutralized) ───────────────────────────────
    if context_ret is not None:
        y = daily_ret.dropna()
        x = context_ret.reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) > 60:
            y_c, x_c = y.loc[common], x.loc[common]
            roll_cov = y_c.rolling(60).cov(x_c)
            roll_var = x_c.rolling(60).var()
            beta = (roll_cov / roll_var.replace(0, np.nan)).fillna(1.0)
            mkt_ret_1m = context_ret.rolling(21).sum().reindex(result.index).ffill()
            result["residual_momentum"] = (
                result["return_1m"] - (beta.reindex(result.index).ffill() * mkt_ret_1m)
            )
        else:
            result["residual_momentum"] = result["return_1m"]
    else:
        result["residual_momentum"] = result["return_1m"]

    # ── 3. Mean Reversion Signals ─────────────────────────────────────────────
    result["return_5d"]  = px / px.shift(5) - 1
    result["return_10d"] = px / px.shift(10) - 1

    ma20  = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    result["bollinger_distance"] = (px - ma20) / std20.replace(0, np.nan)

    # ── 4. Rolling Sharpe (Trend Persistence) ────────────────────────────────
    roll_mean = daily_ret.rolling(60).mean()
    roll_std  = daily_ret.rolling(60).std()
    result["rolling_sharpe"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Momentum factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False
        else:
            nan_pct = result[col].isna().mean()
            if nan_pct > 0.5:
                report["warnings"].append(f"High NaN ratio in {col}: {nan_pct:.1%}")

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
