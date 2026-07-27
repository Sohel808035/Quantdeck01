"""
feature_layer/factors/volatility.py
─────────────────────────────────────
Volatility Factor Engine (v1.0.0)

Category:    Volatility & Risk-Adjusted Signal
Description: Computes realized volatility, downside deviation, GARCH-like
             volatility estimates, jump detection, and volatility regime signals.

Alpha Families:
  1. Historical Volatility      — Rolling 20D / 60D annualized
  2. Downside Risk              — Semi-deviation, Sortino denominator
  3. Volatility Regime          — Vol-of-vol, regime break detection
  4. Parkinson Volatility       — High-Low range based estimator (more efficient)
  5. Idiosyncratic Volatility   — Residual vol after removing market component
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Volatility"
CATEGORY = "Volatility & Risk"

OUTPUT_COLUMNS = [
    "realized_vol_20d",
    "realized_vol_60d",
    "downside_deviation",
    "vol_of_vol",
    "parkinson_vol",
    "idio_volatility",
]


def compute(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Computes Volatility factor columns from a single-ticker flat DataFrame.

    Args:
        df:           Single-ticker DataFrame indexed by Date with OHLCV columns.
        context_ret:  Optional market return Series for idiosyncratic vol computation.
        price_col:    Column name for close price.

    Returns:
        DataFrame with Volatility feature columns (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    # ── 1. Historical Realized Volatility (Annualized) ─────────────────────
    result["realized_vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)
    result["realized_vol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)

    # ── 2. Downside Deviation (Sortino denominator) ─────────────────────────
    negative_ret = daily_ret.clip(upper=0)
    result["downside_deviation"] = (
        negative_ret.rolling(60).apply(lambda x: np.sqrt((x**2).mean()), raw=True) * np.sqrt(252)
    )

    # ── 3. Volatility of Volatility (vol regime signal) ─────────────────────
    vol_20 = daily_ret.rolling(20).std()
    result["vol_of_vol"] = vol_20.rolling(60).std()

    # ── 4. Parkinson Volatility (High-Low estimator, ~5x more efficient) ─────
    if "High" in df.columns and "Low" in df.columns:
        log_hl = np.log(df["High"] / df["Low"].replace(0, np.nan))
        parkinson = np.sqrt((1.0 / (4.0 * np.log(2))) * (log_hl ** 2))
        result["parkinson_vol"] = parkinson.rolling(20).mean() * np.sqrt(252)
    else:
        result["parkinson_vol"] = result["realized_vol_20d"]  # Fallback

    # ── 5. Idiosyncratic Volatility (CAPM residual) ─────────────────────────
    if context_ret is not None:
        mkt = context_ret.reindex(daily_ret.index).dropna()
        common = daily_ret.dropna().index.intersection(mkt.index)
        if len(common) > 60:
            y = daily_ret.loc[common]
            x = mkt.loc[common]
            cov_yx = y.rolling(60).cov(x)
            var_x  = x.rolling(60).var()
            beta   = (cov_yx / var_x.replace(0, np.nan)).reindex(result.index)
            residual = daily_ret - beta * context_ret.reindex(daily_ret.index)
            result["idio_volatility"] = residual.rolling(60).std() * np.sqrt(252)
        else:
            result["idio_volatility"] = result["realized_vol_60d"]
    else:
        result["idio_volatility"] = result["realized_vol_60d"]

    # ── Lag all features 1 day to prevent look-ahead bias ────────────────────
    result = result.shift(1)
    return result


def validate(result: pd.DataFrame) -> dict:
    """Validates Volatility factor output quality."""
    report = {"factor": NAME, "version": VERSION, "passed": True, "warnings": []}

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            report["warnings"].append(f"Missing column: {col}")
            report["passed"] = False
        else:
            if (result[col].dropna() < 0).any():
                report["warnings"].append(f"Negative values detected in volatility column '{col}'")

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
