"""
label_layer/labels/regression.py
──────────────────────────────────
Regression Label Engine (v1.0.0)

Category:    Continuous Supervised Labels
Description: Generates continuous future return targets for regression
             models including XGBoost, LightGBM, and neural networks.
             Supports simple, log, excess, and risk-adjusted returns
             across multiple forecast horizons.

Label Families:
  1. Simple Forward Return      — (Price[t+H] / Price[t]) - 1
  2. Log Forward Return         — log(Price[t+H] / Price[t])
  3. Excess Return              — Forward return minus benchmark return
  4. Risk-Adjusted Return       — Forward return / trailing volatility
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Regression"
CATEGORY = "Continuous Labels"


def compute(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "Close",
    benchmark_ret: Optional[pd.Series] = None,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Computes continuous future return labels for every requested horizon.

    Labels at row t represent returns over (t+1, t+1+H) — fully forward-
    looking with no look-ahead at feature generation time (features are
    already lagged by 1 day in the feature factory).

    Args:
        df:            Single-ticker OHLCV DataFrame indexed by Date.
        horizons:      List of forecast horizons in trading days.
        price_col:     Close price column name.
        benchmark_ret: Benchmark daily return Series for excess return labels.
        vol_window:    Rolling window for trailing volatility (risk-adj labels).

    Returns:
        DataFrame of label columns indexed by Date (same index as input).
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    for H in horizons:
        future_px = px.shift(-H)
        label_base = f"fwd_{H}d"

        # ── 1. Simple Forward Return ──────────────────────────────────────────
        result[f"ret_{label_base}"] = future_px / px.replace(0, np.nan) - 1

        # ── 2. Log Forward Return ─────────────────────────────────────────────
        result[f"log_ret_{label_base}"] = np.log(
            future_px / px.replace(0, np.nan)
        )

        # ── 3. Excess Return over Benchmark ──────────────────────────────────
        if benchmark_ret is not None:
            bm = benchmark_ret.reindex(df.index).fillna(0)
            # Compound benchmark return over horizon H
            bm_compound = (1 + bm).rolling(H).apply(np.prod, raw=True).shift(-(H - 1))
            excess = result[f"ret_{label_base}"] - (bm_compound - 1)
            result[f"excess_ret_{label_base}"] = excess

        # ── 4. Risk-Adjusted Forward Return (Sharpe analog) ──────────────────
        trailing_vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)
        result[f"risk_adj_ret_{label_base}"] = (
            result[f"ret_{label_base}"] / trailing_vol.replace(0, np.nan)
        )

    return result


def validate(result: pd.DataFrame, horizons: List[int]) -> dict:
    """Validates regression label output for expected columns and value ranges."""
    report = {"label_engine": NAME, "version": VERSION, "passed": True, "warnings": []}

    for H in horizons:
        col = f"ret_fwd_{H}d"
        if col not in result.columns:
            report["warnings"].append(f"Missing required column: {col}")
            report["passed"] = False
        else:
            non_null_pct = result[col].notna().mean()
            if non_null_pct < 0.5:
                report["warnings"].append(
                    f"Low valid ratio in '{col}': {non_null_pct:.1%} — "
                    f"last {H} rows will always be NaN."
                )

    return report
