"""
label_layer/labels/custom.py
──────────────────────────────
Custom Label Engine (v1.0.0)

Category:    User-Defined & Extensible Labels
Description: Provides a plug-in interface for defining arbitrary custom
             labels without modifying any core factory code. Supports
             callable label functions, threshold-based classification,
             and composite multi-signal labels.

Usage Pattern:
    from label_layer.labels.custom import CustomLabelBuilder

    builder = CustomLabelBuilder()
    builder.register("my_label", lambda df: (df["Close"].pct_change(10) > 0.05).astype(float))

    result = builder.compute(df)
"""

from __future__ import annotations
import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
NAME = "Custom"
CATEGORY = "User-Defined Labels"

# Type alias
LabelFn = Callable[[pd.DataFrame], pd.Series]


class CustomLabelBuilder:
    """
    Registry and executor for user-defined custom label functions.

    Each label is a pure function:
        (df: pd.DataFrame) -> pd.Series

    where df is a single-ticker OHLCV DataFrame indexed by Date.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, LabelFn] = {}

    def register(self, name: str, fn: LabelFn) -> "CustomLabelBuilder":
        """
        Registers a custom label function.

        Args:
            name: Column name for the generated label.
            fn:   Callable accepting a single-ticker DataFrame, returning a Series.

        Returns:
            Self for method chaining.
        """
        if not callable(fn):
            raise ValueError(f"Label function for '{name}' must be callable.")
        self._registry[name] = fn
        logger.debug(f"Custom label '{name}' registered.")
        return self

    def unregister(self, name: str) -> "CustomLabelBuilder":
        """Removes a previously registered custom label."""
        self._registry.pop(name, None)
        return self

    def list_labels(self) -> List[str]:
        """Returns names of all registered custom labels."""
        return list(self._registry.keys())

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes all registered custom label functions on the input DataFrame.

        Args:
            df: Single-ticker OHLCV DataFrame indexed by Date.

        Returns:
            DataFrame with one column per registered custom label.
        """
        result = pd.DataFrame(index=df.index)

        for name, fn in self._registry.items():
            try:
                series = fn(df)
                if not isinstance(series, pd.Series):
                    series = pd.Series(series, index=df.index)
                result[f"custom_{name}"] = series.reindex(df.index)
            except Exception as e:
                logger.warning(f"Custom label '{name}' failed: {e}")
                result[f"custom_{name}"] = np.nan

        return result

    def validate(self, result: pd.DataFrame) -> dict:
        """Validates custom label output columns exist and have data."""
        report = {"label_engine": NAME, "version": VERSION, "passed": True, "warnings": []}

        for name in self._registry:
            col = f"custom_{name}"
            if col not in result.columns:
                report["warnings"].append(f"Missing custom label column: {col}")
                report["passed"] = False
            else:
                null_pct = result[col].isna().mean()
                if null_pct > 0.9:
                    report["warnings"].append(f"Custom label '{col}' is {null_pct:.0%} NaN.")

        return report


# ── Built-in Custom Label Presets ─────────────────────────────────────────────

def make_threshold_label(
    price_col: str = "Close",
    horizon: int = 21,
    threshold: float = 0.05,
) -> LabelFn:
    """
    Factory: Returns a label function that classifies stocks as 1 if their
    H-day forward return exceeds the given threshold, else 0.

    Args:
        price_col:  Column for close price.
        horizon:    Forecast horizon in trading days.
        threshold:  Return threshold for positive classification.
    """
    def _fn(df: pd.DataFrame) -> pd.Series:
        px = df[price_col]
        fwd_ret = px.shift(-horizon) / px.replace(0, np.nan) - 1
        label = (fwd_ret > threshold).astype(float)
        label[fwd_ret.isna()] = np.nan
        return label
    return _fn


def make_drawdown_label(
    price_col: str = "Close",
    horizon: int = 21,
    drawdown_threshold: float = -0.10,
) -> LabelFn:
    """
    Factory: Returns a label function that flags extreme drawdown risk.
    Label = 1 if max drawdown within horizon exceeds threshold, else 0.

    Args:
        price_col:           Column for close price.
        horizon:             Look-ahead window in trading days.
        drawdown_threshold:  Negative return threshold (default: -10%).
    """
    def _fn(df: pd.DataFrame) -> pd.Series:
        px = df[price_col]
        labels = pd.Series(np.nan, index=df.index)
        for i in range(len(px) - horizon):
            future_slice = px.iloc[i + 1 : i + horizon + 1]
            if len(future_slice) == 0:
                continue
            peak = px.iloc[i]
            max_dd = (future_slice.min() / peak) - 1
            labels.iloc[i] = float(max_dd < drawdown_threshold)
        return labels
    return _fn


def make_composite_label(
    label_fns: List[LabelFn],
    weights: Optional[List[float]] = None,
) -> LabelFn:
    """
    Factory: Combines multiple label functions into a weighted composite label.

    Args:
        label_fns:  List of label callables.
        weights:    Optional weights. If None, equal weights are applied.
    """
    if weights is None:
        weights = [1.0 / len(label_fns)] * len(label_fns)

    if len(weights) != len(label_fns):
        raise ValueError("weights length must match label_fns length")

    def _fn(df: pd.DataFrame) -> pd.Series:
        parts = []
        for fn, w in zip(label_fns, weights):
            try:
                parts.append(fn(df) * w)
            except Exception:
                pass
        if not parts:
            return pd.Series(np.nan, index=df.index)
        return pd.concat(parts, axis=1).sum(axis=1)

    return _fn
