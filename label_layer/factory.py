"""
label_layer/factory.py
───────────────────────
QuantSphereX Label Factory Orchestrator (v1.0.0)

Provides a unified entry point for generating all label types
(Regression, Classification, Ranking) across multiple forecast
horizons for any single-ticker or multi-ticker panel DataFrame.

Backward-compatible with alpha_layer/targets.py:
  - build_label_panel() mirrors build_target_panel() interface
  - TARGET_COL constant preserved for drop-in replacement

Design Principles:
  - No look-ahead bias: all labels use shift(-H) on price only
  - Feature lag is handled at feature factory level (shift +1)
  - Custom labels registered at runtime, no code changes needed
"""

from __future__ import annotations
import logging
import time
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from label_layer.config import LabelConfig, STANDARD_HORIZONS
from label_layer.labels import regression, classification, ranking, horizons
from label_layer.labels.custom import CustomLabelBuilder

logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# Backward-compatibility alias — mirrors alpha_layer/targets.TARGET_COL
TARGET_COL = "label_rank_pct_fwd_21d"


# ─────────────────────────────────────────────────────────────────────────────
# Single-Ticker Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_labels(
    df: pd.DataFrame,
    config: Optional[LabelConfig] = None,
    benchmark_ret: Optional[pd.Series] = None,
    custom_builder: Optional[CustomLabelBuilder] = None,
) -> pd.DataFrame:
    """
    Generates all label types for a single-ticker flat OHLCV DataFrame.

    Args:
        df:              Single-ticker DataFrame indexed by Date with OHLCV columns.
        config:          LabelConfig (defaults applied if not provided).
        benchmark_ret:   Benchmark daily return Series (for excess return labels).
        custom_builder:  Optional CustomLabelBuilder with user-defined labels.

    Returns:
        DataFrame with all label columns (same Date index as input).
    """
    if config is None:
        config = LabelConfig()

    frames = []

    # ── Regression Labels ─────────────────────────────────────────────────────
    if config.apply_regression_labels:
        reg_labels = regression.compute(
            df,
            horizons=config.horizons,
            price_col=config.price_col,
            benchmark_ret=benchmark_ret,
        )
        frames.append(reg_labels)

    # ── Classification Labels ─────────────────────────────────────────────────
    if config.apply_classification_labels:
        cls_labels = classification.compute(
            df,
            horizons=config.horizons,
            price_col=config.price_col,
            benchmark_ret=benchmark_ret,
            binary_threshold=config.binary_threshold,
            tertile_thresholds=config.tertile_thresholds,
            quintile_count=config.quintile_count,
        )
        frames.append(cls_labels)

    # ── Raw forward returns for cross-sectional ranking ───────────────────────
    raw_fwd = ranking.compute_single_ticker(
        df, horizons=config.horizons, price_col=config.price_col
    )
    frames.append(raw_fwd)

    # ── Custom Labels ─────────────────────────────────────────────────────────
    if custom_builder is not None and custom_builder.list_labels():
        custom_labels = custom_builder.compute(df)
        frames.append(custom_labels)

    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Ticker Panel Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_label_panel(
    stock_panel: pd.DataFrame,
    config: Optional[LabelConfig] = None,
    benchmark_ret: Optional[pd.Series] = None,
    custom_builder: Optional[CustomLabelBuilder] = None,
) -> pd.DataFrame:
    """
    Generates all labels for a multi-ticker (Date, Ticker) MultiIndex panel.

    Ranking labels are applied cross-sectionally after per-ticker forward
    returns are computed — matching the original build_target_panel() approach.

    Args:
        stock_panel:     Multi-ticker OHLCV panel with (Date, Ticker) MultiIndex.
        config:          LabelConfig (defaults applied if not provided).
        benchmark_ret:   Benchmark daily return Series.
        custom_builder:  Optional CustomLabelBuilder with user-defined labels.

    Returns:
        Panel DataFrame with (Date, Ticker) MultiIndex and all label columns.
    """
    if config is None:
        config = LabelConfig()

    t0 = time.perf_counter()
    logger.info(
        f"Label Factory building labels for "
        f"{stock_panel.index.get_level_values('Ticker').nunique()} tickers | "
        f"horizons={config.horizons}"
    )

    all_frames = []
    tickers = stock_panel.index.get_level_values("Ticker").unique()

    for ticker in tickers:
        try:
            ticker_df = stock_panel.xs(ticker, level="Ticker")
            ticker_labels = build_ticker_labels(
                ticker_df,
                config=config,
                benchmark_ret=benchmark_ret,
                custom_builder=custom_builder,
            )
            ticker_labels["Ticker"] = ticker
            ticker_labels = ticker_labels.reset_index().rename(
                columns={"index": "Date"}
            ).set_index(["Date", "Ticker"])
            all_frames.append(ticker_labels)
        except Exception as e:
            logger.warning(f"Label generation failed for {ticker}: {e}")

    if not all_frames:
        logger.error("Label Factory produced no output.")
        return pd.DataFrame()

    panel = pd.concat(all_frames).sort_index()

    # ── Apply Cross-Sectional Ranking Labels ──────────────────────────────────
    if config.apply_ranking_labels:
        logger.info("Applying cross-sectional ranking labels...")
        panel = ranking.compute_panel(panel, horizons=config.horizons)

    elapsed = time.perf_counter() - t0
    n_labels = len(panel.columns)
    n_valid  = panel.notna().all(axis=1).sum()
    logger.info(
        f"Label Factory complete | {n_labels} label columns | "
        f"{n_valid:,} fully-valid rows | {elapsed:.2f}s"
    )

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Backward-Compatibility Shim
# Mirrors alpha_layer/targets.build_target_panel() exactly
# ─────────────────────────────────────────────────────────────────────────────

def build_target_panel(
    stock_panel: pd.DataFrame,
    price_col: str = "Close",
    horizon: int = 21,
) -> pd.DataFrame:
    """
    Backward-compatible drop-in replacement for alpha_layer/targets.build_target_panel().

    Generates cross-sectional percentile rank labels for a single horizon
    and exposes them under the legacy TARGET_COL name for the ML pipeline.

    Args:
        stock_panel:  Multi-ticker panel with (Date, Ticker) MultiIndex.
        price_col:    Close price column.
        horizon:      Forecast horizon in trading days (default: 21D).

    Returns:
        stock_panel with a TARGET_COL column appended (range [0, 1]).
    """
    logger.info(f"[Compat] Building {horizon}-day cross-sectional rank targets...")

    config = LabelConfig(
        horizons=[horizon],
        price_col=price_col,
        apply_regression_labels=False,
        apply_classification_labels=False,
        apply_ranking_labels=True,
    )

    label_panel = build_label_panel(stock_panel, config=config)

    rank_col = f"rank_pct_fwd_{horizon}d"
    result = stock_panel.copy()

    if rank_col in label_panel.columns:
        result[TARGET_COL] = label_panel[rank_col].reindex(result.index)
    else:
        # Fallback: direct percentile rank
        raw_fwd = label_panel.get(f"_raw_fwd_{horizon}d", pd.Series(dtype=float))
        result[TARGET_COL] = raw_fwd.groupby(level=0).rank(pct=True)

    n_valid = result[TARGET_COL].notna().sum()
    logger.info(
        f"  [Compat] Rank target created. Valid rows: {n_valid:,} "
        f"(Range: [{result[TARGET_COL].min():.2f}, {result[TARGET_COL].max():.2f}])"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Registry Info
# ─────────────────────────────────────────────────────────────────────────────

def get_label_registry() -> Dict:
    """Returns metadata for all label engine modules."""
    return {
        "factory_version": VERSION,
        "engines": [
            {"name": regression.NAME, "category": regression.CATEGORY, "version": regression.VERSION},
            {"name": classification.NAME, "category": classification.CATEGORY, "version": classification.VERSION},
            {"name": ranking.NAME, "category": ranking.CATEGORY, "version": ranking.VERSION},
            {"name": horizons.NAME, "category": horizons.CATEGORY, "version": horizons.VERSION},
            {"name": "Custom", "category": "User-Defined Labels", "version": "1.0.0"},
        ],
        "standard_horizons": STANDARD_HORIZONS,
        "default_target_col": TARGET_COL,
    }
