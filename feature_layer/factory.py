"""
feature_layer/factory.py
──────────────────────────
QuantSphereX Feature Factory Orchestrator (v2.0.0)

Refactored from the monolithic feature_layer/implementations.py.
Each factor family is now an independent, versioned module.
The factory discovers and executes all registered engines, combining
results into a single merged feature panel.

Preserved backward compatibility with existing ML pipeline.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from feature_layer.factors import (
    momentum,
    quality,
    value,
    growth,
    liquidity,
    volatility,
    macro,
    sentiment,
    alternative,
)
from feature_layer.config import FeatureFactoryConfig

logger = logging.getLogger(__name__)

VERSION = "2.0.0"

# Factor module registry — ordered by category
FACTOR_REGISTRY = [
    momentum,
    quality,
    value,
    growth,
    liquidity,
    volatility,
    macro,
    sentiment,
    alternative,
]


def get_registry_info() -> List[Dict]:
    """Returns metadata for all registered factor engines."""
    return [
        {"name": m.NAME, "category": m.CATEGORY, "version": m.VERSION, "columns": m.OUTPUT_COLUMNS}
        for m in FACTOR_REGISTRY
    ]


def compute_all_factors(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    config: Optional[FeatureFactoryConfig] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Executes all registered factor engines on a single-ticker DataFrame
    and merges results into one wide feature matrix.

    Args:
        df:           Single-ticker OHLCV DataFrame indexed by Date.
        context_ret:  Market benchmark return Series for beta-based factors.
        config:       FeatureFactoryConfig (uses defaults if not provided).
        price_col:    Price column name. Defaults to 'Close'.

    Returns:
        DataFrame with all factor columns (same Date index as input).
    """
    if config is None:
        config = FeatureFactoryConfig()

    frames = []
    benchmarks = []

    for module in FACTOR_REGISTRY:
        try:
            bm = module.benchmark(df, context_ret=context_ret)
            result = module.compute(df, context_ret=context_ret)
            frames.append(result)
            benchmarks.append(bm)
            status = "✓" if bm["validation"]["passed"] else "⚠"
            logger.debug(
                f"{status} [{module.NAME}] {bm['columns']} cols | {bm['execution_seconds']:.4f}s"
            )
        except Exception as e:
            logger.warning(f"Factor module {module.NAME} failed: {e}")

    if not frames:
        logger.error("No factor modules produced output.")
        return pd.DataFrame(index=df.index)

    combined = pd.concat(frames, axis=1)

    # ── Post-Processing ────────────────────────────────────────────────────────
    if config.apply_cross_sectional_ranking:
        combined = _apply_zscore(combined, config)

    _log_benchmark_report(benchmarks)
    return combined


def _apply_zscore(df: pd.DataFrame, config: FeatureFactoryConfig) -> pd.DataFrame:
    """Rolling Z-score normalization with winsorization per feature column."""
    result = df.copy()
    for col in result.columns:
        s = result[col]
        # Winsorize
        lo = s.quantile(config.winsorize_lower_pct)
        hi = s.quantile(config.winsorize_upper_pct)
        s = s.clip(lower=lo, upper=hi)
        # Z-score
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        result[col] = (s - mu) / (sd + config.zscore_epsilon)

    return result


def _log_benchmark_report(benchmarks: List[Dict]) -> None:
    """Logs a structured benchmark table for all factor engines."""
    logger.info("=" * 72)
    logger.info(f"{'Factor Engine':<22} {'Version':<10} {'Cols':>6} {'Time(s)':>9} {'Status':<10}")
    logger.info("-" * 72)
    for bm in benchmarks:
        status = "PASS" if bm["validation"]["passed"] else "WARN"
        logger.info(
            f"{bm['factor']:<22} v{bm['version']:<9} {bm['columns']:>6} "
            f"{bm['execution_seconds']:>8.4f}s {status:<10}"
        )
    logger.info("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-Compatibility Shim
# Preserves original compute_features(panel_df) API for the ML pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(panel_df: pd.DataFrame, config: Optional[FeatureFactoryConfig] = None) -> pd.DataFrame:
    """
    Backward-compatible entry point for the ML pipeline.

    Accepts a (Date, Ticker) MultiIndex panel DataFrame and returns
    a wide feature matrix per (Date, Ticker) row.

    Args:
        panel_df:  Multi-ticker panel with (Date, Ticker) MultiIndex.
        config:    Feature factory configuration.

    Returns:
        Wide feature DataFrame with same MultiIndex as input.
    """
    if config is None:
        config = FeatureFactoryConfig()

    all_frames = []

    if isinstance(panel_df.index, pd.MultiIndex):
        tickers = panel_df.index.get_level_values("Ticker").unique()

        for ticker in tickers:
            try:
                ticker_df = panel_df.xs(ticker, level="Ticker")
                features = compute_all_factors(ticker_df, config=config)
                features["Ticker"] = ticker
                features = features.reset_index().set_index(["Date", "Ticker"])
                all_frames.append(features)
            except Exception as e:
                logger.warning(f"Skipping ticker {ticker}: {e}")

    else:
        # Flat single-ticker DataFrame
        all_frames.append(compute_all_factors(panel_df, config=config))

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames).sort_index()
