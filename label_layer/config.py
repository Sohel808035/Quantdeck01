"""
label_layer/config.py
──────────────────────
Configuration DTO for the QuantSphereX Label Factory.
Controls forecast horizons, classification thresholds,
ranking methods, and output naming conventions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import pandas as pd


# ── Standard institutional forecast horizons (trading days) ──────────────────
HORIZON_1D  = 1
HORIZON_5D  = 5
HORIZON_21D = 21
HORIZON_63D = 63

# Built-in horizon presets
STANDARD_HORIZONS = [HORIZON_1D, HORIZON_5D, HORIZON_21D, HORIZON_63D]


@dataclass
class LabelConfig:
    """
    Centralized configuration for the Label Factory.

    Args:
        horizons:               Forecast horizons in trading days.
        price_col:              OHLCV column used to compute returns.
        benchmark_col:          Benchmark column for excess return labels (optional).
        binary_threshold:       Min forward return for a positive binary label.
        tertile_thresholds:     Two quantile cutpoints for tertile labels [low_cut, high_cut].
        quintile_count:         Number of quantile buckets for ranking.
        apply_rank_labels:      Whether to produce cross-sectional percentile rank labels.
        apply_regression_labels: Whether to produce continuous return labels.
        apply_classification_labels: Whether to produce discrete class labels.
        apply_ranking_labels:   Whether to produce ordinal rank-bucket labels.
        drop_future_nans:       Whether to drop rows where future return is NaN.
    """
    horizons: List[int] = field(default_factory=lambda: STANDARD_HORIZONS)
    price_col: str = "Close"
    benchmark_col: Optional[str] = None

    # Classification thresholds
    binary_threshold: float = 0.0
    tertile_thresholds: List[float] = field(default_factory=lambda: [0.333, 0.667])
    quintile_count: int = 5

    # Label generation switches
    apply_regression_labels: bool = True
    apply_classification_labels: bool = True
    apply_ranking_labels: bool = True

    # Output behaviour
    drop_future_nans: bool = False
    label_prefix: str = "label"
