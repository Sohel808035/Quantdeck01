"""
feature_layer/config.py
───────────────────────
Configuration Parameters for Feature Factory Factor Engines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureFactoryConfig:
    """Configuration DTO for Feature Factory generation and post-processing."""
    # Winsorization & Z-Scoring
    winsorize_lower_pct: float = 0.01  # 1% tail
    winsorize_upper_pct: float = 0.99  # 99% tail
    zscore_epsilon: float = 1e-6

    # Orthogonalization
    correlation_drop_threshold: float = 0.60
    min_family_protected: bool = True

    # Factor Windows
    momentum_short_window: int = 21   # 1M
    momentum_medium_window: int = 63  # 3M
    momentum_long_window: int = 126   # 6M
    volatility_short_window: int = 20
    volatility_long_window: int = 60

    # Sector Neutralization
    apply_sector_neutralization: bool = True
    apply_cross_sectional_ranking: bool = True
