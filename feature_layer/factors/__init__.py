"""
feature_layer/factors — Independent Factor Engine Registry.

Each module is a standalone quantitative factor family with its own:
  - compute()     : Pure function to compute factor columns from OHLCV data
  - validate()    : Quality checks on output columns
  - benchmark()   : Timed execution with validation report
  - VERSION       : Semantic version string
  - OUTPUT_COLUMNS: List of generated column names
"""

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

__all__ = [
    "momentum",
    "quality",
    "value",
    "growth",
    "liquidity",
    "volatility",
    "macro",
    "sentiment",
    "alternative",
]
