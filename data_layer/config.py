"""
data_layer/config.py
────────────────────
Centralized Configuration for QuantSphereX V2 Data Domain.
Provides strongly-typed parameters for batching, caching, retry logic,
rate limiting, validation thresholds, and storage paths.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _get_default_cache_dir() -> Path:
    """Resolves cross-platform data cache directory."""
    env_dir = os.environ.get("QUANTSPHEREX_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent.parent / "data_cache"


@dataclass
class DataConfig:
    """
    Configuration DTO for data ingestion, caching, and validation.
    """
    # Directory Settings
    cache_dir: Path = field(default_factory=_get_default_cache_dir)
    
    # Ingestion & Provider Settings
    batch_size: int = 50
    yfinance_suffix: str = ".NS"
    request_timeout_seconds: int = 30
    
    # Resilience & Rate Limiting Settings
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_initial_pause_seconds: float = 1.0
    rate_limit_pause_seconds: float = 0.5
    
    # Validation Thresholds
    max_price_jump_threshold: float = 0.30  # 30% daily jump
    stale_price_days_threshold: int = 5     # 5 identical closing prices
    max_corrupted_row_pct: float = 0.05     # Halt execution if > 5% corrupted
    allow_stale_data: bool = False           # Drop stale data if False
    
    # Default Historical Ranges
    default_start_date: str = "2005-01-01"
    
    def __post_init__(self):
        """Ensure paths are resolved Path objects and valid directory exists."""
        self.cache_dir = Path(self.cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
