"""
data_layer/storage.py
─────────────────────
Persistent local Parquet cache for raw OHLCV data.

Why Parquet?
• ~10× smaller than CSV, millisecond I/O for 200 stocks × 20 years
• Lossless numeric types (no CSV float round-trip problems)
• Eliminates redundant yfinance network calls
"""

from __future__ import annotations
import hashlib
import logging
import os
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# Default cache directory: project_root/data_cache/
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data_cache"


class ParquetCache:
    """
    Stores and retrieves raw OHLCV DataFrames as Parquet files.
    
    Cache key is an MD5 hash of (sorted_tickers, start_date, end_date).
    This means: if the same universe + date range is requested again, we
    load from disk instantly and never hit yfinance.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, tickers: list[str], start: str, end: str) -> str:
        key_str = ",".join(sorted(tickers)) + f"|{start}|{end}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _path(self, key: str, name: str) -> Path:
        return self.cache_dir / f"{name}_{key}.parquet"

    # ── public interface ──────────────────────────────────────────────────────

    def exists(self, tickers: list[str], start: str, end: str, name: str = "stock") -> bool:
        key = self._make_key(tickers, start, end)
        return self._path(key, name).exists()

    def save(self, df: pd.DataFrame, tickers: list[str], start: str, end: str, name: str = "stock") -> None:
        key = self._make_key(tickers, start, end)
        path = self._path(key, name)
        df.to_parquet(path)
        logger.info(f"[Cache] Saved {name} → {path.name}")

    def load(self, tickers: list[str], start: str, end: str, name: str = "stock") -> pd.DataFrame:
        key = self._make_key(tickers, start, end)
        path = self._path(key, name)
        logger.info(f"[Cache] Loading {name} ← {path.name}")
        return pd.read_parquet(path)

    def invalidate(self, tickers: list[str], start: str, end: str, name: str = "stock") -> None:
        """Force a fresh download by deleting the cache file."""
        key = self._make_key(tickers, start, end)
        path = self._path(key, name)
        if path.exists():
            path.unlink()
            logger.info(f"[Cache] Invalidated {path.name}")
