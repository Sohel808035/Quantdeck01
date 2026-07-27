"""
data_layer/storage.py
─────────────────────
Persistent Local Parquet Cache for Market & Index Panels.

Features:
  • Implement IDataCache interface
  • Thread-safe atomic file writes (temp file + rename)
  • MD5 hashing of sorted universe and date parameters
  • Metadata sidecar inspection
  • Lossless numeric type retention via PyArrow
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd  # type: ignore

from data_layer.config import DataConfig
from data_layer.interfaces import IDataCache

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
_LOCK = threading.Lock()


class ParquetCache(IDataCache):
    """
    Stores and retrieves raw OHLCV and Feature DataFrames as Parquet files.
    
    Cache key is an MD5 hash of (sorted_tickers, start_date, end_date).
    Atomic writes ensure cache files are never corrupted during unexpected interruptions.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or DataConfig()
        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = self.config.cache_dir
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, tickers: List[str], start: str, end: str) -> str:
        """Generates deterministic MD5 hash key for dataset parameters."""
        sorted_tickers = sorted(list(set(tickers)))
        key_str = ",".join(sorted_tickers) + f"|{start}|{end}"
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str, name: str) -> Path:
        return self.cache_dir / f"{name}_{key}.parquet"

    def _meta_path(self, key: str, name: str) -> Path:
        return self.cache_dir / f"{name}_{key}.json"

    # ── Public Interface Implementation ────────────────────────────────────────

    def exists(self, tickers: List[str], start: str, end: str, name: str = "stock") -> bool:
        """Returns True if the requested dataset exists in Parquet cache."""
        key = self._make_key(tickers, start, end)
        return self._path(key, name).exists()

    def save(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start: str,
        end: str,
        name: str = "stock",
    ) -> Path:
        """
        Saves DataFrame atomically using a temporary file to prevent partial corruption.
        Also writes a small JSON metadata sidecar.
        """
        if df.empty:
            logger.warning(f"[Cache] Refusing to save empty DataFrame for '{name}'.")
            key = self._make_key(tickers, start, end)
            return self._path(key, name)

        key = self._make_key(tickers, start, end)
        target_path = self._path(key, name)
        temp_path = self.cache_dir / f"{target_path.name}.tmp_{os.getpid()}"
        meta_path = self._meta_path(key, name)

        with _LOCK:
            try:
                # 1. Atomic write to temporary Parquet file
                df.to_parquet(temp_path, compression="snappy")
                if target_path.exists():
                    target_path.unlink()
                temp_path.replace(target_path)

                # 2. Write metadata sidecar
                metadata = {
                    "key": key,
                    "name": name,
                    "rows": len(df),
                    "columns": [str(c) for c in df.columns],
                    "tickers_count": len(tickers),
                    "start": start,
                    "end": end,
                    "created_at": pd.Timestamp.now().isoformat(),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"[Cache] Saved {name} → {target_path.name} ({len(df):,} rows)")
                return target_path

            except Exception as exc:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                logger.error(f"[Cache] Atomic save failed for '{name}': {exc}")
                raise exc

    def load(self, tickers: List[str], start: str, end: str, name: str = "stock") -> pd.DataFrame:
        """Loads dataset from local Parquet storage."""
        key = self._make_key(tickers, start, end)
        target_path = self._path(key, name)

        if not target_path.exists():
            raise FileNotFoundError(f"[Cache] Key not found in Parquet storage: {target_path}")

        logger.info(f"[Cache] Loading {name} ← {target_path.name}")
        df = pd.read_parquet(target_path)
        return df

    def invalidate(self, tickers: List[str], start: str, end: str, name: str = "stock") -> bool:
        """Deletes cached Parquet file and metadata sidecar."""
        key = self._make_key(tickers, start, end)
        target_path = self._path(key, name)
        meta_path = self._meta_path(key, name)
        removed = False

        with _LOCK:
            if target_path.exists():
                target_path.unlink()
                removed = True
                logger.info(f"[Cache] Invalidated {target_path.name}")
            if meta_path.exists():
                meta_path.unlink()

        return removed

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns statistics on local cache usage."""
        parquet_files = list(self.cache_dir.glob("*.parquet"))
        total_size_bytes = sum(f.stat().st_size for f in parquet_files)
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": len(parquet_files),
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "files": [f.name for f in parquet_files],
        }
