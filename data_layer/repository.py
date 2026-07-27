"""
data_layer/repository.py
────────────────────────
Repository Pattern Implementation for QuantSphereX Data Domain.
Provides a unified domain interface (`IMarketDataRepository`) decoupled from underlying
storage technologies (Parquet, PostgreSQL/TimescaleDB, Redis, SQLite).
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd

from data_layer.config import DataConfig
from data_layer.storage import ParquetCache
from data_layer.versioning import DatasetVersionManager
from db.redis_client import RedisCacheManager

logger = logging.getLogger(__name__)


class IMarketDataRepository(ABC):
    """Abstract Repository Interface for Market & Macro Data Queries."""

    @abstractmethod
    def get_stock_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieves stock OHLCV panel as a MultiIndex (Date, Ticker) DataFrame."""
        pass

    @abstractmethod
    def save_stock_panel(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        """Persists stock OHLCV panel and returns storage location/version ID."""
        pass

    @abstractmethod
    def get_nifty50(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieves NIFTY 50 macro index data."""
        pass

    @abstractmethod
    def get_india_vix(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieves India VIX macro index data."""
        pass


class ParquetRepository(IMarketDataRepository):
    """
    Concrete Parquet Repository with Dataset Versioning support.
    """

    def __init__(
        self,
        cache: Optional[ParquetCache] = None,
        version_mgr: Optional[DatasetVersionManager] = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or DataConfig()
        self.cache = cache or ParquetCache(config=self.config)
        self.version_mgr = version_mgr or DatasetVersionManager(config=self.config)

    def get_stock_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        end_str = end_date or "today"
        if self.cache.exists(tickers, start_date, end_str, name="stock"):
            return self.cache.load(tickers, start_date, end_str, name="stock")
        return pd.DataFrame()

    def save_stock_panel(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        end_str = end_date or "today"
        self.cache.save(df, tickers, start_date, end_str, name="stock")
        version_id = self.version_mgr.create_version(
            dataset_name="stock_panel",
            df=df,
            start_date=start_date,
            end_date=end_str,
        )
        return version_id

    def get_nifty50(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        end_str = end_date or "today"
        if self.cache.exists(["^NSEI"], start_date, end_str, name="nifty50"):
            return self.cache.load(["^NSEI"], start_date, end_str, name="nifty50")
        return pd.DataFrame()

    def get_india_vix(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        end_str = end_date or "today"
        if self.cache.exists(["^INDIAVIX"], start_date, end_str, name="vix"):
            return self.cache.load(["^INDIAVIX"], start_date, end_str, name="vix")
        return pd.DataFrame()


class HybridDataRepository(IMarketDataRepository):
    """
    Master Institutional Hybrid Repository.
    Orchestrates L1 (Redis RAM Cache) -> L2 (Parquet / Versioned Storage) for multi-tiered performance.
    """

    def __init__(
        self,
        parquet_repo: Optional[ParquetRepository] = None,
        redis_cache: Optional[RedisCacheManager] = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or DataConfig()
        self.parquet_repo = parquet_repo or ParquetRepository(config=self.config)
        self.redis_cache = redis_cache or RedisCacheManager()

    def _cache_key(self, name: str, tickers: List[str], start: str, end: str) -> str:
        t_hash = hash(",".join(sorted(tickers))) & 0xFFFFFFFF
        return f"repo:{name}:{t_hash}:{start}:{end}"

    def get_stock_panel(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        end_str = end_date or "today"
        ckey = self._cache_key("stock", tickers, start_date, end_str)

        # 1. Query L1 Redis Cache
        cached_df = self.redis_cache.get_dataframe(ckey)
        if cached_df is not None and not cached_df.empty:
            logger.info(f"[HybridRepo] L1 RAM Cache Hit for stock panel ({len(cached_df):,} rows)")
            return cached_df

        # 2. Query L2 Parquet Storage
        df = self.parquet_repo.get_stock_panel(tickers, start_date, end_str)
        if not df.empty:
            # Populate L1 RAM Cache
            self.redis_cache.set_dataframe(ckey, df, ttl=3600)
            logger.info(f"[HybridRepo] L2 Parquet Storage Hit for stock panel ({len(df):,} rows)")
            return df

        return pd.DataFrame()

    def save_stock_panel(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> str:
        end_str = end_date or "today"
        version_id = self.parquet_repo.save_stock_panel(df, tickers, start_date, end_str)
        ckey = self._cache_key("stock", tickers, start_date, end_str)
        self.redis_cache.set_dataframe(ckey, df, ttl=3600)
        return version_id

    def get_nifty50(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        end_str = end_date or "today"
        ckey = self._cache_key("nifty50", ["^NSEI"], start_date, end_str)

        cached_df = self.redis_cache.get_dataframe(ckey)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        df = self.parquet_repo.get_nifty50(start_date, end_str)
        if not df.empty:
            self.redis_cache.set_dataframe(ckey, df, ttl=3600)
        return df

    def get_india_vix(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        end_str = end_date or "today"
        ckey = self._cache_key("vix", ["^INDIAVIX"], start_date, end_str)

        cached_df = self.redis_cache.get_dataframe(ckey)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        df = self.parquet_repo.get_india_vix(start_date, end_str)
        if not df.empty:
            self.redis_cache.set_dataframe(ckey, df, ttl=3600)
        return df
