"""
data_layer/interfaces.py
────────────────────────
Abstract Base Interfaces and Data Transfer Objects (DTOs) for QuantSphereX Data Domain.
Enforces SOLID principles (Interface Segregation & Dependency Inversion).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed audit metrics produced by the validation pipeline."""
    initial_rows: int
    final_rows: int
    dropped_duplicates: int = 0
    dropped_nans: int = 0
    dropped_outliers: int = 0
    dropped_stale: int = 0
    dropped_non_positive: int = 0
    fail_percentage: float = 0.0
    passed: bool = True
    critical_error: Optional[str] = None


class IDataCache(ABC):
    """Abstract Storage Interface for Parquet / Database Caching."""
    
    @abstractmethod
    def exists(self, tickers: List[str], start: str, end: str, name: str = "stock") -> bool:
        """Check if cached dataset exists for the given universe and date range."""
        pass
        
    @abstractmethod
    def save(self, df: pd.DataFrame, tickers: List[str], start: str, end: str, name: str = "stock") -> Path:
        """Persist DataFrame to binary storage."""
        pass

    @abstractmethod
    def load(self, tickers: List[str], start: str, end: str, name: str = "stock") -> pd.DataFrame:
        """Load DataFrame from binary storage."""
        pass

    @abstractmethod
    def invalidate(self, tickers: List[str], start: str, end: str, name: str = "stock") -> bool:
        """Remove cached dataset from binary storage."""
        pass


class IDataProvider(ABC):
    """Abstract Interface for Financial Data Ingestion Providers."""

    @abstractmethod
    def fetch_daily_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV and fundamental metrics as a long-format MultiIndex DataFrame."""
        pass


class IUniverseProvider(ABC):
    """Abstract Interface for Universe Membership Management."""

    @abstractmethod
    def get_universe(self, date: Optional[pd.Timestamp] = None) -> List[str]:
        """Return list of active universe tickers for a given snapshot date."""
        pass

    @abstractmethod
    def get_sector_mapping(self) -> Dict[str, str]:
        """Return dictionary mapping tickers to industry sectors."""
        pass

    @abstractmethod
    def get_benchmark_sector_weights(self) -> Dict[str, float]:
        """Return target benchmark sector weights."""
        pass
