"""
QuantSphereX Data Layer Domain Module.
"""

from data_layer.config import DataConfig
from data_layer.interfaces import (
    IDataCache,
    IDataProvider,
    IUniverseProvider,
    ValidationReport,
)
from data_layer.storage import ParquetCache
from data_layer.ingestor import (
    MarketDataIngestor,
    YFinanceIngestor,
    MacroDataIngestor,
    validate_panel,
)
from data_layer.universe import (
    UniverseManager,
    get_universe,
    get_yfinance_tickers,
    NIFTY200_STATIC_LIST,
)
from data_layer.versioning import DatasetVersionManager
from data_layer.repository import (
    IMarketDataRepository,
    ParquetRepository,
    HybridDataRepository,
)

__all__ = [
    "DataConfig",
    "IDataCache",
    "IDataProvider",
    "IUniverseProvider",
    "ValidationReport",
    "ParquetCache",
    "MarketDataIngestor",
    "YFinanceIngestor",
    "MacroDataIngestor",
    "validate_panel",
    "UniverseManager",
    "get_universe",
    "get_yfinance_tickers",
    "NIFTY200_STATIC_LIST",
    "DatasetVersionManager",
    "IMarketDataRepository",
    "ParquetRepository",
    "HybridDataRepository",
]
