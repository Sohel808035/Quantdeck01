"""
tests/test_data_layer.py
────────────────────────
Unit and Integration Test Suite for QuantSphereX Data Layer Domain.
Uses standard library unittest.
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from data_layer.config import DataConfig
from data_layer.interfaces import ValidationReport, IDataCache
from data_layer.storage import ParquetCache
from data_layer.ingestor import validate_panel, YFinanceIngestor, MacroDataIngestor
from data_layer.universe import UniverseManager, get_universe, get_yfinance_tickers


class TestDataLayer(unittest.TestCase):

    def test_data_config_initialization(self):
        """Verify DataConfig defaults and custom directory resolution."""
        config = DataConfig(batch_size=25, max_retries=5)
        self.assertEqual(config.batch_size, 25)
        self.assertEqual(config.max_retries, 5)
        self.assertIsInstance(config.cache_dir, Path)
        self.assertTrue(config.cache_dir.exists())

    def test_parquet_cache_operations(self):
        """Verify ParquetCache atomic save, exists, load, and invalidate lifecycle."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = DataConfig(cache_dir=Path(tmp_dir))
            cache = ParquetCache(config=config)

            # Mock panel
            dates = pd.date_range("2024-01-01", periods=5)
            idx = pd.MultiIndex.from_product([dates, ["RELIANCE", "TCS"]], names=["Date", "Ticker"])
            df_mock = pd.DataFrame({
                "Open": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [95.0] * 10,
                "Close": [102.0] * 10,
                "Volume": [1000] * 10,
            }, index=idx)

            tickers = ["RELIANCE", "TCS"]
            start = "2024-01-01"
            end = "2024-01-05"

            # Initially should not exist
            self.assertFalse(cache.exists(tickers, start, end, name="test_stock"))

            # Save
            saved_path = cache.save(df_mock, tickers, start, end, name="test_stock")
            self.assertTrue(saved_path.exists())
            self.assertTrue(cache.exists(tickers, start, end, name="test_stock"))

            # Load
            loaded_df = cache.load(tickers, start, end, name="test_stock")
            self.assertEqual(len(loaded_df), 10)
            self.assertEqual(list(loaded_df.columns), list(df_mock.columns))

            # Invalidate
            removed = cache.invalidate(tickers, start, end, name="test_stock")
            self.assertTrue(removed)
            self.assertFalse(cache.exists(tickers, start, end, name="test_stock"))

    def test_validate_panel_dupes_and_nans(self):
        """Verify validate_panel detects duplicates and NaNs."""
        dates = pd.date_range("2024-01-01", periods=3)
        idx = pd.MultiIndex.from_tuples([
            (dates[0], "TCS"),
            (dates[0], "TCS"),  # Duplicate
            (dates[1], "TCS"),
            (dates[2], "TCS"),
        ], names=["Date", "Ticker"])

        df = pd.DataFrame({
            "Open": [100.0, 100.0, np.nan, 102.0],  # One NaN
            "High": [105.0, 105.0, 106.0, 107.0],
            "Low": [95.0, 95.0, 96.0, 97.0],
            "Close": [102.0, 102.0, 103.0, 104.0],
        }, index=idx)

        validated = validate_panel(df, max_fail_pct=0.50)
        # Duplicate removed (1 dropped), NaN in Open removed (1 dropped) -> 2 rows remain
        self.assertEqual(len(validated), 2)
        self.assertFalse(validated.index.duplicated().any())
        self.assertFalse(validated["Open"].isna().any())

    def test_validate_panel_outliers_and_corruption_halt(self):
        """Verify price jump outlier removal and corruption threshold error throwing."""
        dates = pd.date_range("2024-01-01", periods=10)
        idx = pd.MultiIndex.from_product([dates, ["TCS"]], names=["Date", "Ticker"])

        prices = [100.0, 101.0, 102.0, 500.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]  # 500 is >30% jump
        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
        }, index=idx)

        # With high fail tolerance, outlier is dropped cleanly
        validated = validate_panel(df, threshold_move=0.30, max_fail_pct=0.50)
        self.assertLess(len(validated), 10)

        # With tight fail tolerance (e.g. max_fail_pct=0.01), ValueError is raised
        with self.assertRaises(ValueError):
            validate_panel(df, threshold_move=0.30, max_fail_pct=0.01)

    def test_universe_manager_methods(self):
        """Verify UniverseManager static fallback, sector mapping, and benchmark weights."""
        mgr = UniverseManager()

        universe = mgr.get_universe()
        self.assertIsInstance(universe, list)
        self.assertGreater(len(universe), 100)
        self.assertIn("RELIANCE", universe)

        sector_map = mgr.get_sector_mapping()
        self.assertIsInstance(sector_map, dict)
        self.assertEqual(sector_map.get("HDFCBANK"), "Financial Services")
        self.assertEqual(sector_map.get("TCS"), "IT")

        benchmark_w = mgr.get_benchmark_sector_weights()
        self.assertIsInstance(benchmark_w, dict)
        self.assertIn("Financial Services", benchmark_w)
        self.assertAlmostEqual(sum(benchmark_w.values()), 1.0, places=4)

    def test_legacy_universe_wrappers(self):
        """Verify legacy get_universe and get_yfinance_tickers helper functions."""
        univ = get_universe()
        self.assertGreater(len(univ), 50)

        yf_tickers = get_yfinance_tickers(suffix=".NS")
        self.assertTrue(yf_tickers[0].endswith(".NS"))


if __name__ == "__main__":
    unittest.main()
