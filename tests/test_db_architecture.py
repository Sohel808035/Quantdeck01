"""
tests/test_db_architecture.py
──────────────────────────────
Unit and Integration Test Suite for QuantSphereX Database & Storage Domain.
Tests Repository Pattern, Redis Fallback Cache, Dataset Versioning, and SQL Schemas.
"""

import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from data_layer.config import DataConfig
from data_layer.versioning import DatasetVersionManager
from data_layer.repository import ParquetRepository, HybridDataRepository
from db.redis_client import RedisCacheManager, InMemoryFallbackCache


class TestDatabaseArchitecture(unittest.TestCase):

    def test_schema_files_exist(self):
        """Verify DDL schema and migration SQL files exist."""
        migration_file = Path("db/migrations/001_initial_schema.sql")
        schema_file = Path("db/schema.sql")
        self.assertTrue(migration_file.exists())
        self.assertTrue(schema_file.exists())

        with open(migration_file, "r", encoding="utf-8") as f:
            sql_text = f.read()
            self.assertIn("CREATE TABLE IF NOT EXISTS market_data", sql_text)
            self.assertIn("create_hypertable", sql_text)
            self.assertIn("dataset_metadata", sql_text)

    def test_redis_fallback_cache(self):
        """Verify Redis cache manager fallback in-memory behavior."""
        cache_mgr = RedisCacheManager()
        
        # Test basic set/get
        cache_mgr.set("test_key", "hello_quantspherex")
        val = cache_mgr.get("test_key")
        self.assertEqual(val, b"hello_quantspherex")

        # Test DataFrame serialization
        df_mock = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        cache_mgr.set_dataframe("df_key", df_mock)

        df_loaded = cache_mgr.get_dataframe("df_key")
        self.assertIsNotNone(df_loaded)
        self.assertEqual(len(df_loaded), 3)

        # Test delete
        cache_mgr.delete("test_key")
        self.assertIsNone(cache_mgr.get("test_key"))

    def test_dataset_versioning_manager(self):
        """Verify DatasetVersionManager manifest creation, hashing, and version loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = DataConfig(cache_dir=Path(tmp_dir))
            version_mgr = DatasetVersionManager(config=config)

            df_mock = pd.DataFrame({"Close": [10.0, 20.0, 30.0]})
            version_id = version_mgr.create_version(
                dataset_name="test_dataset",
                df=df_mock,
                start_date="2024-01-01",
                end_date="2024-01-03",
            )

            self.assertIsNotNone(version_id)
            manifest = version_mgr.get_manifest(version_id)
            self.assertEqual(manifest["dataset_name"], "test_dataset")
            self.assertEqual(manifest["row_count"], 3)
            self.assertIn("hash_key", manifest)

            # Load version
            df_loaded = version_mgr.load_version(version_id)
            self.assertEqual(len(df_loaded), 3)

            # List versions
            versions_list = version_mgr.list_versions("test_dataset")
            self.assertEqual(len(versions_list), 1)

    def test_repository_pattern_hybrid(self):
        """Verify ParquetRepository and HybridDataRepository integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = DataConfig(cache_dir=Path(tmp_dir))
            repo = HybridDataRepository(config=config)

            dates = pd.date_range("2024-01-01", periods=3)
            idx = pd.MultiIndex.from_product([dates, ["TCS"]], names=["Date", "Ticker"])
            df_mock = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

            tickers = ["TCS"]
            start = "2024-01-01"
            end = "2024-01-03"

            # Initially empty
            df_init = repo.get_stock_panel(tickers, start, end)
            self.assertTrue(df_init.empty)

            # Save via hybrid repository
            version_id = repo.save_stock_panel(df_mock, tickers, start, end)
            self.assertIsNotNone(version_id)

            # Query (should hit L1 RAM cache or L2 storage)
            df_retrieved = repo.get_stock_panel(tickers, start, end)
            self.assertEqual(len(df_retrieved), 3)


if __name__ == "__main__":
    unittest.main()
