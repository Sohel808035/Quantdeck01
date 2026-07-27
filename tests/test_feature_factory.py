"""
tests/test_feature_factory.py
──────────────────────────────
Unit Test Suite for QuantSphereX Feature Factory (v2.0.0)

Covers:
  - All 9 independent factor modules (compute, validate, benchmark)
  - Feature Factory Orchestrator (compute_all_factors, compute_features)
  - Output column presence and type validation
  - Benchmark execution time contracts
  - Configuration handling
"""

import unittest
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ─── Factor Module Imports ─────────────────────────────────────────────────
from feature_layer.factors import (
    momentum, quality, value, growth, liquidity,
    volatility, macro, sentiment, alternative,
)
from feature_layer.factory import (
    compute_all_factors, compute_features, get_registry_info, FACTOR_REGISTRY,
)
from feature_layer.config import FeatureFactoryConfig


# ─── Shared Fixtures ──────────────────────────────────────────────────────────
def _make_ohlcv(n: int = 500, base_price: float = 1000.0) -> pd.DataFrame:
    """Generates a realistic synthetic OHLCV DataFrame for unit testing."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    ret = np.random.normal(0.0004, 0.015, size=n)
    px = base_price * (1 + pd.Series(ret)).cumprod()
    px.index = dates

    high  = px * (1 + np.abs(np.random.normal(0.005, 0.003, n)))
    low   = px * (1 - np.abs(np.random.normal(0.005, 0.003, n)))
    open_ = px.shift(1).fillna(base_price) * (1 + np.random.normal(0, 0.003, n))
    vol   = (np.abs(np.random.normal(1e6, 2e5, n))).astype(int)

    df = pd.DataFrame({
        "Open": open_.values,
        "High": high.values,
        "Low": low.values,
        "Close": px.values,
        "Volume": vol,
    }, index=dates)
    df.index.name = "Date"
    return df


def _make_market_ret(n: int = 500) -> pd.Series:
    """Generates a synthetic Nifty 50 return series for beta-based factors."""
    np.random.seed(99)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0.0003, 0.01, n), index=dates, name="mkt_ret")


# ─── Individual Factor Engine Tests ──────────────────────────────────────────
class TestMomentumFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.mkt = _make_market_ret()

    def test_compute_returns_correct_columns(self):
        result = momentum.compute(self.df)
        for col in momentum.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_compute_with_context_ret(self):
        result = momentum.compute(self.df, context_ret=self.mkt)
        self.assertIn("residual_momentum", result.columns)

    def test_validate_passes_on_valid_data(self):
        result = momentum.compute(self.df)
        report = momentum.validate(result)
        self.assertFalse(any("Missing column" in w for w in report["warnings"]))

    def test_benchmark_returns_metrics(self):
        bm = momentum.benchmark(self.df)
        self.assertIn("execution_seconds", bm)
        self.assertIn("rows", bm)
        self.assertGreater(bm["rows"], 0)
        self.assertGreater(bm["execution_seconds"], 0)

    def test_lookback_lag_applied(self):
        """First row should be all NaN due to shift(1) anti-lookahead bias."""
        result = momentum.compute(self.df)
        self.assertTrue(result.iloc[0].isna().all())


class TestQualityFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_computed_columns(self):
        result = quality.compute(self.df)
        for col in ["earnings_stability", "gross_margin", "accruals_ratio"]:
            self.assertIn(col, result.columns)

    def test_validate_runs_without_error(self):
        result = quality.compute(self.df)
        report = quality.validate(result)
        self.assertIn("factor", report)

    def test_benchmark_runs(self):
        bm = quality.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


class TestValueFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_all_columns(self):
        result = value.compute(self.df)
        for col in value.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_distance_52w_high_is_non_positive(self):
        result = value.compute(self.df)
        # Price can never be above its 52W high, so this should be ≤ 0 (after lag, some NaN)
        non_nan = result["distance_52w_high"].dropna()
        self.assertTrue((non_nan <= 0.001).all(), "distance_52w_high should be ≤ 0")

    def test_benchmark_runs(self):
        bm = value.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


class TestGrowthFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_all_columns(self):
        result = growth.compute(self.df)
        for col in growth.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_validate_passes(self):
        result = growth.compute(self.df)
        report = growth.validate(result)
        self.assertNotIn(False, [col in result.columns for col in growth.OUTPUT_COLUMNS])

    def test_benchmark_runs(self):
        bm = growth.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


class TestLiquidityFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_all_columns(self):
        result = liquidity.compute(self.df)
        for col in liquidity.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_amihud_is_non_negative(self):
        result = liquidity.compute(self.df)
        non_nan = result["amihud_illiquidity"].dropna()
        self.assertTrue((non_nan >= 0).all(), "Amihud illiquidity must be non-negative")

    def test_benchmark_runs(self):
        bm = liquidity.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


class TestVolatilityFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.mkt = _make_market_ret()

    def test_compute_returns_all_columns(self):
        result = volatility.compute(self.df)
        for col in volatility.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_realized_vol_is_non_negative(self):
        result = volatility.compute(self.df)
        for col in ["realized_vol_20d", "realized_vol_60d"]:
            non_nan = result[col].dropna()
            self.assertTrue((non_nan >= 0).all())

    def test_idio_vol_with_market_ret(self):
        result = volatility.compute(self.df, context_ret=self.mkt)
        self.assertIn("idio_volatility", result.columns)

    def test_validate_no_negative_vols(self):
        result = volatility.compute(self.df)
        report = volatility.validate(result)
        no_neg_warnings = [w for w in report["warnings"] if "Negative" in w]
        self.assertEqual(len(no_neg_warnings), 0)


class TestMacroFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.mkt = _make_market_ret()

    def test_compute_returns_all_columns(self):
        result = macro.compute(self.df)
        for col in macro.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_above_200dma_is_binary(self):
        result = macro.compute(self.df)
        non_nan = result["above_200dma"].dropna()
        self.assertTrue(set(non_nan.unique()).issubset({0.0, 1.0}))

    def test_market_beta_with_context(self):
        result = macro.compute(self.df, context_ret=self.mkt)
        non_nan = result["market_beta"].dropna()
        self.assertGreater(len(non_nan), 0)

    def test_benchmark_runs(self):
        bm = macro.benchmark(self.df, context_ret=self.mkt)
        self.assertGreater(bm["execution_seconds"], 0)


class TestSentimentFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_all_columns(self):
        result = sentiment.compute(self.df)
        for col in sentiment.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_rsi_in_valid_range(self):
        result = sentiment.compute(self.df)
        non_nan = result["rsi_14"].dropna()
        self.assertTrue((non_nan >= 0).all() and (non_nan <= 100).all())

    def test_validate_passes(self):
        result = sentiment.compute(self.df)
        report = sentiment.validate(result)
        # RSI should never be out of range on clean data
        rsi_warnings = [w for w in report["warnings"] if "RSI out of" in w]
        self.assertEqual(len(rsi_warnings), 0)

    def test_benchmark_runs(self):
        bm = sentiment.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


class TestAlternativeFactor(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()

    def test_compute_returns_all_columns(self):
        result = alternative.compute(self.df)
        for col in alternative.OUTPUT_COLUMNS:
            self.assertIn(col, result.columns)

    def test_weekday_effect_in_valid_range(self):
        result = alternative.compute(self.df)
        non_nan = result["weekday_effect"].dropna()
        self.assertTrue((non_nan >= 0).all() and (non_nan <= 6).all())

    def test_benchmark_runs(self):
        bm = alternative.benchmark(self.df)
        self.assertGreater(bm["execution_seconds"], 0)


# ─── Feature Factory Orchestrator Tests ──────────────────────────────────────
class TestFeatureFactory(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.mkt = _make_market_ret()
        self.config = FeatureFactoryConfig()

    def test_registry_has_all_9_factor_engines(self):
        registry = get_registry_info()
        self.assertEqual(len(registry), 9)
        names = [r["name"] for r in registry]
        for expected in ["Momentum", "Quality", "Value", "Growth", "Liquidity",
                         "Volatility", "Macro", "Sentiment", "Alternative"]:
            self.assertIn(expected, names)

    def test_compute_all_factors_returns_wide_dataframe(self):
        result = compute_all_factors(self.df, context_ret=self.mkt, config=self.config)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 20)
        self.assertEqual(len(result), len(self.df))

    def test_compute_features_backward_compatibility(self):
        """Legacy API must accept MultiIndex panel and return MultiIndex result."""
        tickers = ["AAA", "BBB"]
        frames = []
        for t in tickers:
            sub = self.df.copy()
            sub["Ticker"] = t
            sub = sub.reset_index().set_index(["Date", "Ticker"])
            frames.append(sub)
        panel = pd.concat(frames)

        features = compute_features(panel, config=self.config)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 10)
        result_tickers = features.index.get_level_values("Ticker").unique().tolist()
        for t in tickers:
            self.assertIn(t, result_tickers)

    def test_configuration_applied(self):
        """Verify config DTO with custom winsorization works without error."""
        custom_config = FeatureFactoryConfig(
            winsorize_lower_pct=0.02,
            winsorize_upper_pct=0.98,
            apply_cross_sectional_ranking=True,
        )
        result = compute_all_factors(self.df, config=custom_config)
        self.assertGreater(len(result.columns), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
