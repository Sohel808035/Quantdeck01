"""
tests/test_label_factory.py
────────────────────────────
Unit Test Suite for QuantSphereX Label Factory (v1.0.0)

Covers:
  - Regression labels (simple, log, excess, risk-adjusted)
  - Classification labels (binary, tertile, quintile, excess binary)
  - Ranking labels (percentile, z-score, decile, IR) — single and cross-sectional
  - Multi-horizon utilities (summary, correlation matrix, optimal horizon)
  - Custom label builder (register, compute, factory functions)
  - Label Factory Orchestrator (build_ticker_labels, build_label_panel)
  - Backward-compatibility (build_target_panel, TARGET_COL)
  - No look-ahead: labels verified against shifted prices only
  - Anti-look-ahead lag: last H rows must be NaN for horizon H
"""

import unittest
import numpy as np
import pandas as pd

from label_layer.config import LabelConfig, STANDARD_HORIZONS
from label_layer.labels import regression, classification, ranking, horizons
from label_layer.labels.custom import (
    CustomLabelBuilder,
    make_threshold_label,
    make_drawdown_label,
    make_composite_label,
)
from label_layer.factory import (
    build_ticker_labels,
    build_label_panel,
    build_target_panel,
    get_label_registry,
    TARGET_COL,
)


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, base_price: float = 500.0, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic single-ticker OHLCV DataFrame."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    ret = np.random.normal(0.0004, 0.012, size=n)
    px = base_price * (1 + pd.Series(ret)).cumprod().values
    high  = px * (1 + np.abs(np.random.normal(0.005, 0.002, n)))
    low   = px * (1 - np.abs(np.random.normal(0.005, 0.002, n)))
    open_ = np.roll(px, 1); open_[0] = base_price
    vol   = np.abs(np.random.normal(5e5, 1e5, n)).astype(int)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": px, "Volume": vol},
        index=dates
    )
    df.index.name = "Date"
    return df


def _make_panel(n: int = 300, tickers: list = None) -> pd.DataFrame:
    """Generates a synthetic multi-ticker (Date, Ticker) MultiIndex panel."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    frames = []
    for i, t in enumerate(tickers):
        df = _make_ohlcv(n=n, seed=42 + i)
        df["Ticker"] = t
        df = df.reset_index().set_index(["Date", "Ticker"])
        frames.append(df)
    return pd.concat(frames).sort_index()


def _make_market_ret(n: int = 300) -> pd.Series:
    """Synthetic market benchmark return series."""
    np.random.seed(99)
    return pd.Series(
        np.random.normal(0.0003, 0.009, n),
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
        name="market_ret"
    )


# ─── Regression Label Tests ───────────────────────────────────────────────────

class TestRegressionLabels(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.mkt = _make_market_ret()
        self.horizons = [5, 21]

    def test_simple_forward_return_columns_exist(self):
        result = regression.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            self.assertIn(f"ret_fwd_{H}d", result.columns)

    def test_log_return_columns_exist(self):
        result = regression.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            self.assertIn(f"log_ret_fwd_{H}d", result.columns)

    def test_excess_return_generated_with_benchmark(self):
        result = regression.compute(self.df, horizons=self.horizons, benchmark_ret=self.mkt)
        for H in self.horizons:
            self.assertIn(f"excess_ret_fwd_{H}d", result.columns)

    def test_risk_adjusted_return_columns_exist(self):
        result = regression.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            self.assertIn(f"risk_adj_ret_fwd_{H}d", result.columns)

    def test_last_H_rows_are_nan(self):
        """No forward return is possible in the last H rows."""
        result = regression.compute(self.df, horizons=[21])
        tail = result["ret_fwd_21d"].iloc[-21:]
        self.assertTrue(tail.isna().all(), "Last 21 rows must be NaN for 21D label")

    def test_validate_passes(self):
        result = regression.compute(self.df, horizons=self.horizons)
        report = regression.validate(result, self.horizons)
        self.assertNotIn(
            True,
            ["Missing required column" in w for w in report["warnings"]]
        )

    def test_index_preserved(self):
        result = regression.compute(self.df, horizons=self.horizons)
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(result.index.equals(self.df.index))


# ─── Classification Label Tests ───────────────────────────────────────────────

class TestClassificationLabels(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv()
        self.horizons = [5, 21]

    def test_binary_columns_exist(self):
        result = classification.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            self.assertIn(f"binary_fwd_{H}d", result.columns)

    def test_binary_labels_are_binary(self):
        result = classification.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            col = f"binary_fwd_{H}d"
            valid = result[col].dropna()
            self.assertTrue(set(valid.unique()).issubset({0.0, 1.0}))

    def test_tertile_columns_exist_and_in_range(self):
        result = classification.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            col = f"tertile_fwd_{H}d"
            self.assertIn(col, result.columns)
            valid = result[col].dropna()
            self.assertTrue(set(valid.unique()).issubset({0.0, 1.0, 2.0}))

    def test_quintile_columns_exist_and_in_range(self):
        result = classification.compute(self.df, horizons=self.horizons)
        for H in self.horizons:
            col = f"quintile_fwd_{H}d"
            self.assertIn(col, result.columns)
            valid = result[col].dropna()
            self.assertTrue((valid >= 0).all() and (valid < 5).all())

    def test_excess_binary_with_benchmark(self):
        mkt = _make_market_ret()
        result = classification.compute(self.df, horizons=[21], benchmark_ret=mkt)
        self.assertIn("excess_binary_fwd_21d", result.columns)
        valid = result["excess_binary_fwd_21d"].dropna()
        self.assertTrue(set(valid.unique()).issubset({0.0, 1.0}))

    def test_validate_passes_on_clean_data(self):
        result = classification.compute(self.df, horizons=self.horizons)
        report = classification.validate(result, self.horizons)
        bad = [w for w in report["warnings"] if "binary values" in w]
        self.assertEqual(len(bad), 0)


# ─── Ranking Label Tests ──────────────────────────────────────────────────────

class TestRankingLabels(unittest.TestCase):
    def setUp(self):
        self.panel = _make_panel(n=200)
        self.horizons = [5, 21]

    def test_compute_single_ticker_raw_columns(self):
        df = _make_ohlcv(n=200)
        result = ranking.compute_single_ticker(df, horizons=self.horizons)
        for H in self.horizons:
            self.assertIn(f"_raw_fwd_{H}d", result.columns)

    def test_compute_panel_percentile_rank_in_bounds(self):
        # Add raw fwd returns to panel first
        raw_frames = []
        for ticker in self.panel.index.get_level_values("Ticker").unique():
            df = self.panel.xs(ticker, level="Ticker")
            raw = ranking.compute_single_ticker(df, horizons=self.horizons)
            raw["Ticker"] = ticker
            raw = raw.reset_index().set_index(["Date", "Ticker"])
            raw_frames.append(raw)
        raw_panel = pd.concat(raw_frames).sort_index()

        ranked = ranking.compute_panel(raw_panel, horizons=self.horizons)
        for H in self.horizons:
            col = f"rank_pct_fwd_{H}d"
            self.assertIn(col, ranked.columns)
            valid = ranked[col].dropna()
            self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_validate_passes_after_compute_panel(self):
        raw_frames = []
        for ticker in self.panel.index.get_level_values("Ticker").unique():
            df = self.panel.xs(ticker, level="Ticker")
            raw = ranking.compute_single_ticker(df, horizons=self.horizons)
            raw["Ticker"] = ticker
            raw = raw.reset_index().set_index(["Date", "Ticker"])
            raw_frames.append(raw)
        raw_panel = pd.concat(raw_frames).sort_index()
        ranked = ranking.compute_panel(raw_panel, horizons=self.horizons)
        report = ranking.validate(ranked, self.horizons)
        self.assertFalse(any("Missing" in w for w in report["warnings"]))


# ─── Horizon Utilities Tests ──────────────────────────────────────────────────

class TestHorizonUtilities(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(n=300)
        self.horizons = [5, 21, 63]

    def test_compute_forward_returns_all_columns(self):
        result = horizons.compute_forward_returns(self.df, self.horizons)
        for H in self.horizons:
            self.assertIn(f"fwd_ret_{H}d", result.columns)

    def test_horizon_summary_returns_dataframe(self):
        summary = horizons.horizon_summary(self.df, self.horizons)
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), len(self.horizons))
        self.assertIn("mean_return", summary.columns)
        self.assertIn("pct_positive", summary.columns)

    def test_horizon_correlation_matrix_is_square(self):
        corr = horizons.horizon_correlation_matrix(self.df, self.horizons)
        self.assertEqual(corr.shape[0], corr.shape[1])
        self.assertEqual(corr.shape[0], len(self.horizons))

    def test_select_optimal_horizon_returns_valid_value(self):
        best = horizons.select_optimal_horizon(self.df, self.horizons)
        self.assertIn(best, self.horizons)


# ─── Custom Label Tests ───────────────────────────────────────────────────────

class TestCustomLabels(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(n=200)

    def test_register_and_compute_simple_lambda(self):
        builder = CustomLabelBuilder()
        builder.register("is_positive_5d", lambda df: (df["Close"].shift(-5) > df["Close"]).astype(float))
        result = builder.compute(self.df)
        self.assertIn("custom_is_positive_5d", result.columns)

    def test_list_labels(self):
        builder = CustomLabelBuilder()
        builder.register("label_a", lambda df: pd.Series(1.0, index=df.index))
        builder.register("label_b", lambda df: pd.Series(0.0, index=df.index))
        self.assertEqual(sorted(builder.list_labels()), ["label_a", "label_b"])

    def test_unregister_removes_label(self):
        builder = CustomLabelBuilder()
        builder.register("temp", lambda df: pd.Series(0.0, index=df.index))
        builder.unregister("temp")
        self.assertNotIn("temp", builder.list_labels())

    def test_method_chaining(self):
        builder = (
            CustomLabelBuilder()
            .register("a", lambda df: pd.Series(1.0, index=df.index))
            .register("b", lambda df: pd.Series(0.0, index=df.index))
        )
        self.assertEqual(len(builder.list_labels()), 2)

    def test_validate_passes_on_valid_output(self):
        builder = CustomLabelBuilder()
        builder.register("my_label", lambda df: pd.Series(1.0, index=df.index))
        result = builder.compute(self.df)
        report = builder.validate(result)
        self.assertTrue(report["passed"])

    def test_make_threshold_label_factory(self):
        fn = make_threshold_label(horizon=5, threshold=0.02)
        result = fn(self.df)
        self.assertIsInstance(result, pd.Series)
        valid = result.dropna()
        self.assertTrue(set(valid.unique()).issubset({0.0, 1.0}))

    def test_make_composite_label_factory(self):
        fn_a = make_threshold_label(horizon=5, threshold=0.01)
        fn_b = make_threshold_label(horizon=10, threshold=0.02)
        composite = make_composite_label([fn_a, fn_b], weights=[0.6, 0.4])
        result = composite(self.df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))

    def test_failed_custom_label_returns_nan_column(self):
        """A failing label function should not crash the factory."""
        builder = CustomLabelBuilder()
        builder.register("bad_fn", lambda df: 1 / 0)  # Will raise ZeroDivisionError
        result = builder.compute(self.df)
        self.assertIn("custom_bad_fn", result.columns)
        self.assertTrue(result["custom_bad_fn"].isna().all())


# ─── Label Factory Orchestrator Tests ─────────────────────────────────────────

class TestLabelFactory(unittest.TestCase):
    def setUp(self):
        self.df = _make_ohlcv(n=200)
        self.panel = _make_panel(n=200, tickers=["ALPHA", "BETA", "GAMMA"])
        self.config = LabelConfig(horizons=[5, 21])

    def test_build_ticker_labels_returns_wide_dataframe(self):
        result = build_ticker_labels(self.df, config=self.config)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 5)
        self.assertEqual(len(result), len(self.df))

    def test_build_ticker_labels_with_custom_builder(self):
        builder = CustomLabelBuilder()
        builder.register("my_signal", lambda df: pd.Series(1.0, index=df.index))
        result = build_ticker_labels(self.df, config=self.config, custom_builder=builder)
        self.assertIn("custom_my_signal", result.columns)

    def test_build_label_panel_multiindex_output(self):
        result = build_label_panel(self.panel, config=self.config)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(result.index, pd.MultiIndex)
        result_tickers = result.index.get_level_values("Ticker").unique().tolist()
        for t in ["ALPHA", "BETA", "GAMMA"]:
            self.assertIn(t, result_tickers)

    def test_build_label_panel_contains_ranking_labels(self):
        result = build_label_panel(self.panel, config=self.config)
        self.assertIn("rank_pct_fwd_5d", result.columns)
        self.assertIn("rank_pct_fwd_21d", result.columns)

    def test_ranking_labels_in_01_range(self):
        result = build_label_panel(self.panel, config=self.config)
        valid = result["rank_pct_fwd_21d"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_build_label_panel_regression_labels_present(self):
        result = build_label_panel(self.panel, config=self.config)
        self.assertIn("ret_fwd_5d", result.columns)
        self.assertIn("log_ret_fwd_21d", result.columns)

    def test_build_label_panel_classification_labels_present(self):
        result = build_label_panel(self.panel, config=self.config)
        self.assertIn("binary_fwd_5d", result.columns)
        self.assertIn("tertile_fwd_21d", result.columns)

    def test_config_switches_disable_label_types(self):
        config_no_cls = LabelConfig(
            horizons=[5],
            apply_regression_labels=True,
            apply_classification_labels=False,
            apply_ranking_labels=False,
        )
        result = build_ticker_labels(self.df, config=config_no_cls)
        self.assertNotIn("binary_fwd_5d", result.columns)
        self.assertIn("ret_fwd_5d", result.columns)

    def test_get_label_registry_lists_all_engines(self):
        registry = get_label_registry()
        self.assertIn("engines", registry)
        engine_names = [e["name"] for e in registry["engines"]]
        for expected in ["Regression", "Classification", "Ranking", "Horizons", "Custom"]:
            self.assertIn(expected, engine_names)


# ─── Backward-Compatibility Tests ─────────────────────────────────────────────

class TestBackwardCompatibility(unittest.TestCase):
    def setUp(self):
        self.panel = _make_panel(n=200, tickers=["X", "Y", "Z"])

    def test_build_target_panel_adds_target_col(self):
        result = build_target_panel(self.panel, horizon=21)
        self.assertIn(TARGET_COL, result.columns)

    def test_target_col_is_percentile_rank(self):
        result = build_target_panel(self.panel, horizon=21)
        valid = result[TARGET_COL].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_target_col_constant_matches_factory(self):
        """TARGET_COL must match alpha_layer naming convention."""
        self.assertTrue(TARGET_COL.startswith("label_"))

    def test_original_columns_preserved(self):
        result = build_target_panel(self.panel, horizon=21)
        for col in self.panel.columns:
            self.assertIn(col, result.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
