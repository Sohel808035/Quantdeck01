"""
tests/test_backtesting_engine.py
──────────────────────────────────
Unit Test Suite for QuantSphereX Modular Backtesting Engine.

Covers:
  1. SignalPipeline   - Weight expansion, lag, and regime scaling
  2. CostModel        - Fixed and market impact costs
  3. ExecutionEngine  - Gross/net return series and equity curve
  4. PortfolioTracker - HHI, N_eff, and exposure panel
  5. MetricsEngine    - CAGR, Sharpe, Sortino, MaxDD, rolling metrics
  6. FactorAttributionEngine - OLS regression, return decomposition
  7. ReportGenerator  - Tearsheet generation and JSON/CSV export
  8. BacktestEngine   - Full end-to-end orchestration
  9. Legacy Backtester backward compatibility
"""

import os
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from execution_layer.backtesting.config import BacktestConfig
from execution_layer.backtesting.signals import SignalPipeline
from execution_layer.backtesting.costs import CostModel
from execution_layer.backtesting.execution import ExecutionEngine
from execution_layer.backtesting.portfolio import PortfolioTracker
from execution_layer.backtesting.metrics import MetricsEngine
from execution_layer.backtesting.factor_attribution import FactorAttributionEngine
from execution_layer.backtesting.reports import ReportGenerator
from execution_layer.backtesting.engine import BacktestEngine, BacktestResult
from execution_layer.backtester import Backtester


# ─── Shared Fixtures ──────────────────────────────────────────────────────────

def _make_stock_returns(n_dates=252, n_tickers=5, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = np.random.normal(0.0003, 0.012, size=(n_dates, n_tickers))
    return pd.DataFrame(rets, index=dates, columns=tickers)


def _make_weights_schedule(stock_returns, rebalance="MS"):
    """Monthly equal-weight schedule."""
    monthly_dates = pd.date_range(
        stock_returns.index[0], stock_returns.index[-1], freq=rebalance
    )
    monthly_dates = monthly_dates[monthly_dates.isin(stock_returns.index)]
    n = len(stock_returns.columns)
    w = pd.DataFrame(1.0 / n, index=monthly_dates, columns=stock_returns.columns)
    return w


def _make_regime_exposure(stock_returns):
    """Constant full exposure regime."""
    return pd.Series(1.0, index=stock_returns.index, name="regime_exposure")


def _make_adv_data(stock_returns, adv_value=5e7):
    """Flat ADV data panel."""
    return pd.DataFrame(adv_value, index=stock_returns.index, columns=stock_returns.columns)


def _make_config():
    return BacktestConfig(
        initial_capital=1_000_000.0,
        apply_vol_targeting=False,  # Disable for determinism in unit tests
        transaction_cost_pct=0.001,
        impact_coeff=0.05,
    )


# ─── 1. SignalPipeline ───────────────────────────────────────────────────────

class TestSignalPipeline(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns()
        self.weights = _make_weights_schedule(self.returns)
        self.config = _make_config()

    def test_build_holding_weights_shape(self):
        pipe = SignalPipeline(config=self.config)
        hw = pipe.build_holding_weights(self.weights, self.returns.index)
        self.assertEqual(hw.shape[0], len(self.returns))
        self.assertEqual(hw.shape[1], len(self.returns.columns))

    def test_signal_lag_first_row_is_zero(self):
        pipe = SignalPipeline(config=self.config)
        hw = pipe.build_holding_weights(self.weights, self.returns.index)
        self.assertEqual(hw.iloc[0].sum(), 0.0)  # First row must be zero (lag)

    def test_regime_scaling_reduces_exposure(self):
        pipe = SignalPipeline(config=self.config)
        regime = pd.Series(0.5, index=self.returns.index)  # 50% exposure
        hw = pipe.build_holding_weights(self.weights, self.returns.index, regime_exposure=regime)
        # Use a row well after the warm-up lag (weights fill in on first rebalance date)
        non_zero_rows = hw[hw.sum(axis=1) > 0]
        if len(non_zero_rows) > 0:
            row = non_zero_rows.iloc[0]
            self.assertAlmostEqual(row.sum(), 0.5, places=3)


# ─── 2. CostModel ────────────────────────────────────────────────────────────

class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns()
        self.weights = _make_weights_schedule(self.returns)
        self.config = _make_config()
        pipe = SignalPipeline(config=self.config)
        self.hw = pipe.build_holding_weights(self.weights, self.returns.index)
        self.adv = _make_adv_data(self.returns)

    def test_fixed_costs_non_negative(self):
        cm = CostModel(config=self.config)
        fixed, impact, total = cm.compute_trade_costs(self.hw)
        self.assertTrue((fixed >= 0).all())

    def test_impact_costs_non_negative_with_adv(self):
        cm = CostModel(config=self.config)
        fixed, impact, total = cm.compute_trade_costs(self.hw, adv_data=self.adv)
        self.assertTrue((impact >= 0).all())

    def test_total_costs_equal_fixed_plus_impact(self):
        cm = CostModel(config=self.config)
        fixed, impact, total = cm.compute_trade_costs(self.hw, adv_data=self.adv)
        pd.testing.assert_series_equal(total, fixed + impact, check_names=False)


# ─── 3. ExecutionEngine ──────────────────────────────────────────────────────

class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns()
        self.weights = _make_weights_schedule(self.returns)
        self.config = _make_config()
        pipe = SignalPipeline(config=self.config)
        self.hw = pipe.build_holding_weights(self.weights, self.returns.index)

    def test_equity_curve_starts_near_initial_capital(self):
        eng = ExecutionEngine(config=self.config)
        _, _, equity, *_ = eng.run(self.hw, self.returns)
        self.assertAlmostEqual(equity.iloc[0], self.config.initial_capital, delta=self.config.initial_capital * 0.05)

    def test_net_ret_less_than_gross(self):
        eng = ExecutionEngine(config=self.config)
        gross, net, equity, turnover, fc, ic = eng.run(self.hw, self.returns)
        self.assertLessEqual(float(net.sum()), float(gross.sum()))

    def test_equity_curve_length(self):
        eng = ExecutionEngine(config=self.config)
        _, _, equity, *_ = eng.run(self.hw, self.returns)
        self.assertEqual(len(equity), len(self.returns))


# ─── 4. PortfolioTracker ─────────────────────────────────────────────────────

class TestPortfolioTracker(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns()
        self.weights = _make_weights_schedule(self.returns)
        self.config = _make_config()
        pipe = SignalPipeline(config=self.config)
        self.hw = pipe.build_holding_weights(self.weights, self.returns.index)
        eng = ExecutionEngine(config=self.config)
        _, _, self.equity, *_ = eng.run(self.hw, self.returns)

    def test_hhi_between_0_and_1(self):
        pt = PortfolioTracker(config=self.config)
        hhi = pt.concentration_hhi(self.hw)
        valid = hhi.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_effective_n_stocks_gte_1(self):
        pt = PortfolioTracker(config=self.config)
        n_eff = pt.effective_n_stocks(self.hw)
        valid = n_eff[n_eff > 0].dropna()
        self.assertTrue((valid >= 1.0).all())

    def test_exposure_panel_shape(self):
        pt = PortfolioTracker(config=self.config)
        exp = pt.build_exposure_panel(self.hw, self.equity)
        self.assertEqual(exp.shape, self.hw.shape)


# ─── 5. MetricsEngine ────────────────────────────────────────────────────────

class TestMetricsEngine(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns()
        self.weights = _make_weights_schedule(self.returns)
        self.config = _make_config()
        pipe = SignalPipeline(config=self.config)
        hw = pipe.build_holding_weights(self.weights, self.returns.index)
        eng = ExecutionEngine(config=self.config)
        _, net_ret, equity, turnover, fc, ic = eng.run(hw, self.returns)
        self.net_ret = net_ret
        self.equity = equity
        self.turnover = turnover
        self.fc = fc
        self.ic = ic

    def test_full_period_metrics_keys(self):
        me = MetricsEngine(config=self.config)
        m = me.full_period_metrics(self.equity, self.net_ret, self.turnover, self.fc, self.ic)
        for key in ["cagr", "ann_vol", "sharpe_ratio", "max_drawdown", "calmar_ratio"]:
            self.assertIn(key, m)

    def test_max_drawdown_is_negative(self):
        me = MetricsEngine(config=self.config)
        m = me.full_period_metrics(self.equity, self.net_ret, self.turnover, self.fc, self.ic)
        self.assertLessEqual(m["max_drawdown"], 0.0)

    def test_rolling_metrics_returns_dataframe(self):
        me = MetricsEngine(config=self.config)
        rm = me.rolling_metrics(self.net_ret, window=21)
        self.assertIsInstance(rm, pd.DataFrame)
        self.assertIn("rolling_sharpe", rm.columns)
        self.assertIn("rolling_vol", rm.columns)

    def test_rolling_metrics_length_matches_returns(self):
        me = MetricsEngine(config=self.config)
        rm = me.rolling_metrics(self.net_ret, window=21)
        self.assertEqual(len(rm), len(self.net_ret))


# ─── 6. FactorAttributionEngine ──────────────────────────────────────────────

class TestFactorAttributionEngine(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=252, freq="B")
        self.port_ret = pd.Series(np.random.normal(0.0003, 0.01, 252), index=dates)
        self.factor_returns = pd.DataFrame({
            "Market": np.random.normal(0.0002, 0.008, 252),
            "Momentum": np.random.normal(0.0001, 0.005, 252),
        }, index=dates)

    def test_factor_regression_returns_dict(self):
        fa = FactorAttributionEngine()
        result = fa.compute_factor_regression(self.port_ret, self.factor_returns)
        self.assertIn("alpha_daily", result)
        self.assertIn("r_squared", result)
        self.assertIn("beta_Market", result)
        self.assertIn("beta_Momentum", result)

    def test_r_squared_between_0_and_1(self):
        fa = FactorAttributionEngine()
        result = fa.compute_factor_regression(self.port_ret, self.factor_returns)
        self.assertGreaterEqual(result["r_squared"], 0.0)
        self.assertLessEqual(result["r_squared"], 1.0)

    def test_return_attribution_has_residual_column(self):
        fa = FactorAttributionEngine()
        df = fa.return_attribution(self.port_ret, self.factor_returns)
        self.assertIn("residual", df.columns)
        self.assertIn("portfolio_ret", df.columns)


# ─── 7. ReportGenerator ──────────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    def _make_metrics(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=252, freq="B")
        ret = pd.Series(np.random.normal(0.0003, 0.01, 252), index=dates)
        equity = (1_000_000 * (1 + ret).cumprod())
        dd = (equity - equity.cummax()) / equity.cummax()
        monthly = (ret + 1).resample("ME").prod() - 1
        monthly_tbl = monthly.to_frame("Monthly Return")
        monthly_tbl.index = monthly_tbl.index.to_period("M")
        return {
            "cagr": 0.12, "ann_vol": 0.15, "sharpe_ratio": 0.80,
            "sortino_ratio": 1.10, "information_ratio": 0.50,
            "max_drawdown": -0.12, "calmar_ratio": 1.0, "total_return": 0.25,
            "final_equity": 1_250_000.0, "ann_turnover": 3.5,
            "ann_fixed_cost_bp": 45.0, "ann_impact_cost_bp": 5.0,
            "equity_curve": equity, "daily_returns": ret, "drawdown_series": dd,
            "monthly_returns": monthly_tbl,
        }

    def test_tearsheet_returns_dataframe(self):
        rg = ReportGenerator()
        m = self._make_metrics()
        ts = rg.build_tearsheet(m)
        self.assertIsInstance(ts, pd.DataFrame)
        self.assertIn("sharpe_ratio", ts.columns)

    def test_export_json_creates_file(self):
        rg = ReportGenerator()
        m = self._make_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.json")
            rg.export_json(m, output_path=path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("sharpe_ratio", data)

    def test_export_csv_creates_files(self):
        rg = ReportGenerator()
        m = self._make_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            rg.export_csv(m, output_dir=tmpdir)
            files = os.listdir(tmpdir)
            self.assertTrue(any("equity_curve" in f for f in files))
            self.assertTrue(any("monthly_returns" in f for f in files))


# ─── 8. BacktestEngine (Full End-to-End) ─────────────────────────────────────

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.returns = _make_stock_returns(n_dates=252, n_tickers=5)
        self.weights = _make_weights_schedule(self.returns)
        self.regime = _make_regime_exposure(self.returns)
        self.adv = _make_adv_data(self.returns)
        self.config = _make_config()

    def test_end_to_end_returns_backtest_result(self):
        engine = BacktestEngine(config=self.config)
        result = engine.run(
            weights_schedule=self.weights,
            stock_returns=self.returns,
            regime_exposure=self.regime,
            adv_data=self.adv,
        )
        self.assertIsInstance(result, BacktestResult)

    def test_metrics_dict_populated(self):
        engine = BacktestEngine(config=self.config)
        result = engine.run(self.weights, self.returns, self.regime)
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("cagr", result.metrics)
        self.assertIn("max_drawdown", result.metrics)

    def test_rolling_metrics_shape(self):
        engine = BacktestEngine(config=self.config)
        result = engine.run(self.weights, self.returns, self.regime)
        self.assertGreater(len(result.rolling_metrics), 0)
        self.assertIn("rolling_sharpe", result.rolling_metrics.columns)

    def test_factor_attribution_with_factors(self):
        np.random.seed(42)
        factor_ret = pd.DataFrame({
            "Market": np.random.normal(0.0002, 0.008, len(self.returns)),
        }, index=self.returns.index)
        engine = BacktestEngine(config=self.config)
        result = engine.run(
            self.weights, self.returns, self.regime,
            factor_returns=factor_ret,
        )
        self.assertIn("beta_Market", result.factor_attribution)

    def test_tearsheet_is_dataframe(self):
        engine = BacktestEngine(config=self.config)
        result = engine.run(self.weights, self.returns, self.regime)
        self.assertIsInstance(result.tearsheet, pd.DataFrame)
        self.assertGreater(len(result.tearsheet), 0)

    def test_export_report_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = BacktestEngine(config=self.config)
            engine.run(
                self.weights, self.returns, self.regime,
                export_dir=tmpdir,
            )
            files = os.listdir(tmpdir)
            self.assertTrue(any(".csv" in f for f in files))
            self.assertTrue(any(".json" in f for f in files))


# ─── 9. Legacy Backtester Backward Compatibility ─────────────────────────────

class TestLegacyBacktesterCompat(unittest.TestCase):
    def test_legacy_backtester_still_runs(self):
        returns = _make_stock_returns()
        weights = _make_weights_schedule(returns)
        regime = _make_regime_exposure(returns)
        bt = Backtester(initial_capital=1_000_000.0, apply_vol_targeting=False)
        result = bt.run_backtest(
            weights_schedule=weights,
            stock_returns=returns,
            regime_exposure=regime,
        )
        self.assertIn("sharpe_ratio", result)
        self.assertIn("cagr", result)
        self.assertIn("equity_curve", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
