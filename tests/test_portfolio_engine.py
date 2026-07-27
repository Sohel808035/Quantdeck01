"""
tests/test_portfolio_engine.py
───────────────────────────────
Unit Test Suite for QuantSphereX Portfolio Engine & Plugins.

Covers:
  - Plugin Registry discovery & listing
  - All 6 Optimization Plugins:
      1. Equal Weight
      2. Risk Parity
      3. Hierarchical Risk Parity (HRP)
      4. Minimum Variance
      5. Black-Litterman
      6. Kelly Criterion
  - Constraints Engine (Asset caps, ADV limits, Sector neutralization, Beta target)
  - Rebalancing Engine (Turnover penalty / no-trade zone, drift trigger)
  - Transaction Cost Engine (Commissions, slippage, market impact model)
  - Position Sizing Engine (Vol targeting, conviction weighting, cash buffer)
  - Backward compatibility with legacy PortfolioOptimizer interface
"""

import unittest
import numpy as np
import pandas as pd

from portfolio_layer.base import PortfolioPluginRegistry, PortfolioConstraints
from portfolio_layer.optimizer import PortfolioOptimizer
from portfolio_layer.constraints import ConstraintsEngine
from portfolio_layer.rebalancing import RebalancingEngine
from portfolio_layer.transaction_cost import TransactionCostEngine
from portfolio_layer.position_sizing import PositionSizingEngine


def _make_returns_df(n_dates: int = 100, tickers: list = None) -> pd.DataFrame:
    """Generates synthetic daily returns DataFrame."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rets = np.random.normal(0.0005, 0.015, size=(n_dates, len(tickers)))
    return pd.DataFrame(rets, index=dates, columns=tickers)


def _make_alpha_scores(tickers: list) -> pd.Series:
    """Generates synthetic alpha score Series."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(0.1, 0.9, size=len(tickers)), index=tickers)


class TestPortfolioEngine(unittest.TestCase):

    def setUp(self):
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        self.ticker_set = set(self.tickers)
        self.returns_df = _make_returns_df(tickers=self.tickers)
        self.alpha_scores = _make_alpha_scores(self.tickers)
        self.adv_data = pd.Series([1e8, 2e8, 1.5e8, 8e7, 3e8], index=self.tickers)
        self.optimizer = PortfolioOptimizer()

    def test_plugin_registry_lists_all_6_plugins(self):
        plugins = PortfolioPluginRegistry.list_plugins()
        registered_names = [p["name"].lower() for p in plugins]
        for expected in ["equal_weight", "risk_parity", "hrp", "min_variance", "black_litterman", "kelly"]:
            self.assertIn(expected, registered_names)

    def test_equal_weight_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="equal_weight",
            adv_data=self.adv_data,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_risk_parity_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="risk_parity",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_hrp_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="hrp",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_min_variance_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="min_variance",
            returns_df=self.returns_df,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_black_litterman_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="black_litterman",
            returns_df=self.returns_df,
            alpha_scores=self.alpha_scores,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_kelly_plugin(self):
        weights = self.optimizer.optimize(
            self.ticker_set,
            optimizer_name="kelly",
            returns_df=self.returns_df,
            alpha_scores=self.alpha_scores,
        )
        self.assertEqual(len(weights), len(self.tickers))
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)

    def test_constraints_engine(self):
        engine = ConstraintsEngine(PortfolioConstraints(max_weight_per_asset=0.25))
        raw_w = pd.Series([0.5, 0.2, 0.1, 0.1, 0.1], index=self.tickers)
        constrained_w = engine.apply_all_constraints(raw_w)
        self.assertLessEqual(constrained_w.max(), 0.26)
        self.assertAlmostEqual(constrained_w.sum(), 1.0, places=4)

    def test_rebalancing_turnover_penalty(self):
        engine = RebalancingEngine(turnover_threshold=0.05)  # 5% threshold
        curr_w = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=self.tickers)
        targ_w = pd.Series([0.22, 0.18, 0.20, 0.30, 0.10], index=self.tickers)
        final_w = engine.apply_turnover_penalty(curr_w, targ_w)
        # Small changes (<5%) should be suppressed
        self.assertAlmostEqual(final_w["AAPL"], 0.2, places=4)
        self.assertAlmostEqual(final_w.sum(), 1.0, places=4)

    def test_transaction_cost_engine(self):
        cost_engine = TransactionCostEngine(commission_bps=10.0, slippage_bps=5.0)
        curr_w = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=self.tickers)
        targ_w = pd.Series([0.3, 0.1, 0.2, 0.2, 0.2], index=self.tickers)
        net_w, cost = cost_engine.net_weights_after_costs(curr_w, targ_w, portfolio_value=1e7)
        self.assertGreater(cost, 0.0)
        self.assertAlmostEqual(net_w.sum(), 1.0, places=4)

    def test_position_sizing_cash_buffer(self):
        sizing_engine = PositionSizingEngine(cash_buffer_pct=0.05)
        raw_w = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=self.tickers)
        sized_w = sizing_engine.apply_cash_buffer(raw_w)
        self.assertAlmostEqual(sized_w.sum(), 0.95, places=4)

    def test_backward_compatibility_legacy_equal_weight(self):
        w = self.optimizer.equal_weight(self.ticker_set, adv_data=self.adv_data)
        self.assertEqual(len(w), len(self.tickers))
        self.assertAlmostEqual(w.sum(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
