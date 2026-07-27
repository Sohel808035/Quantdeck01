"""
tests/test_risk_engine.py
──────────────────────────
Unit Test Suite for QuantSphereX Institutional Risk Engine.

Covers:
  1. Value at Risk (VaR) & Conditional VaR (CVaR) — Historical, Parametric, Monte Carlo
  2. Stress Testing Engine (Historical Crisis Replay)
  3. Liquidity Risk Engine (Days-to-Liquidate, LVaR)
  4. Factor Risk Engine (Factor Beta & MCR)
  5. Sector & Country Exposure Risk Engine
  6. Correlation Analysis Engine (Pairwise, PCA, Condition Number)
  7. Tail Risk & EVT Engine (Skewness, Kurtosis, Tail Index, ETL)
  8. Macro Scenario Analysis Engine
  9. Position & Concentration Limits Audit (HHI, N_eff)
 10. Master InstitutionalRiskEngine Orchestrator
 11. Preserved Legacy Regime Models
"""

import unittest
import numpy as np
import pandas as pd

from risk_layer.config import RiskConfig
from risk_layer.base import RiskMetricsReport
from risk_layer.engine import InstitutionalRiskEngine
from risk_layer.var_cvar import VaRCVaREngine
from risk_layer.stress_testing import StressTestingEngine
from risk_layer.liquidity_risk import LiquidityRiskEngine
from risk_layer.factor_risk import FactorRiskEngine
from risk_layer.sector_country_exposure import ExposureRiskEngine
from risk_layer.correlation_analysis import CorrelationAnalysisEngine
from risk_layer.tail_risk import TailRiskEngine
from risk_layer.scenario_analysis import ScenarioAnalysisEngine
from risk_layer.limits import LimitsAuditEngine
from risk_layer.regime_model import compute_regime_exposure


def _make_returns_data(n_dates: int = 252, tickers: list = None):
    """Generates synthetic stock panel and portfolio return series."""
    if tickers is None:
        tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rets = np.random.normal(0.0005, 0.015, size=(n_dates, len(tickers)))
    df = pd.DataFrame(rets, index=dates, columns=tickers)
    weights = pd.Series(1.0 / len(tickers), index=tickers)
    port_ret = (df * weights).sum(axis=1)
    return df, weights, port_ret


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        self.returns_df, self.weights, self.port_ret = _make_returns_data(tickers=self.tickers)
        self.adv_data = pd.Series([2e8, 1.5e8, 3e8, 1.8e8, 2.5e8], index=self.tickers)
        self.sector_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "Technology",
            "HDFCBANK.NS": "Financials",
            "INFY.NS": "Technology",
            "ICICIBANK.NS": "Financials",
        }

    def test_var_cvar_engine(self):
        engine = VaRCVaREngine()

        var95_h, cvar95_h = engine.historical_var_cvar(self.port_ret, confidence=0.95)
        self.assertGreater(var95_h, 0.0)
        self.assertGreaterEqual(cvar95_h, var95_h)

        var95_p, cvar95_p = engine.parametric_var_cvar(self.port_ret, confidence=0.95)
        self.assertGreater(var95_p, 0.0)
        self.assertGreaterEqual(cvar95_p, var95_p)

        var95_mc, cvar95_mc = engine.monte_carlo_var_cvar(self.port_ret, confidence=0.95, n_simulations=1000)
        self.assertGreater(var95_mc, 0.0)

    def test_stress_testing_engine(self):
        engine = StressTestingEngine()
        results = engine.run_historical_replay(self.weights, portfolio_value=1e7)
        self.assertIn("2008_Global_Financial_Crisis", results)
        self.assertIn("2020_COVID_Crash", results)
        self.assertGreater(results["2008_Global_Financial_Crisis"], 0.0)

    def test_liquidity_risk_engine(self):
        engine = LiquidityRiskEngine()
        dtl = engine.portfolio_days_to_liquidate(self.weights, self.adv_data, portfolio_value=1e7)
        self.assertGreater(dtl, 0.0)

        lvar = engine.liquidity_adjusted_var(0.02, self.weights, self.adv_data, portfolio_value=1e7)
        self.assertGreaterEqual(lvar, 0.02)

    def test_factor_risk_engine(self):
        engine = FactorRiskEngine()
        beta_df = pd.DataFrame({
            "Momentum": [0.8, 0.5, 0.2, 0.6, 0.3],
            "Value": [-0.1, 0.2, 0.5, 0.1, 0.4],
        }, index=self.tickers)

        exposures = engine.compute_factor_exposures(self.weights, beta_df)
        self.assertIn("Momentum", exposures)
        self.assertIn("Value", exposures)

        cov = self.returns_df.cov()
        mcr = engine.marginal_contribution_to_risk(self.weights, cov)
        self.assertIn("pcr", mcr.columns)

    def test_sector_country_exposure_engine(self):
        engine = ExposureRiskEngine()
        sec_df = engine.compute_sector_exposure(self.weights, self.sector_map)
        self.assertIn("portfolio_weight", sec_df.columns)
        self.assertIn("Financials", sec_df.index)

        cntry = engine.compute_country_exposure(self.weights)
        self.assertIn("India", cntry.index)

    def test_correlation_analysis_engine(self):
        engine = CorrelationAnalysisEngine()
        metrics = engine.compute_correlation_metrics(self.returns_df)
        self.assertIn("avg_pairwise_correlation", metrics)
        self.assertIn("pca_top1_var", metrics)
        self.assertIn("condition_number", metrics)

    def test_tail_risk_engine(self):
        engine = TailRiskEngine()
        metrics = engine.compute_tail_metrics(self.port_ret)
        self.assertIn("skewness", metrics)
        self.assertIn("kurtosis", metrics)
        self.assertIn("etl_95", metrics)

    def test_scenario_analysis_engine(self):
        engine = ScenarioAnalysisEngine()
        impacts = engine.run_scenario_matrix(self.weights, portfolio_value=1e7)
        self.assertIn("Crude_Oil_Price_Spike_30pct", impacts)
        self.assertGreater(impacts["Crude_Oil_Price_Spike_30pct"], 0.0)

    def test_limits_audit_engine(self):
        engine = LimitsAuditEngine()
        conc = engine.concentration_metrics(self.weights)
        self.assertIn("hhi_index", conc)
        self.assertIn("effective_n_stocks", conc)
        self.assertAlmostEqual(conc["effective_n_stocks"], 5.0, places=1)

        passed, checks = engine.audit_limits(self.weights, sector_map=self.sector_map)
        self.assertIn("single_position_limit", checks)

    def test_institutional_risk_engine_orchestrator(self):
        engine = InstitutionalRiskEngine()
        report = engine.audit_portfolio_risk(
            weights=self.weights,
            returns_df=self.returns_df,
            adv_data=self.adv_data,
            sector_map=self.sector_map,
        )
        self.assertIsInstance(report, RiskMetricsReport)
        self.assertGreater(report.var_95_historical, 0.0)
        self.assertGreater(report.cvar_95, 0.0)
        summary = report.summary()
        self.assertIn("VaR_95_Hist", summary)

    def test_preserved_legacy_regime_model(self):
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        px = pd.Series(np.linspace(100, 200, 250), index=dates)
        nifty_df = pd.DataFrame({"Close": px}, index=dates)

        exp = compute_regime_exposure(nifty_df)
        self.assertEqual(len(exp), 250)
        self.assertGreaterEqual(float(exp.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
