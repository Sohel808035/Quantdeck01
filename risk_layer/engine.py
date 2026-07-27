"""
risk_layer/engine.py
────────────────────
Master Institutional Risk Engine Orchestrator.
Combines VaR/CVaR, Stress Testing, Liquidity Risk, Factor Risk, Sector/Country Exposures,
Correlation Analysis, Tail Risk, Scenario Analysis, and Limits Auditing into a single API.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Set
import pandas as pd
import numpy as np

from risk_layer.base import RiskMetricsReport
from risk_layer.config import RiskConfig
from risk_layer.var_cvar import VaRCVaREngine
from risk_layer.stress_testing import StressTestingEngine
from risk_layer.liquidity_risk import LiquidityRiskEngine
from risk_layer.factor_risk import FactorRiskEngine
from risk_layer.sector_country_exposure import ExposureRiskEngine
from risk_layer.correlation_analysis import CorrelationAnalysisEngine
from risk_layer.tail_risk import TailRiskEngine
from risk_layer.scenario_analysis import ScenarioAnalysisEngine
from risk_layer.limits import LimitsAuditEngine

logger = logging.getLogger(__name__)


class InstitutionalRiskEngine:
    """Master Institutional Risk Management Suite."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.var_engine = VaRCVaREngine(confidence_levels=self.config.confidence_levels)
        self.stress_engine = StressTestingEngine()
        self.liquidity_engine = LiquidityRiskEngine(max_adv_participation=self.config.max_adv_participation_pct)
        self.factor_engine = FactorRiskEngine()
        self.exposure_engine = ExposureRiskEngine()
        self.correlation_engine = CorrelationAnalysisEngine()
        self.tail_engine = TailRiskEngine(evt_quantile=self.config.evt_threshold_quantile)
        self.scenario_engine = ScenarioAnalysisEngine()
        self.limits_engine = LimitsAuditEngine(config=self.config)

    def audit_portfolio_risk(
        self,
        weights: pd.Series,
        returns_df: Optional[pd.DataFrame] = None,
        adv_data: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        country_map: Optional[Dict[str, str]] = None,
        stock_betas: Optional[pd.Series] = None,
        factor_beta_matrix: Optional[pd.DataFrame] = None,
    ) -> RiskMetricsReport:
        """
        Executes comprehensive 12-factor institutional risk audit across portfolio positions.
        """
        if weights.empty:
            return RiskMetricsReport(portfolio_value=self.config.portfolio_value)

        w = weights / weights.sum()
        tickers = list(w.index)
        adv_series = adv_data if adv_data is not None else pd.Series(dtype=float)

        # Portfolio return series
        if returns_df is not None and not returns_df.empty:
            common = list(set(tickers) & set(returns_df.columns))
            if common:
                port_returns = (returns_df[common].dropna(how="all") * w.reindex(common).fillna(0)).sum(axis=1)
            else:
                port_returns = pd.Series(dtype=float)
        else:
            port_returns = pd.Series(dtype=float)

        # 1. VaR & CVaR
        var95_h, cvar95 = self.var_engine.historical_var_cvar(port_returns, confidence=0.95)
        var99_h, cvar99 = self.var_engine.historical_var_cvar(port_returns, confidence=0.99)
        var95_p, _      = self.var_engine.parametric_var_cvar(port_returns, confidence=0.95)
        var99_p, _      = self.var_engine.parametric_var_cvar(port_returns, confidence=0.99)
        var95_mc, _     = self.var_engine.monte_carlo_var_cvar(port_returns, confidence=0.95)

        # 2. Liquidity Risk & LVaR
        dtl = self.liquidity_engine.portfolio_days_to_liquidate(w, adv_series, self.config.portfolio_value)
        lvar95 = self.liquidity_engine.liquidity_adjusted_var(var95_h, w, adv_series, self.config.portfolio_value)

        # 3. Concentration & Limits
        conc = self.limits_engine.concentration_metrics(w)
        passed, _ = self.limits_engine.audit_limits(w, sector_map=sector_map)

        # 4. Tail Risk
        tail_metrics = self.tail_engine.compute_tail_metrics(port_returns)

        # 5. Correlation Metrics
        corr_metrics = self.correlation_engine.compute_correlation_metrics(returns_df if returns_df is not None else pd.DataFrame(), tickers=tickers)

        # 6. Sector & Country Exposure
        sec_df = self.exposure_engine.compute_sector_exposure(w, sector_map or {})
        sec_dict = sec_df["portfolio_weight"].to_dict() if not sec_df.empty else {}
        cntry_series = self.exposure_engine.compute_country_exposure(w, country_map)
        cntry_dict = cntry_series.to_dict()

        # 7. Factor Risk
        if factor_beta_matrix is not None:
            factor_exp = self.factor_engine.compute_factor_exposures(w, factor_beta_matrix).to_dict()
        else:
            factor_exp = {}

        # 8. Stress Testing & Scenario Analysis
        stress_losses = self.stress_engine.run_historical_replay(w, stock_betas=stock_betas, portfolio_value=self.config.portfolio_value)
        scenario_losses = self.scenario_engine.run_scenario_matrix(w, stock_betas=stock_betas, portfolio_value=self.config.portfolio_value)

        return RiskMetricsReport(
            portfolio_value=self.config.portfolio_value,
            var_95_historical=var95_h,
            var_99_historical=var99_h,
            var_95_parametric=var95_p,
            var_99_parametric=var99_p,
            var_95_monte_carlo=var95_mc,
            cvar_95=cvar95,
            cvar_99=cvar99,
            lvar_95=lvar95,
            hhi_index=conc["hhi_index"],
            effective_n_stocks=conc["effective_n_stocks"],
            top_5_concentration=conc["top_5_concentration"],
            top_10_concentration=conc["top_10_concentration"],
            max_position_weight=float(w.max()),
            position_limits_passed=passed,
            skewness=tail_metrics["skewness"],
            kurtosis=tail_metrics["kurtosis"],
            evt_tail_index=tail_metrics["evt_tail_index"],
            avg_pairwise_correlation=corr_metrics["avg_pairwise_correlation"],
            pca_top3_variance_pct=corr_metrics["pca_top3_var"],
            days_to_liquidate_95pct=dtl,
            sector_exposures=sec_dict,
            country_exposures=cntry_dict,
            factor_exposures=factor_exp,
            stress_test_losses=stress_losses,
            scenario_impacts=scenario_losses,
        )
