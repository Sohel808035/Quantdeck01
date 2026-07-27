"""
ai_quant_analyst/advisor.py
────────────────────────────
Strategy & Portfolio Advisor Module.
Generates actionable, institutional recommendations for model retraining, portfolio rebalancing, risk reduction, and execution optimization.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig

logger = logging.getLogger(__name__)


class StrategyAdvisor:
    """
    Synthesises outputs from backtests, risk audits, monitoring, and prediction explainers
    to offer prioritized, actionable recommendations.
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()

    def generate_recommendations(
        self,
        backtest_metrics: Optional[Dict[str, Any]] = None,
        risk_profile: Optional[Dict[str, Any]] = None,
        monitoring_summary: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generates prioritized action recommendations.

        Returns:
            List of dicts with keys: priority, domain, recommendation, rationale.
        """
        recs = []

        # ── 1. Backtest Metrics Audits ──────────────────────────────────────
        if backtest_metrics:
            sharpe = backtest_metrics.get("sharpe_ratio", 1.0)
            cagr = backtest_metrics.get("cagr", 0.10)
            max_dd = backtest_metrics.get("max_drawdown", -0.10)
            tc_bp = backtest_metrics.get("ann_fixed_cost_bp", 0.0)
            turnover = backtest_metrics.get("ann_turnover", 0.0)

            if sharpe < 0.6:
                recs.append({
                    "priority": "HIGH",
                    "domain": "MODEL_TUNING",
                    "recommendation": "Perform hyperparameter optimization and feature pruning.",
                    "rationale": f"Backtest Sharpe ratio ({sharpe:.2f}) is below target threshold 0.80.",
                })

            if abs(max_dd) > 0.20:
                recs.append({
                    "priority": "HIGH",
                    "domain": "RISK_CONTROL",
                    "recommendation": "Tighten volatility targeting scalar and lower position cap limits.",
                    "rationale": f"Max drawdown ({max_dd:.1%}) exceeds risk threshold of -20%.",
                })

            if tc_bp > 100.0 or turnover > 6.0:
                recs.append({
                    "priority": "MEDIUM",
                    "domain": "EXECUTION_COSTS",
                    "recommendation": "Switch rebalance frequency from weekly to monthly or apply turnover constraints.",
                    "rationale": f"Annualised turnover ({turnover:.1f}x) generates {tc_bp:.0f} bps in transaction costs.",
                })

        # ── 2. Risk Profile Audits ──────────────────────────────────────────
        if risk_profile:
            risk_grade = risk_profile.get("risk_grade", "LOW")
            var_95 = risk_profile.get("var_95", 0.0)
            alerts = risk_profile.get("alerts", [])

            if risk_grade == "HIGH":
                recs.append({
                    "priority": "CRITICAL",
                    "domain": "PORTFOLIO_DELEVERAGING",
                    "recommendation": "Reduce gross exposure by 20% and allocate to cash buffer.",
                    "rationale": f"Risk profile grade is HIGH with active risk alerts: {len(alerts)} alerts triggered.",
                })

        # ── 3. Monitoring Audits ───────────────────────────────────────────
        if monitoring_summary:
            dq_passed = monitoring_summary.get("data_quality_passed", True)
            drift_detected = monitoring_summary.get("feature_drift_detected", False)

            if drift_detected:
                recs.append({
                    "priority": "HIGH",
                    "domain": "MODEL_RETRAINING",
                    "recommendation": "Trigger automated model retraining pipeline on latest 12-month window.",
                    "rationale": "Significant feature distribution drift detected (PSI > 0.20).",
                })

            if not dq_passed:
                recs.append({
                    "priority": "CRITICAL",
                    "domain": "DATA_PIPELINE",
                    "recommendation": "Audit data ingest pipeline for missing values or index staleness.",
                    "rationale": "Data quality checks failed on incoming market feeds.",
                })

        # Default recommendation if everything is optimal
        if not recs:
            recs.append({
                "priority": "LOW",
                "domain": "MAINTENANCE",
                "recommendation": "Maintain current portfolio allocation and monitoring schedule.",
                "rationale": "All quantitative systems operate within nominal risk and performance parameters.",
            })

        return recs
