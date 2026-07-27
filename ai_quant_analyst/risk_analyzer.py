"""
ai_quant_analyst/risk_analyzer.py
──────────────────────────────────
Risk Interpretation Module.
Translates complex VaR/CVaR, stress test outputs, factor exposures, and liquidity metrics into institutional risk commentary.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    Translates quantitative risk engine outputs into institutional risk narratives and alert commentary.
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()

    def analyze_risk_profile(
        self,
        var_cvar_dict: Dict[str, float],
        factor_exposures: Optional[Dict[str, float]] = None,
        concentration_dict: Optional[Dict[str, float]] = None,
        stress_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesises a comprehensive institutional risk report.

        Args:
            var_cvar_dict:      Dict with 'hist_var', 'hist_cvar', 'param_var', 'param_cvar'.
            factor_exposures:   Dict of factor -> beta.
            concentration_dict: Dict with 'top_5_weight', 'hhi', 'effective_n'.
            stress_results:     Dict of scenario_name -> stress_sharpe.

        Returns:
            Dict containing risk grade, executive summary, and detailed sections.
        """
        sections = []
        alerts = []

        # ── 1. Tail Risk Assessment (VaR / CVaR) ──────────────────────────
        var_95 = var_cvar_dict.get("hist_var", var_cvar_dict.get("param_var", 0.0))
        cvar_95 = var_cvar_dict.get("hist_cvar", var_cvar_dict.get("param_cvar", 0.0))

        cvar_ratio = (cvar_95 / var_95) if var_95 > 0 else 1.0
        tail_fatness = "Fat-tailed (Heavy Crash Risk)" if cvar_ratio > 1.4 else "Normal Tail Behavior"

        tail_summary = (
            f"Daily 95% Value-at-Risk (VaR) is {var_95:.2%}, with an Expected Shortfall (CVaR) of {cvar_95:.2%}. "
            f"The CVaR/VaR ratio of {cvar_ratio:.2f} indicates {tail_fatness}."
        )
        sections.append(("Tail Risk & VaR", tail_summary))

        if var_95 > 0.03:
            alerts.append(f"HIGH TAIL RISK: Daily VaR ({var_95:.2%}) exceeds 3% threshold.")

        # ── 2. Factor Exposures ─────────────────────────────────────────────
        factor_summary = "No factor exposure data provided."
        if factor_exposures:
            dominant_factor = max(factor_exposures.items(), key=lambda x: abs(x[1]))
            factor_summary = (
                f"Portfolio maintains primary exposure to '{dominant_factor[0]}' (beta = {dominant_factor[1]:+.2f}). "
                f"Total factor coverage: {len(factor_exposures)} factors tracked."
            )
            sections.append(("Factor Risk", factor_summary))

        # ── 3. Portfolio Concentration ────────────────────────────────────
        conc_summary = "No concentration data provided."
        if concentration_dict:
            top5 = concentration_dict.get("top_5_weight", 0.0)
            eff_n = concentration_dict.get("effective_n", 0.0)
            conc_summary = (
                f"Top 5 assets account for {top5:.1%} of total AUM. "
                f"Effective number of positions (1/HHI): {eff_n:.1f} assets."
            )
            sections.append(("Concentration Risk", conc_summary))
            if top5 > 0.40:
                alerts.append(f"HIGH CONCENTRATION: Top 5 holdings account for {top5:.1%} of portfolio.")

        # ── 4. Stress Test Resilience ──────────────────────────────────────
        stress_summary = "No stress test results provided."
        if stress_results:
            scenarios = stress_results.get("scenarios", {})
            mandate_met = stress_results.get("mandate_met", True)
            resilient_count = sum(1 for s in scenarios.values() if s >= 0.8)
            total_scenarios = len(scenarios)

            stress_summary = (
                f"Passed {resilient_count}/{total_scenarios} stress scenarios. "
                f"CQRO Mandate (Sharpe ≥ 0.8 under stress): {'MET ✅' if mandate_met else 'BREACHED ❌'}."
            )
            sections.append(("Stress Test Audit", stress_summary))
            if not mandate_met:
                alerts.append("CQRO MANDATE BREACH: Sharpe dropped below 0.8 under stress scenarios.")

        # Overall Risk Grade
        if len(alerts) == 0:
            risk_grade = "LOW"
            overall_assessment = "Portfolio risk profile is well-balanced with resilient tail metrics and controlled concentration."
        elif len(alerts) == 1:
            risk_grade = "MODERATE"
            overall_assessment = "Portfolio displays moderate risk exposure with 1 area requiring monitoring."
        else:
            risk_grade = "HIGH"
            overall_assessment = "Portfolio displays elevated systemic risk across multiple dimensions."

        return {
            "risk_grade": risk_grade,
            "overall_assessment": overall_assessment,
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "alerts": alerts,
            "sections": dict(sections),
        }
