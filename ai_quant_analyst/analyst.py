"""
ai_quant_analyst/analyst.py
────────────────────────────
Master AI Quant Analyst Orchestrator.
Single entry point wiring all analyst capability engines:
  - Explain Predictions
  - Interpret SHAP
  - Generate Investment Reports
  - Analyze Risk
  - Summarize Earnings & News
  - Detect Anomalies
  - Suggest Improvements
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig
from ai_quant_analyst.prediction_explainer import PredictionExplainer
from ai_quant_analyst.risk_analyzer import RiskAnalyzer
from ai_quant_analyst.text_summarizer import TextSummarizer
from ai_quant_analyst.anomaly_detector import AnomalyDetector
from ai_quant_analyst.advisor import StrategyAdvisor
from ai_quant_analyst.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class AIQuantAnalyst:
    """
    QuantSphereX AI Quant Analyst.

    Capabilities:
      1. explain_prediction()     - Translates model signals to narratives
      2. interpret_shap()         - Explains feature attributions
      3. analyze_risk()           - Synthesises VaR, CVaR, stress & factors
      4. summarize_news()         - Sentiment scoring & impact rating
      5. summarize_earnings()     - Transcript highlights & Guidance evaluation
      6. detect_anomalies()       - Flags returns, prediction shifts, factor outliers
      7. suggest_improvements()   - Prioritized optimization recommendations
      8. generate_report()        - Institutional investment memos

    Usage:
        analyst = AIQuantAnalyst()
        explanation = analyst.explain_prediction("RELIANCE", 0.045, 0.78)
        report = analyst.generate_report("Alpha-v1", backtest_metrics=metrics)
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()
        self.explainer = PredictionExplainer(self.config)
        self.risk_analyzer = RiskAnalyzer(self.config)
        self.text_summarizer = TextSummarizer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.advisor = StrategyAdvisor(self.config)
        self.report_generator = ReportGenerator(self.config)

    # ── 1. Explain Predictions ───────────────────────────────────────────────

    def explain_prediction(
        self,
        symbol: str,
        predicted_score: float,
        probability: Optional[float] = None,
        confidence_interval: Optional[Tuple[float, float]] = None,
        feature_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Explains model prediction score for a stock."""
        return self.explainer.explain_prediction(
            symbol, predicted_score, probability, confidence_interval, feature_values
        )

    # ── 2. Interpret SHAP ───────────────────────────────────────────────────

    def interpret_shap(
        self,
        symbol: str,
        shap_values: Dict[str, float],
        base_value: float = 0.0,
    ) -> Dict[str, Any]:
        """Interprets SHAP values for a stock."""
        return self.explainer.interpret_shap(symbol, shap_values, base_value)

    # ── 3. Analyze Risk ─────────────────────────────────────────────────────

    def analyze_risk(
        self,
        var_cvar_dict: Dict[str, float],
        factor_exposures: Optional[Dict[str, float]] = None,
        concentration_dict: Optional[Dict[str, float]] = None,
        stress_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesises quantitative risk engines into an institutional risk narrative."""
        return self.risk_analyzer.analyze_risk_profile(
            var_cvar_dict, factor_exposures, concentration_dict, stress_results
        )

    # ── 4. Summarize News & Earnings ─────────────────────────────────────────

    def summarize_news(
        self,
        headline: str,
        content: str,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Computes sentiment score and impact rating for a news item."""
        return self.text_summarizer.summarize_news(headline, content, symbol)

    def summarize_earnings(
        self,
        symbol: str,
        transcript_text: str,
        quarter: str = "Q4",
        year: int = 2025,
    ) -> Dict[str, Any]:
        """Summarizes earnings call transcript."""
        return self.text_summarizer.summarize_earnings(symbol, transcript_text, quarter, year)

    # ── 5. Detect Anomalies ─────────────────────────────────────────────────

    def detect_anomalies(
        self,
        returns_df: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.Series] = None,
        baseline_predictions: Optional[pd.Series] = None,
        current_betas: Optional[Dict[str, float]] = None,
        baseline_betas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Runs multi-dimensional anomaly detection across returns, predictions, and factor shifts."""
        results = {}
        if returns_df is not None:
            results["return_anomalies"] = self.anomaly_detector.detect_return_anomalies(returns_df)
        if predictions is not None and baseline_predictions is not None:
            results["prediction_anomalies"] = self.anomaly_detector.detect_prediction_anomalies(predictions, baseline_predictions)
        if current_betas is not None and baseline_betas is not None:
            results["factor_shifts"] = self.anomaly_detector.detect_factor_shift(current_betas, baseline_betas)
        return results

    # ── 6. Suggest Improvements ─────────────────────────────────────────────

    def suggest_improvements(
        self,
        backtest_metrics: Optional[Dict[str, Any]] = None,
        risk_profile: Optional[Dict[str, Any]] = None,
        monitoring_summary: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Generates prioritized quantitative improvement recommendations."""
        return self.advisor.generate_recommendations(
            backtest_metrics, risk_profile, monitoring_summary
        )

    # ── 7. Generate Full Investment Report ───────────────────────────────────

    def generate_report(
        self,
        strategy_name: str = "QuantSphereX Alpha Strategy",
        predictions_explanation: Optional[Dict[str, Any]] = None,
        shap_explanation: Optional[Dict[str, Any]] = None,
        risk_analysis: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[Dict[str, str]]] = None,
        backtest_metrics: Optional[Dict[str, Any]] = None,
        export_path: Optional[str] = None,
    ) -> str:
        """Assembles and exports a complete Markdown investment report."""
        report = self.report_generator.build_investment_report(
            strategy_name=strategy_name,
            predictions_explanation=predictions_explanation,
            shap_explanation=shap_explanation,
            risk_analysis=risk_analysis,
            recommendations=recommendations or self.suggest_improvements(backtest_metrics, risk_analysis),
            backtest_metrics=backtest_metrics,
        )
        if export_path:
            self.report_generator.export_report(report, export_path)
        return report
