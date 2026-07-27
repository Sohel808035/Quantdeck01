"""
QuantSphereX AI Quant Analyst Package.
Provides automated narrative explanations for model predictions, SHAP interpretations,
institutional risk analysis, earnings/news summarization, statistical anomaly detection,
prioritized strategy recommendations, and Markdown/JSON report generation.
"""

from ai_quant_analyst.config import AIAnalystConfig
from ai_quant_analyst.prediction_explainer import PredictionExplainer
from ai_quant_analyst.risk_analyzer import RiskAnalyzer
from ai_quant_analyst.text_summarizer import TextSummarizer
from ai_quant_analyst.anomaly_detector import AnomalyDetector
from ai_quant_analyst.advisor import StrategyAdvisor
from ai_quant_analyst.report_generator import ReportGenerator
from ai_quant_analyst.analyst import AIQuantAnalyst

__all__ = [
    "AIAnalystConfig",
    "PredictionExplainer",
    "RiskAnalyzer",
    "TextSummarizer",
    "AnomalyDetector",
    "StrategyAdvisor",
    "ReportGenerator",
    "AIQuantAnalyst",
]
