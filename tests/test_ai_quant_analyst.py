"""
tests/test_ai_quant_analyst.py
────────────────────────────────
Unit Test Suite for QuantSphereX AI Quant Analyst.

Covers:
  1. PredictionExplainer  - Prediction narratives & SHAP interpretations
  2. RiskAnalyzer         - Risk profile analysis, VaR/CVaR, factor exposures
  3. TextSummarizer       - News sentiment scoring & earnings call summarization
  4. AnomalyDetector      - Return anomalies, prediction anomalies, factor shifts
  5. StrategyAdvisor      - Prioritized improvement recommendations
  6. ReportGenerator      - Markdown investment report generation & export
  7. AIQuantAnalyst       - Master orchestrator end-to-end capabilities
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig
from ai_quant_analyst.prediction_explainer import PredictionExplainer
from ai_quant_analyst.risk_analyzer import RiskAnalyzer
from ai_quant_analyst.text_summarizer import TextSummarizer
from ai_quant_analyst.anomaly_detector import AnomalyDetector
from ai_quant_analyst.advisor import StrategyAdvisor
from ai_quant_analyst.report_generator import ReportGenerator
from ai_quant_analyst.analyst import AIQuantAnalyst


# ─── 1. PredictionExplainer ──────────────────────────────────────────────────

class TestPredictionExplainer(unittest.TestCase):
    def setUp(self):
        self.explainer = PredictionExplainer()

    def test_explain_bullish_prediction(self):
        res = self.explainer.explain_prediction("RELIANCE", 0.045, probability=0.82)
        self.assertEqual(res["direction"], "BULLISH")
        self.assertIn("RELIANCE", res["narrative"])

    def test_explain_bearish_prediction(self):
        res = self.explainer.explain_prediction("TCS", -0.055, probability=0.75)
        self.assertEqual(res["direction"], "BEARISH")
        self.assertIn("BEARISH", res["narrative"])

    def test_interpret_shap_values(self):
        shap_dict = {
            "mom_60": 0.035,
            "vol_20": -0.012,
            "rsi_14": 0.008,
        }
        res = self.explainer.interpret_shap("INFY", shap_dict, base_value=0.01)
        self.assertIn("executive_summary", res)
        self.assertGreater(len(res["top_positive_drivers"]), 0)
        self.assertGreater(len(res["top_negative_drivers"]), 0)


# ─── 2. RiskAnalyzer ─────────────────────────────────────────────────────────

class TestRiskAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = RiskAnalyzer()

    def test_analyze_low_risk_profile(self):
        var_cvar = {"hist_var": 0.015, "hist_cvar": 0.022}
        factors = {"Market": 0.95, "Momentum": 0.10}
        conc = {"top_5_weight": 0.25, "effective_n": 18.5}
        stress = {"scenarios": {"2x TC": 1.1}, "mandate_met": True}

        res = self.analyzer.analyze_risk_profile(var_cvar, factors, conc, stress)
        self.assertEqual(res["risk_grade"], "LOW")
        self.assertEqual(len(res["alerts"]), 0)

    def test_analyze_high_risk_profile_triggers_alerts(self):
        var_cvar = {"hist_var": 0.045, "hist_cvar": 0.072}  # High VaR > 3%
        conc = {"top_5_weight": 0.65, "effective_n": 4.2}    # High conc > 40%
        stress = {"scenarios": {"2x TC": 0.5}, "mandate_met": False}

        res = self.analyzer.analyze_risk_profile(var_cvar, concentration_dict=conc, stress_results=stress)
        self.assertEqual(res["risk_grade"], "HIGH")
        self.assertGreater(len(res["alerts"]), 0)


# ─── 3. TextSummarizer ───────────────────────────────────────────────────────

class TestTextSummarizer(unittest.TestCase):
    def setUp(self):
        self.ts = TextSummarizer()

    def test_positive_news_sentiment(self):
        headline = "Company announces record profit and guidance raise"
        content = "Strong demand drove margin expansion and robust cash flow."
        res = self.ts.summarize_news(headline, content, symbol="TATA")
        self.assertEqual(res["sentiment_label"], "POSITIVE")
        self.assertGreater(res["sentiment_score"], 0.0)

    def test_negative_news_sentiment(self):
        headline = "Earnings miss due to supply chain disruption"
        content = "Margin compression led to guidance cut and lawsuit risks."
        res = self.ts.summarize_news(headline, content, symbol="WIPRO")
        self.assertEqual(res["sentiment_label"], "NEGATIVE")
        self.assertLess(res["sentiment_score"], 0.0)

    def test_earnings_summarizer(self):
        text = "Management discussed Q4 results, strong margin expansion, and raised guidance."
        res = self.ts.summarize_earnings("HDFCBANK", text, quarter="Q4", year=2025)
        self.assertTrue(res["guidance_discussed"])
        self.assertIn("HDFCBANK", res["executive_memo"])


# ─── 4. AnomalyDetector ──────────────────────────────────────────────────────

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.ad = AnomalyDetector()

    def test_detect_return_anomalies(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        rets = pd.DataFrame({"STOCK_A": np.random.normal(0, 0.01, 100)}, index=dates)
        rets.iloc[80] = 0.15  # 15% spike

        res = self.ad.detect_return_anomalies(rets, window=60)
        self.assertGreater(res["total_anomalies"], 0)

    def test_detect_prediction_anomalies(self):
        preds = pd.Series([0.01, 0.02, 0.015, 0.25], index=["S1", "S2", "S3", "S4"])
        baseline = pd.Series(np.random.normal(0.01, 0.01, 100))
        res = self.ad.detect_prediction_anomalies(preds, baseline)
        self.assertTrue(res["has_anomalies"])

    def test_detect_factor_shift(self):
        curr = {"Market": 1.5, "Value": -0.2}
        base = {"Market": 0.9, "Value": -0.1}
        res = self.ad.detect_factor_shift(curr, base, threshold=0.4)
        self.assertTrue(res["significant_rotation"])


# ─── 5. StrategyAdvisor ──────────────────────────────────────────────────────

class TestStrategyAdvisor(unittest.TestCase):
    def setUp(self):
        self.advisor = StrategyAdvisor()

    def test_generate_recommendations_low_sharpe(self):
        metrics = {"sharpe_ratio": 0.4, "max_drawdown": -0.10, "cagr": 0.08}
        recs = self.advisor.generate_recommendations(backtest_metrics=metrics)
        domains = [r["domain"] for r in recs]
        self.assertIn("MODEL_TUNING", domains)

    def test_generate_recommendations_high_turnover(self):
        metrics = {"sharpe_ratio": 1.2, "ann_fixed_cost_bp": 150.0, "ann_turnover": 8.0}
        recs = self.advisor.generate_recommendations(backtest_metrics=metrics)
        domains = [r["domain"] for r in recs]
        self.assertIn("EXECUTION_COSTS", domains)


# ─── 6. ReportGenerator ──────────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.rg = ReportGenerator()

    def test_build_investment_report(self):
        metrics = {"cagr": 0.18, "sharpe_ratio": 1.45, "max_drawdown": -0.12}
        report = self.rg.build_investment_report("Alpha-v1", backtest_metrics=metrics)
        self.assertIn("Alpha-v1", report)
        self.assertIn("1.45", report)

    def test_export_report(self):
        report_text = "# Test Report\nContent here."
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            exported = self.rg.export_report(report_text, path)
            self.assertTrue(os.path.exists(exported))


# ─── 7. AIQuantAnalyst (Full Orchestration) ──────────────────────────────────

class TestAIQuantAnalyst(unittest.TestCase):
    def setUp(self):
        self.analyst = AIQuantAnalyst()

    def test_explain_prediction(self):
        res = self.analyst.explain_prediction("RELIANCE", 0.05, probability=0.80)
        self.assertIn("BULLISH", res["direction"])

    def test_interpret_shap(self):
        shap_map = {"momentum": 0.02, "volatility": -0.01}
        res = self.analyst.interpret_shap("RELIANCE", shap_map)
        self.assertIn("executive_summary", res)

    def test_analyze_risk(self):
        var_dict = {"hist_var": 0.02, "hist_cvar": 0.03}
        res = self.analyst.analyze_risk(var_dict)
        self.assertIn("risk_grade", res)

    def test_summarize_news(self):
        res = self.analyst.summarize_news("Quarterly revenue beats expectation", "Record sales recorded.", symbol="TCS")
        self.assertEqual(res["sentiment_label"], "POSITIVE")

    def test_summarize_earnings(self):
        res = self.analyst.summarize_earnings("TCS", "Record growth and guidance raise discussed.")
        self.assertIn("TCS", res["executive_memo"])

    def test_detect_anomalies(self):
        np.random.seed(42)
        df = pd.DataFrame({"A": np.random.randn(100)}, index=pd.date_range("2022-01-01", periods=100, freq="B"))
        res = self.analyst.detect_anomalies(returns_df=df)
        self.assertIn("return_anomalies", res)

    def test_suggest_improvements(self):
        metrics = {"sharpe_ratio": 0.5}
        recs = self.analyst.suggest_improvements(backtest_metrics=metrics)
        self.assertGreater(len(recs), 0)

    def test_generate_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "invest_report.md")
            report = self.analyst.generate_report(
                strategy_name="QuantSphereX Master Strategy",
                backtest_metrics={"cagr": 0.22, "sharpe_ratio": 1.65, "max_drawdown": -0.09},
                export_path=out_path,
            )
            self.assertIn("QuantSphereX Master Strategy", report)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
