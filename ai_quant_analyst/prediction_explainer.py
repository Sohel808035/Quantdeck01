"""
ai_quant_analyst/prediction_explainer.py
────────────────────────────────────────
Prediction & SHAP Explanation Module.
Translates raw model signals, probabilities, and SHAP feature attributions into natural language investment rationales.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig

logger = logging.getLogger(__name__)


class PredictionExplainer:
    """
    Translates raw ML predictions and SHAP values into quantitative narrative explanations.
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()

    def explain_prediction(
        self,
        symbol: str,
        predicted_score: float,
        probability: Optional[float] = None,
        confidence_interval: Optional[Tuple[float, float]] = None,
        feature_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Explains a single stock prediction.

        Returns:
            Dict containing signal_direction, signal_strength, summary, and detail.
        """
        # Determine direction
        if predicted_score > 0.02:
            direction = "BULLISH"
            stance = "Strong Outperform" if predicted_score > 0.05 else "Moderate Outperform"
        elif predicted_score < -0.02:
            direction = "BEARISH"
            stance = "Strong Underperform" if predicted_score < -0.05 else "Moderate Underperform"
        else:
            direction = "NEUTRAL"
            stance = "Market Perform"

        # Signal confidence
        prob_str = f" with {probability:.1%} probability" if probability is not None else ""
        ci_str = f" [95% CI: {confidence_interval[0]:.3f} to {confidence_interval[1]:.3f}]" if confidence_interval else ""

        narrative = (
            f"The model maintains a {direction} stance ({stance}) on {symbol} "
            f"with an expected return of {predicted_score:+.2%}{prob_str}{ci_str}."
        )

        drivers = []
        if feature_values:
            sorted_feats = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feats = sorted_feats[:self.config.max_top_features]
            for feat, val in top_feats:
                impact = "positive" if val > 0 else "negative"
                drivers.append(f"• {feat}: {val:+.4f} ({impact} contribution)")

        return {
            "symbol": symbol,
            "direction": direction,
            "stance": stance,
            "predicted_score": round(predicted_score, 4),
            "probability": round(probability, 4) if probability is not None else None,
            "narrative": narrative,
            "key_drivers": drivers,
        }

    def interpret_shap(
        self,
        symbol: str,
        shap_values: Dict[str, float],
        base_value: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Interprets SHAP values for a stock. Categorises positive vs negative drivers.

        Args:
            symbol:      Stock ticker.
            shap_values: Dict of feature_name -> SHAP attribution value.
            base_value:  Model expected base value E[f(x)].

        Returns:
            Dict with SHAP interpretation breakdown.
        """
        if not shap_values:
            return {"symbol": symbol, "interpretation": "No SHAP values provided."}

        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        top_shap = sorted_shap[:self.config.max_top_features]

        positive_drivers = [(k, v) for k, v in top_shap if v > 0]
        negative_drivers = [(k, v) for k, v in top_shap if v < 0]

        total_attribution = sum(shap_values.values())
        final_prediction = base_value + total_attribution

        lines = [
            f"SHAP attribution summary for {symbol}:",
            f"  Base Model Value: {base_value:+.4f}",
            f"  Total Attribution: {total_attribution:+.4f}",
            f"  Final Prediction:  {final_prediction:+.4f}",
        ]

        if positive_drivers:
            lines.append("  Top Positive Drivers:")
            for k, v in positive_drivers:
                lines.append(f"    + {k}: {v:+.4f} impact")

        if negative_drivers:
            lines.append("  Top Negative Drivers:")
            for k, v in negative_drivers:
                lines.append(f"    - {k}: {v:+.4f} impact")

        pos_str = ", ".join(k for k, _ in positive_drivers[:2]) or "none"
        neg_str = ", ".join(k for k, _ in negative_drivers[:2]) or "none"

        executive_summary = (
            f"{symbol} prediction ({final_prediction:+.2%}) is driven primarily by "
            f"positive momentum in [{pos_str}] and offset by negative headwinds in [{neg_str}]."
        )

        return {
            "symbol": symbol,
            "base_value": round(base_value, 4),
            "total_attribution": round(total_attribution, 4),
            "final_prediction": round(final_prediction, 4),
            "top_positive_drivers": positive_drivers,
            "top_negative_drivers": negative_drivers,
            "executive_summary": executive_summary,
            "full_breakdown": "\n".join(lines),
        }
