"""
ai_quant_analyst/anomaly_detector.py
───────────────────────────────────
Quantitative Anomaly Detection Module.
Detects statistical anomalies across asset returns, model prediction distributions, factor exposures, and portfolio weights.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ai_quant_analyst.config import AIAnalystConfig

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Identifies statistical anomalies using Z-scores, IQR bounds, and rolling volatility spikes.
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()
        self.z_thresh = self.config.anomaly_z_threshold

    def detect_return_anomalies(
        self,
        returns_df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Detects historical return spikes (|Z-score| > threshold).

        Returns:
            Dict of anomaly events per asset.
        """
        anomalies = []
        for col in returns_df.columns:
            series = returns_df[col].dropna()
            if len(series) < window:
                continue

            roll_mean = series.rolling(window).mean()
            roll_std = series.rolling(window).std().replace(0, np.nan)
            z_scores = (series - roll_mean) / roll_std

            spikes = series[z_scores.abs() > self.z_thresh]
            for date, val in spikes.items():
                z_val = float(z_scores.loc[date])
                direction = "SPIKE UP" if z_val > 0 else "CRASH DOWN"
                anomalies.append({
                    "symbol": col,
                    "date": str(date.date()) if hasattr(date, "date") else str(date),
                    "return": round(float(val), 4),
                    "z_score": round(z_val, 2),
                    "anomaly_type": direction,
                })

        return {
            "total_anomalies": len(anomalies),
            "anomalies": anomalies,
            "anomaly_rate": round(len(anomalies) / max(returns_df.size, 1), 6),
        }

    def detect_prediction_anomalies(
        self,
        predictions: pd.Series,
        historical_baseline: pd.Series,
    ) -> Dict[str, Any]:
        """
        Detects extreme prediction values relative to baseline expectations.
        """
        baseline_mean = historical_baseline.mean()
        baseline_std = historical_baseline.std() if historical_baseline.std() > 0 else 1.0

        z_scores = (predictions - baseline_mean) / baseline_std
        anomalous_preds = predictions[z_scores.abs() > self.z_thresh]

        flagged = []
        for idx, val in anomalous_preds.items():
            z = float(z_scores.loc[idx])
            flagged.append({
                "identifier": str(idx),
                "predicted_value": round(float(val), 4),
                "z_score": round(z, 2),
                "severity": "HIGH" if abs(z) > 4.0 else "MEDIUM",
            })

        return {
            "anomalous_count": len(flagged),
            "flagged_predictions": flagged,
            "has_anomalies": len(flagged) > 0,
        }

    def detect_factor_shift(
        self,
        current_betas: Dict[str, float],
        baseline_betas: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detects sudden shifts in portfolio factor exposures.
        """
        shifts = []
        for factor, curr_beta in current_betas.items():
            base_beta = baseline_betas.get(factor, 0.0)
            diff = curr_beta - base_beta
            if abs(diff) > threshold:
                shifts.append({
                    "factor": factor,
                    "baseline_beta": round(base_beta, 4),
                    "current_beta": round(curr_beta, 4),
                    "shift": round(diff, 4),
                    "direction": "ROTATION IN" if diff > 0 else "ROTATION OUT",
                })

        return {
            "shift_count": len(shifts),
            "shifts": shifts,
            "significant_rotation": len(shifts) > 0,
        }
