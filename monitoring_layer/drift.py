"""
monitoring_layer/drift.py
──────────────────────────
Feature Drift and Prediction Drift Monitor.
Uses Population Stability Index (PSI) and Kolmogorov-Smirnov test.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10 → No drift | 0.10–0.20 → Slight | > 0.20 → Significant drift.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    # Define bins from reference distribution
    breaks = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breaks = np.unique(breaks)
    if len(breaks) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breaks)
    cur_counts, _ = np.histogram(current, bins=breaks)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) — clip to small value
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


class DriftMonitor:
    """
    Monitors distributional shift between a reference window and a current window.

    Feature Drift:   Applied to ML model inputs (features).
    Prediction Drift: Applied to model output scores / predictions.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self.cfg = self.config.drift

    def check_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Computes PSI and KS test for each feature column.

        Args:
            reference_df: Historical/baseline feature DataFrame.
            current_df:   Recent feature DataFrame.
            feature_cols: Subset of columns to check (default: all numeric).

        Returns:
            Per-feature drift report dict.
        """
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        cols = feature_cols or numeric_cols
        cols = [c for c in cols if c in current_df.columns and c in reference_df.columns]

        report: Dict[str, Any] = {"type": "feature_drift", "features": {}, "drifted_features": []}

        for col in cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) < self.cfg.min_samples_for_drift or len(cur_vals) < self.cfg.min_samples_for_drift:
                continue

            psi = _compute_psi(ref_vals, cur_vals)
            ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)

            feature_result = {
                "psi": psi,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
                "drifted": False,
            }

            if psi > self.cfg.psi_critical_threshold or ks_pval < self.cfg.ks_pvalue_threshold:
                feature_result["drifted"] = True
                report["drifted_features"].append(col)
                sev = AlertSeverity.CRITICAL if psi > self.cfg.psi_critical_threshold else AlertSeverity.WARNING
                self.alert_engine.fire(
                    sev, "DRIFT", f"feature_drift.{col}",
                    value=psi, threshold=self.cfg.psi_warning_threshold,
                    message=f"Feature drift detected: '{col}' PSI={psi:.3f}, KS p={ks_pval:.3f}.",
                )

            report["features"][col] = feature_result

        report["drift_rate"] = round(
            len(report["drifted_features"]) / max(len(cols), 1), 4
        )
        return report

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Checks for distributional shift in model prediction scores.

        Args:
            reference_predictions: Baseline model outputs (e.g. scores from last N days).
            current_predictions:   Recent model outputs.
            model_name:            Name of the model (for alert messages).

        Returns:
            Prediction drift report dict.
        """
        ref = np.array(reference_predictions).flatten()
        cur = np.array(current_predictions).flatten()
        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]

        if len(ref) < self.cfg.min_samples_for_drift or len(cur) < self.cfg.min_samples_for_drift:
            return {"type": "prediction_drift", "status": "insufficient_data"}

        psi = _compute_psi(ref, cur)
        ks_stat, ks_pval = ks_2samp(ref, cur)

        mean_shift = float(cur.mean() - ref.mean())
        std_shift = float(cur.std() - ref.std())

        drifted = psi > self.cfg.psi_warning_threshold or ks_pval < self.cfg.ks_pvalue_threshold

        if drifted:
            sev = AlertSeverity.CRITICAL if psi > self.cfg.psi_critical_threshold else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "DRIFT", f"prediction_drift.{model_name}",
                value=psi, threshold=self.cfg.psi_warning_threshold,
                message=(
                    f"Prediction drift for '{model_name}': "
                    f"PSI={psi:.3f}, KS p={ks_pval:.3f}, mean shift={mean_shift:+.4f}."
                ),
            )

        return {
            "type": "prediction_drift",
            "model": model_name,
            "psi": psi,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "mean_shift": round(mean_shift, 6),
            "std_shift": round(std_shift, 6),
            "drifted": drifted,
        }
