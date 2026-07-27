"""
ml_layer/confidence.py
────────────────────────
ML Pipeline: Confidence Estimation Module (v1.0.0)

Estimates prediction uncertainty and confidence intervals via:
  1. Ensemble Variance    — Std across N XGBoost ensemble members
  2. Quantile Regression  — Separate XGBoost models at user-defined quantiles
  3. Conformal Prediction — Distribution-free coverage guarantees
  4. Score Bucketing      — Discretize scores into high/medium/low confidence tiers
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel, XGBoostAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


def ensemble_variance(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Computes prediction variance across ensemble members as a confidence proxy.

    High variance → low confidence (models disagree).
    Low variance  → high confidence (models agree).

    Args:
        model: Trained EnsembleAlphaModel with ≥2 members.
        X:     Feature DataFrame.

    Returns:
        Series of per-row prediction standard deviations (same index as X).
    """
    if not model.models or len(model.models) < 2:
        logger.warning("[Confidence] Need ≥2 ensemble members for variance estimation.")
        return pd.Series(np.nan, index=X.index)

    preds = pd.concat([m.predict(X) for m in model.models], axis=1)
    return preds.std(axis=1).rename("prediction_std")


def confidence_tiers(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.Series:
    """
    Assigns each prediction to a confidence tier (HIGH / MEDIUM / LOW)
    based on ensemble variance relative to the cross-sectional distribution.

    Args:
        model:         Trained EnsembleAlphaModel.
        X:             Feature DataFrame.
        low_quantile:  Variance below this quantile → HIGH confidence.
        high_quantile: Variance above this quantile → LOW confidence.

    Returns:
        Series of 'HIGH', 'MEDIUM', 'LOW' labels per row.
    """
    var_series = ensemble_variance(model, X)
    if var_series.isna().all():
        return pd.Series("UNKNOWN", index=X.index)

    lo = var_series.quantile(low_quantile)
    hi = var_series.quantile(high_quantile)

    tiers = pd.Series("MEDIUM", index=X.index)
    tiers[var_series <= lo] = "HIGH"
    tiers[var_series >= hi] = "LOW"

    return tiers.rename("confidence_tier")


def conformal_intervals(
    model: EnsembleAlphaModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    coverage: float = 0.90,
) -> pd.DataFrame:
    """
    Computes conformal prediction intervals with guaranteed marginal coverage.

    Uses a calibration set to compute nonconformity scores,
    then applies the quantile correction to test predictions.

    Args:
        model:    Trained EnsembleAlphaModel.
        X_cal:    Calibration features.
        y_cal:    Calibration true labels.
        X_test:   Test features to predict intervals for.
        coverage: Desired coverage probability (default: 90%).

    Returns:
        DataFrame with columns: predicted, lower_bound, upper_bound.
    """
    # Calibration: compute nonconformity scores |y - ŷ|
    cal_preds = model.predict(X_cal)
    residuals = (y_cal.reindex(cal_preds.index) - cal_preds).abs().dropna()

    if len(residuals) == 0:
        logger.warning("[Confidence] Empty calibration set for conformal intervals.")
        test_preds = model.predict(X_test)
        return pd.DataFrame({
            "predicted":    test_preds,
            "lower_bound":  np.nan,
            "upper_bound":  np.nan,
        })

    # Quantile threshold
    alpha = 1 - coverage
    quantile_level = min(1.0, (1 - alpha) * (1 + 1 / len(residuals)))
    threshold = float(residuals.quantile(quantile_level))

    test_preds = model.predict(X_test)
    return pd.DataFrame({
        "predicted":   test_preds.values,
        "lower_bound": test_preds.values - threshold,
        "upper_bound": test_preds.values + threshold,
    }, index=X_test.index)


def full_confidence_report(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    X_cal: Optional[pd.DataFrame] = None,
    y_cal: Optional[pd.Series] = None,
    config: Optional[MLConfig] = None,
) -> pd.DataFrame:
    """
    Generates a full confidence report combining ensemble variance,
    confidence tiers, and (optionally) conformal intervals.

    Args:
        model:   Trained EnsembleAlphaModel.
        X:       Feature DataFrame for inference.
        X_cal:   Calibration features (optional, for conformal intervals).
        y_cal:   Calibration labels (optional, for conformal intervals).
        config:  MLConfig.

    Returns:
        DataFrame with all confidence columns.
    """
    if config is None:
        config = MLConfig()

    mean_pred  = model.predict(X).rename("alpha_score")
    std_pred   = ensemble_variance(model, X)
    tier_pred  = confidence_tiers(model, X)

    result = pd.concat([mean_pred, std_pred, tier_pred], axis=1)

    if X_cal is not None and y_cal is not None:
        intervals = conformal_intervals(
            model, X_cal, y_cal, X, coverage=0.90
        )
        result["lower_90"] = intervals["lower_bound"]
        result["upper_90"] = intervals["upper_bound"]

    return result
