"""
ml_layer/prediction.py
────────────────────────
ML Pipeline: Prediction Module (v1.0.0)

Handles inference from trained EnsembleAlphaModel with:
  - Cross-sectional score generation
  - Score normalization (percentile rank, z-score)
  - Date-stamped prediction panel output
  - Drift detection against training distribution
"""

from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


def predict(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    normalize: bool = True,
    method: str = "rank",
) -> pd.Series:
    """
    Generates cross-sectional alpha scores from a trained ensemble model.

    Args:
        model:     Trained EnsembleAlphaModel.
        X:         Feature DataFrame with (Date, Ticker) MultiIndex.
        normalize: Whether to apply cross-sectional normalization.
        method:    Normalization method: 'rank' (percentile) or 'zscore'.

    Returns:
        Series of alpha scores indexed by (Date, Ticker).
    """
    if not model.models:
        raise RuntimeError("Model has no trained members. Run training.train() first.")

    raw_scores = model.predict(X)

    if not normalize:
        return raw_scores

    # ── Cross-sectional normalization by date ─────────────────────────────────
    score_df = raw_scores.to_frame("score")

    if isinstance(X.index, pd.MultiIndex):
        if method == "rank":
            normalized = score_df["score"].groupby(level=0).rank(pct=True)
        else:  # zscore
            def _zscore(s: pd.Series) -> pd.Series:
                mu, sd = s.mean(), s.std()
                return (s - mu) / (sd + 1e-8)
            normalized = score_df["score"].groupby(level=0).transform(_zscore)
        return normalized.rename("alpha_score")

    return raw_scores


def predict_at_date(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    pred_date: pd.Timestamp,
    feature_cols: Optional[List[str]] = None,
) -> pd.Series:
    """
    Generates alpha scores for all tickers on a specific prediction date.

    Args:
        model:        Trained EnsembleAlphaModel.
        X:            Full feature panel with (Date, Ticker) MultiIndex.
        pred_date:    The date to generate predictions for.
        feature_cols: Optional subset of features to use.

    Returns:
        Series of alpha scores indexed by Ticker.
    """
    try:
        X_date = X.loc[pred_date]
    except KeyError:
        logger.warning(f"[Prediction] Date {pred_date} not found in feature panel.")
        return pd.Series(dtype=float)

    if feature_cols is not None:
        X_date = X_date[[c for c in feature_cols if c in X_date.columns]]

    raw = model.predict(X_date)
    return raw.rank(pct=True).rename("alpha_score")


def build_score_panel(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    pred_dates: Optional[List[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Constructs a (Date × Ticker) score panel for a list of prediction dates.

    Args:
        model:       Trained EnsembleAlphaModel.
        X:           Full feature panel with (Date, Ticker) MultiIndex.
        pred_dates:  Dates to generate predictions. Defaults to all unique dates.

    Returns:
        DataFrame with Date as index and Ticker as columns (wide format).
    """
    if pred_dates is None:
        pred_dates = X.index.get_level_values(0).unique().tolist()

    rows = {}
    for d in pred_dates:
        scores = predict_at_date(model, X, d)
        if not scores.empty:
            rows[d] = scores

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).T.rename_axis("Date")


def detect_drift(
    model: EnsembleAlphaModel,
    X_train_sample: pd.DataFrame,
    X_pred: pd.DataFrame,
    threshold: float = 0.15,
) -> dict:
    """
    Detects feature distribution drift between training and prediction data.

    Returns a drift report dict with per-feature z-score shifts.
    """
    features = model.models[0].features if model.models else []
    if not features:
        return {"drift_detected": False, "mean_shift": 0.0, "per_feature": {}}

    feat_cols = [f for f in features if f in X_train_sample.columns and f in X_pred.columns]
    train_mu = X_train_sample[feat_cols].mean()
    train_sd = X_train_sample[feat_cols].std().replace(0, np.nan)
    pred_mu  = X_pred[feat_cols].mean()

    z_shifts = ((pred_mu - train_mu) / train_sd).abs()
    mean_shift = float(z_shifts.mean())
    drift_detected = mean_shift > threshold

    if drift_detected:
        top_drifted = z_shifts.nlargest(5).to_dict()
        logger.warning(
            f"[Drift] Feature distribution shift={mean_shift:.3f} > {threshold}. "
            f"Top features: {top_drifted}"
        )

    return {
        "drift_detected": drift_detected,
        "mean_shift": round(mean_shift, 4),
        "per_feature": z_shifts.to_dict(),
    }
