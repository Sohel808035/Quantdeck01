"""
ml_layer/explainability.py
────────────────────────────
ML Pipeline: SHAP Explainability Module (v1.0.0)

Provides institutional-grade model explainability using SHAP with:
  - TreeExplainer for XGBoost ensemble members
  - Aggregated SHAP values across ensemble
  - Global feature importance (mean |SHAP|)
  - Local explanation for individual predictions
  - SHAP waterfall and summary plot generation
  - Feature interaction detection via SHAP interaction values
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> Optional[np.ndarray]:
    """
    Computes SHAP values averaged across all ensemble members.

    Args:
        model:       Trained EnsembleAlphaModel.
        X:           Feature DataFrame (can be full panel or a subset).
        max_samples: Maximum number of rows to compute SHAP for (speed cap).

    Returns:
        NumPy array of SHAP values shape (n_samples, n_features),
        or None if SHAP is not installed.
    """
    try:
        import shap
    except ImportError:
        logger.warning("[SHAP] shap library not installed. Run: pip install shap")
        return None

    if not model.models:
        raise RuntimeError("No trained models found. Run training.train() first.")

    features = model.models[0].features
    X_shap = X[features].dropna().head(max_samples)

    all_shap = []
    for m in model.models:
        if m.model is None:
            continue
        try:
            explainer = shap.TreeExplainer(m.model)
            sv = explainer.shap_values(X_shap)
            all_shap.append(sv)
        except Exception as e:
            logger.warning(f"[SHAP] Member explainer failed: {e}")

    if not all_shap:
        return None

    mean_shap = np.mean(all_shap, axis=0)
    return mean_shap


def global_shap_importance(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> pd.DataFrame:
    """
    Computes global SHAP feature importance (mean |SHAP value| per feature).

    Args:
        model:       Trained EnsembleAlphaModel.
        X:           Feature DataFrame.
        max_samples: Max rows for SHAP computation.

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank.
        Sorted by mean_abs_shap descending.
    """
    shap_values = compute_shap_values(model, X, max_samples=max_samples)
    if shap_values is None:
        return pd.DataFrame()

    features = model.models[0].features
    X_shap = X[features].dropna().head(max_samples)

    mean_abs = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame({
        "feature":       features,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result


def local_explanation(
    model: EnsembleAlphaModel,
    X_row: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Generates a local SHAP explanation for a single row/prediction.

    Args:
        model:  Trained EnsembleAlphaModel.
        X_row:  Single-row DataFrame to explain.

    Returns:
        Series of feature → SHAP contribution, sorted by absolute value.
        Returns None if SHAP is not available.
    """
    shap_values = compute_shap_values(model, X_row, max_samples=1)
    if shap_values is None:
        return None

    features = model.models[0].features
    if shap_values.ndim == 2:
        shap_row = shap_values[0]
    else:
        shap_row = shap_values

    contributions = pd.Series(shap_row, index=features, name="shap_value")
    return contributions.reindex(contributions.abs().sort_values(ascending=False).index)


def shap_summary_plot(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    max_samples: int = 500,
    save_path: Optional[str] = None,
) -> None:
    """
    Generates a SHAP beeswarm summary plot showing feature impact distribution.

    Args:
        model:       Trained EnsembleAlphaModel.
        X:           Feature DataFrame.
        max_samples: Max rows for SHAP computation.
        save_path:   If provided, saves figure to this path (PNG).
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[SHAP] shap or matplotlib not installed.")
        return

    features = model.models[0].features
    X_shap = X[features].dropna().head(max_samples)
    shap_values = compute_shap_values(model, X_shap, max_samples=max_samples)
    if shap_values is None:
        return

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Feature Contributions — QuantSphereX Ensemble")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[SHAP] Summary plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def shap_interaction_scores(
    model: EnsembleAlphaModel,
    X: pd.DataFrame,
    top_n: int = 10,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    Computes SHAP pairwise interaction scores (mean |SHAP interaction|) for top features.

    Args:
        model:       Trained EnsembleAlphaModel (first member only for speed).
        X:           Feature DataFrame.
        top_n:       Number of top features to include.
        max_samples: Max rows to evaluate.

    Returns:
        Square DataFrame (feature × feature) of mean |SHAP interaction values|.
    """
    try:
        import shap
    except ImportError:
        logger.warning("[SHAP] shap not installed.")
        return pd.DataFrame()

    if not model.models or model.models[0].model is None:
        return pd.DataFrame()

    features = model.models[0].features[:top_n]
    X_shap = X[features].dropna().head(max_samples)

    try:
        explainer = shap.TreeExplainer(model.models[0].model)
        interaction_vals = explainer.shap_interaction_values(X_shap)
        mean_inter = np.abs(interaction_vals).mean(axis=0)
        return pd.DataFrame(mean_inter, index=features, columns=features)
    except Exception as e:
        logger.warning(f"[SHAP] Interaction computation failed: {e}")
        return pd.DataFrame()
