"""
ml_layer/feature_importance.py
────────────────────────────────
ML Pipeline: Feature Importance Module (v1.0.0)

Computes and reports feature importances from trained XGBoost ensemble with:
  - XGBoost native gain/weight/cover importances
  - Cross-member average and std across ensemble
  - Top-N feature selection helper
  - Importance stability analysis (variance across ensemble members)
"""

from __future__ import annotations
import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel

logger = logging.getLogger(__name__)


def get_importance(
    model: EnsembleAlphaModel,
    importance_type: str = "gain",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extracts feature importances averaged across all ensemble members.

    Args:
        model:           Trained EnsembleAlphaModel.
        importance_type: XGBoost importance type: 'gain', 'weight', 'cover'.
        top_n:           If set, returns only the top N features.

    Returns:
        DataFrame with columns: feature, mean_importance, std_importance, rank.
    """
    if not model.models:
        raise RuntimeError("No trained models. Run training.train() first.")

    all_importances = []
    for m in model.models:
        if m.model is None:
            continue
        imp = m.model.get_booster().get_score(importance_type=importance_type)
        all_importances.append(pd.Series(imp))

    if not all_importances:
        return pd.DataFrame()

    imp_df = pd.concat(all_importances, axis=1).fillna(0)
    result = pd.DataFrame({
        "feature":          imp_df.index,
        "mean_importance":  imp_df.mean(axis=1).values,
        "std_importance":   imp_df.std(axis=1).fillna(0).values,
    })
    result = result.sort_values("mean_importance", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    if top_n is not None:
        result = result.head(top_n)

    return result


def get_top_features(
    model: EnsembleAlphaModel,
    n: int = 20,
    importance_type: str = "gain",
) -> List[str]:
    """
    Returns the top N feature names by average gain importance.

    Args:
        model:           Trained EnsembleAlphaModel.
        n:               Number of top features to return.
        importance_type: XGBoost importance type.

    Returns:
        Ordered list of feature column names.
    """
    imp_df = get_importance(model, importance_type=importance_type, top_n=n)
    return imp_df["feature"].tolist()


def importance_stability(model: EnsembleAlphaModel) -> pd.DataFrame:
    """
    Computes stability of feature importances across ensemble members.
    High std/mean ratio indicates an unstable, noisy feature.

    Returns:
        DataFrame with feature, cv (coeff. of variation), and stability label.
    """
    imp_df = get_importance(model, importance_type="gain")
    if imp_df.empty:
        return imp_df

    imp_df["cv"] = imp_df["std_importance"] / (imp_df["mean_importance"] + 1e-8)
    imp_df["stability"] = pd.cut(
        imp_df["cv"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["stable", "moderate", "unstable"],
    )
    return imp_df[["feature", "mean_importance", "std_importance", "cv", "stability", "rank"]]


def plot_importance(
    model: EnsembleAlphaModel,
    top_n: int = 20,
    importance_type: str = "gain",
    save_path: Optional[str] = None,
) -> None:
    """
    Generates a horizontal bar chart of feature importances.

    Args:
        model:           Trained EnsembleAlphaModel.
        top_n:           Number of features to display.
        importance_type: XGBoost importance type.
        save_path:       If provided, saves figure to this path (PNG).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Skipping importance plot.")
        return

    imp_df = get_importance(model, importance_type=importance_type, top_n=top_n)
    if imp_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4))
    ax.barh(
        imp_df["feature"][::-1],
        imp_df["mean_importance"][::-1],
        xerr=imp_df["std_importance"][::-1],
        color="#2196F3",
        alpha=0.85,
        capsize=3,
    )
    ax.set_xlabel(f"XGBoost {importance_type.capitalize()} Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Ensemble Average ± Std)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
