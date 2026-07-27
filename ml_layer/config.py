"""
ml_layer/config.py
───────────────────
Configuration DTO for the QuantSphereX ML Pipeline.
Controls training, evaluation, hyperparameter tuning,
experiment tracking, and SHAP explainability settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class MLConfig:
    """
    Centralized configuration for the ML Pipeline.

    Args:
        experiment_name:     Human-readable label for the experiment run.
        target_col:          Name of the label column to predict.
        n_ensemble:          Number of XGBoost ensemble members.
        val_split:           Time-based validation fraction.
        cv_n_splits:         Number of time-series cross-validation folds.
        tune_n_trials:       Optuna hyperparameter tuning trials (0 = disabled).
        shap_max_samples:    Max training rows fed to SHAP explainer (capped for speed).
        confidence_quantiles: Quantiles for prediction interval estimation.
        registry_dir:        Path to persist model artifacts.
        tracking_dir:        Path to persist experiment run logs.
        overfit_threshold:   Train IC minus Val IC limit before self-healing triggers.
    """
    experiment_name: str = "quantspherex_xgb_v1"
    target_col: str = "label_rank_pct_fwd_21d"
    n_ensemble: int = 5
    val_split: float = 0.20
    cv_n_splits: int = 5

    # Hyperparameter tuning
    tune_n_trials: int = 0               # 0 = skip tuning; set >0 to enable Optuna

    # SHAP
    shap_max_samples: int = 2000

    # Confidence estimation
    confidence_quantiles: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90]
    )

    # Persistence
    registry_dir: str = "ml_layer/registry"
    tracking_dir: str = "ml_layer/experiments"

    # Model hyperparameters (base)
    base_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective":        "reg:squarederror",
        "max_depth":        4,
        "learning_rate":    0.04,
        "n_estimators":     400,
        "min_child_weight": 40,
        "gamma":            4.0,
        "subsample":        0.75,
        "colsample_bytree": 0.75,
        "reg_alpha":        1.2,
        "reg_lambda":       4.0,
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    })

    overfit_threshold: float = 0.05
