"""
ml_layer/__init__.py
──────────────────────
QuantSphereX ML Pipeline Package (v1.0.0)

Separated modules:
  - training            : Model training with time-split and self-healing
  - prediction          : Score generation and drift detection
  - evaluation          : IC metrics, decile spread, institutional quality gate
  - feature_importance  : XGBoost gain/weight/cover importances + stability
  - registry            : File-based model versioning and champion promotion
  - experiment_tracker  : Lightweight run logging and comparison
  - cross_validation    : Expanding and sliding window time-series CV
  - hyperparameter_tuning: Optuna Bayesian optimization + grid search fallback
  - confidence          : Ensemble variance, tiers, conformal intervals
  - explainability      : SHAP TreeExplainer for global and local insights

Backward-compatible with alpha_layer (EnsembleAlphaModel unchanged).
"""

from ml_layer.config import MLConfig
from ml_layer import (
    training,
    prediction,
    evaluation,
    feature_importance,
    registry,
    experiment_tracker,
    cross_validation,
    hyperparameter_tuning,
    confidence,
    explainability,
)
from ml_layer.training import TrainResult, train
from ml_layer.evaluation import EvalResult, evaluate, compare_experiments
from ml_layer.registry import ModelRegistry, RegistryEntry
from ml_layer.experiment_tracker import ExperimentTracker, RunRecord
from ml_layer.cross_validation import CVResult, expanding_window_cv, sliding_window_cv
from ml_layer.confidence import full_confidence_report

VERSION = "1.0.0"

__all__ = [
    "MLConfig",
    "VERSION",
    # Modules
    "training", "prediction", "evaluation", "feature_importance",
    "registry", "experiment_tracker", "cross_validation",
    "hyperparameter_tuning", "confidence", "explainability",
    # Classes & functions
    "TrainResult", "train",
    "EvalResult", "evaluate", "compare_experiments",
    "ModelRegistry", "RegistryEntry",
    "ExperimentTracker", "RunRecord",
    "CVResult", "expanding_window_cv", "sliding_window_cv",
    "full_confidence_report",
]
