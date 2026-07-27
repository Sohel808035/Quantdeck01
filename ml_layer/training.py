"""
ml_layer/training.py
──────────────────────
ML Pipeline: Training Module (v1.0.0)

Wraps the existing EnsembleAlphaModel with:
  - Clean train/val time-split
  - Self-healing overfitting loop (preserved from xgboost_trainer.py)
  - Structured TrainResult DTO
  - Optional hyperparameter injection from tuner

Backward-compatible: imports and delegates to alpha_layer.xgboost_trainer.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel, XGBoostAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Structured output DTO from a single training run."""
    experiment_name: str
    model: EnsembleAlphaModel
    features: List[str]
    train_ic: float
    val_ic: float
    overfit_score: float
    n_train_rows: int
    n_val_rows: int
    elapsed_seconds: float
    params_used: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def train(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[MLConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    """
    Trains an EnsembleAlphaModel on (X, y) using a time-based val split.

    Preserves the existing self-healing overfitting control from
    alpha_layer/xgboost_trainer.py without modification.

    Args:
        X:             Feature DataFrame with (Date, Ticker) MultiIndex.
        y:             Target Series aligned to X.
        config:        MLConfig (defaults applied if not provided).
        custom_params: Override base XGBoost hyperparameters (from tuner).

    Returns:
        TrainResult DTO with model, metrics, and metadata.
    """
    if config is None:
        config = MLConfig()

    params = {**config.base_params, **(custom_params or {})}
    t0 = time.perf_counter()

    # ── Time-based train/val split ────────────────────────────────────────────
    dates = X.index.get_level_values(0).unique().sort_values()
    split_idx = int(len(dates) * (1 - config.val_split))
    split_idx = max(1, min(split_idx, len(dates) - 1))
    cutoff = dates[split_idx]

    X_tr = X.loc[:cutoff - pd.Timedelta(days=1)]
    X_vl = X.loc[cutoff:]
    y_tr = y.loc[:cutoff - pd.Timedelta(days=1)]
    y_vl = y.loc[cutoff:]

    n_train = len(X_tr)
    n_val   = len(X_vl)

    if X_tr.empty:
        raise ValueError("Training set is empty after time split. Check date range and val_split.")

    logger.info(
        f"[Training] '{config.experiment_name}' | "
        f"train={n_train:,} rows | val={n_val:,} rows | "
        f"n_ensemble={config.n_ensemble}"
    )

    # ── Build & train ensemble ────────────────────────────────────────────────
    model = EnsembleAlphaModel(
        n_models=config.n_ensemble,
        params=params,
    )

    # Feed pre-split data directly (override internal split in EnsembleAlphaModel)
    model.models = []
    train_ics, val_ics, overfit_scores = [], [], []

    for i in range(config.n_ensemble):
        seed_params = {**params, "random_state": 42 + i}
        m = XGBoostAlphaModel(params=seed_params)
        res = m.fit(
            X_tr, y_tr,
            X_val=X_vl if not X_vl.empty else None,
            y_val=y_vl if not X_vl.empty else None,
        )
        model.models.append(m)
        train_ics.append(res.get("train_ic", 0.0))
        val_ics.append(res.get("val_ic", 0.0))
        overfit_scores.append(res.get("overfit_score", 0.0))

    avg_train    = float(np.mean(train_ics))
    avg_val      = float(np.mean(val_ics))
    avg_overfit  = float(np.mean(overfit_scores))
    elapsed      = time.perf_counter() - t0

    logger.info(
        f"[Training] Done | Train IC={avg_train:.4f} | "
        f"Val IC={avg_val:.4f} | Overfit={avg_overfit:.4f} | {elapsed:.1f}s"
    )

    features = model.models[0].features if model.models else []

    return TrainResult(
        experiment_name=config.experiment_name,
        model=model,
        features=features,
        train_ic=avg_train,
        val_ic=avg_val,
        overfit_score=avg_overfit,
        n_train_rows=n_train,
        n_val_rows=n_val,
        elapsed_seconds=elapsed,
        params_used=params,
    )
