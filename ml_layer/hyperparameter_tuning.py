"""
ml_layer/hyperparameter_tuning.py
───────────────────────────────────
ML Pipeline: Hyperparameter Tuning Module (v1.0.0)

Optuna-based Bayesian hyperparameter optimization with:
  - IC-maximizing objective (Spearman correlation on validation fold)
  - XGBoost search space covering depth, regularization, and sampling
  - Pruning of unpromising trials via MedianPruner
  - Best-params extraction and logging
  - Falls back gracefully when Optuna is not installed
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import XGBoostAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


def _ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    if len(df) < 10:
        return 0.0
    return float(df["t"].corr(df["p"], method="spearman"))


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[MLConfig] = None,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs Bayesian hyperparameter optimization using Optuna.

    If Optuna is not installed, falls back to a fast grid search over
    5 predefined configurations, returning the best-performing params.

    Args:
        X:        Feature panel with (Date, Ticker) MultiIndex.
        y:        Target Series.
        config:   MLConfig (reads tune_n_trials if n_trials not provided).
        n_trials: Override number of Optuna trials.

    Returns:
        Dict of best hyperparameters ready for injection into training.
    """
    if config is None:
        config = MLConfig()

    n_trials = n_trials if n_trials is not None else config.tune_n_trials

    if n_trials == 0:
        logger.info("[Tuner] Hyperparameter tuning skipped (tune_n_trials=0).")
        return config.base_params

    # ── Time-based val split ──────────────────────────────────────────────────
    dates = X.index.get_level_values(0).unique().sort_values()
    split_idx = int(len(dates) * 0.80)
    cutoff = dates[split_idx]

    X_tr = X.loc[:cutoff - pd.Timedelta(days=1)]
    X_vl = X.loc[cutoff:]
    y_tr = y.reindex(X_tr.index)
    y_vl = y.reindex(X_vl.index)

    if X_tr.empty or X_vl.empty:
        logger.warning("[Tuner] Cannot tune: empty train or val split.")
        return config.base_params

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _optuna_tune(X_tr, y_tr, X_vl, y_vl, config, n_trials)
    except ImportError:
        logger.warning("[Tuner] Optuna not installed. Falling back to grid search.")
        return _grid_search(X_tr, y_tr, X_vl, y_vl, config)


def _optuna_tune(
    X_tr, y_tr, X_vl, y_vl,
    config: MLConfig,
    n_trials: int,
) -> Dict[str, Any]:
    """Optuna Bayesian optimization with IC objective."""
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective":        "reg:squarederror",
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=100),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 80),
            "gamma":            trial.suggest_float("gamma", 1.0, 20.0),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.95),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "tree_method":      "hist",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        try:
            m = XGBoostAlphaModel(params=params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_vl)
            return _ic(y_vl, preds)
        except Exception:
            return -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    logger.info(
        f"[Tuner] Best Val IC={study.best_value:.4f} after {n_trials} trials. "
        f"Best params: {best}"
    )
    return best


def _grid_search(
    X_tr, y_tr, X_vl, y_vl,
    config: MLConfig,
) -> Dict[str, Any]:
    """Fast 5-point grid search fallback when Optuna is unavailable."""
    candidates = [
        {**config.base_params, "max_depth": 2, "gamma": 8.0, "n_estimators": 200},
        {**config.base_params, "max_depth": 3, "gamma": 6.0, "n_estimators": 300},
        {**config.base_params, "max_depth": 4, "gamma": 4.0, "n_estimators": 400},
        {**config.base_params, "max_depth": 4, "learning_rate": 0.02, "n_estimators": 600},
        {**config.base_params, "max_depth": 5, "reg_lambda": 8.0, "n_estimators": 300},
    ]

    best_ic = -999.0
    best_params = config.base_params

    for i, params in enumerate(candidates):
        try:
            m = XGBoostAlphaModel(params=params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_vl)
            ic = _ic(y_vl, preds)
            logger.info(f"  [Grid Search {i+1}/5] Val IC={ic:.4f}")
            if ic > best_ic:
                best_ic = ic
                best_params = params
        except Exception as e:
            logger.warning(f"  [Grid Search {i+1}] Failed: {e}")

    logger.info(f"[Grid Search] Best Val IC={best_ic:.4f}")
    return best_params
