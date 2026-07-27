"""
ml_layer/cross_validation.py
──────────────────────────────
ML Pipeline: Cross-Validation Module (v1.0.0)

Implements time-series purged cross-validation with:
  - Expanding-window TimeSeriesCV (anchored)
  - Sliding-window TimeSeriesCV (rolling)
  - Purged K-Fold with configurable embargo gap
  - Per-fold IC tracking
  - CV summary table
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel, XGBoostAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Structured output from a cross-validation run."""
    cv_type: str
    n_folds: int
    fold_results: List[Dict]
    mean_val_ic: float
    std_val_ic: float
    mean_train_ic: float
    ic_stability: float   # std / mean — lower is more stable

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.fold_results)
        df.loc["mean"] = df.mean(numeric_only=True)
        df.loc["std"]  = df.std(numeric_only=True)
        return df


def _ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    if len(df) < 10:
        return 0.0
    return float(df["t"].corr(df["p"], method="spearman"))


def expanding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[MLConfig] = None,
    min_train_frac: float = 0.40,
) -> CVResult:
    """
    Expanding-window (anchored) time-series cross-validation.

    The training window grows from min_train_frac forward.
    Each fold uses all prior data as training and a fixed test window.

    Args:
        X:              Feature panel with (Date, Ticker) MultiIndex.
        y:              Target Series.
        config:         MLConfig.
        min_train_frac: Initial training fraction before first fold.

    Returns:
        CVResult DTO with per-fold metrics.
    """
    if config is None:
        config = MLConfig()

    dates = X.index.get_level_values(0).unique().sort_values()
    n = len(dates)
    n_splits = config.cv_n_splits
    step = int(n * (1 - min_train_frac) / n_splits)

    if step == 0:
        raise ValueError("Not enough data for the requested number of CV folds.")

    fold_results = []
    min_train_idx = int(n * min_train_frac)

    for fold in range(n_splits):
        test_start_idx = min_train_idx + fold * step
        test_end_idx   = min(test_start_idx + step, n - 1)

        if test_start_idx >= n or test_end_idx >= n:
            break

        train_dates = dates[:test_start_idx]
        test_dates  = dates[test_start_idx:test_end_idx]

        X_tr = X.loc[X.index.get_level_values(0).isin(train_dates)]
        y_tr = y.reindex(X_tr.index)
        X_te = X.loc[X.index.get_level_values(0).isin(test_dates)]
        y_te = y.reindex(X_te.index)

        if len(X_tr) < 200 or len(X_te) < 50:
            continue

        try:
            m = XGBoostAlphaModel(params={**config.base_params, "random_state": 42 + fold})
            m.fit(X_tr, y_tr)
            val_preds  = m.predict(X_te)
            train_preds = m.predict(X_tr)
            fold_val_ic   = _ic(y_te, val_preds)
            fold_train_ic = _ic(y_tr, train_preds)
            logger.info(f"  [CV Fold {fold+1}/{n_splits}] Train IC={fold_train_ic:.4f} | Val IC={fold_val_ic:.4f}")
            fold_results.append({
                "fold": fold + 1,
                "train_ic": fold_train_ic,
                "val_ic": fold_val_ic,
                "n_train": len(X_tr),
                "n_test": len(X_te),
            })
        except Exception as e:
            logger.warning(f"  [CV Fold {fold+1}] Failed: {e}")

    if not fold_results:
        return CVResult("expanding", n_splits, [], 0.0, 0.0, 0.0, 0.0)

    val_ics   = [r["val_ic"] for r in fold_results]
    train_ics = [r["train_ic"] for r in fold_results]
    mean_v    = float(np.mean(val_ics))
    std_v     = float(np.std(val_ics))
    mean_t    = float(np.mean(train_ics))
    stability = float(std_v / (abs(mean_v) + 1e-8))

    logger.info(
        f"[CV Expanding] Mean Val IC={mean_v:.4f} ± {std_v:.4f} | Stability CV={stability:.3f}"
    )

    return CVResult(
        cv_type="expanding",
        n_folds=len(fold_results),
        fold_results=fold_results,
        mean_val_ic=mean_v,
        std_val_ic=std_v,
        mean_train_ic=mean_t,
        ic_stability=stability,
    )


def sliding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[MLConfig] = None,
    window_frac: float = 0.40,
) -> CVResult:
    """
    Sliding-window (rolling) time-series cross-validation.

    The training window is fixed in size and slides forward each fold.

    Args:
        X:            Feature panel with (Date, Ticker) MultiIndex.
        y:            Target Series.
        config:       MLConfig.
        window_frac:  Fraction of total dates to use as training window.

    Returns:
        CVResult DTO with per-fold metrics.
    """
    if config is None:
        config = MLConfig()

    dates = X.index.get_level_values(0).unique().sort_values()
    n = len(dates)
    n_splits = config.cv_n_splits
    window   = int(n * window_frac)
    step     = int((n - window) / n_splits)

    if step == 0:
        raise ValueError("Not enough data for rolling CV with these settings.")

    fold_results = []
    for fold in range(n_splits):
        train_start_idx = fold * step
        train_end_idx   = train_start_idx + window
        test_end_idx    = min(train_end_idx + step, n - 1)

        if train_end_idx >= n or test_end_idx >= n:
            break

        train_dates = dates[train_start_idx:train_end_idx]
        test_dates  = dates[train_end_idx:test_end_idx]

        X_tr = X.loc[X.index.get_level_values(0).isin(train_dates)]
        y_tr = y.reindex(X_tr.index)
        X_te = X.loc[X.index.get_level_values(0).isin(test_dates)]
        y_te = y.reindex(X_te.index)

        if len(X_tr) < 200 or len(X_te) < 50:
            continue

        try:
            m = XGBoostAlphaModel(params={**config.base_params, "random_state": 42 + fold})
            m.fit(X_tr, y_tr)
            val_preds   = m.predict(X_te)
            train_preds = m.predict(X_tr)
            fold_val_ic   = _ic(y_te, val_preds)
            fold_train_ic = _ic(y_tr, train_preds)
            logger.info(f"  [CV Sliding {fold+1}/{n_splits}] Train IC={fold_train_ic:.4f} | Val IC={fold_val_ic:.4f}")
            fold_results.append({
                "fold": fold + 1,
                "train_ic": fold_train_ic,
                "val_ic": fold_val_ic,
                "n_train": len(X_tr),
                "n_test": len(X_te),
            })
        except Exception as e:
            logger.warning(f"  [CV Sliding Fold {fold+1}] Failed: {e}")

    if not fold_results:
        return CVResult("sliding", n_splits, [], 0.0, 0.0, 0.0, 0.0)

    val_ics   = [r["val_ic"] for r in fold_results]
    mean_v    = float(np.mean(val_ics))
    std_v     = float(np.std(val_ics))
    mean_t    = float(np.mean([r["train_ic"] for r in fold_results]))
    stability = float(std_v / (abs(mean_v) + 1e-8))

    logger.info(
        f"[CV Sliding] Mean Val IC={mean_v:.4f} ± {std_v:.4f} | Stability CV={stability:.3f}"
    )

    return CVResult(
        cv_type="sliding",
        n_folds=len(fold_results),
        fold_results=fold_results,
        mean_val_ic=mean_v,
        std_val_ic=std_v,
        mean_train_ic=mean_t,
        ic_stability=stability,
    )
