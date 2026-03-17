"""
alpha_layer/xgboost_trainer.py  (CQRO v4 — Clean, No Retry Loop)
══════════════════════════════════════════════════════════════════
XGBoost ensemble alpha model.

Fixes:
  - Removed infinite retry loop: regularization is applied ONCE if overfit > threshold
  - Val IC < 0 does NOT trigger retrain (it's expected in early windows; 
    the corrective action is simply to use a shallower tree via params)
  - Model is trained once per window per ensemble member, cleanly and fast
"""

from __future__ import annotations
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

# ── Base hyperparameters (deterministic, conservative) ────────────────────────
BASE_PARAMS: Dict[str, Any] = {
    "objective":        "reg:squarederror",
    "max_depth":        3,            # shallower trees for institutional robustness
    "learning_rate":    0.03,         # slower, more stable learning
    "n_estimators":     300,          # more estimators with lower learning rate
    "min_child_weight": 50,           # significantly higher to prevent outlier-driven leaves
    "gamma":            5.0,
    "subsample":        0.7,          # more stochasticity
    "colsample_bytree": 0.7,
    "reg_alpha":        1.5,
    "reg_lambda":       5.0,
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# ── Tighter params applied ONCE if overfitting score > threshold ──────────────
REGULARIZED_PARAMS: Dict[str, Any] = {
    **BASE_PARAMS,
    "max_depth":        2,
    "gamma":            10.0,
    "min_child_weight": 30,
    "n_estimators":     80,
    "tree_method":      "hist",
    "reg_alpha":        1.5,
    "reg_lambda":       5.0,
}


class XGBoostAlphaModel:
    """Single deterministic XGBoost model with one-shot overfitting control."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**BASE_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBRegressor] = None
        self.features: List[str] = []
        self.overfit_score: float = 0.0
        self.train_ic: float = 0.0
        self.val_ic: float = 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
    ) -> Dict[str, float]:
        """
        Train the model. If X_val/y_val provided:
          1. Train with base params
          2. Evaluate IC on both sets
          3. If overfit_score > 0.05, re-fit ONCE with REGULARIZED_PARAMS (not in a loop)
        Returns {train_ic, val_ic, overfit_score}.
        """
        self.features = sorted(X_train.columns.tolist())
        Xtr = X_train[self.features]

        # ── Train ─────────────────────────────────────────────────────────────
        params_to_use = self.params
        self.model = xgb.XGBRegressor(**params_to_use)
        self.model.fit(Xtr, y_train)

        if X_val is not None and y_val is not None:
            Xvl = X_val[self.features]

            train_preds = pd.Series(self.model.predict(Xtr), index=Xtr.index)
            val_preds   = pd.Series(self.model.predict(Xvl), index=Xvl.index)

            self.train_ic = self._ic(y_train, train_preds)
            self.val_ic   = self._ic(y_val,   val_preds)
            self.overfit_score = self.train_ic - self.val_ic

            logger.info(
                f"  [Overfit Audit] Train IC: {self.train_ic:.4f} | "
                f"Val IC: {self.val_ic:.4f} | Score: {self.overfit_score:.4f}"
            )

            # ── ONE corrective re-fit (no loop) ───────────────────────────────
            if self.overfit_score > 0.05:
                logger.warning(
                    f"  [Regularize] Overfit={self.overfit_score:.3f} > 0.05. "
                    f"Switching to regularized params and re-fitting once."
                )
                reg_params = {**REGULARIZED_PARAMS, "random_state": params_to_use.get("random_state", 42)}
                self.model = xgb.XGBRegressor(**reg_params)
                self.model.fit(Xtr, y_train)
                # Re-evaluate after regularization (informational only)
                val_preds2 = pd.Series(self.model.predict(Xvl), index=Xvl.index)
                self.val_ic = self._ic(y_val, val_preds2)
                logger.info(f"  [Regularize] Post-regularization Val IC: {self.val_ic:.4f}")

        return {
            "train_ic":      self.train_ic,
            "val_ic":        self.val_ic,
            "overfit_score": self.overfit_score,
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        preds = self.model.predict(X[self.features])
        return pd.Series(preds, index=X.index, name="predicted_score")

    @staticmethod
    def _ic(y_true: pd.Series, y_pred: pd.Series) -> float:
        df = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
        if len(df) < 10:
            return 0.0
        return float(df["t"].corr(df["p"], method="spearman"))


class EnsembleAlphaModel:
    """
    Ensemble of N deterministic XGBoost models for variance reduction.
    Each model uses a unique seed. Val split is done ONCE at the ensemble level
    (not inside each sub-model loop), so computation is clean and fast.
    """

    def __init__(self, n_models: int = 5, params: Optional[Dict[str, Any]] = None):
        self.n_models    = n_models
        self.base_params = params or {}
        self.models: List[XGBoostAlphaModel] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, val_split: float = 0.2) -> Dict[str, float]:
        """
        1. Split train/val once by time (no shuffling)
        2. Train all N models
        3. Report average train_ic / val_ic
        """
        # ── Time-based train/val split ─────────────────────────────────────────
        dates     = X.index.get_level_values(0).unique().sort_values()
        split_idx = int(len(dates) * (1 - val_split))

        if split_idx >= len(dates):
            split_idx = len(dates) - 1

        cutoff = dates[split_idx]
        # Strict: train ends BEFORE cutoff, val starts AT cutoff
        X_tr, X_vl = X.loc[:cutoff - pd.Timedelta(days=1)], X.loc[cutoff:]
        y_tr, y_vl = y.loc[:cutoff - pd.Timedelta(days=1)], y.loc[cutoff:]

        if X_tr.empty or X_vl.empty:
            logger.warning("  [Ensemble] Val split produced empty fold, training without validation.")
            X_vl, y_vl = None, None

        logger.info(
            f"  [Ensemble] {self.n_models} models | "
            f"Train: {len(X_tr):,} rows | Val: {len(X_vl):,} rows"
            if X_vl is not None
            else f"  [Ensemble] {self.n_models} models | Train: {len(X_tr):,} rows | No val"
        )

        self.models = []
        train_ics, val_ics = [], []
        for i in range(self.n_models):
            seed_params = {**self.base_params, "random_state": 42 + i}
            m = XGBoostAlphaModel(params=seed_params)
            res = m.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
            self.models.append(m)
            train_ics.append(res.get("train_ic", 0.0))
            val_ics.append(res.get("val_ic", 0.0))

        avg_train = float(np.mean(train_ics))
        avg_val   = float(np.mean(val_ics))
        logger.info(
            f"  [Ensemble Avg] Train IC: {avg_train:.4f} | Val IC: {avg_val:.4f}"
        )

        return {"train_ic": avg_train, "val_ic": avg_val}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            raise RuntimeError("No models trained yet.")
        preds = pd.concat([m.predict(X) for m in self.models], axis=1).mean(axis=1)
        return preds.rename("predicted_score")

    def get_version_hash(self) -> str:
        h = hashlib.md5(str([m.params for m in self.models]).encode()).hexdigest()
        return h[:8]

    def compute_drift(self, X_train_sample: pd.DataFrame, X_pred: pd.DataFrame) -> None:
        """Optional: Log feature mean shift between train and live data."""
        feats = self.models[0].features if self.models else []
        if not feats:
            return
        train_means = X_train_sample[feats].mean()
        pred_means  = X_pred.reindex(columns=feats).mean()
        drift = (pred_means - train_means).abs().mean()
        if drift > 0.15:
            logger.warning(f"  [Drift] Feature distribution shift = {drift:.3f} (threshold: 0.15)")
