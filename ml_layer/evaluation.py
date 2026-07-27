"""
ml_layer/evaluation.py
────────────────────────
ML Pipeline: Evaluation Module (v1.0.0)

Wraps and extends alpha_layer/pure_alpha_validator.py with:
  - Structured EvalResult DTO
  - IC series, t-stat, % positive, decile spread
  - Sub-period breakdown (pre/post 2008, COVID period)
  - Institutional quality gate flags
  - Comparison table across multiple runs
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from alpha_layer.pure_alpha_validator import evaluate_pure_alpha, compute_daily_ic
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Structured evaluation result DTO."""
    experiment_name: str
    ic_mean: float
    ic_std: float
    ic_tstat: float
    pct_positive: float
    ic_series: pd.Series
    rolling_6m_ic: pd.Series
    sub_period_ic: Dict[str, float]
    decile_sharpe: float
    decile_maxdd: float
    weak_alpha: bool
    institutional_grade: bool  # True if IC > 0.04 and t-stat > 2.5
    notes: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "ic_mean":    round(self.ic_mean, 4),
            "ic_tstat":   round(self.ic_tstat, 2),
            "pct_pos":    round(self.pct_positive, 3),
            "decile_sharpe": round(self.decile_sharpe, 2),
            "decile_maxdd":  round(self.decile_maxdd, 4),
            "weak_alpha":    self.weak_alpha,
            "inst_grade":    self.institutional_grade,
        }


def evaluate(
    scores_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    config: Optional[MLConfig] = None,
    forward_days: int = 21,
) -> EvalResult:
    """
    Runs the full pure alpha evaluation suite on model predictions.

    Wraps evaluate_pure_alpha() from alpha_layer with a structured DTO output.

    Args:
        scores_df:    Wide (Date × Ticker) DataFrame of alpha scores.
        stock_panel:  Full OHLCV panel with (Date, Ticker) MultiIndex.
        config:       MLConfig for experiment labeling.
        forward_days: Horizon for IC computation.

    Returns:
        EvalResult DTO with all evaluation metrics.
    """
    if config is None:
        config = MLConfig()

    raw = evaluate_pure_alpha(
        scores_df=scores_df,
        stock_panel=stock_panel,
        transaction_cost=0.0015,
    )

    ic_mean  = raw.get("ic_mean", 0.0)
    ic_tstat = raw.get("ic_tstat", 0.0)
    institutional_grade = (ic_mean > 0.04) and (ic_tstat > 2.5)

    return EvalResult(
        experiment_name=config.experiment_name,
        ic_mean=ic_mean,
        ic_std=raw.get("ic_std", 0.0),
        ic_tstat=ic_tstat,
        pct_positive=raw.get("pct_positive", 0.0),
        ic_series=raw.get("ic_series", pd.Series(dtype=float)),
        rolling_6m_ic=raw.get("rolling_6m_ic", pd.Series(dtype=float)),
        sub_period_ic=raw.get("sub_period_ic", {}),
        decile_sharpe=raw.get("decile_sharpe", 0.0),
        decile_maxdd=raw.get("decile_maxdd", 0.0),
        weak_alpha=raw.get("weak_alpha", True),
        institutional_grade=institutional_grade,
    )


def compare_experiments(results: List[EvalResult]) -> pd.DataFrame:
    """
    Generates a comparison table across multiple EvalResult objects.

    Args:
        results: List of EvalResult from multiple runs / experiments.

    Returns:
        DataFrame with one row per experiment and key metrics as columns.
    """
    rows = [r.summary() for r in results]
    df = pd.DataFrame(rows).set_index("experiment")
    df = df.sort_values("ic_tstat", ascending=False)
    return df


def ic_decay_analysis(
    model,
    X: pd.DataFrame,
    stock_panel: pd.DataFrame,
    horizons: List[int] = None,
) -> pd.DataFrame:
    """
    Measures IC at multiple forward horizons to detect signal decay.
    Useful for choosing optimal label horizon.

    Args:
        model:       Trained EnsembleAlphaModel.
        X:           Feature panel (Date, Ticker) MultiIndex.
        stock_panel: OHLCV panel for forward return computation.
        horizons:    List of horizons to test.

    Returns:
        DataFrame with horizon → IC mean mapping.
    """
    if horizons is None:
        horizons = [5, 10, 21, 42, 63]

    from ml_layer.prediction import predict, build_score_panel
    scores_panel = build_score_panel(model, X)

    rows = []
    for H in horizons:
        ic_s = compute_daily_ic(scores_panel, stock_panel, forward_days=H)
        rows.append({
            "horizon_days": H,
            "ic_mean": round(float(ic_s.mean()), 5) if not ic_s.empty else np.nan,
            "ic_std": round(float(ic_s.std()), 5) if not ic_s.empty else np.nan,
        })

    return pd.DataFrame(rows).set_index("horizon_days")
