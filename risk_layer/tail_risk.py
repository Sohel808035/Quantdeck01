"""
risk_layer/tail_risk.py
────────────────────────
Tail Risk & Extreme Value Theory (EVT) Engine.
Computes Extreme Value Theory (EVT) Peak-Over-Threshold (POT) tail index ξ, skewness, and kurtosis.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, genpareto

logger = logging.getLogger(__name__)


class TailRiskEngine:
    """Evaluates non-Gaussian left-tail risk, EVT parameters, and extreme loss probability."""

    def __init__(self, evt_quantile: float = 0.05):
        self.evt_quantile = evt_quantile

    def compute_tail_metrics(
        self,
        returns_series: pd.Series,
        evt_quantile: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes skewness, kurtosis, Expected Tail Loss (ETL), and EVT POT tail index.
        """
        if evt_quantile is None:
            evt_quantile = self.evt_quantile

        s = returns_series.dropna()
        if len(s) < 30:
            return {"skewness": 0.0, "kurtosis": 0.0, "etl_95": 0.0, "evt_tail_index": 0.0}

        sk = float(skew(s))
        kt = float(kurtosis(s))  # Excess kurtosis

        # 5% Left-Tail Expected Loss
        cutoff = np.percentile(s, evt_quantile * 100)
        tail = s[s <= cutoff]
        etl = -float(tail.mean()) if len(tail) > 0 else -cutoff

        # Extreme Value Theory (EVT) Peak-Over-Threshold (POT) fit via Generalized Pareto
        losses = -s[s < 0]
        if len(losses) < 10:
            return {"skewness": round(sk, 4), "kurtosis": round(kt, 4), "etl_95": round(max(0.0, etl), 4), "evt_tail_index": 0.0}

        pot_threshold = np.percentile(losses, (1.0 - evt_quantile) * 100)
        exceedances = losses[losses > pot_threshold] - pot_threshold

        if len(exceedances) > 10:
            try:
                c, loc, scale = genpareto.fit(exceedances, floc=0)
                tail_index = float(c)  # Shape parameter \xi (xi > 0 = heavy tail)
            except Exception:
                tail_index = 0.0
        else:
            tail_index = 0.0

        return {
            "skewness": round(sk, 4),
            "kurtosis": round(kt, 4),
            "etl_95": round(max(0.0, etl), 4),
            "evt_tail_index": round(tail_index, 4),
        }
