"""
risk_layer/correlation_analysis.py
───────────────────────────────────
Correlation Analysis & Multi-collinearity Risk Engine.
Computes asset correlation matrix, average pairwise correlation, condition number, and PCA variance.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class CorrelationAnalysisEngine:
    """Evaluates correlation risk, PCA factor dominance, and matrix condition number."""

    def compute_correlation_metrics(
        self,
        returns_df: pd.DataFrame,
        tickers: Optional[list[str]] = None,
    ) -> Dict[str, float]:
        """
        Computes summary correlation metrics across selected tickers.
        """
        if returns_df.empty:
            return {"avg_pairwise_correlation": 0.0, "pca_top1_var": 0.0, "pca_top3_var": 0.0, "condition_number": 0.0}

        sub = returns_df[tickers] if tickers is not None else returns_df
        sub = sub.dropna(how="all")
        if sub.empty or len(sub.columns) < 2:
            return {"avg_pairwise_correlation": 0.0, "pca_top1_var": 0.0, "pca_top3_var": 0.0, "condition_number": 0.0}

        corr = sub.corr().fillna(0).values
        n = corr.shape[0]

        # 1. Average Pairwise Correlation (excluding diagonal 1.0)
        upper_triangle = corr[np.triu_indices(n, k=1)]
        avg_corr = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0

        # 2. PCA Variance Explained
        pca = PCA(n_components=min(n, 5))
        pca.fit(sub.fillna(0))
        top1_var = float(pca.explained_variance_ratio_[0])
        top3_var = float(np.sum(pca.explained_variance_ratio_[:3]))

        # 3. Condition Number (Matrix Multicollinearity Risk)
        cond_num = float(np.linalg.cond(corr))

        return {
            "avg_pairwise_correlation": round(avg_corr, 4),
            "pca_top1_var": round(top1_var, 4),
            "pca_top3_var": round(top3_var, 4),
            "condition_number": round(cond_num, 2),
        }
