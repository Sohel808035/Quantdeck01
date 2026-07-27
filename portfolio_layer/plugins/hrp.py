"""
portfolio_layer/plugins/hrp.py
───────────────────────────────
Hierarchical Risk Parity (HRP) Portfolio Plugin.
Implements Marcos López de Prado (2016) tree clustering & recursive bisection.
"""

from __future__ import annotations
import logging
from typing import Set, Optional, List
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

from portfolio_layer.base import BasePortfolioPlugin, PortfolioPluginRegistry, PortfolioConstraints

logger = logging.getLogger(__name__)


@PortfolioPluginRegistry.register
class HierarchicalRiskParityPlugin(BasePortfolioPlugin):
    """Hierarchical Risk Parity (HRP) optimizer using machine learning tree clustering."""

    @property
    def name(self) -> str:
        return "hrp"

    @property
    def description(self) -> str:
        return "Hierarchical Risk Parity (López de Prado 2016) using correlation matrix hierarchical clustering."

    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(list(selected_tickers))

        if returns_df is None or returns_df.empty:
            logger.warning("[HRP] No returns_df provided; falling back to Equal Weight.")
            return pd.Series(1.0 / len(tickers), index=tickers)

        sub_ret = returns_df[[t for t in tickers if t in returns_df.columns]].dropna()
        if sub_ret.empty or len(sub_ret.columns) < 2:
            return pd.Series(1.0 / len(tickers), index=tickers)

        cov = sub_ret.cov()
        corr = sub_ret.corr().fillna(0)

        # 1. Distance matrix
        dist = np.sqrt(0.5 * (1 - corr.values))
        # Condensed distance matrix
        condensed_dist = dist[np.triu_indices(len(dist), k=1)]
        if len(condensed_dist) == 0:
            return pd.Series(1.0 / len(tickers), index=tickers)

        link = linkage(condensed_dist, method="single")

        # 2. Quasi-diagonalization (reordering columns)
        sort_idx = self._get_quasi_diag(link)
        sorted_cols = [sub_ret.columns[i] for i in sort_idx]

        # 3. Recursive bisection
        weights = self._get_rec_bisection(cov.loc[sorted_cols, sorted_cols])
        weights = weights.reindex(tickers).fillna(1.0 / len(tickers))
        return weights / weights.sum()

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        link = link.astype(int)
        sort_idx = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]

        while max(sort_idx) >= num_items:
            sort_idx_copy = list(sort_idx)
            for i, item in enumerate(sort_idx_copy):
                if item >= num_items:
                    sort_idx[i] = link[item - num_items, 0]
                    sort_idx.insert(i + 1, link[item - num_items, 1])
        return sort_idx

    def _get_rec_bisection(self, cov: pd.DataFrame) -> pd.Series:
        w = pd.Series(1.0, index=cov.index)
        items = [cov.index.tolist()]

        while len(items) > 0:
            items = [
                i[j:k] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1
            ]
            for i in range(0, len(items), 2):
                c_items0 = items[i]
                c_items1 = items[i + 1]

                cov0 = cov.loc[c_items0, c_items0]
                cov1 = cov.loc[c_items1, c_items1]

                iv0 = 1.0 / np.diag(cov0.values)
                w0 = iv0 / np.sum(iv0)
                var0 = np.dot(w0, np.dot(cov0.values, w0))

                iv1 = 1.0 / np.diag(cov1.values)
                w1 = iv1 / np.sum(iv1)
                var1 = np.dot(w1, np.dot(cov1.values, w1))

                alpha = 1 - var0 / (var0 + var1)
                w[c_items0] *= alpha
                w[c_items1] *= (1 - alpha)

        return w
