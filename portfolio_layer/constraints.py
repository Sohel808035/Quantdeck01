"""
portfolio_layer/constraints.py
───────────────────────────────
QuantSphereX Portfolio Constraints Engine.
Applies asset caps, sector neutralization, beta targeting, and ADV liquidity filters.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Set
import pandas as pd
import numpy as np

from portfolio_layer.base import PortfolioConstraints

logger = logging.getLogger(__name__)


class ConstraintsEngine:
    """Applies institutional constraints and filters to raw optimization weights."""

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        self.constraints = constraints or PortfolioConstraints()

    def apply_all_constraints(
        self,
        weights: pd.Series,
        adv_data: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        benchmark_sector_weights: Optional[Dict[str, float]] = None,
        stock_betas: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Applies sequential constraint pipeline:
          1. Min/Max asset weight bounds (with iterative re-normalization)
          2. ADV Liquidity participation cap
          3. Sector allocation bounds
          4. Portfolio Beta target bounds
          5. Sum normalization to 1.0
        """
        if weights.empty:
            return weights

        w = weights.copy()

        # 1. ADV Liquidity participation cap
        if adv_data is not None and len(adv_data) > 0:
            adv_subset = adv_data.reindex(w.index).fillna(0)
            max_w = (adv_subset * self.constraints.max_adv_pct) / self.constraints.portfolio_value
            w = np.minimum(w, max_w)

        # 2. Sector allocation bounds
        if sector_map is not None and benchmark_sector_weights is not None:
            w = self.sector_neutralize(w, sector_map, benchmark_sector_weights)

        # 3. Portfolio Beta bounds
        if stock_betas is not None and len(stock_betas) > 0:
            w = self.beta_target(
                w, stock_betas, target_range=(self.constraints.target_beta_min, self.constraints.target_beta_max)
            )

        # 4. Min/Max asset weight bounds (iterative clipping)
        w = self._clip_and_normalize(
            w,
            min_w=self.constraints.min_weight_per_asset,
            max_w=self.constraints.max_weight_per_asset,
        )

        return w

    def _clip_and_normalize(
        self,
        weights: pd.Series,
        min_w: float,
        max_w: float,
        max_iter: int = 10,
    ) -> pd.Series:
        w = weights.copy()
        for _ in range(max_iter):
            w = w.clip(lower=min_w, upper=max_w)
            total = w.sum()
            if total == 0:
                return pd.Series(1.0 / len(w), index=w.index)
            if abs(total - 1.0) < 1e-5:
                break
            # Scale uncapped weights
            uncapped = (w > min_w) & (w < max_w)
            if not uncapped.any():
                w = w / total
                break
            excess = 1.0 - w[~uncapped].sum()
            if excess <= 0:
                w[uncapped] = 0.0
                break
            w[uncapped] = (w[uncapped] / w[uncapped].sum()) * excess
        return w

    def sector_neutralize(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        benchmark_weights: Dict[str, float],
        max_deviation: float = 0.05,
    ) -> pd.Series:
        """Adjusts weights to ensure sector allocations stay within benchmark +/- max_deviation."""
        if weights.empty:
            return weights

        df = pd.DataFrame({"weight": weights})
        df["Sector"] = df.index.map(sector_map).fillna("Other")
        port_sector_w = df.groupby("Sector")["weight"].sum()

        for sector, b_weight in benchmark_weights.items():
            current_w = port_sector_w.get(sector, 0.0)
            if current_w > 0:
                target_w = max(min(current_w, b_weight + max_deviation), b_weight - max_deviation)
                scalar = target_w / current_w
                df.loc[df["Sector"] == sector, "weight"] *= scalar

        result = df["weight"]
        if result.sum() > 0:
            result = result / result.sum()
        return result

    def beta_target(
        self,
        weights: pd.Series,
        stock_betas: pd.Series,
        target_range: tuple[float, float] = (0.8, 1.2),
    ) -> pd.Series:
        """Rescales weights if portfolio beta exceeds target range."""
        if weights.empty or stock_betas.empty:
            return weights

        betas = stock_betas.reindex(weights.index).fillna(1.0)
        port_beta = (weights * betas).sum()

        if port_beta < target_range[0] and port_beta > 0:
            scalar = target_range[0] / port_beta
            return weights * scalar
        elif port_beta > target_range[1]:
            scalar = target_range[1] / port_beta
            return weights * scalar

        return weights
