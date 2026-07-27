"""
risk_layer/limits.py
────────────────────
Position & Concentration Limits Audit Engine.
Evaluates Herfindahl-Hirschman Index (HHI), Effective Number of Assets (N_eff), and position limits.
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from risk_layer.config import RiskConfig

logger = logging.getLogger(__name__)


class LimitsAuditEngine:
    """Audits portfolio weights against regulatory and institutional position & concentration limits."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def concentration_metrics(self, weights: pd.Series) -> Dict[str, float]:
        """
        Computes Herfindahl-Hirschman Index (HHI), Effective Number of Assets (N_eff),
        Top-5, and Top-10 concentration ratios.
        """
        if weights.empty or weights.sum() == 0:
            return {"hhi_index": 0.0, "effective_n_stocks": 0.0, "top_5_concentration": 0.0, "top_10_concentration": 0.0}

        w = (weights / weights.sum()).values

        # 1. HHI = sum(w_i ^ 2)
        hhi = float(np.sum(w**2))

        # 2. Effective Number of Stocks N_eff = 1 / HHI
        n_eff = float(1.0 / hhi) if hhi > 0 else 0.0

        # 3. Top-5 and Top-10 concentration ratios
        sorted_w = np.sort(w)[::-1]
        top5 = float(np.sum(sorted_w[:5]))
        top10 = float(np.sum(sorted_w[:10]))

        return {
            "hhi_index": round(hhi, 4),
            "effective_n_stocks": round(n_eff, 2),
            "top_5_concentration": round(top5, 4),
            "top_10_concentration": round(top10, 4),
        }

    def audit_limits(
        self,
        weights: pd.Series,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Audits portfolio weights against configured limit thresholds:
          - Max single position weight cap
          - Max sector concentration cap
          - Min Effective N_eff threshold
          - Max HHI threshold

        Returns (all_limits_passed, dict_of_individual_checks).
        """
        if weights.empty:
            return True, {}

        w = weights / weights.sum()
        max_pos = float(w.max())
        hhi_info = self.concentration_metrics(w)

        checks = {
            "single_position_limit": max_pos <= self.config.max_single_position_pct,
            "hhi_limit": hhi_info["hhi_index"] <= self.config.max_hhi_threshold,
            "effective_n_limit": hhi_info["effective_n_stocks"] >= self.config.min_effective_n_stocks,
        }

        if sector_map is not None:
            df = pd.DataFrame({"weight": w})
            df["Sector"] = df.index.map(sector_map).fillna("Other")
            sector_w = df.groupby("Sector")["weight"].sum()
            max_sec = float(sector_w.max())
            checks["sector_limit"] = max_sec <= self.config.max_sector_exposure_pct

        all_passed = all(checks.values())
        return all_passed, checks
