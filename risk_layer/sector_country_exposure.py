"""
risk_layer/sector_country_exposure.py
──────────────────────────────────────
Sector and Country Exposure Risk Engine.
Computes portfolio breakdown, active bets vs benchmark, and concentration.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExposureRiskEngine:
    """Computes sector, country, and geographic concentration exposures."""

    def compute_sector_exposure(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        benchmark_sector_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Computes portfolio sector exposures and active bet vs benchmark.
        """
        if weights.empty:
            return pd.DataFrame()

        df = pd.DataFrame({"weight": weights})
        df["Sector"] = df.index.map(sector_map).fillna("Other")

        port_sectors = df.groupby("Sector")["weight"].sum()

        result = pd.DataFrame({"portfolio_weight": port_sectors})
        if benchmark_sector_weights is not None:
            result["benchmark_weight"] = pd.Series(benchmark_sector_weights)
            result["benchmark_weight"] = result["benchmark_weight"].fillna(0.0)
            result["active_bet"] = result["portfolio_weight"] - result["benchmark_weight"]
        else:
            result["benchmark_weight"] = 0.0
            result["active_bet"] = result["portfolio_weight"]

        return result.sort_values("portfolio_weight", ascending=False)

    def compute_country_exposure(
        self,
        weights: pd.Series,
        country_map: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """
        Computes portfolio country exposures. Defaults to 'India' (NSE) if unmapped.
        """
        if weights.empty:
            return pd.Series(dtype=float)

        if country_map is None:
            country_map = {}

        countries = pd.Series(weights.index).map(country_map).fillna("India").values
        df = pd.DataFrame({"weight": weights.values, "Country": countries})
        return df.groupby("Country")["weight"].sum().sort_values(ascending=False)
