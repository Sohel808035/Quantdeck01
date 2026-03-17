"""
portfolio_layer/optimizer.py  (v2)
────────────────────────────────────
Equal weight within selected portfolio: 1/N per stock.
With top_n = 25 → exactly 4% per position.
"""

from __future__ import annotations

import logging
from typing import Set, Optional, Dict
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

class PortfolioOptimizer:

    def equal_weight(
        self, 
        selected_tickers: Set[str],
        adv_data: Optional[pd.Series] = None,
        portfolio_value: float = 1e7,
        max_adv_pct: float = 0.05
    ) -> pd.Series:
        """
        V3 Institutional: Equal weight with ADV constraints.
        1. Base weight = 1/N.
        2. Cap each weight at max_adv_pct * ADV / portfolio_value.
        3. Redistribute residual weight equally.
        """
        if not selected_tickers:
            return pd.Series(dtype=float)

        tickers = sorted(selected_tickers)
        n = len(tickers)
        weights = pd.Series(1.0 / n, index=tickers, name="weight")

        if adv_data is not None:
            # 1. Cap weights by ADV
            adv_subset = adv_data.reindex(tickers).fillna(0)
            max_weights = (adv_subset * max_adv_pct) / portfolio_value
            
            # 2. Identify capped stocks
            capped_mask = weights > max_weights
            if capped_mask.any():
                weights[capped_mask] = max_weights[capped_mask]
                
                # 3. Redistribute remaining weight to uncapped stocks
                residual = 1.0 - weights.sum()
                uncapped = ~capped_mask
                if uncapped.any() and residual > 0.01:
                    weights[uncapped] += (residual / uncapped.sum())
        
        return weights

    def sector_neutralize(
        self, 
        weights: pd.Series, 
        sector_map: Dict[str, str], 
        benchmark_weights: Dict[str, float]
    ) -> pd.Series:
        """
        V3 Institutional: Adjusts portfolio weights to match sector exposure.
        Simple heuristic: scaled to +/- 5% deviation from benchmark.
        """
        if weights.empty: return weights
        
        df = pd.DataFrame({"weight": weights})
        df["Sector"] = df.index.map(sector_map).fillna("Other")
        
        port_sector_w = df.groupby("Sector")["weight"].sum()
        
        # Scaling adjustment
        for sector, b_weight in benchmark_weights.items():
            current_w = port_sector_w.get(sector, 0.0)
            if current_w > 0:
                # Target: current_w should be within [b-0.05, b+0.05]
                target_w = max(min(current_w, b_weight + 0.05), b_weight - 0.05)
                scalar = target_w / current_w
                df.loc[df["Sector"] == sector, "weight"] *= scalar
                
        # Final normalize to 100%
        result = df["weight"] / df["weight"].sum()
        return result

    def beta_target(
        self, 
        weights: pd.Series, 
        stock_betas: pd.Series, 
        target_range: tuple = (0.8, 1.2)
    ) -> pd.Series:
        """
        V3 Institutional: Rescale weights to keep portfolio beta within range.
        """
        if weights.empty or stock_betas.empty: return weights
        
        # Align betas
        betas = stock_betas.reindex(weights.index).fillna(1.0)
        port_beta = (weights * betas).sum()
        
        if port_beta < target_range[0]:
            scalar = target_range[0] / port_beta
            return weights * scalar
        elif port_beta > target_range[1]:
            scalar = target_range[1] / port_beta
            return weights * scalar
        
        return weights
    def apply_turnover_penalty(
        self, 
        current_weights: pd.Series, 
        target_weights: pd.Series, 
        threshold: float = 0.005
    ) -> pd.Series:
        """
        V3 Institutional: Turnover Penalty (No-Trade Zone).
        If the required change for a stock is < threshold (default 50bps), 
        keep the current weight to avoid expensive micro-trades.
        """
        if current_weights.empty:
            return target_weights

        # Align both series
        all_tickers = sorted(list(set(current_weights.index) | set(target_weights.index)))
        curr = current_weights.reindex(all_tickers).fillna(0.0)
        targ = target_weights.reindex(all_tickers).fillna(0.0)
        
        diff = abs(targ - curr)
        
        # Only trade if diff > threshold
        final_weights = curr.copy()
        trade_mask = diff > threshold
        final_weights[trade_mask] = targ[trade_mask]
        
        # Final normalize to ensure we are still 100% (or less if cash is allowed)
        # Note: In institutional L/S, normally we'd allow cash. 
        # Here we re-normalize to maintain exposure.
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()
            
        return final_weights
