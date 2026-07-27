"""
portfolio_layer/optimizer.py  (v3 — Modular Plugin Architecture)
────────────────────────────────────────────────────────────────
Backward-compatible wrapper around PortfolioPluginRegistry.

Maintains existing equal_weight(), sector_neutralize(), beta_target(),
and apply_turnover_penalty() interfaces while enabling plugin execution.
"""

from __future__ import annotations
import logging
from typing import Set, Optional, Dict, Any
import pandas as pd
import numpy as np

from portfolio_layer.base import (
    BasePortfolioPlugin,
    PortfolioPluginRegistry,
    PortfolioConstraints,
)
from portfolio_layer.constraints import ConstraintsEngine
from portfolio_layer.rebalancing import RebalancingEngine
from portfolio_layer.transaction_cost import TransactionCostEngine
from portfolio_layer.position_sizing import PositionSizingEngine

# Import plugins to force self-registration in registry
import portfolio_layer.plugins.equal_weight
import portfolio_layer.plugins.risk_parity
import portfolio_layer.plugins.hrp
import portfolio_layer.plugins.min_variance
import portfolio_layer.plugins.black_litterman
import portfolio_layer.plugins.kelly

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Modular Portfolio Optimizer supporting dynamic plugin execution,
    constraints, rebalancing penalties, transaction cost models, and position sizing.
    """

    def __init__(self, default_optimizer: str = "equal_weight"):
        self.default_optimizer = default_optimizer
        self.constraints_engine = ConstraintsEngine()
        self.rebalancing_engine = RebalancingEngine()
        self.cost_engine = TransactionCostEngine()
        self.sizing_engine = PositionSizingEngine()

    def optimize(
        self,
        selected_tickers: Set[str],
        optimizer_name: Optional[str] = None,
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Executes the requested optimizer plugin by name.

        Supported Optimizers:
          - 'equal_weight'    : Equal Weight (1/N)
          - 'risk_parity'     : Risk Parity / Equal Risk Contribution (ERC)
          - 'hrp'             : Hierarchical Risk Parity (López de Prado 2016)
          - 'min_variance'    : Minimum Variance Portfolio
          - 'black_litterman' : Black-Litterman model
          - 'kelly'           : Fractional Kelly Criterion
        """
        name = optimizer_name or self.default_optimizer
        plugin = PortfolioPluginRegistry.get(name)

        weights = plugin.optimize(
            selected_tickers=selected_tickers,
            returns_df=returns_df,
            alpha_scores=alpha_scores,
            adv_data=adv_data,
            constraints=constraints or self.constraints_engine.constraints,
            **kwargs,
        )

        return weights

    # ── Backward-Compatibility Wrappers ───────────────────────────────────────

    def equal_weight(
        self,
        selected_tickers: Set[str],
        adv_data: Optional[pd.Series] = None,
        portfolio_value: float = 1e7,
        max_adv_pct: float = 0.05,
    ) -> pd.Series:
        """Backward-compatible equal_weight entry point."""
        constraints = PortfolioConstraints(
            portfolio_value=portfolio_value,
            max_adv_pct=max_adv_pct,
        )
        return self.optimize(
            selected_tickers=selected_tickers,
            optimizer_name="equal_weight",
            adv_data=adv_data,
            constraints=constraints,
        )

    def sector_neutralize(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str],
        benchmark_weights: Dict[str, float],
    ) -> pd.Series:
        """Backward-compatible sector_neutralize wrapper."""
        return self.constraints_engine.sector_neutralize(
            weights, sector_map, benchmark_weights
        )

    def beta_target(
        self,
        weights: pd.Series,
        stock_betas: pd.Series,
        target_range: tuple = (0.8, 1.2),
    ) -> pd.Series:
        """Backward-compatible beta_target wrapper."""
        return self.constraints_engine.beta_target(
            weights, stock_betas, target_range=target_range
        )

    def apply_turnover_penalty(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        threshold: float = 0.005,
    ) -> pd.Series:
        """Backward-compatible apply_turnover_penalty wrapper."""
        return self.rebalancing_engine.apply_turnover_penalty(
            current_weights, target_weights, threshold=threshold
        )
