"""
portfolio_layer/base.py
────────────────────────
Abstract Base Class and Plugin Registry for QuantSphereX Portfolio Optimization.
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Constraints DTO for portfolio construction."""
    max_weight_per_asset: float = 0.10      # Max 10% in any single stock
    min_weight_per_asset: float = 0.00      # Long-only default
    max_sector_weight: float = 0.30         # Max 30% per sector
    max_adv_pct: float = 0.05               # Max 5% of 20-day ADV
    portfolio_value: float = 1e7            # Portfolio AUM
    target_beta_min: float = 0.8
    target_beta_max: float = 1.2
    turnover_threshold: float = 0.005       # 50bps no-trade zone


class BasePortfolioPlugin(ABC):
    """Abstract Base Class for all portfolio optimization plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the optimization algorithm."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief explanation of the optimization methodology."""
        pass

    @abstractmethod
    def optimize(
        self,
        selected_tickers: Set[str],
        returns_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.Series] = None,
        adv_data: Optional[pd.Series] = None,
        constraints: Optional[PortfolioConstraints] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Computes asset weights given inputs and constraints.

        Args:
            selected_tickers: Set of ticker symbols in the target universe.
            returns_df:       Historical daily return panel (columns = tickers).
            alpha_scores:     Model predicted alpha scores for tickers.
            adv_data:         20-day Average Daily Volume (in currency).
            constraints:      PortfolioConstraints DTO.

        Returns:
            pd.Series of weights indexed by ticker (summing to 1.0 for long-only).
        """
        pass


class PortfolioPluginRegistry:
    """Central registry discovering and managing portfolio optimizer plugins."""

    _plugins: Dict[str, Type[BasePortfolioPlugin]] = {}

    @classmethod
    def register(cls, plugin_class: Type[BasePortfolioPlugin]) -> Type[BasePortfolioPlugin]:
        """Decorator or method to register a portfolio optimization plugin."""
        instance = plugin_class()
        cls._plugins[instance.name.lower()] = plugin_class
        logger.debug(f"[Portfolio Registry] Registered plugin: {instance.name}")
        return plugin_class

    @classmethod
    def get(cls, name: str) -> BasePortfolioPlugin:
        """Retrieves an instantiated plugin by name."""
        key = name.lower()
        if key not in cls._plugins:
            raise KeyError(
                f"Portfolio plugin '{name}' not found. Available: {list(cls._plugins.keys())}"
            )
        return cls._plugins[key]()

    @classmethod
    def list_plugins(cls) -> List[Dict[str, str]]:
        """Lists all registered plugins and their metadata."""
        result = []
        for name, cls_type in cls._plugins.items():
            instance = cls_type()
            result.append({
                "name": instance.name,
                "description": instance.description,
            })
        return result
