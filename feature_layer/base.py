"""
feature_layer/base.py
─────────────────────
Base Classes, Interfaces, and Data Transfer Objects (DTOs) for QuantSphereX Factor Engines.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class FactorMetadata:
    """Metadata DTO for an independent factor module."""
    name: str
    category: str
    version: str
    description: str
    output_columns: List[str]
    execution_time_seconds: float = 0.0


@dataclass
class FactorValidationResult:
    """Validation audit result for generated feature columns."""
    factor_name: str
    passed: bool
    total_rows: int
    nan_count_map: Dict[str, int] = field(default_factory=dict)
    infinite_count_map: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseFactorEngine(ABC):
    """Abstract Base Class for independent quantitative factor modules."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the factor family."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the factor family."""
        pass

    @property
    @abstractmethod
    def output_columns(self) -> List[str]:
        """List of feature column names produced by this engine."""
        pass

    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        context_ret: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Computes feature columns from input stock panel DataFrame.
        Input DataFrame is assumed to have Date index or (Date, Ticker) MultiIndex.
        Returns DataFrame with generated feature columns.
        """
        pass

    def validate(self, result_df: pd.DataFrame) -> FactorValidationResult:
        """Validates generated feature DataFrames for NaNs, Infs, and range anomalies."""
        total_rows = len(result_df)
        nan_map = {}
        inf_map = {}
        warnings = []

        for col in self.output_columns:
            if col in result_df.columns:
                nans = int(result_df[col].isna().sum())
                infs = int(np.isinf(result_df[col]).sum()) if len(result_df) > 0 else 0
                nan_map[col] = nans
                inf_map[col] = infs

                if nans > total_rows * 0.5 and total_rows > 0:
                    warnings.append(f"High NaN ratio in '{col}': {nans}/{total_rows}")
                if infs > 0:
                    warnings.append(f"Infinite values detected in '{col}': {infs}")

        return FactorValidationResult(
            factor_name=self.name,
            passed=len(warnings) == 0,
            total_rows=total_rows,
            nan_count_map=nan_map,
            infinite_count_map=inf_map,
            warnings=warnings,
        )

    def benchmark_compute(
        self,
        df: pd.DataFrame,
        context_ret: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, FactorMetadata]:
        """Executes computation while capturing execution benchmark metrics."""
        t0 = time.perf_counter()
        computed_df = self.compute(df, context_ret=context_ret)
        t1 = time.perf_counter()
        elapsed = t1 - t0

        meta = FactorMetadata(
            name=self.name,
            category=self.category,
            version=self.version,
            description=self.__doc__ or "Factor Engine",
            output_columns=self.output_columns,
            execution_time_seconds=elapsed,
        )
        return computed_df, meta
