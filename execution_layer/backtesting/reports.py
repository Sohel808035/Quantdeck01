"""
execution_layer/backtesting/reports.py
───────────────────────────────────────
Performance Report Generator and Export Module.
Produces institutional-quality backtest tearsheets with CSV and JSON export.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates institutional performance reports from backtest results and exports them.
    Supports CSV, JSON, and Markdown exports.
    """

    def build_tearsheet(
        self,
        metrics: Dict[str, Any],
        rolling_metrics: Optional[pd.DataFrame] = None,
        factor_attribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Builds a structured tearsheet DataFrame summarising all key metrics.

        Returns:
            Single-row summary DataFrame suitable for display or export.
        """
        row = {}
        # Core metrics
        scalar_keys = [
            "cagr", "ann_vol", "sharpe_ratio", "sortino_ratio", "information_ratio",
            "max_drawdown", "calmar_ratio", "total_return", "final_equity",
            "ann_turnover", "ann_fixed_cost_bp", "ann_impact_cost_bp",
        ]
        for k in scalar_keys:
            if k in metrics:
                row[k] = metrics[k]

        # Rolling metrics summary
        if rolling_metrics is not None and not rolling_metrics.empty:
            row["avg_rolling_sharpe"] = float(rolling_metrics["rolling_sharpe"].mean())
            row["min_rolling_sharpe"] = float(rolling_metrics["rolling_sharpe"].min())
            row["pct_positive_rolling"] = float((rolling_metrics["rolling_sharpe"] > 0).mean())

        # Factor attribution summary
        if factor_attribution is not None:
            for k, v in factor_attribution.items():
                row[f"factor_{k}"] = v

        return pd.DataFrame([row])

    def monthly_returns_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Returns the monthly returns pivot table from metrics dict."""
        return metrics.get("monthly_returns", pd.DataFrame())

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Prints a formatted performance summary to logger."""
        logger.info("=" * 60)
        logger.info("  QUANTSPHEREX BACKTEST PERFORMANCE REPORT")
        logger.info("=" * 60)
        logger.info(f"  CAGR:          {metrics.get('cagr', 0):.2%}")
        logger.info(f"  Ann. Vol:      {metrics.get('ann_vol', 0):.2%}")
        logger.info(f"  Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown:  {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Calmar Ratio:  {metrics.get('calmar_ratio', 0):.3f}")
        logger.info(f"  Ann. Turnover: {metrics.get('ann_turnover', 0):.2f}x")
        logger.info(f"  TC Cost (bp):  {metrics.get('ann_fixed_cost_bp', 0):.2f}")
        logger.info(f"  Impact (bp):   {metrics.get('ann_impact_cost_bp', 0):.2f}")
        logger.info("=" * 60)

    def export_csv(self, metrics: Dict[str, Any], output_dir: str = "reports") -> str:
        """Exports equity curve, daily returns, and monthly returns table to CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, series_name in [
            ("equity_curve", "equity"), ("daily_returns", "returns"), ("drawdown_series", "drawdown")
        ]:
            if name in metrics:
                path = os.path.join(output_dir, f"backtest_{name}.csv")
                metrics[name].to_csv(path, header=[series_name])
                paths[name] = path
                logger.info(f"[Report] Exported {name} to {path}")

        if "monthly_returns" in metrics:
            path = os.path.join(output_dir, "backtest_monthly_returns.csv")
            metrics["monthly_returns"].to_csv(path)
            paths["monthly_returns"] = path

        return output_dir

    def export_json(self, metrics: Dict[str, Any], output_path: str = "reports/backtest_summary.json") -> str:
        """Exports all scalar metrics to a JSON summary file."""
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        scalar_keys = [
            "cagr", "ann_vol", "sharpe_ratio", "sortino_ratio", "information_ratio",
            "max_drawdown", "calmar_ratio", "total_return", "final_equity",
            "ann_turnover", "ann_fixed_cost_bp", "ann_impact_cost_bp",
        ]
        summary = {k: round(float(metrics[k]), 6) for k in scalar_keys if k in metrics}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[Report] Exported JSON summary to {output_path}")
        return output_path
