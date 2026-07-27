"""
execution_layer/backtesting/metrics.py
────────────────────────────────────────
Performance Metrics Module.
Computes full-period and rolling performance metrics.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from execution_layer.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class MetricsEngine:
    """Computes institutional-grade full-period and rolling performance metrics."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def full_period_metrics(
        self,
        equity_curve: pd.Series,
        daily_ret: pd.Series,
        turnover: pd.Series,
        fixed_costs: pd.Series,
        impact_costs: pd.Series,
        benchmark_ret: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Computes comprehensive single-period performance statistics."""
        if len(equity_curve) < 2:
            return {}

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

        # Active period only (ignore warm-up zeros)
        active_mask = daily_ret.abs() > 1e-9
        if active_mask.any():
            active_start = active_mask.idxmax()
            active_ret = daily_ret.loc[active_start:]
            active_equity = equity_curve.loc[active_start:]
            active_years = len(active_ret) / 252.0
            cagr = float((active_equity.iloc[-1] / active_equity.iloc[0]) ** (1.0 / max(active_years, 1e-3)) - 1)
        else:
            active_ret = daily_ret
            active_years = len(daily_ret) / 252.0
            cagr = 0.0

        ann_vol = float(active_ret.std() * np.sqrt(252))
        rf_daily = (1 + self.config.risk_free_rate) ** (1 / 252) - 1
        excess_ret = active_ret - rf_daily
        sharpe = float((excess_ret.mean() / active_ret.std()) * np.sqrt(252)) if active_ret.std() > 0 else 0.0

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = float(drawdown.min())
        calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

        # Sortino Ratio
        downside = active_ret[active_ret < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 1e-8
        sortino = float((active_ret.mean() * 252 - self.config.risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        # Information Ratio vs benchmark
        if benchmark_ret is not None:
            aligned_bench = benchmark_ret.reindex(active_ret.index).fillna(0)
            active_bets = active_ret - aligned_bench
            ir = float(active_bets.mean() / active_bets.std() * np.sqrt(252)) if active_bets.std() > 0 else 0.0
        else:
            ir = np.nan

        # Cost attribution
        years = max(len(daily_ret) / 252, 1e-4)
        ann_turnover = float(turnover.mean() * 252)
        ann_fixed_cost_bp = float((fixed_costs.sum() * 10_000) / years)
        ann_impact_cost_bp = float((impact_costs.sum() * 10_000) / years)

        # Monthly returns
        monthly = (daily_ret + 1).resample("ME").prod() - 1
        monthly_table = monthly.to_frame("Monthly Return")
        monthly_table.index = monthly_table.index.to_period("M")

        return {
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "information_ratio": ir,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_return": total_return,
            "final_equity": float(equity_curve.iloc[-1]),
            "ann_turnover": ann_turnover,
            "ann_fixed_cost_bp": ann_fixed_cost_bp,
            "ann_impact_cost_bp": ann_impact_cost_bp,
            "equity_curve": equity_curve,
            "daily_returns": daily_ret,
            "drawdown_series": drawdown,
            "monthly_returns": monthly_table,
        }

    def rolling_metrics(
        self,
        daily_ret: pd.Series,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Computes rolling Sharpe, Sortino, Volatility, and Max Drawdown.

        Returns:
            DataFrame with rolling metrics indexed by date.
        """
        w = window or self.config.rolling_window
        rf_daily = (1 + self.config.risk_free_rate) ** (1 / 252) - 1

        roll = daily_ret.rolling(w, min_periods=max(20, w // 3))

        rolling_mean = roll.mean()
        rolling_std = roll.std().replace(0, np.nan)

        r_sharpe = ((rolling_mean - rf_daily) / rolling_std * np.sqrt(252)).rename("rolling_sharpe")
        r_vol = (rolling_std * np.sqrt(252)).rename("rolling_vol")

        # Rolling Sortino
        def _sortino(x):
            down = x[x < 0]
            ds = down.std() if len(down) > 1 else 1e-8
            return (x.mean() * 252 - self.config.risk_free_rate) / (ds * np.sqrt(252)) if ds > 0 else np.nan

        r_sortino = roll.apply(_sortino, raw=False).rename("rolling_sortino")

        # Rolling Max Drawdown
        def _max_dd(x):
            eq = (1 + x).cumprod()
            dd = (eq - eq.cummax()) / eq.cummax()
            return float(dd.min())

        r_max_dd = roll.apply(_max_dd, raw=False).rename("rolling_max_dd")

        return pd.DataFrame({
            "rolling_sharpe":  r_sharpe,
            "rolling_vol":     r_vol,
            "rolling_sortino": r_sortino,
            "rolling_max_dd":  r_max_dd,
        })
