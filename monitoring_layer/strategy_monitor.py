"""
monitoring_layer/strategy_monitor.py
──────────────────────────────────────
Strategy Performance Monitor.
Tracks rolling Sharpe, rolling IC, drawdown breaches, and portfolio health in real-time.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from monitoring_layer.config import MonitoringConfig
from monitoring_layer.alert_engine import AlertEngine, AlertSeverity

logger = logging.getLogger(__name__)


class StrategyMonitor:
    """
    Monitors live strategy performance metrics:
      - Rolling Sharpe (breach alert if < min_sharpe)
      - Rolling IC (breach alert if < min_ic)
      - Rolling Drawdown (breach alert if < max_drawdown_breach)
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        alert_engine: Optional[AlertEngine] = None,
    ):
        self.config = config or MonitoringConfig()
        self.alert_engine = alert_engine or AlertEngine(config=self.config)
        self.cfg = self.config.strategy

    def check_rolling_sharpe(
        self,
        daily_returns: pd.Series,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Computes the rolling Sharpe ratio and fires alerts if below the threshold.

        Args:
            daily_returns: Daily net return Series.
            window:        Rolling window in days.

        Returns:
            Rolling Sharpe summary dict.
        """
        w = window or self.cfg.sharpe_window
        if len(daily_returns) < w:
            return {"status": "insufficient_data", "window": w}

        rf_daily = (1 + self.config.system.latency_warning_ms) ** (1 / 252) - 1
        rf_daily = (1 + 0.065) ** (1 / 252) - 1  # 6.5% annualised RFR

        roll_mean = daily_returns.rolling(w).mean()
        roll_std = daily_returns.rolling(w).std().replace(0, np.nan)
        rolling_sharpe = ((roll_mean - rf_daily) / roll_std * np.sqrt(252))

        latest_sharpe = float(rolling_sharpe.iloc[-1]) if not np.isnan(rolling_sharpe.iloc[-1]) else 0.0
        mean_sharpe = float(rolling_sharpe.dropna().mean())

        if latest_sharpe < self.cfg.min_sharpe:
            sev = AlertSeverity.CRITICAL if latest_sharpe < 0 else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "STRATEGY", "rolling_sharpe",
                value=latest_sharpe, threshold=self.cfg.min_sharpe,
                message=(
                    f"Rolling Sharpe ({w}d) = {latest_sharpe:.3f} "
                    f"below minimum threshold {self.cfg.min_sharpe:.2f}."
                ),
            )

        return {
            "window": w,
            "latest_sharpe": round(latest_sharpe, 4),
            "mean_sharpe": round(mean_sharpe, 4),
            "min_sharpe": round(float(rolling_sharpe.dropna().min()), 4),
            "pct_positive": round(float((rolling_sharpe.dropna() > 0).mean()), 4),
            "rolling_series": rolling_sharpe,
            "breach": latest_sharpe < self.cfg.min_sharpe,
        }

    def check_rolling_ic(
        self,
        alpha_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Computes rolling Information Coefficient (IC) using Spearman rank correlation.
        Fires alert if mean IC drops below threshold.

        Args:
            alpha_scores:    DataFrame of per-stock alpha scores indexed by date.
            forward_returns: DataFrame of next-day returns indexed by date.
            window:          Rolling lookback window.

        Returns:
            Rolling IC summary dict.
        """
        w = window or self.cfg.ic_window
        common_idx = alpha_scores.index.intersection(forward_returns.index)
        if len(common_idx) < w:
            return {"status": "insufficient_data", "window": w}

        daily_ics = []
        for d in common_idx:
            try:
                a = alpha_scores.loc[d].dropna()
                f = forward_returns.loc[d].reindex(a.index).dropna()
                shared = a.index.intersection(f.index)
                if len(shared) >= 5:
                    ic, _ = spearmanr(a[shared], f[shared])
                    daily_ics.append(ic if not np.isnan(ic) else 0.0)
                else:
                    daily_ics.append(0.0)
            except Exception:
                daily_ics.append(0.0)

        ic_series = pd.Series(daily_ics, index=common_idx)
        rolling_ic = ic_series.rolling(w).mean()

        latest_ic = float(rolling_ic.iloc[-1]) if not np.isnan(rolling_ic.iloc[-1]) else 0.0
        mean_ic = float(rolling_ic.dropna().mean())

        if latest_ic < self.cfg.min_ic:
            sev = AlertSeverity.CRITICAL if latest_ic < 0 else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "STRATEGY", "rolling_ic",
                value=latest_ic, threshold=self.cfg.min_ic,
                message=(
                    f"Rolling IC ({w}d) = {latest_ic:.4f} below minimum threshold {self.cfg.min_ic:.4f}."
                ),
            )

        return {
            "window": w,
            "latest_ic": round(latest_ic, 4),
            "mean_ic": round(mean_ic, 4),
            "ic_ir": round(float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0, 4),
            "pct_positive": round(float((ic_series > 0).mean()), 4),
            "rolling_series": rolling_ic,
            "breach": latest_ic < self.cfg.min_ic,
        }

    def check_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> Dict[str, Any]:
        """
        Monitors current drawdown level. Fires alert if below max_drawdown_breach.

        Returns:
            Drawdown metrics dict.
        """
        if len(equity_curve) < 2:
            return {"status": "insufficient_data"}

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        current_dd = float(drawdown.iloc[-1])
        max_dd = float(drawdown.min())

        if current_dd < self.cfg.max_drawdown_breach:
            sev = AlertSeverity.CRITICAL if current_dd < self.cfg.max_drawdown_breach * 1.5 else AlertSeverity.WARNING
            self.alert_engine.fire(
                sev, "STRATEGY", "drawdown",
                value=current_dd, threshold=self.cfg.max_drawdown_breach,
                message=(
                    f"Active drawdown {current_dd:.2%} breached threshold {self.cfg.max_drawdown_breach:.2%}. "
                    f"Max historical drawdown: {max_dd:.2%}."
                ),
            )

        # Time-to-recovery stats
        in_dd = drawdown < -1e-4
        if in_dd.any():
            diff = in_dd.iloc[::-1].idxmax() - drawdown.idxmin()
            dd_duration = diff.days if hasattr(diff, "days") else int(diff)
        else:
            dd_duration = 0

        return {
            "current_drawdown": round(current_dd, 4),
            "max_drawdown": round(max_dd, 4),
            "drawdown_duration_days": max(dd_duration, 0),
            "breach": current_dd < self.cfg.max_drawdown_breach,
            "drawdown_series": drawdown,
        }
