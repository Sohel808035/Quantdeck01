"""
execution_layer/backtester.py  (v2 — Production rewrite)
──────────────────────────────────────────────────────────
Vectorised backtester with:
  • 0.15% transaction cost per trade (one-way)
  • Monthly rebalancing (weights only change on rebalance dates — 0 turnover otherwise)
  • Regime exposure scaling applied to raw weights
  • Volatility targeting applied iteratively (approximate one-step approach)
  • Full metrics: CAGR, Vol, Sharpe, Calmar, Max DD, Turnover, Monthly returns
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from risk_layer.vol_targeting import compute_vol_target_scalar  # type: ignore

logger = logging.getLogger(__name__)

TRANSACTION_COST = 0.0015  # 0.15% one-way


class Backtester:

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,   # ₹1 crore default
        transaction_cost: float = TRANSACTION_COST,
        target_vol: float = 0.18,
        apply_vol_targeting: bool = True,
    ):
        self.initial_capital   = initial_capital
        self.tc                = transaction_cost
        self.target_vol        = target_vol
        self.apply_vol_targeting = apply_vol_targeting

    # ── public interface ───────────────────────────────────────────────────────

    def run_backtest(
        self,
        weights_schedule: pd.DataFrame,
        stock_returns:    pd.DataFrame,
        regime_exposure:  pd.Series,
        adv_data:         Optional[pd.DataFrame] = None,
        impact_coeff:     float = 0.1,  # V3 impact penalty
    ) -> Dict[str, Any]:
        """
        V3 Institutional Backtest:
        Adds nonlinear market impact cost model.
        """
        # ── expand monthly weights into a daily weight frame ─────────────────
        all_dates = stock_returns.index
        daily_weights = (
            weights_schedule
            .reindex(all_dates)          # insert NaN on non-rebalance days
            .ffill()                     # hold weights forward
            .fillna(0.0)
        )

        # ── apply regime exposure ─────────────────────────────────────────────
        regime_aligned = regime_exposure.reindex(all_dates).ffill().fillna(1.0)
        daily_weights = daily_weights.multiply(regime_aligned, axis=0)

        # ── signal lag: weight set at close of day t → applies to day t+1 ────
        holding_weights = daily_weights.shift(1).fillna(0.0)

        # ── align returns ─────────────────────────────────────────────────────
        aligned_returns = stock_returns.reindex(columns=holding_weights.columns).fillna(0.0)
        aligned_returns = aligned_returns.reindex(index=holding_weights.index).fillna(0.0)

        # ── gross portfolio return ─────────────────────────────────────────────
        gross_daily_ret = (holding_weights * aligned_returns).sum(axis=1)

        # ── vol targeting ───────────────────────────────────────────────────
        if self.apply_vol_targeting:
            scalar = compute_vol_target_scalar(
                gross_daily_ret,
                target_vol=self.target_vol,
                lookback=60,
            )
            # Scale weights day by day
            holding_weights = holding_weights.multiply(scalar, axis=0)
            gross_daily_ret = (holding_weights * aligned_returns).sum(axis=1)

        # ── V3 Transaction Costs & Impact ────────────────────────────────────
        # 1. Fixed Cost (0.15% per trade)
        trades = holding_weights.diff().abs()
        turnover = trades.sum(axis=1)
        fixed_costs = turnover * self.tc
        
        # 2. Market Impact (nonlinear)
        impact_costs = pd.Series(0.0, index=all_dates)
        if adv_data is not None:
            # Align ADV to the same tickers/dates. Floor at 1 to prevent division by zero.
            adv_aligned = adv_data.reindex(index=all_dates, columns=holding_weights.columns).fillna(1.0).clip(lower=1.0)
            
            # Participation Rate = (TradeValue) / (DailyADV)
            participation = (trades * self.initial_capital) / adv_aligned
            
            # Cost = ImpactCoeff * Participation^2
            impact_matrix = pd.DataFrame(impact_coeff * (participation ** 2))
            impact_costs = impact_matrix.sum(axis=1)

        daily_total_cost = fixed_costs + impact_costs
        net_daily_ret = gross_daily_ret - daily_total_cost

        # ── equity curve ──────────────────────────────────────────────────────
        equity = self.initial_capital * pd.Series(1.0 + net_daily_ret).cumprod()

        # ── metrics ───────────────────────────────────────────────────────────
        metrics = self._compute_metrics(
            equity, 
            pd.Series(net_daily_ret), 
            pd.Series(turnover),
            pd.Series(fixed_costs),
            pd.Series(impact_costs)
        )
        return metrics

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        equity:       pd.Series,
        daily_ret:    pd.Series,
        turnover:     pd.Series,
        fixed_costs:  pd.Series,
        impact_costs: pd.Series
    ) -> Dict[str, Any]:

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_days       = (equity.index[-1] - equity.index[0]).days
        years        = max(n_days / 365.25, 1 / 365)
        
        # Institutional Adjustment: Stats for active period only
        active_mask = (daily_ret.abs() > 1e-9)
        if active_mask.any():
            active_start = active_mask.idxmax()
            active_returns = daily_ret.loc[active_start:]
            active_years = len(active_returns) / 252
            cagr = (equity.iloc[-1] / equity.loc[active_start]) ** (1 / active_years) - 1
            sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(252) if active_returns.std() > 0 else 0.0
        else:
            cagr = 0.0
            sharpe = 0.0

        ann_vol = daily_ret.std() * np.sqrt(252)

        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())
        calmar      = cagr / abs(max_dd) if max_dd != 0 else np.nan

        ann_turnover = turnover.mean() * 252
        
        # Cost Attribution
        total_tc_bp = (fixed_costs.sum() * 10000) / years  # annualised bp
        total_impact_bp = (impact_costs.sum() * 10000) / years  # annualised bp

        # Monthly returns table
        monthly = (daily_ret + 1).resample("ME").prod() - 1
        monthly_table = monthly.to_frame("Monthly Return")
        monthly_table.index = monthly_table.index.to_period("M")

        return {
            "cagr":           cagr,
            "ann_vol":        ann_vol,
            "sharpe_ratio":   sharpe,
            "max_drawdown":   max_dd,
            "calmar_ratio":   calmar,
            "total_return":   total_return,
            "final_equity":   float(equity.iloc[-1]),
            "ann_turnover":   ann_turnover,
            "ann_fixed_cost_bp": total_tc_bp,
            "ann_impact_cost_bp": total_impact_bp,
            "equity_curve":   equity,
            "daily_returns":  daily_ret,
            "monthly_returns": monthly_table,
            "drawdown_series": drawdown,
        }
