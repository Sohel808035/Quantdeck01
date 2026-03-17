"""
execution_layer/stress_tester.py  (CQRO Mandate §VIII)
═══════════════════════════════════════════════════════════
Stress Testing Suite.

Scenarios:
  1. 2x Transaction Cost
  2. 30% Liquidity Reduction (ADV shocks)
  3. Volatility Spike Regime
  4. 2008 Crash Replay (Jan 2008 – Mar 2009)
  5. 2020 Crash Replay (Feb 2020 – May 2020)

Evaluates Sharpe degradation.
Mandate: Sharpe must not collapse below 0.8 under stress.
"""

from __future__ import annotations
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run_scenario_backtest(
    weight_df: pd.DataFrame,
    stock_returns: pd.DataFrame,
    regime_exposure: pd.Series,
    adv_data: pd.DataFrame,
    transaction_cost: float,
    impact_coeff: float,
    adv_multiplier: float = 1.0,
    date_filter: tuple = None,
    initial_capital: float = 100_000.0,
) -> Dict[str, float]:
    """Simple vectorized backtest for stress scenario."""
    from risk_layer.vol_targeting import compute_vol_target_scalar

    all_dates = stock_returns.index
    daily_weights = weight_df.reindex(all_dates).ffill().fillna(0.0)
    regime_aligned = regime_exposure.reindex(all_dates).ffill().fillna(1.0)
    daily_weights = daily_weights.multiply(regime_aligned, axis=0)
    holding_weights = daily_weights.shift(1).fillna(0.0)

    aligned_returns = stock_returns.reindex(
        columns=holding_weights.columns
    ).fillna(0.0).reindex(index=holding_weights.index).fillna(0.0)

    gross = (holding_weights * aligned_returns).sum(axis=1)

    trades = holding_weights.diff().abs()
    turnover = trades.sum(axis=1)
    fixed_costs = turnover * transaction_cost

    impact_costs = pd.Series(0.0, index=all_dates)
    if adv_data is not None:
        adv_adj = adv_data.reindex(index=all_dates, columns=holding_weights.columns).fillna(1.0).clip(lower=1.0)
        adv_adj = adv_adj * adv_multiplier  # Apply liquidity reduction
        participation = (trades * initial_capital) / adv_adj
        impact_costs = (impact_coeff * (participation ** 2)).sum(axis=1)

    net = gross - fixed_costs - impact_costs

    # Restrict to date window if specified
    if date_filter:
        s, e = date_filter
        net = net.loc[s:e]

    if len(net) < 30 or net.std() == 0:
        return {"sharpe": 0.0}

    sharpe = float((net.mean() / net.std()) * np.sqrt(252))
    return {"sharpe": sharpe}


def run_stress_tests(
    weight_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    regime_exposure: pd.Series,
    adv_data: pd.DataFrame,
    base_sharpe: float,
    transaction_cost: float = 0.0015,
    impact_coeff: float = 0.1,
    initial_capital: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Runs all 5 CQRO stress scenarios and reports Sharpe degradation.
    """
    logger.info("=" * 70)
    logger.info("STRESS TESTING SUITE (§VIII)")
    logger.info("=" * 70)

    close_prices = stock_panel["Close"].unstack(level=1).sort_index()
    stock_returns = close_prices.pct_change().fillna(0.0)

    scenarios = {
        "2x Transaction Cost":    dict(transaction_cost=transaction_cost * 2.0, impact_coeff=impact_coeff),
        "30% Liquidity Reduction": dict(transaction_cost=transaction_cost, impact_coeff=impact_coeff, adv_multiplier=0.70),
        "2008 Crash Replay":       dict(transaction_cost=transaction_cost, impact_coeff=impact_coeff,
                                        date_filter=(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-03-31"))),
        "2020 Crash Replay":       dict(transaction_cost=transaction_cost, impact_coeff=impact_coeff,
                                        date_filter=(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-31"))),
    }

    results = {}
    mandate_met = True

    for name, cfg in scenarios.items():
        res = _run_scenario_backtest(
            weight_df=weight_df,
            stock_returns=stock_returns,
            regime_exposure=regime_exposure,
            adv_data=adv_data,
            transaction_cost=cfg.get("transaction_cost", transaction_cost),
            impact_coeff=cfg.get("impact_coeff", impact_coeff),
            adv_multiplier=cfg.get("adv_multiplier", 1.0),
            date_filter=cfg.get("date_filter", None),
            initial_capital=initial_capital,
        )
        s = res["sharpe"]
        results[name] = s
        status = "✅ Resilient" if s >= 0.8 else "❌ Collapsed"
        if s < 0.8:
            mandate_met = False
        logger.info(f"  [{name}]: Sharpe = {s:.2f}  {status}")

    # Volatility Spike — use periods where realized vol > 90th pct
    port_ret = (weight_df.reindex(stock_returns.index).ffill().fillna(0.0).shift(1) * 
                stock_returns.reindex(columns=weight_df.columns).fillna(0.0)).sum(axis=1)
    vol_90th = port_ret.rolling(20).std().quantile(0.90)
    high_vol_dates = port_ret.index[port_ret.rolling(20).std() > vol_90th]
    if len(high_vol_dates) > 30:
        hv_ret = port_ret.loc[high_vol_dates]
        hv_sharpe = float((hv_ret.mean() / hv_ret.std()) * np.sqrt(252)) if hv_ret.std() > 0 else 0.0
        results["High-Vol Regime"] = hv_sharpe
        status = "✅ Resilient" if hv_sharpe >= 0.8 else "❌ Collapsed"
        if hv_sharpe < 0.8: mandate_met = False
        logger.info(f"  [High-Vol Regime]: Sharpe = {hv_sharpe:.2f}  {status}")

    logger.info(f"\n  Mandate (Sharpe ≥ 0.8 under all stress): {'✅ PASSED' if mandate_met else '❌ FAILED'}")
    return {"scenarios": results, "mandate_met": mandate_met}
