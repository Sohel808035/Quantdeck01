"""
risk_layer/regime_robustness.py  (CQRO Mandate §VII)
═══════════════════════════════════════════════════════
Regime Robustness Testing.

Segments full backtest into three structural periods:
  - 2005–2012  (Crash + Recovery)
  - 2013–2018  (Bull + EM Stress)
  - 2019–2026  (COVID + Rate Shock)

For each: IC mean, Sharpe, Max DD.
Mandate: profitable in at least 2 of 3 periods.

Also computes Bull / Bear / High-Vol IC.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SUB_PERIODS: Dict[str, Tuple[str, str]] = {
    "2005-2012": ("2005-01-01", "2012-12-31"),
    "2013-2018": ("2013-01-01", "2018-12-31"),
    "2019-2026": ("2019-01-01", "2026-12-31"),
}


def run_regime_robustness(
    ic_series: pd.Series,
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    nifty_df: pd.DataFrame,
    regime_exposure: pd.Series,
) -> Dict[str, Any]:
    """
    Full regime robustness diagnostic per CQRO §VII.
    """
    logger.info("=" * 70)
    logger.info("REGIME ROBUSTNESS TESTING (§VII)")
    logger.info("=" * 70)

    sub_results = {}
    profitable_periods = 0

    for name, (s, e) in SUB_PERIODS.items():
        sub_ic  = ic_series.loc[s:e]
        sub_ret = daily_returns.loc[s:e]
        sub_eq  = equity_curve.loc[s:e]

        ic_mean = float(sub_ic.mean()) if not sub_ic.empty else 0.0

        if sub_ret.empty or sub_ret.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float((sub_ret.mean() / sub_ret.std()) * np.sqrt(252))

        if sub_eq.empty:
            maxdd = 0.0
        else:
            rolling_max = sub_eq.cummax()
            maxdd = float(((sub_eq - rolling_max) / rolling_max).min())

        if sharpe > 0:
            profitable_periods += 1

        sub_results[name] = {"ic_mean": ic_mean, "sharpe": sharpe, "max_dd": maxdd}
        logger.info(
            f"  [{name}]  IC: {ic_mean:+.4f}  |  Sharpe: {sharpe:.2f}  |  Max DD: {maxdd:.2%}"
        )

    mandate_met = profitable_periods >= 2
    logger.info(
        f"\n  Profitable periods: {profitable_periods}/3  "
        f"({'✅ Mandate Met' if mandate_met else '❌ Mandate Breached'})"
    )

    # ── Bull / Bear / High-Vol IC ────────────────────────────────────────────
    bull_mask = regime_exposure.reindex(ic_series.index).ffill() == 1.0
    bear_mask = ~bull_mask

    nifty_px = nifty_df["Close"]
    vol_20 = nifty_px.pct_change().rolling(20).std() * np.sqrt(252)
    vol_90th = vol_20.quantile(0.90)
    hv_mask = vol_20.reindex(ic_series.index).ffill() > vol_90th

    bull_ic = float(ic_series[bull_mask].mean()) if bull_mask.any() else 0.0
    bear_ic = float(ic_series[bear_mask].mean()) if bear_mask.any() else 0.0
    hv_ic   = float(ic_series[hv_mask].mean())   if hv_mask.any()   else 0.0

    logger.info(f"\n  Regime-Segmented IC:")
    logger.info(f"    Bull regime  IC: {bull_ic:+.4f}")
    logger.info(f"    Bear regime  IC: {bear_ic:+.4f}")
    logger.info(f"    High-vol     IC: {hv_ic:+.4f}")

    return {
        "sub_period_results": sub_results,
        "profitable_periods": profitable_periods,
        "mandate_met":        mandate_met,
        "bull_ic":            bull_ic,
        "bear_ic":            bear_ic,
        "highvol_ic":         hv_ic,
    }
