"""
alpha_layer/pure_alpha_validator.py  (CQRO Mandate §IV)
═══════════════════════════════════════════════════════════
Pure Alpha Validation — BEFORE any risk overlays.

Computes:
  1. Daily cross-sectional Spearman IC
  2. IC mean, IC t-stat, % positive IC days
  3. Rolling 6M IC
  4. Decile spread portfolio (L10 - S10, monthly rebalance, 15bps cost)
  5. Flags model as "statistically weak" if IC < 0.03 or t-stat < 2
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_daily_ic(
    scores_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    forward_days: int = 21,
    price_col: str = "Close",
) -> pd.Series:
    """
    Computes the daily cross-sectional Spearman IC.
    scores_df: (Date × Ticker) DataFrame of predicted scores
    Returns: Series indexed by Date
    """
    logger.info(f"  Computing daily IC (horizon={forward_days}D)...")
    close = stock_panel[price_col].unstack(level=1).sort_index()
    fwd_ret = (close.shift(-forward_days) / close - 1).dropna(how="all")

    ic_series = {}
    for date in scores_df.index:
        if date not in fwd_ret.index:
            continue
        scores = scores_df.loc[date].dropna()
        rets   = fwd_ret.loc[date].dropna()
        common = scores.index.intersection(rets.index)
        if len(common) < 20:
            continue
        ic, _ = stats.spearmanr(scores[common], rets[common])
        if not np.isnan(ic):
            ic_series[date] = ic

    return pd.Series(ic_series, name="IC")


def evaluate_pure_alpha(
    scores_df: pd.DataFrame,
    stock_panel: pd.DataFrame,
    transaction_cost: float = 0.0015,
    initial_capital: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Full Pure Alpha evaluation per CQRO §IV mandate.
    Returns dict with IC stats + decile spread metrics.
    """
    logger.info("=" * 70)
    logger.info("PURE ALPHA VALIDATION (§IV — Pre-Overlay)")
    logger.info("=" * 70)

    # ── 1. Daily IC ──────────────────────────────────────────────────────────
    ic = compute_daily_ic(scores_df, stock_panel, forward_days=21)

    if ic.empty:
        logger.error("  IC computation returned empty Series. Check score/price alignment.")
        return {"ic_mean": 0, "ic_tstat": 0, "pct_positive": 0, "weak_alpha": True}

    ic_mean    = float(ic.mean())
    ic_std     = float(ic.std())
    ic_tstat   = float((ic_mean / (ic_std / np.sqrt(len(ic)))) if ic_std > 0 else 0.0)
    pct_pos    = float((ic > 0).mean())
    rolling_6m = ic.rolling(126, min_periods=63).mean()

    logger.info(f"  IC Mean          : {ic_mean:.4f}  (bar: > 0.030)")
    logger.info(f"  IC Std Dev       : {ic_std:.4f}")
    logger.info(f"  IC t-stat        : {ic_tstat:.2f}   (bar: > 2.0)")
    logger.info(f"  % Positive IC    : {pct_pos:.1%}   (bar: > 55%)")

    # ── 2. Sub-period IC breakdown ───────────────────────────────────────────
    sub_periods = {
        "2005-2012": (pd.Timestamp("2005-01-01"), pd.Timestamp("2012-12-31")),
        "2013-2018": (pd.Timestamp("2013-01-01"), pd.Timestamp("2018-12-31")),
        "2019-2026": (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-12-31")),
    }
    sub_ic = {}
    profitable_periods = 0
    for name, (s, e) in sub_periods.items():
        sub = ic.loc[s:e]
        m = float(sub.mean()) if not sub.empty else 0.0
        sub_ic[name] = m
        if m > 0: profitable_periods += 1
        logger.info(f"  Sub-period IC [{name}]: {m:.4f}")

    # ── 3. Decile Spread (L/S) ───────────────────────────────────────────────
    logger.info("  Computing Decile Spread Portfolio (Long Top 10% / Short Bottom 10%)...")
    close = stock_panel["Close"].unstack(level=1).sort_index()
    rets  = close.pct_change().fillna(0.0)

    port_returns = []
    prev_long_w  = pd.Series(dtype=float)
    prev_short_w = pd.Series(dtype=float)

    rebal_dates = scores_df.index

    for i, date in enumerate(rebal_dates):
        if date not in rets.index:
            continue
        scores = scores_df.loc[date].dropna()
        if scores.empty or len(scores) < 20:
            continue

        n_dec = max(1, int(len(scores) * 0.10))
        long_tickers  = scores.nlargest(n_dec).index.tolist()
        short_tickers = scores.nsmallest(n_dec).index.tolist()

        # Equal weight within each leg
        long_w  = pd.Series(1.0 / n_dec, index=long_tickers)
        short_w = pd.Series(1.0 / n_dec, index=short_tickers)

        # Next rebalance date for holding period
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rets.index[-1]
        hold_rets = rets.loc[date:next_date]

        for _, day_ret in hold_rets.iterrows():
            long_r  = (long_w  * day_ret.reindex(long_tickers).fillna(0)).sum()
            short_r = (short_w * day_ret.reindex(short_tickers).fillna(0)).sum()
            spread_r = long_r - short_r
            port_returns.append(spread_r)

        # Turnover cost on rebalance
        long_to  = (long_w.reindex(prev_long_w.index.union(long_w.index)).fillna(0) -
                    prev_long_w.reindex(prev_long_w.index.union(long_w.index)).fillna(0)).abs().sum()
        port_returns[-1] -= (long_to + 0.02) * transaction_cost  # crude cost deduction
        prev_long_w  = long_w
        prev_short_w = short_w

    if not port_returns:
        logger.warning("  Decile spread computation failed — no returns generated.")
        decile_sharpe = 0.0
        decile_maxdd  = 0.0
    else:
        pr = pd.Series(port_returns)
        decile_sharpe = float((pr.mean() / pr.std()) * np.sqrt(252)) if pr.std() > 0 else 0.0
        equity = (1 + pr).cumprod()
        decile_maxdd  = float(((equity - equity.cummax()) / equity.cummax()).min())

    logger.info(f"  Decile Spread Sharpe : {decile_sharpe:.2f}  (bar: > 0.9)")
    logger.info(f"  Decile Spread Max DD : {decile_maxdd:.2%}")

    # ── 4. Weakness Flag ─────────────────────────────────────────────────────
    weak_alpha = (ic_mean < 0.03) or (ic_tstat < 2.0)
    if weak_alpha:
        logger.warning("  ⚠️  WEAK ALPHA SIGNAL: IC or t-stat below institutional threshold.")
        logger.warning("  ⚠️  Risk overlays WILL NOT compensate for poor cross-sectional alpha.")
    else:
        logger.info("  ✅  Alpha signal is statistically significant.")

    return {
        "ic_series":        ic,
        "ic_mean":          ic_mean,
        "ic_std":           ic_std,
        "ic_tstat":         ic_tstat,
        "pct_positive":     pct_pos,
        "rolling_6m_ic":    rolling_6m,
        "sub_period_ic":    sub_ic,
        "profitable_periods": profitable_periods,
        "decile_sharpe":    decile_sharpe,
        "decile_maxdd":     decile_maxdd,
        "weak_alpha":       weak_alpha,
    }
