"""
QuantDeck — CQRO Institutional Alpha Engine
════════════════════════════════════════════════════════════════════════════════
Chief Quantitative Research Officer Mandate Implementation.

Execution Order:
  §I   Data Integrity & Leakage Control
  §II  Alpha Engine Design (Orthogonal & Regime-Aware)
  §III Model Training Protocol (Ensemble XGBoost, Walk-Forward, Overfitting Guard)
  §IV  Pure Alpha Validation (IC, t-stat, Decile Spread)
  §V   Portfolio Defensive Construction
  §VI  Transaction Cost & Execution Realism
  §VII Regime Robustness Testing
  §VIII Stress Testing
  §IX  Final Institutional Decision Matrix

Absolute Rule:
  Alpha quality precedes engineering.
  Capital preservation > headline Sharpe.
  Robustness > peak return.
"""

from __future__ import annotations
import logging
import warnings
import sys
import os

# Force UTF-8 output on Windows terminals
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Local imports ─────────────────────────────────────────────────────────────
from data_layer.universe   import get_universe, UniverseManager
from data_layer.ingestor   import YFinanceIngestor, MacroDataIngestor
from data_layer.storage    import ParquetCache
from feature_layer.implementations import (
    compute_stock_features,
    post_process_features,
    apply_cross_sectional_rank,
    apply_sector_neutralization,
    drop_highly_correlated_features,
    FEATURE_COLS,
)
from alpha_layer.targets              import build_target_panel, TARGET_COL
from alpha_layer.xgboost_trainer      import EnsembleAlphaModel
from alpha_layer.walk_forward         import WalkForwardEngine
from alpha_layer.pure_alpha_validator import evaluate_pure_alpha
from portfolio_layer.ranking     import CrossSectionalRanker
from portfolio_layer.optimizer   import PortfolioOptimizer
from risk_layer.regime_model     import compute_regime_exposure
from risk_layer.vol_targeting    import compute_vol_target_scalar
from risk_layer.regime_robustness import run_regime_robustness
from execution_layer.backtester  import Backtester
from execution_layer.stress_tester import run_stress_tests

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
START_DATE          = "2005-01-01"
TRAIN_WINDOW_YEARS  = 3        
REBALANCE_MONTHS    = 3        # Step 4 mandate: 6m -> 3m
REBALANCE_HORIZON   = 60       # Rank target horizon
TOP_N               = 45       # Baseline (dynamic override in Step 3)
BUFFER_N            = 65       # Baseline (dynamic override in Step 3)
INITIAL_CAPITAL     = 100_000.0
TARGET_VOL          = 0.14
TRANSACTION_COST    = 0.0015   
IMPACT_COEFF        = 0.1
IC_EXPOSURE_THRESH  = 0.03     
VOL_EXPOSURE_THRESH = 25.0     
TURNOVER_PENALTY    = 0.015    # Step 3 mandate

pd.set_option("display.max_columns", None)


# ══════════════════════════════════════════════════════════════════════════════
# §I — DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
def step1_fetch_data():
    logger.info("=" * 70)
    logger.info("§I — DATA FETCH & INTEGRITY CHECKS")
    logger.info("=" * 70)
    
    # ── Institutional Step: Clear existing caches for fresh feature engine ───
    cache_dir = Path("e:/Quantdeck01/data_cache")
    if cache_dir.exists():
        for f in cache_dir.glob("stock_*.parquet"): f.unlink()
        for f in cache_dir.glob("walk_forward_cache.pkl"): f.unlink()
        logger.info("  [Cache] Cleared stock and walk-forward caches for fresh run.")

    cache  = ParquetCache()
    ingest = YFinanceIngestor(cache=cache)
    macro  = MacroDataIngestor(cache=cache)

    tickers = get_universe()
    logger.info(f"Universe: {len(tickers)} tickers")

    stock_panel = ingest.fetch_daily_data(tickers, start_date=START_DATE)
    nifty_df    = macro.fetch_nifty50(start_date=START_DATE)
    vix_df      = macro.fetch_india_vix(start_date=START_DATE)

    # §I.6 — Log missing data events
    n_tickers    = stock_panel.index.get_level_values("Ticker").nunique()
    n_days_total = stock_panel.index.get_level_values("Date").nunique()
    missing_ct   = stock_panel["Close"].isna().sum()
    logger.info(f"Stock panel: {len(stock_panel):,} rows | {n_tickers} tickers | {n_days_total} days")
    if missing_ct > 0:
        logger.warning(f"  §I.6 Missing data: {missing_ct:,} NaN Close prices detected.")

    # §I.3 — Verify index alignment
    stock_dates = stock_panel.index.get_level_values("Date").unique()
    nifty_dates = nifty_df.index
    overlap = len(stock_dates.intersection(nifty_dates))
    if overlap < 100:
        raise ValueError(f"§I.3 ALIGNMENT MISMATCH: Only {overlap} common dates during fetch.")
    
    return stock_panel, nifty_df, vix_df


# ══════════════════════════════════════════════════════════════════════════════
# §II — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def step2_build_features(stock_panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("§II — ALPHA ENGINE DESIGN (Step 1-2 Upgrade)")
    logger.info("=" * 70)

    # ── Institutional Step: Quality Factors are already in stock_panel from ingestor ──
    tickers  = stock_panel.index.get_level_values(1).unique()
    univ_mgr = UniverseManager()
    sector_map = univ_mgr.get_sector_mapping()

    daily_rets = stock_panel["Close"].unstack().pct_change().fillna(0)
    sector_rets = pd.DataFrame(index=daily_rets.index)
    for sec in sorted(set(sector_map.values())):
        sec_t = [t for t, s in sector_map.items() if s == sec and t in daily_rets.columns]
        if sec_t:
            sector_rets[sec] = daily_rets[sec_t].mean(axis=1)

    logger.info(f"  Processing {len(tickers)} tickers with upgraded library factors...")
    feature_frames = []
    for ticker in tickers:
        try:
            tkr_df = stock_panel.xs(ticker, level="Ticker")
            if len(tkr_df.dropna(subset=["Close"])) < 252:
                continue

            sec         = sector_map.get(ticker, "Other")
            context_ret = sector_rets[sec] if sec in sector_rets.columns else daily_rets.mean(axis=1)
            
            # Pass full tkr_df to include ROE/ROA/Earnings_Growth
            feat = compute_stock_features(tkr_df, context_ret=context_ret)
            feat["Ticker"] = ticker
            feat = feat.set_index("Ticker", append=True)
            
            # Carry over data for backtest
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in tkr_df.columns:
                    feat[col] = tkr_df[col].values
            feature_frames.append(feat)
        except Exception as exc:
            logger.warning(f"  Skipping {ticker}: {exc}")

    panel = pd.concat(feature_frames).sort_index()
    
    # §II (post): Post-Process (Winsor/Z), Sector Neutralization & Orthogonalization
    panel = post_process_features(panel)
    panel = apply_sector_neutralization(panel, sector_map)
    panel = apply_cross_sectional_rank(panel)
    panel = drop_highly_correlated_features(panel, threshold=0.6) # Step 2 mandate
    
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# §III — WALK-FORWARD MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def step3_walk_forward(full_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 70)
    logger.info("§III — WALK-FORWARD TRAINING (Ensemble XGBoost, Overfitting Guard)")
    logger.info("=" * 70)

    features_cols = sorted([c for c in full_panel.columns if c in FEATURE_COLS])
    logger.info(f"  Institutional active features: {features_cols}")

    engine   = WalkForwardEngine(train_years=TRAIN_WINDOW_YEARS, rebalance_months=REBALANCE_MONTHS, horizon_days=60, embargo_days=10)
    ranker   = CrossSectionalRanker(top_n=TOP_N, buffer_n=BUFFER_N)
    opt      = PortfolioOptimizer()
    univ_mgr = UniverseManager()

    # ADV for liquidity cap
    adv_pivot = (full_panel["Close"] * full_panel["Volume"]).unstack(level=1)

    weight_schedule: dict = {}
    scores_history:  dict = {}
    current_holdings: set = set()
    prev_weights = pd.Series(dtype=float)
    overfitting_count = 0
    total_windows     = 0

    for i, (train_df, pred_date) in enumerate(engine.generate_splits(full_panel)):
        total_windows += 1
        X_train = train_df[features_cols].dropna()
        y_train = train_df.loc[X_train.index, TARGET_COL]

        if len(X_train) < 500:
            logger.warning(f"  Skipping {pred_date.date()}: insufficient training rows ({len(X_train)}).")
            continue

        # §III — Ensemble of 2 deterministic XGBoost models
        model = EnsembleAlphaModel(n_models=2)
        fit_res = model.fit(X_train, y_train, val_split=0.2)
        if fit_res.get("overfit_score", 0) > 0.05:
            overfitting_count += 1

        # §V — Signal Smoothing (5-day MA)
        all_dates = full_panel.index.get_level_values("Date").unique().sort_values()
        try:
            idx_rebal = all_dates.get_loc(pred_date)
        except KeyError:
            continue
        smooth_window = all_dates[max(0, idx_rebal - 4): idx_rebal + 1]

        monthly_scores_list = []
        for d in smooth_window:
            try:
                cs_feat = full_panel.xs(d, level="Date")[features_cols].dropna()
                pit_univ = univ_mgr.get_universe(d)
                cs_feat  = cs_feat.reindex([t for t in cs_feat.index if t in pit_univ])
                if not cs_feat.empty:
                    monthly_scores_list.append(model.predict(cs_feat))
            except KeyError:
                continue

        if not monthly_scores_list:
            continue
        scores = pd.concat(monthly_scores_list, axis=1).mean(axis=1)

        # §V — Portfolio construction with Hysteresis (Step 3: Top 20% Quintile)
        dynamic_top_n = max(10, int(len(scores) * 0.20))
        dynamic_buffer = int(dynamic_top_n * 1.4) # Preserve hysteresis
        
        # Override baseline configs with dynamic quintile logic
        ranker.top_n = dynamic_top_n
        ranker.buffer_n = dynamic_buffer
        
        new_portfolio = ranker.select_portfolio(scores, current_holdings, pred_date)
        adv_current   = adv_pivot.loc[pred_date] if pred_date in adv_pivot.index else None
        weights       = opt.equal_weight(new_portfolio, adv_data=adv_current, portfolio_value=INITIAL_CAPITAL)
        
        # Sector Constraint Implementation
        univ_mgr = UniverseManager()
        weights  = opt.sector_neutralize(weights, univ_mgr.get_sector_mapping(), univ_mgr.get_benchmark_sector_weights())
        
        weights       = opt.apply_turnover_penalty(prev_weights, weights, threshold=TURNOVER_PENALTY)

        weight_schedule[pred_date] = weights
        scores_history[pred_date]  = scores
        current_holdings           = new_portfolio
        prev_weights               = weights

    # §III — Overfitting audit summary
    if total_windows > 0:
        overfit_pct = overfitting_count / total_windows
        logger.info(f"\n  [Overfitting Audit] {overfitting_count}/{total_windows} windows triggered regularization ({overfit_pct:.1%})")
        if overfit_pct > 0.30:
            logger.warning(f"  WARNING: >30% windows overfitting. Base model complexity may be too high.")
        else:
            logger.info("  OK: Overfitting within acceptable bounds.")

    all_d = sorted(weight_schedule.keys())
    all_t = sorted(set().union(*[w.index for w in weight_schedule.values()]))
    weight_df = pd.DataFrame(index=all_d, columns=all_t, dtype=float).fillna(0.0)
    scores_df = pd.DataFrame(index=all_d, columns=all_t, dtype=float).fillna(0.0)
    for date in all_d:
        weight_df.loc[date, weight_schedule[date].index] = weight_schedule[date].values
        scores_df.loc[date, scores_history[date].index]  = scores_history[date].values

    return weight_df, scores_df


# ══════════════════════════════════════════════════════════════════════════════
# §VI — PRODUCTION BACKTEST (Full Cost Model)
# ══════════════════════════════════════════════════════════════════════════════
def step4_production_backtest(
    weight_df:       pd.DataFrame,
    stock_panel:     pd.DataFrame,
    nifty_df:        pd.DataFrame,
    vix_df:          pd.DataFrame,
    ic_series:       pd.Series,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("§VI — PRODUCTION BACKTEST (Full Cost & Execution Realism)")
    logger.info("=" * 70)

    close_prices  = stock_panel["Close"].unstack(level="Ticker").sort_index()
    stock_returns = close_prices.pct_change().fillna(0.0)
    regime_exposure = compute_regime_exposure(nifty_df, vix_df=vix_df)
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()

    # §I.3 — Verify alignment before backtesting
    common_dates = weight_df.index.intersection(stock_returns.index)
    if len(common_dates) < 10:
        raise ValueError(f"§I.3 ALIGNMENT MISMATCH: Only {len(common_dates)} common dates in weight↔returns.")
    logger.info(f"  [I.3] Weight <-> Returns date overlap: {len(common_dates)} dates OK")

    bt = Backtester(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        target_vol=TARGET_VOL,
        apply_vol_targeting=True,
    )
    results = bt.run_backtest(weight_df, stock_returns, regime_exposure, adv_data=adv_data, impact_coeff=IMPACT_COEFF)

    # §VI — Cost sanity check
    if results["ann_turnover"] > 1e-6 and results["ann_fixed_cost_bp"] < 1.0:
        raise ValueError("§VI COST BUG: Positive turnover detected but zero fixed cost reported.")

    eq = results["equity_curve"]
    logger.info(f"  CAGR             : {results['cagr']:.2%}")
    logger.info(f"  Net Sharpe       : {results['sharpe_ratio']:.2f}")
    logger.info(f"  Ann. Volatility  : {results['ann_vol']:.2%}")
    logger.info(f"  Max Drawdown     : {results['max_drawdown']:.2%}")
    logger.info(f"  Ann. Turnover    : {results['ann_turnover']:.1%}")
    logger.info(f"  Fixed Cost       : {results['ann_fixed_cost_bp']:.1f} bps/yr")
    logger.info(f"  Impact Cost      : {results['ann_impact_cost_bp']:.1f} bps/yr")
    logger.info(f"  Total Cost Drag  : {results['ann_fixed_cost_bp'] + results['ann_impact_cost_bp']:.1f} bps/yr")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# §IX — INSTITUTIONAL DECISION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def step5_decision_matrix(
    alpha_stats:   Dict[str, Any],
    prod_results:  Dict[str, Any],
    regime_res:    Dict[str, Any],
    stress_res:    Dict[str, Any],
):
    logger.info("=" * 70)
    logger.info("§IX — FINAL INSTITUTIONAL DECISION MATRIX")
    logger.info("=" * 70)

    ic_mean     = alpha_stats["ic_mean"]
    ic_tstat    = alpha_stats["ic_tstat"]
    net_sharpe  = prod_results["sharpe_ratio"]
    turnover    = prod_results["ann_turnover"]
    cost_drag   = prod_results["ann_fixed_cost_bp"] + prod_results["ann_impact_cost_bp"]
    max_dd      = abs(prod_results["max_drawdown"])

    checks = {
        "IC Mean > 0.035":           ic_mean > 0.035,
        "IC t-stat > 2.0":           ic_tstat > 2.0,
        "Net Sharpe > 1.5":          net_sharpe > 1.5,
        "Turnover < 400%":           (turnover * 100) < 400,
        "Cost Drag Realistic":       0 < cost_drag < 500,
        "Sub-period Stable (≥ 2/3)": regime_res["mandate_met"],
        "Stress Resilient":          stress_res["mandate_met"],
        "Max DD < 20%":              max_dd < 0.20,
    }

    passed = sum(checks.values())
    total  = len(checks)

    logger.info("\n" + "═" * 60)
    logger.info("   CQRO INSTITUTIONAL DECISION MATRIX")
    logger.info("═" * 60)
    for label, result in checks.items():
        mark = "✅ PASS" if result else "❌ FAIL"
        print(f"  {label:<35}: {mark}")
    print("─" * 60)
    print(f"  Score: {passed}/{total}")
    print("─" * 60)

    if passed == total:
        verdict = "[A] INSTITUTIONAL GRADE - DEPLOYMENT ELIGIBLE"
    elif passed >= total - 2:
        verdict = "[B] RESEARCH GRADE - REFINEMENT REQUIRED"
    else:
        verdict = "[C] STATISTICALLY WEAK - REBUILD ALPHA"

    print(f"\n  VERDICT: {verdict}")
    print("═" * 60 + "\n")

    # Key metrics summary
    print("  KEY METRICS SUMMARY")
    print(f"  {'IC Mean':<30}: {ic_mean:.4f}")
    print(f"  {'IC t-stat':<30}: {ic_tstat:.2f}")
    print(f"  {'% Positive IC':<30}: {alpha_stats['pct_positive']:.1%}")
    print(f"  {'Decile Spread Sharpe':<30}: {alpha_stats['decile_sharpe']:.2f}")
    print(f"  {'Net Sharpe (Full System)':<30}: {net_sharpe:.2f}")
    print(f"  {'Annual Turnover':<30}: {turnover:.1%}")
    print(f"  {'Annual Cost Drag':<30}: {cost_drag:.1f} bps")
    print(f"  {'Max Drawdown':<30}: {max_dd:.2%}")
    print(f"  {'Profitable Sub-periods':<30}: {regime_res['profitable_periods']}/3")
    print(f"  {'Bull Regime IC':<30}: {regime_res['bull_ic']:+.4f}")
    print(f"  {'Bear Regime IC':<30}: {regime_res['bear_ic']:+.4f}")
    print(f"  {'High-Vol Regime IC':<30}: {regime_res['highvol_ic']:+.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def step6_save_reports(prod_results: Dict[str, Any]):
    Path("reports").mkdir(exist_ok=True)

    eq = prod_results["equity_curve"]
    eq.plot(title="QuantDeck CQRO — Net Equity Curve (2005–2026)", grid=True, figsize=(14, 6))
    plt.tight_layout()
    plt.savefig("reports/cqro_equity_curve.png", dpi=150)
    plt.close()

    # Monthly returns table
    monthly = prod_results["monthly_returns"]["Monthly Return"]
    monthly_pct = monthly.map(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Ret":   monthly_pct.values,
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Ret").fillna("")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    logger.info("\n" + "─" * 70)
    logger.info("MONTHLY RETURNS TABLE")
    logger.info("─" * 70)
    logger.info("\n" + pivot.to_string())

    logger.info("\n  Reports saved to ./reports/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("QuantDeck — CQRO Institutional Alpha Engine Starting")
    logger.info("=" * 70)

    # # Clear old weight cache (force full re-run with new features)
    # cache_f = Path("data_cache/weight_schedule_latest.parquet")
    # if cache_f.exists():
    #     cache_f.unlink()
    #     logger.info("  Cleared weight schedule cache → full re-run.")

    # §I   Data
    stock_panel, nifty_df, vix_df = step1_fetch_data()

    # §II  Features
    feature_panel = step2_build_features(stock_panel)
    full_panel    = build_target_panel(feature_panel, price_col="Close", horizon=REBALANCE_HORIZON)

    # §III Walk-Forward (with Persistence)
    cache_path = Path("data_cache/walk_forward_cache.pkl")
    if cache_path.exists():
        logger.info(">>> LOADING CACHED WALK-FORWARD RESULTS (Step III Skip) <<<")
        import pickle
        with open(cache_path, "rb") as f:
            weight_df, scores_df = pickle.load(f)
    else:
        weight_df, scores_df = step3_walk_forward(full_panel)
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump((weight_df, scores_df), f)
        logger.info(f"  Step III results cached to {cache_path}")

    # §IV  Pure Alpha Validation (BEFORE any risk overlays)
    alpha_stats = evaluate_pure_alpha(scores_df, stock_panel, transaction_cost=TRANSACTION_COST, initial_capital=INITIAL_CAPITAL)

    # §V/VI Production Backtest
    prod_results = step4_production_backtest(weight_df, stock_panel, nifty_df, vix_df, alpha_stats.get("ic_series", pd.Series()))

    # §VII Regime Robustness
    regime_exposure = compute_regime_exposure(nifty_df, vix_df=vix_df)
    regime_res = run_regime_robustness(
        ic_series      = alpha_stats.get("ic_series", pd.Series(dtype=float)),
        equity_curve   = prod_results["equity_curve"],
        daily_returns  = prod_results["daily_returns"],
        nifty_df       = nifty_df,
        regime_exposure= regime_exposure,
    )

    # §VIII Stress Tests
    adv_data = (stock_panel["Close"] * stock_panel["Volume"]).unstack(level=1).rolling(20, min_periods=5).mean().ffill()
    stress_res = run_stress_tests(
        weight_df       = weight_df,
        stock_panel     = stock_panel,
        regime_exposure = regime_exposure,
        adv_data        = adv_data,
        base_sharpe     = prod_results["sharpe_ratio"],
        transaction_cost= TRANSACTION_COST,
        impact_coeff    = IMPACT_COEFF,
        initial_capital = INITIAL_CAPITAL,
    )

    # §IX  Decision Matrix
    step5_decision_matrix(alpha_stats, prod_results, regime_res, stress_res)

    # Save reports
    step6_save_reports(prod_results)

    logger.info("CQRO Engine Run Complete.")


if __name__ == "__main__":
    main()
