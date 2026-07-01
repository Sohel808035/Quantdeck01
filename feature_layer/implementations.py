"""
feature_layer/implementations.py  (CQRO Mandate — Full Orthogonal v4)
════════════════════════════════════════════════════════════════════════
Implements all 5 orthogonal alpha families per the CQRO mandate:

  1. Residual Momentum         — Beta-adjusted sector residual
  2. Volatility Structure      — 20D/120D ratio, Downside Deviation Ratio
  3. Distribution Shape        — 60D Skewness, 60D Kurtosis
  4. Liquidity Dynamics        — Volume Shock, Turnover Ratio
  5. Trend Persistence         — Rolling Sharpe, Breakout Intensity

Data Integrity:
  - All features lagged 1 day (shift+1) → zero look-ahead
  - context_ret passed externally for Residual Momentum
  - Missing data events are logged
  i have made the self healing system to heal its pipeline also 
  
"""

from __future__ import annotations
import logging
from typing import Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_stock_features(
    df: pd.DataFrame,
    context_ret: Optional[pd.Series] = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    V5 Institutional Alpha Library:
    Implements momentum, mean reversion, liquidity, volatility, and quality factors.
    """
    result = pd.DataFrame(index=df.index)
    px = df[price_col].copy()
    daily_ret = px.pct_change()

    # 1. Momentum Factors
    result["return_1m"] = px / px.shift(21) - 1
    result["return_3m"] = px / px.shift(63) - 1
    result["return_6m"] = px / px.shift(126) - 1
    
    # Residual Momentum (Beta-neutralized)
    if context_ret is not None:
        y = daily_ret.dropna()
        x = context_ret.reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) > 60:
            y_c, x_c = y.loc[common], x.loc[common]
            roll_cov = y_c.rolling(60).cov(x_c)
            roll_var = x_c.rolling(60).var()
            beta = (roll_cov / roll_var.replace(0, np.nan)).fillna(1.0)
            mkt_ret_1m = context_ret.rolling(21).sum().reindex(result.index).ffill()
            result["residual_momentum"] = result["return_1m"] - (beta.reindex(result.index).ffill() * mkt_ret_1m)
        else:
            result["residual_momentum"] = result["return_1m"]
    else:
        result["residual_momentum"] = result["return_1m"]

    # 2. Mean Reversion Factors
    result["return_5d"]  = px / px.shift(5) - 1
    result["return_10d"] = px / px.shift(10) - 1
    ma20  = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    result["bollinger_distance"] = (px - ma20) / std20.replace(0, np.nan)

    # 3. Liquidity Factors
    if "Volume" in df.columns:
        vol20 = df["Volume"].rolling(20).mean()
        result["volume_shock"] = df["Volume"] / vol20.replace(0, np.nan)
        # Amihud Illiquidity: |Ret| / Volume
        result["amihud_illiquidity"] = daily_ret.abs() / (df["Volume"] * px).replace(0, np.nan)
    else:
        result["volume_shock"] = np.nan
        result["amihud_illiquidity"] = np.nan

    # 4. Volatility Factors
    std20_ret = daily_ret.rolling(20).std()
    std60_ret = daily_ret.rolling(60).std()
    result["volatility_regime"] = std20_ret / std60_ret.replace(0, np.nan)
    
    # Idiosyncratic Volatility (Residual volatility)
    if context_ret is not None:
        # Standard proxy: sqrt(Var(stock) - Beta^2 * Var(market))
        y_var = daily_ret.rolling(60).var()
        x_var = context_ret.reindex(daily_ret.index).rolling(60).var()
        # beta already calculated above for same window? No, let's recalibrate
        common_v = daily_ret.dropna().index.intersection(context_ret.dropna().index)
        if len(common_v) > 60:
            y_v, x_v = daily_ret.loc[common_v], context_ret.loc[common_v]
            b_v = (y_v.rolling(60).cov(x_v) / x_v.rolling(60).var().replace(0, np.nan)).fillna(1.0)
            b_v = b_v.reindex(result.index).ffill()
            resid_var = y_var - (b_v**2 * x_var.reindex(result.index))
            result["idiosyncratic_volatility"] = np.sqrt(resid_var.clip(lower=0))
        else:
            result["idiosyncratic_volatility"] = std20_ret
    else:
        result["idiosyncratic_volatility"] = std20_ret

    # 5. Quality Factors (from ingestor)
    for qf in ["ROE", "ROA", "Earnings_Growth"]:
        if qf in df.columns:
            result[qf.lower()] = df[qf]
        else:
            result[qf.lower()] = np.nan

    # ── LAG EVERYTHING 1 DAY (Rule §I.2) ─────────────────────────
    result = result.shift(1)
    return result

# ════════════════════════════════════════════════════════════════════════
# Post-Processing: Winsorize & Z-Score
# ════════════════════════════════════════════════════════════════════════

def post_process_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    V5 Institutional: Winsorize at 1% tails and Z-score daily.
    Vectorized replacement for slow groupby.apply structure.
    """
    logger.info("Applying winsorization (1%) and daily Z-scoring...")
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    processed = panel.copy()
    
    # Process each feature column using vectorized groupby transform for speed
    for col in feats:
        # Winsorize: cross-sectional limits per date
        lower = processed[col].groupby("Date").transform(lambda x: x.quantile(0.01))
        upper = processed[col].groupby("Date").transform(lambda x: x.quantile(0.99))
        processed[col] = processed[col].clip(lower, upper)
        
        # Z-Score: cross-sectional normalization
        mu = processed[col].groupby("Date").transform("mean")
        sigma = processed[col].groupby("Date").transform("std")
        # Shift sigma slightly to avoid division by zero
        processed[col] = (processed[col] - mu) / sigma.replace(0, 1e-6)
        
    return processed

# ════════════════════════════════════════════════════════════════════════
# Feature Registry
# ════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "return_1m", "return_3m", "return_6m", "residual_momentum",
    "return_5d", "return_10d", "bollinger_distance",
    "volume_shock", "amihud_illiquidity",
    "idiosyncratic_volatility", "volatility_regime",
    "roe", "roa", "earnings_growth"
]

DESCENDING_FEATURES = [
    "return_5d", "return_10d", "bollinger_distance", 
    "amihud_illiquidity", "idiosyncratic_volatility", "volatility_regime"
]

def apply_cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Applies [0, 1] percentile ranking to features.
    """
    logger.info("Applying cross-sectional percentile ranking...")
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    ranked = panel.copy()

    for col in feats:
        ascending = col not in DESCENDING_FEATURES
        wide = panel[col].unstack(level="Ticker")
        wide_ranked = wide.rank(axis=1, pct=True, ascending=ascending)
        ranked[col] = wide_ranked.stack(future_stack=True).reindex(panel.index)

    return ranked

def apply_sector_neutralization(panel: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """
    Demean features within sectors.
    """
    logger.info("Applying institutional sector neutralization...")
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    panel["Sector"] = panel.index.get_level_values("Ticker").map(sector_map).fillna("Other")
    
    for col in feats:
        panel[col] = panel[col] - panel.groupby(["Date", "Sector"])[col].transform("mean")
    
    return panel.drop(columns=["Sector"])

def drop_highly_correlated_features(panel: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    V4 Institutional Orthogonalization:
    Drops redundant features but ensures at least one member of each family survives.
    """
    families = {
        "Momentum": ["return_1m", "return_3m", "return_6m", "residual_momentum"],
        "Reversion": ["return_5d", "return_10d", "bollinger_distance"],
        "Liquidity": ["volume_shock", "amihud_illiquidity"],
        "Volatility": ["idiosyncratic_volatility", "volatility_regime"],
        "Quality": ["roe", "roa", "earnings_growth"]
    }
    
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    # Filter only features that have at least some data to avoid empty dropna()
    feats = [c for c in feats if panel[c].notna().any()]
    
    if len(panel) < 500 or not feats: 
        return panel
    
    # Only drop rows where these specific features exist
    available_data = panel[feats].dropna()
    if len(available_data) == 0:
        logger.warning("  [Orthogonalization] No rows with complete feature data. Skipping correlation drop.")
        return panel
        
    sample = available_data.sample(min(50_000, len(available_data)), random_state=42)
    corr = sample.corr().abs()
    
    to_drop = set()
    protected = set()
    
    # Prioritize keeping one from each family
    for family, members in families.items():
        found = [m for m in members if m in corr.columns]
        if found:
            protected.add(found[0]) # Protect the first one (primary signal)
            
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    for col in upper.columns:
        if any(upper[col] > threshold):
            if col not in protected:
                to_drop.add(col)
    
    if to_drop:
        logger.info(f"  [Orthogonalization] Dropping {len(to_drop)} redundant features (threshold={threshold}): {list(to_drop)}")
        return panel.drop(columns=list(to_drop))
    
    return panel
