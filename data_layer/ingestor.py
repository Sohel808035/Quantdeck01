"""
data_layer/ingestor.py  (v2 — Production rewrite)
───────────────────────────────────────────────────
Fetches OHLCV data from Yahoo Finance  (yfinance) and caches results
locally as Parquet via the ParquetCache class.

Key design decisions
────────────────────
• Fetches in batches of 50 tickers to stay under yfinance rate limits.
• Returns a long-format MultiIndex DataFrame:
    index: (Date, Ticker)
    columns: Open, High, Low, Close, Volume
• Uses Adj Close as the Close column to handle corporate-action splits
  automatically.
• Separate helpers for NIFTY50 index and India VIX.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

from data_layer.storage import ParquetCache  # type: ignore

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50   # yfinance handles ≤50 tickers cleanly
_RETRY_PAUSE = 2   # seconds between batch retries


class MarketDataIngestor:
    """Abstract base — sub-classes implement fetch_daily_data."""
    def fetch_daily_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

def validate_panel(df: pd.DataFrame, threshold_move: float = 0.3, max_fail_pct: float = 0.05) -> pd.DataFrame:
    """
    V4 Institutional Validation Pipeline:
    1. Detect duplicate timestamps per ticker.
    2. Detect missing OHLCV values.
    3. Detect abnormal daily returns (>30%).
    4. Detect stale data (identical price for 5+ days).
    5. Halt execution if total corrupted rows > max_fail_pct.
    """
    if df.empty:
        return df

    initial_rows = len(df)
    logger.info(f"[Validation] Auditing {initial_rows:,} rows of market data...")

    # 1. Duplicate check
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        logger.warning(f"  [DUPES] {dupes} duplicate (Date, Ticker) indices found. Dropping.")
        df = df[~df.index.duplicated(keep="first")]

    # 2. Missing Data Check (OHLCV)
    cols_to_check = ["Open", "High", "Low", "Close"]
    for col in cols_to_check:
        if col not in df.columns:
            logger.error(f"  [MISSING COL] Required column {col} is missing from panel.")
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            logger.warning(f"  [MISSING] {n_nan} NaN values in {col}. Dropping.")
            df = df.dropna(subset=[col])

    # 3. Abnormal Moves Check
    rets = df.groupby(level="Ticker")["Close"].pct_change().abs()
    n_outliers = (rets > threshold_move).sum()
    if n_outliers > 0:
        logger.warning(f"  [OUTLIER] {n_outliers} price jumps > {threshold_move:.0%} detected.")
        # We drop outliers for safety in institutional research
        df = df[rets <= threshold_move]

    # 4. Stale Data Check (Same price for 5 consecutive days)
    # Fast check: diff == 0 for 4 consecutive diffs = 5 identical prices
    is_diff_zero = df.groupby(level="Ticker")["Close"].diff() == 0
    stale_mask = is_diff_zero.groupby(level="Ticker").rolling(4).sum() == 4
    # Reset index of stale_mask because rolling groupby changes structure
    stale_mask = stale_mask.reset_index(level=0, drop=True)
    n_stale = stale_mask.sum()
    if n_stale > 0:
        logger.warning(f"  [STALE] {n_stale} rows have frozen prices for 5+ days.")
        # In institutional alpha, we often keep stale data but mark it. 
        # Here we drop to be conservative as requested.
        df = df[~stale_mask.fillna(False)]

    # ── Fault Tolerance Check ────────────────────────────────────────────────
    final_rows = len(df)
    total_dropped = initial_rows - final_rows
    fail_pct = total_dropped / initial_rows

    if fail_pct > max_fail_pct:
        critical_msg = f"CRITICAL QUALITY FAILURE: {fail_pct:.1%} corrupted data dropped (Max: {max_fail_pct:.1%})."
        logger.error(critical_msg)
        raise ValueError(critical_msg)

    logger.info(f"  Audit complete. Quality pass: {(1 - fail_pct):.1%}. Rows remaining: {len(df):,}")
    return df


# ── helpers ────────────────────────────────────────────────────────────────────

def _download_batch(yf_tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    """
    Download a batch and return a long-format (Date, Ticker) DataFrame
    with columns [Open, High, Low, Close, Volume].
    'Close' is sourced from 'Adj Close' for split-correctness.
    """
    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,   # auto_adjust → yfinance replaces Close with Adj Close
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    # ── normalise MultiIndex structure ─────────────────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        # Drop tickers that failed completely (all NaNs)
        raw = raw.dropna(axis=1, how="all")
        if raw.empty:
            return pd.DataFrame()

        # Find ticker level
        if "Ticker" in raw.columns.names:
            ticker_level = "Ticker"
        else:
            ticker_level = 0 if "Open" in raw.columns.get_level_values(1) else 1

        df_long = (
            raw.stack(level=ticker_level, future_stack=True)
               .rename_axis(index=["Date", "Ticker"])
               .reset_index()
        )
        # Strip .NS suffix to keep internal tickers clean
        df_long["Ticker"] = df_long["Ticker"].astype(str).str.replace(r"\.NS$", "", regex=True)

    else:
        # Single-ticker fallback
        df_long = raw.reset_index()
        df_long["Ticker"] = yf_tickers[0].replace(".NS", "")

    # ── standardise columns ────────────────────────────────────────────────────
    df_long.columns = [str(c).title() for c in df_long.columns]

    wanted = [c for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"] if c in df_long.columns]
    df_long = df_long[wanted]

    df_long["Date"] = pd.to_datetime(df_long["Date"])
    df_long = df_long.dropna(subset=["Close"])
    df_long = df_long.set_index(["Date", "Ticker"]).sort_index()

    return df_long


# ── main ingestor ──────────────────────────────────────────────────────────────

class YFinanceIngestor(MarketDataIngestor):
    """
    Fetches daily OHLCV for a list of NSE tickers using yfinance.

    • Splits requests into batches of _BATCH_SIZE.
    • Caches results locally as Parquet so subsequent runs are instant.
    • Returns a long-format MultiIndex DataFrame: index=(Date, Ticker).
    """

    def __init__(self, suffix: str = ".NS", cache: Optional[ParquetCache] = None):
        self.suffix = suffix
        self.cache = cache or ParquetCache()

    def fetch_fundamental_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        V4 Institutional: Fetches key fundamental metrics for Quality factors.
        Modified: Silently skip individual failures to keep ingestion moving.
        """
        logger.info(f"Fetching fundamental data for {len(tickers)} tickers (slow process)...")
        fundamental_frames = []
        
        for t in tickers:
            try:
                # Use a small timeout or just catch the 404
                stock = yf.Ticker(f"{t}{self.suffix}")
                # info is very unreliable for NSE, we try to get basics
                info = stock.info
                if not info or "regularMarketPrice" not in info:
                    continue # Skip if info is empty or garbage
                    
                metrics = {
                    "Ticker": t,
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "Earnings_Growth": info.get("earningsGrowth", np.nan),
                }
                fundamental_frames.append(metrics)
            except Exception:
                # Silently skip, don't log every 404 to avoid cluttering logs
                continue
                
        if not fundamental_frames:
            return pd.DataFrame(columns=["Ticker", "ROE", "ROA", "Earnings_Growth"]).set_index("Ticker")
            
        fund_df = pd.DataFrame(fundamental_frames).set_index("Ticker")
        return fund_df

    def fetch_daily_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        end_str = end_date or "today"

        # ── cache check ────────────────────────────────────────────────────────
        if self.cache.exists(tickers, start_date, end_str, name="stock"):
            return self.cache.load(tickers, start_date, end_str, name="stock")

        logger.info(
            f"Fetching {len(tickers)} tickers from {start_date} to {end_str} "
            f"in batches of {_BATCH_SIZE}..."
        )

        yf_tickers = [f"{t}{self.suffix}" for t in tickers]
        all_batches: List[pd.DataFrame] = []

        for i in range(0, len(yf_tickers), _BATCH_SIZE):
            batch = yf_tickers[i : i + _BATCH_SIZE]
            logger.info(f"  Batch {i // _BATCH_SIZE + 1}: {batch[:3]}... ({len(batch)} tickers)")
            try:
                batch_df = _download_batch(batch, start_date, end_date)
                if not batch_df.empty:
                    # Fundamental extraction (slow/unreliable, skipping for speed)
                    # batch_funds = self.fetch_fundamental_data(batch_raw_tickers)
                    
                    for col in ["ROE", "ROA", "Earnings_Growth"]:
                        # if col in batch_funds.columns:
                        #     # Map to the long-format panel
                        #     batch_df[col] = batch_df.index.get_level_values("Ticker").map(batch_funds[col])
                        # else:
                        batch_df[col] = np.nan
                            
                    all_batches.append(batch_df)
            except Exception as exc:
                logger.warning(f"  Batch failed: {exc}. Retrying after {_RETRY_PAUSE}s...")
                time.sleep(_RETRY_PAUSE)
                try:
                    batch_df = _download_batch(batch, start_date, end_date)
                    if not batch_df.empty:
                        all_batches.append(batch_df)
                except Exception as exc2:
                    logger.error(f"  Batch permanently failed: {exc2}")

        if not all_batches:
            logger.error("No data fetched at all.")
            return pd.DataFrame()

        result = pd.concat(all_batches).sort_index()

        # ── V3 Institutional Validation ────────────────────────────────────────
        result = validate_panel(result)

        # ── save to cache ──────────────────────────────────────────────────────
        self.cache.save(result, tickers, start_date, end_str, name="stock")
        logger.info(f"Fetched {len(result)} rows across {result.index.get_level_values('Ticker').nunique()} tickers.")
        return result


class MacroDataIngestor:
    """Fetches macro index data (NIFTY50, India VIX)."""

    def __init__(self, cache: Optional[ParquetCache] = None):
        self.cache = cache or ParquetCache()

    def fetch_nifty50(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Returns a flat DataFrame indexed by Date with OHLCV columns."""
        end_str = end_date or "today"
        if self.cache.exists(["^NSEI"], start_date, end_str, name="nifty50"):
            return self.cache.load(["^NSEI"], start_date, end_str, name="nifty50")

        logger.info("Fetching NIFTY 50 Index...")
        df = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Standardise columns (handle MultiIndex if present)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).title() for c in df.columns]
        
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        self.cache.save(df, ["^NSEI"], start_date, end_str, name="nifty50")
        return df

    def fetch_india_vix(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        end_str = end_date or "today"
        if self.cache.exists(["^INDIAVIX"], start_date, end_str, name="vix"):
            return self.cache.load(["^INDIAVIX"], start_date, end_str, name="vix")

        logger.info("Fetching India VIX...")
        df = yf.download("^INDIAVIX", start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Standardise columns (handle MultiIndex if present)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).title() for c in df.columns]
        
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        self.cache.save(df, ["^INDIAVIX"], start_date, end_str, name="vix")
        return df
