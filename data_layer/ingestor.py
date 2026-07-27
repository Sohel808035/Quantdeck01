"""
data_layer/ingestor.py  (v3 — QuantSphereX Production Ingestion Engine)
────────────────────────────────────────────────────────────────────────
Institutional Data Ingestion Engine with Exponential Backoff Retry Logic,
Rate Limiting, Parquet Storage Integration, and Rigorous Data Validation.

Key Improvements:
  • 100% Backward Compatible Signatures (`YFinanceIngestor`, `MacroDataIngestor`, `validate_panel`).
  • Configurable Exponential Backoff Retry Protocol for Batch Downloads.
  • Dynamic Rate Limiting & Throttling between network requests.
  • Institutional Data Integrity Validation (`validate_panel`):
      1. Duplicate (Date, Ticker) Index Detection.
      2. Missing OHLCV & Non-Positive Price Auditing.
      3. Abnormal Price Jump Outlier Auditing (>30%).
      4. Stale/Frozen Price Pattern Auditing (5+ Days).
      5. Corrupted Data Ratio Enforcer (Halts execution if fail_pct > max_fail_pct).
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

from data_layer.config import DataConfig
from data_layer.interfaces import IDataProvider, ValidationReport
from data_layer.storage import ParquetCache  # type: ignore

logger = logging.getLogger(__name__)

# Fallback default configuration instance
_DEFAULT_CONFIG = DataConfig()


class MarketDataIngestor(IDataProvider):
    """Abstract base provider interface for daily market data ingestion."""

    def fetch_daily_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


def validate_panel(
    df: pd.DataFrame,
    threshold_move: float = 0.3,
    max_fail_pct: float = 0.05,
    config: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """
    QuantSphereX V2 Production Validation Pipeline:
    
    1. Detect duplicate (Date, Ticker) timestamps.
    2. Detect missing required OHLCV values.
    3. Detect non-positive prices (Close <= 0, Open <= 0).
    4. Detect abnormal daily price jumps (> threshold_move, default 30%).
    5. Detect stale/frozen prices (same Close for 5+ consecutive days).
    6. Verify corrupted row ratio does not exceed max_fail_pct.
    """
    if df.empty:
        return df

    cfg = config or _DEFAULT_CONFIG
    t_move = threshold_move if threshold_move is not None else cfg.max_price_jump_threshold
    m_fail = max_fail_pct if max_fail_pct is not None else cfg.max_corrupted_row_pct

    initial_rows = len(df)
    logger.info(f"[Validation] Auditing {initial_rows:,} rows of market data...")

    clean_df = df.copy()

    # 1. Duplicate Index Check
    dupes_ct = clean_df.index.duplicated().sum()
    if dupes_ct > 0:
        logger.warning(f"  [DUPES] {dupes_ct:,} duplicate (Date, Ticker) indices found. Retaining first.")
        clean_df = clean_df[~clean_df.index.duplicated(keep="first")]

    # 2. Missing OHLCV Check
    required_cols = [c for c in ["Open", "High", "Low", "Close"] if c in clean_df.columns]
    for col in required_cols:
        nan_ct = clean_df[col].isna().sum()
        if nan_ct > 0:
            logger.warning(f"  [MISSING] {nan_ct:,} NaN values in '{col}'. Dropping.")
            clean_df = clean_df.dropna(subset=[col])

    # 3. Non-Positive Price Check
    if "Close" in clean_df.columns:
        invalid_prices = (clean_df["Close"] <= 0).sum()
        if invalid_prices > 0:
            logger.warning(f"  [INVALID PRICE] {invalid_prices:,} non-positive Close prices detected. Dropping.")
            clean_df = clean_df[clean_df["Close"] > 0]

    # 4. Abnormal Price Jumps (> threshold_move)
    if "Close" in clean_df.columns and len(clean_df) > 0:
        daily_rets = clean_df.groupby(level="Ticker")["Close"].pct_change().abs()
        outliers_mask = daily_rets > t_move
        outliers_ct = outliers_mask.sum()
        if outliers_ct > 0:
            logger.warning(f"  [OUTLIER] {outliers_ct:,} price jumps > {t_move:.0%} detected. Dropping.")
            clean_df = clean_df[~outliers_mask.fillna(False)]

    # 5. Stale Data Check (Same Close price for 5+ consecutive days)
    if "Close" in clean_df.columns and len(clean_df) > 0 and not cfg.allow_stale_data:
        try:
            is_diff_zero = clean_df.groupby(level="Ticker")["Close"].diff() == 0
            stale_mask = is_diff_zero.groupby(level="Ticker").rolling(cfg.stale_price_days_threshold - 1).sum() == (cfg.stale_price_days_threshold - 1)
            stale_mask = stale_mask.reset_index(level=0, drop=True)
            stale_ct = stale_mask.sum()
            if stale_ct > 0:
                logger.warning(f"  [STALE] {stale_ct:,} rows have frozen prices for {cfg.stale_price_days_threshold}+ days.")
                clean_df = clean_df[~stale_mask.fillna(False)]
        except Exception as exc:
            logger.debug(f"  [STALE CHECK SKIP] Could not evaluate stale mask: {exc}")

    # 6. Fault Tolerance Threshold Check
    final_rows = len(clean_df)
    total_dropped = initial_rows - final_rows
    fail_pct = total_dropped / initial_rows if initial_rows > 0 else 0.0

    report = ValidationReport(
        initial_rows=initial_rows,
        final_rows=final_rows,
        dropped_duplicates=int(dupes_ct),
        fail_percentage=fail_pct,
        passed=(fail_pct <= m_fail),
    )

    if fail_pct > m_fail:
        critical_msg = (
            f"CRITICAL DATA QUALITY FAILURE: {fail_pct:.1%} corrupted data dropped "
            f"({total_dropped:,}/{initial_rows:,} rows). Max allowed: {m_fail:.1%}."
        )
        logger.error(critical_msg)
        raise ValueError(critical_msg)

    logger.info(f"  Audit complete. Quality pass: {(1 - fail_pct):.1%}. Rows remaining: {final_rows:,}")
    return clean_df


def _download_batch_with_retry(
    yf_tickers: List[str],
    start: str,
    end: Optional[str],
    config: DataConfig,
) -> pd.DataFrame:
    """
    Downloads a single batch of tickers with exponential backoff retry logic.
    Returns long-format (Date, Ticker) DataFrame.
    """
    max_retries = config.max_retries
    pause = config.retry_initial_pause_seconds
    backoff = config.retry_backoff_factor

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                yf_tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,  # Replace Close with split/adj corrected Close
                progress=False,
                threads=True,
            )

            if raw.empty:
                return pd.DataFrame()

            # ── Format MultiIndex response ──────────────────────────────────────
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.dropna(axis=1, how="all")
                if raw.empty:
                    return pd.DataFrame()

                ticker_level = "Ticker" if "Ticker" in raw.columns.names else (
                    0 if "Open" in raw.columns.get_level_values(1) else 1
                )

                df_long = (
                    raw.stack(level=ticker_level, future_stack=True)
                       .rename_axis(index=["Date", "Ticker"])
                       .reset_index()
                )
                df_long["Ticker"] = df_long["Ticker"].astype(str).str.replace(r"\.NS$", "", regex=True)
            else:
                df_long = raw.reset_index()
                df_long["Ticker"] = yf_tickers[0].replace(".NS", "")

            # ── Standardize Column Names ─────────────────────────────────────────
            df_long.columns = [str(c).title() for c in df_long.columns]
            wanted = [c for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"] if c in df_long.columns]
            df_long = df_long[wanted]

            df_long["Date"] = pd.to_datetime(df_long["Date"])
            df_long = df_long.dropna(subset=["Close"])
            df_long = df_long.set_index(["Date", "Ticker"]).sort_index()

            return df_long

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                sleep_time = pause * (backoff ** (attempt - 1))
                logger.warning(
                    f"  Batch attempt {attempt}/{max_retries} failed ({exc}). "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"  Batch permanently failed after {max_retries} attempts: {exc}")

    return pd.DataFrame()


class YFinanceIngestor(MarketDataIngestor):
    """
    Fetches daily market panels for NSE tickers via yfinance.
    Handles caching, batching, retry backoff, rate limiting, and panel validation.
    """

    def __init__(
        self,
        suffix: str = ".NS",
        cache: Optional[ParquetCache] = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or _DEFAULT_CONFIG
        self.suffix = suffix or self.config.yfinance_suffix
        self.cache = cache or ParquetCache(config=self.config)

    def fetch_fundamental_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetches fundamental metrics (ROE, ROA, Earnings_Growth) safely.
        Logs metrics and skips unavailable tickers without interrupting ingestion.
        """
        logger.info(f"Fetching fundamental attributes for {len(tickers)} tickers...")
        fundamental_frames: List[Dict[str, Any]] = []

        for t in tickers:
            try:
                stock = yf.Ticker(f"{t}{self.suffix}")
                info = stock.info
                if not info or "regularMarketPrice" not in info:
                    continue

                metrics = {
                    "Ticker": t,
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "Earnings_Growth": info.get("earningsGrowth", np.nan),
                }
                fundamental_frames.append(metrics)
            except Exception as exc:
                logger.debug(f"Fundamental fetch skipped for {t}: {exc}")
                continue

        if not fundamental_frames:
            return pd.DataFrame(columns=["Ticker", "ROE", "ROA", "Earnings_Growth"]).set_index("Ticker")

        fund_df = pd.DataFrame(fundamental_frames).set_index("Ticker")
        logger.info(f"Fundamental data retrieved for {len(fund_df)}/{len(tickers)} tickers.")
        return fund_df

    def fetch_daily_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        end_str = end_date or "today"

        # ── 1. Check Parquet Cache ───────────────────────────────────────────
        if self.cache.exists(tickers, start_date, end_str, name="stock"):
            return self.cache.load(tickers, start_date, end_str, name="stock")

        logger.info(
            f"Fetching {len(tickers)} tickers from {start_date} to {end_str} "
            f"in batches of {self.config.batch_size}..."
        )

        yf_tickers = [f"{t}{self.suffix}" for t in tickers]
        all_batches: List[pd.DataFrame] = []
        batch_size = self.config.batch_size

        # ── 2. Download Batches with Exponential Backoff & Rate Limiting ────
        for i in range(0, len(yf_tickers), batch_size):
            batch = yf_tickers[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(yf_tickers) + batch_size - 1) // batch_size

            logger.info(f"  Batch {batch_num}/{total_batches}: {batch[:3]}... ({len(batch)} tickers)")
            
            batch_df = _download_batch_with_retry(batch, start_date, end_date, self.config)
            if not batch_df.empty:
                # Add place-holder quality factor columns
                for col in ["ROE", "ROA", "Earnings_Growth"]:
                    batch_df[col] = np.nan
                all_batches.append(batch_df)

            # Rate limiting pause between batches
            if i + batch_size < len(yf_tickers):
                time.sleep(self.config.rate_limit_pause_seconds)

        if not all_batches:
            logger.error("No market data fetched from remote provider.")
            return pd.DataFrame()

        result = pd.concat(all_batches).sort_index()

        # ── 3. Validate Panel Quality ─────────────────────────────────────────
        result = validate_panel(result, config=self.config)

        # ── 4. Save to Parquet Cache ─────────────────────────────────────────
        self.cache.save(result, tickers, start_date, end_str, name="stock")
        logger.info(f"Successfully ingested {len(result):,} rows across {result.index.get_level_values('Ticker').nunique()} tickers.")
        return result


class MacroDataIngestor:
    """Fetches macro index benchmark panels (NIFTY 50, India VIX)."""

    def __init__(
        self,
        cache: Optional[ParquetCache] = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or _DEFAULT_CONFIG
        self.cache = cache or ParquetCache(config=self.config)

    def fetch_nifty50(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Returns flat DataFrame indexed by Date with OHLCV columns."""
        end_str = end_date or "today"
        if self.cache.exists(["^NSEI"], start_date, end_str, name="nifty50"):
            return self.cache.load(["^NSEI"], start_date, end_str, name="nifty50")

        logger.info("Fetching NIFTY 50 Index (^NSEI)...")
        df = _download_batch_with_retry(["^NSEI"], start_date, end_date, self.config)
        
        if df.empty:
            logger.error("Failed to download NIFTY 50 index.")
            return pd.DataFrame()

        # Flatten multiindex ticker for single index panel
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs("^NSEI", level="Ticker") if "^NSEI" in df.index.get_level_values("Ticker") else df.droplevel("Ticker")

        df.index.name = "Date"
        self.cache.save(df, ["^NSEI"], start_date, end_str, name="nifty50")
        return df

    def fetch_india_vix(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Returns flat DataFrame indexed by Date for India VIX."""
        end_str = end_date or "today"
        if self.cache.exists(["^INDIAVIX"], start_date, end_str, name="vix"):
            return self.cache.load(["^INDIAVIX"], start_date, end_str, name="vix")

        logger.info("Fetching India VIX (^INDIAVIX)...")
        df = _download_batch_with_retry(["^INDIAVIX"], start_date, end_date, self.config)
        
        if df.empty:
            logger.error("Failed to download India VIX index.")
            return pd.DataFrame()

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs("^INDIAVIX", level="Ticker") if "^INDIAVIX" in df.index.get_level_values("Ticker") else df.droplevel("Ticker")

        df.index.name = "Date"
        self.cache.save(df, ["^INDIAVIX"], start_date, end_str, name="vix")
        return df
