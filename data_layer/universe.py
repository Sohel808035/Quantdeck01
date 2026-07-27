"""
data_layer/universe.py
──────────────────────
Manages the investable universe of 200 Indian large/mid-cap stocks (NIFTY 200).
Supports Point-In-Time (PIT) historical membership snapshots to eliminate survivorship bias,
and provides industry sector mappings and benchmark sector weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from data_layer.config import DataConfig
from data_layer.interfaces import IUniverseProvider

logger = logging.getLogger(__name__)

# Static fallback list (NIFTY 200 constituents)
NIFTY200_STATIC_LIST: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "KOTAKBANK",
    "AXISBANK", "BAJFINANCE", "BHARTIARTL", "LT", "ASIANPAINT", "MARUTI", "TITAN",
    "NESTLEIND", "ULTRACEMCO", "WIPRO", "HCLTECH", "SUNPHARMA", "ONGC",
    "NTPC", "POWERGRID", "COALINDIA", "SBILIFE", "HDFCLIFE",
    "BAJAJFINSV", "TECHM", "DIVISLAB", "DRREDDY", "CIPLA",
    "EICHERMOT", "HEROMOTOCO", "M&M", "TATAMOTORS", "TATACONSUM",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "BPCL",
    "BRITANNIA", "UPL", "ADANIENT", "ADANIPORTS", "SBIN",
    "APOLLOHOSP", "BAJAJ-AUTO", "INDUSINDBK", "ITC", "LTIM",
    "AMBUJACEM", "ACC", "BANKBARODA", "BERGEPAINT", "BIOCON",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "CONCOR",
    "CUMMINSIND", "DABUR", "DLF", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "GAIL", "GLAND", "GODREJCP", "GODREJPROP",
    "HAVELLS", "IDFCFIRSTB", "IGL", "INDUSTOWER", "IRCTC",
    "JINDALSTEL", "JUBLFOOD", "KEI", "L&TFH", "LICHSGFIN",
    "LUPIN", "MANAPPURAM", "MARICO", "MFSL", "MINDTREE",
    "MOTHERSON", "MPHASIS", "MRF", "NAUKRI", "NMDC",
    "OBEROIRLTY", "OFSS", "PAGEIND", "PEL", "PERSISTENT",
    "PFC", "PHOENIXLTD", "PNB", "POLYCAB", "PVRINOX",
    "RECLTD", "SAIL", "SRF", "SRTRANSFIN", "STAR",
    "SUNDARMFIN", "SUPREMEIND", "SWIGGY", "TATACOMM", "TATACHEM",
    "TATAELXSI", "TATAPOWER", "TRENT", "TRIDENT", "UFLEX",
    "UNIONBANK", "UNITDSPR", "VEDL", "VOLTAS", "WHIRLPOOL",
    "ZOMATO", "ZYDUSLIFE", "PIIND", "TORNTPHARM", "AUROPHARMA",
    "ALKEM", "LALPATHLAB", "METROPOLIS", "MAXHEALTH", "FORTIS",
    "AAVAS", "ABFRL", "AFFLE", "ANGELONE", "APTUS",
    "ASTRAL", "ATUL", "AWFIS", "BAJAJHFL", "BSOFT",
    "CAMPUS", "CARBORUNDUM", "CDSL", "CEATLTD", "CGPOWER",
    "CHALET", "CLEAN", "CRAFTSMAN", "CRISIL", "DALMIA",
    "DEEPAKNTR", "DELTACORP", "DMART", "ERIS", "FINEORG",
    "FLUOROCHEM", "FSL", "GICRE", "GMRAIRPORT", "GPPL",
    "GRINDWELL", "GSPL", "HAPPSTMNDS", "HSCL", "HUDCO",
    "ICICIPRULI", "IIFL", "INDIANB", "INDHOTEL", "INOXINDIA",
    "INTELLECT", "IONEXCHAN", "IPCALAB", "IRB", "ISEC",
    "JKCEMENT", "JSWENERGY", "JYOTHYLAB", "KALYANKJIL", "KARURVYSYA",
    "KAYNES", "KEC", "KIMS", "KNRCON", "KRBL",
    "LAURUSLABS", "LEMONTREE", "LUXIND", "MAPMYINDIA", "MEDANTA",
    "MOTHERSUMI", "NATCOPHARM", "NYKAA", "OLECTRA", "ORIENTELEC",
    "PGHH", "PPLPHARMA", "PRAJIND", "RAINBOW", "RATNAMANI",
    "RITES", "RKFORGE", "ROUTE", "SAFARI", "SAPPHIRE",
    "SCHAEFFLER", "SHOPERSTOP", "SIGNATUREGLO", "SOLARINDS", "SPANDANA",
    "TANLA", "TATAINVEST", "TIINDIA", "TIMKEN", "TTML",
    "UJJIVANSFB", "UNIPARTS", "UTIAMC", "VAIBHAVGBL", "VBL",
    "VIJAYA", "VSTIND", "WCMCABLE", "WELSPUNLIV", "ZFCVINDIA",
]


class UniverseManager(IUniverseProvider):
    """
    QuantSphereX V2 Institutional Universe Manager.
    Supports Point-In-Time (PIT) membership lookup and industry sector metadata.
    """

    def __init__(
        self,
        membership_path: Optional[str | Path] = None,
        config: Optional[DataConfig] = None,
    ):
        self.config = config or DataConfig()
        default_path = self.config.cache_dir / "universe_membership.parquet"
        self.path = Path(membership_path) if membership_path else default_path
        self.history: Optional[pd.DataFrame] = None

        if self.path.exists():
            try:
                self.history = pd.read_parquet(self.path)
                logger.info(f"[Universe] Loaded historical PIT membership from {self.path.name}")
            except Exception as exc:
                logger.error(f"[Universe] Failed to load PIT membership file ({exc}). Falling back to static list.")
        else:
            logger.debug("[Universe] PIT membership file not found. Falling back to static NIFTY 200 list.")

    def get_universe(self, date: Optional[pd.Timestamp] = None) -> List[str]:
        """Returns the list of active tickers for a specific historical date."""
        if date is not None and self.history is not None:
            try:
                hist_dates = self.history.index.get_level_values("Date").unique()
                valid_dates = hist_dates[hist_dates <= pd.to_datetime(date)]

                if not valid_dates.empty:
                    snap_date = valid_dates.max()
                    snap_df = self.history.loc[snap_date]
                    active_tickers = snap_df[snap_df["IsInUniverse"] == True].index.tolist()

                    if active_tickers:
                        return active_tickers
            except Exception as exc:
                logger.error(f"[Universe] Error querying PIT membership for {date}: {exc}")

        return list(NIFTY200_STATIC_LIST)

    def get_sector_mapping(self) -> Dict[str, str]:
        """Provides mapping of ticker symbols to industry sector categories."""
        mapping = {
            "HDFCBANK": "Financial Services", "ICICIBANK": "Financial Services",
            "RELIANCE": "Oil & Gas", "TCS": "IT", "INFY": "IT",
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "BAJFINANCE": "Financial Services",
            "LT": "Construction", "AXISBANK": "Financial Services",
            "KOTAKBANK": "Financial Services", "SBIN": "Financial Services",
            "BHARTIARTL": "Telecom", "ASIANPAINT": "Consumer Goods",
            "MARUTI": "Automobile", "TITAN": "Consumer Goods",
            "SUNPHARMA": "Healthcare", "ULTRACEMCO": "Construction Materials",
        }
        for t in NIFTY200_STATIC_LIST:
            if t not in mapping:
                mapping[t] = "Other / Midcap"
        return mapping

    def get_benchmark_sector_weights(self) -> Dict[str, float]:
        """Provides default benchmark sector target weights for neutralization."""
        return {
            "Financial Services": 0.35, "IT": 0.15, "Oil & Gas": 0.12,
            "FMCG": 0.09, "Automobile": 0.06, "Healthcare": 0.05,
            "Construction": 0.04, "Telecom": 0.03, "Other / Midcap": 0.11
        }


# ── Legacy Compatibility Helpers ──────────────────────────────────────────────

def get_universe(date: Optional[pd.Timestamp] = None) -> List[str]:
    """Legacy helper function for backward compatibility."""
    mgr = UniverseManager()
    return mgr.get_universe(date)


def get_yfinance_tickers(date: Optional[pd.Timestamp] = None, suffix: str = ".NS") -> List[str]:
    """Legacy helper returning yfinance formatted ticker list."""
    return [f"{t}{suffix}" for t in get_universe(date)]
