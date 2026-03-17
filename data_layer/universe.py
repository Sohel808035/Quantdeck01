"""
data_layer/universe.py
──────────────────────
Manages the investable universe of 200 Indian large/mid-cap stocks.
Handles Point-In-Time (PIT) membership to eliminate survivorship bias.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# Static fallback list (NIFTY 200 constituents as of early 2026)
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

class UniverseManager:
    """
    V3 Institutional Universe Management.
    Supports point-in-time membership and provides sector mapping.
    """
    def __init__(self, membership_path: Optional[str] = None):
        self.path = Path(membership_path or "data_cache/universe_membership.parquet")
        self.history: Optional[pd.DataFrame] = None
        
        if self.path.exists():
            try:
                self.history = pd.read_parquet(self.path)
                logger.info(f"[Universe] Loaded historical membership from {self.path}")
            except Exception as e:
                logger.error(f"[Universe] Failed to load membership: {e}")
        else:
            logger.warning("[Universe] Historical membership file NOT found. Falling back to static list. Backtests will have Survivorship Bias.")

    def get_universe(self, date: Optional[pd.Timestamp] = None) -> List[str]:
        """Returns the list of tickers that were in the NIFTY 200 on a specific date."""
        if date is not None and self.history is not None:
            try:
                # Find the nearest snapshot date <= requested date in the index
                hist_dates = self.history.index.get_level_values("Date").unique()
                valid_dates = hist_dates[hist_dates <= pd.to_datetime(date)]
                
                if not valid_dates.empty:
                    snap_date = valid_dates.max()
                    # Query membership for that date
                    snap_df = self.history.loc[snap_date]
                    # Select only active tickers
                    active_tickers = snap_df[snap_df["IsInUniverse"] == True].index.tolist()
                    
                    if active_tickers:
                        return active_tickers
                    
            except Exception as e:
                logger.error(f"[Universe] Error querying membership for {date}: {e}")
        
        # Default fallback to static list if date is None or history fails
        return list(NIFTY200_STATIC_LIST)

    def get_sector_mapping(self) -> Dict[str, str]:
        """Maps Ticker to Industry Sector (Production Metadata)."""
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
        """Provides the NIFTY 200 benchmark sector weights for neutralization logic."""
        return {
            "Financial Services": 0.35, "IT": 0.15, "Oil & Gas": 0.12,
            "FMCG": 0.09, "Automobile": 0.06, "Healthcare": 0.05,
            "Construction": 0.04, "Telecom": 0.03, "Other / Midcap": 0.11
        }

def get_universe(date: Optional[pd.Timestamp] = None) -> List[str]:
    """Legacy wrapper for backward compatibility."""
    mgr = UniverseManager()
    return mgr.get_universe(date)

def get_yfinance_tickers(date: Optional[pd.Timestamp] = None, suffix: str = ".NS") -> List[str]:
    """Returns universe tickers with the yfinance exchange suffix."""
    return [f"{t}{suffix}" for t in get_universe(date)]
