
import pandas as pd
import numpy as np
from pathlib import Path

# Symbols from universe.py
NIFTY200_STATIC_LIST = [
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

def generate_membership():
    # Dates: Semiannual rebalancing (Mar and Sep)
    dates = pd.date_range("2005-01-01", "2026-06-01", freq="6MS")
    rows = []
    
    # We assign a random entry year for each ticker to simulate PIT universe
    np.random.seed(42)
    entry_dates = {}
    for t in NIFTY200_STATIC_LIST:
        # Some are old, some are new
        if t in ["RELIANCE", "TCS", "SBIN", "INFY", "ITC", "HDFCBANK"]:
            entry_dates[t] = pd.Timestamp("2005-01-01")
        elif t in ["ZOMATO", "NYKAA", "SWIGGY", "BAJAJHFL"]:
            entry_dates[t] = pd.Timestamp("2021-01-01")
        else:
            year = np.random.randint(2005, 2022)
            entry_dates[t] = pd.Timestamp(f"{year}-01-01")

    for d in dates:
        for t in NIFTY200_STATIC_LIST:
            is_in = d >= entry_dates[t]
            rows.append({"Date": d, "Ticker": t, "IsInUniverse": is_in})
            
    df = pd.DataFrame(rows).set_index(["Date", "Ticker"])
    
    output_dir = Path("e:/Quantdeck01/data_cache")
    output_dir.mkdir(exist_ok=True)
    df.to_parquet(output_dir / "universe_membership.parquet")
    print(f"Generated universe membership for {len(NIFTY200_STATIC_LIST)} tickers across {len(dates)} rebalance dates.")

if __name__ == "__main__":
    generate_membership()
