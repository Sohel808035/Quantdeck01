import streamlit as st
import pandas as pd
import json
from pathlib import Path

# ------------------------------------------------------------
# Enhanced Streamlit Dashboard – QuantDeck CQRO Engine
# ------------------------------------------------------------
# Features:
#   • Dark glass‑morphism theme with gradient accent
#   • Sidebar navigation with tabs: Equity Curve, Daily Orders, Daily Summary, Metrics, About
#   • Interactive date selector for orders
#   • Styled data tables & download buttons
#   • Metric cards (CAGR, Sharpe, Turnover)
#   • Light entry animation using st.write with markdown placeholders
# ------------------------------------------------------------

# ---- Page configuration ------------------------------------------------
st.set_page_config(
    page_title="QuantDeck CQRO Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS (premium dark theme with glass‑morphism) ----------------
custom_css = """
    <style>
        body {background-color:#0e1117; color:#e0e0e0;}
        .stApp {background: linear-gradient(135deg, #111418 0%, #0e1117 100%);}
        .stSidebar {background-color:#111418;}
        .stButton button {background-color:#0066ff; color:white; border-radius:6px;}
        .stDataFrame {background-color:#1a1e24; border-radius:8px;}
        .stMetric > div {background-color:#21262d; border-radius:8px; padding:10px;}
        .section-header {font-size:1.5rem; color:#66ccff; margin-top:1rem; margin-bottom:0.5rem;}
        .glass {
            backdrop-filter: blur(12px);
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            padding: 1rem;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---- Paths ------------------------------------------------------------
BASE_PATH = Path(".")
REPORTS_PATH = BASE_PATH / "reports"
DAILY_ORDERS_PATH = REPORTS_PATH / "daily_orders.csv"
DAILY_SUMMARY_PATH = REPORTS_PATH / "daily_summary.csv"
EQ_CURVE_IMG = REPORTS_PATH / "cqro_equity_curve.png"
METRICS_JSON = REPORTS_PATH / "engine_metrics.json"

# ---- Sidebar navigation -------------------------------------------------
st.sidebar.title("📊 QuantDeck Dashboard")
st.sidebar.markdown("**Navigate through the engine outputs**")
view_option = st.sidebar.radio(
    "Select view",
    ["Equity Curve", "Daily Orders", "Trade Prices", "Daily Summary", "Metrics", "About"],
    index=0,
)

# ---- Helper functions -------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to read {path.name}: {e}")
    return None

def styled_dataframe(df: pd.DataFrame):
    return df.style.set_properties(**{"background-color": "#1a1e24", "color": "#e0e0e0"}) \
               .set_table_styles([
                   {"selector": "th", "props": [("background-color", "#21262d"), ("color", "#66ccff")]}])

# ---- Main content ------------------------------------------------------
st.title("QuantDeck CQRO Institutional Alpha Engine")

if view_option == "Equity Curve":
    st.subheader("📈 Equity Curve")
    if EQ_CURVE_IMG.exists():
        st.image(str(EQ_CURVE_IMG), use_column_width=True, caption="Net Equity Curve (2005‑2026)")
    else:
        st.warning("Equity curve image not found. Run the engine to generate reports.")

elif view_option == "Daily Orders":
    st.subheader("🛒 Daily Orders – Exact Share Allocation")
    df = load_csv(DAILY_ORDERS_PATH)
    if df is not None:
        # Date filter if column exists
        if "Date" in df.columns:
            selected_date = st.selectbox("Select date", options=sorted(df["Date"].unique()))
            df = df[df["Date"] == selected_date]
        st.dataframe(styled_dataframe(df))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv, file_name="daily_orders.csv", mime="text/csv")
    else:
        st.info("No daily orders have been generated yet. Run the engine first.")

elif view_option == "Trade Prices":
    st.subheader("💱 Trade Prices – Buy/Sell Price Snapshot")
    orders_df = load_csv(DAILY_ORDERS_PATH)
    summary_df = load_csv(DAILY_SUMMARY_PATH)

    if orders_df is None and summary_df is None:
        st.info("No order or summary reports found. Run the engine first.")
    else:
        # If both exist, try to merge on Ticker (and Date if available)
        if orders_df is None:
            merged = summary_df.copy()
        elif summary_df is None:
            merged = orders_df.copy()
        else:
            # Align on Date + Ticker when Date exists in summary
            if "Date" in summary_df.columns and "Date" in orders_df.columns:
                merged = pd.merge(summary_df, orders_df, on=["Date", "Ticker"], how="outer", suffixes=("_summary", "_orders"))
            else:
                merged = pd.merge(summary_df, orders_df, on=["Ticker"], how="outer", suffixes=("_summary", "_orders"))

        # Normalize column names and compute Action
        if "Shares_To_Buy" in merged.columns:
            merged["Action"] = merged["Shares_To_Buy"].apply(lambda x: "Buy" if pd.notna(x) and x > 0 else "Sell/Reduce" if pd.notna(x) and x < 0 else "N/A")
        else:
            merged["Action"] = "N/A"

        # Preferred price columns
        price_cols = []
        if "Latest_Price" in merged.columns:
            price_cols.append("Latest_Price")
        if "Entry_Price" in merged.columns:
            price_cols.append("Entry_Price")
        if "Stop_Loss" in merged.columns:
            price_cols.append("Stop_Loss")
        if "Target" in merged.columns:
            price_cols.append("Target")

        display_cols = [c for c in ["Date", "Ticker", "Action"] if c in merged.columns] + price_cols + [c for c in ["Shares_To_Buy", "Shares"] if c in merged.columns]
        # Reorder and show
        merged = merged.reindex(columns=[c for c in display_cols if c in merged.columns])
        st.dataframe(styled_dataframe(merged.fillna("")))
        csv = merged.to_csv(index=False).encode()
        st.download_button("Download Combined CSV", data=csv, file_name="trade_prices.csv", mime="text/csv")

elif view_option == "Daily Summary":
    st.subheader("📊 Daily Allocation Summary (Weights, SL & Targets)")
    df = load_csv(DAILY_SUMMARY_PATH)
    if df is not None:
        st.dataframe(styled_dataframe(df))
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Summary CSV", data=csv, file_name="daily_summary.csv", mime="text/csv")
    else:
        st.warning("Daily summary not found. Ensure the engine ran with the summary step.")

elif view_option == "Metrics":
    st.subheader("⚙️ Engine Metrics")
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON) as f:
                metrics = json.load(f)
            col1, col2, col3 = st.columns(3)
            col1.metric("CAGR", f"{metrics.get('cagr', 0):.2%}")
            col2.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
            col3.metric("Turnover", f"{metrics.get('turnover', 0):.1%}")
        except Exception as e:
            st.error(f"Failed to load metrics: {e}")
    else:
        st.info("Metrics JSON not yet generated. Future engine runs will create `engine_metrics.json`.")

elif view_option == "About":
    st.subheader("ℹ️ About this Dashboard")
    st.markdown(
        """
        **QuantDeck CQRO** is a **self‑healing** machine‑learning pipeline that
        automatically trains, validates, and generates daily trade allocations.
        
        - **Zero‑touch**: no manual re‑training needed; the engine heals over‑fitting.
        - **Premium UI**: dark glass‑morphism theme, interactive filters and styled tables.
        - **Metrics**: quick view of CAGR, Sharpe, and turnover.
        - **Export**: download exact share allocations and a full daily summary (weights, stop‑loss, target).
        
        Powered by **QuantDeck CQRO** – modern institutional alpha research.
        """
    )

# ---- Footer -----------------------------------------------------------
st.caption("⚡ Powered by QuantDeck CQRO – self‑healing ML pipeline & zero‑touch order generation")
