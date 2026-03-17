# QuantDeck CQRO Institutional Alpha Engine

QuantDeck is an institutional-grade quantitative trading framework and alpha engine. The system is built around strict, professional quantitative research principles (CQRO mandate), heavily emphasizing robust backtesting, walk-forward ensembled machine learning, transaction cost realism, and regime-aware risk management.

It is designed to evaluate raw signals ("alpha") and process them into production-ready portfolio weights. 

## 🏗️ Architecture & Core Layers

### `data_layer/` (Data Extraction & Storage)
Handles all data ingestions and cache management to prevent data leakage and improve performance.
- **`ingestor.py`**: Interacts with data sources (e.g., Yahoo Finance) to fetch daily stock panels and macro indicators (NIFTY 50, India VIX) to contextualize the market state.
- **`storage.py`**: Implements caching using Parquet format to ensure that large financial panels do not have to be re-downloaded every run.
- **`universe.py`**: Manages the trading universe, maintaining active tickers and filtering out delisted or untradable equities.

### `feature_layer/` (Signal Generation & Preprocessing)
Calculates predictive metrics for the ML models.
- **`engineering.py` & `implementations.py`**: Computes core fundamental and technical signals. Applies strict institutional cross-sectional processing (Sector Neutralization, Cross-Sectional Ranking, Feature Pruning).

### `alpha_layer/` (Machine Learning & Modeling)
The core predictive engine that uses the features to forecast target returns.
- **`xgboost_trainer.py`**: Uses an Ensemble XGBoost architecture to predict stock alpha.
- **`walk_forward.py`**: Implements "Walk-Forward" training over a rolling window and predicts the forward horizon with an embargo period (prevents look-ahead bias).
- **`pure_alpha_validator.py`**: Tests the "raw" model outputs using Information Coefficient (IC) tests and decile spread returns to ensure the model has mathematical merit.
- **`targets.py`**: Dynamically constructs forward-looking return targets.

### `portfolio_layer/` (Portfolio Construction)
Converts alpha predictions into investable positions.
- **`ranking.py`**: Identifies the top N stocks to buy using hysteresis buffers to prevent excessive turnover.
- **`optimizer.py`**: Handles sizing constraints, optionally defaulting to equal-weight or volatility-scaled distribution, while imposing sector limits.

### `risk_layer/` (Market Defense & Regime Awareness)
Adapts the strategy to market environments dynamically.
- **`regime_model.py`**: Classifies the market regime (e.g., Bull, Bear, High-Volatility).
- **`vol_targeting.py`**: Dynamically scales total portfolio exposure up or down to target a specific portfolio volatility.
- **`regime_robustness.py` & `filters.py`**: Audits how the alpha performs across different detected regimes.

### `execution_layer/` (Realistic Backtesting)
Simulates trading the portfolio in real life.
- **`backtester.py`**: Processes the historical weights against the actual historical closing prices applying transaction costs and dynamic market impact costs.
- **`stress_tester.py`**: Re-runs the model under extreme simulated parameters.

## 🚀 Execution Pipeline

The `main.py` entry point strings these layers together into a rigorous **9-Step Institutional Execution Pipeline**:

1. **Data Integrity & Leakage Control**
2. **Feature Engineering**
3. **Walk-Forward Model Training**
4. **Pure Validation**
5. **Portfolio Construction**
6. **Production Backtest**
7. **Regime Robustness Verification**
8. **Stress Testing**
9. **Final Institutional Decision Matrix** (Verdicts: Deployment Eligible, Refinement Required, or Rebuild Alpha)

## 🛠️ Usage

Make sure you have a python virtual environment set up and the required dependencies installed (like `pandas`, `xgboost`, `streamlit`, `yfinance`, `pyarrow`, `matplotlib`).

To execute the core pipeline:
```bash
python main.py
```

This will run all steps, cache datasets into the `data_cache/` directory, and generate reports output to the `reports/` folder.