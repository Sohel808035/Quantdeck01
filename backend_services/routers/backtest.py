"""
backend_services/routers/backtest.py
─────────────────────────────────────
Backtesting Engine Service Router.
Runs modular backtests and exports performance metrics and tearsheets.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, status

from backend_services.auth import verify_token_or_key
from backend_services.dependencies import get_backtest_engine
from backend_services.schemas import BacktestRequest, BacktestResponse
from execution_layer.backtesting import BacktestEngine, BacktestConfig

router = APIRouter(prefix="/backtest", tags=["Backtesting Engine Services"])


@router.post(
    "/run",
    response_model=BacktestResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Quant Backtest Simulation",
)
async def run_backtest(
    request: BacktestRequest,
    engine: BacktestEngine = Depends(get_backtest_engine),
    client_id: str = Depends(verify_token_or_key),
) -> BacktestResponse:
    """Runs modular backtest simulation over synthetic/historical returns."""
    # Generate deterministic synthetic panel for backtest API demonstration
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = [f"TICKER_{i:02d}" for i in range(5)]

    rets = pd.DataFrame(np.random.normal(0.0004, 0.011, (252, 5)), index=dates, columns=tickers)
    monthly = pd.date_range("2023-01-01", periods=12, freq="MS")
    monthly = monthly[monthly.isin(dates)]
    weights = pd.DataFrame(0.20, index=monthly, columns=tickers)

    # Configure engine
    engine.config.initial_capital = request.initial_capital
    engine.config.transaction_cost_pct = request.transaction_cost_pct
    engine.config.apply_vol_targeting = request.apply_vol_targeting

    result = engine.run(weights_schedule=weights, stock_returns=rets)
    m = result.metrics

    return BacktestResponse(
        cagr=round(m.get("cagr", 0.0), 4),
        ann_vol=round(m.get("ann_vol", 0.0), 4),
        sharpe_ratio=round(m.get("sharpe_ratio", 0.0), 3),
        sortino_ratio=round(m.get("sortino_ratio", 0.0), 3),
        max_drawdown=round(m.get("max_drawdown", 0.0), 4),
        calmar_ratio=round(m.get("calmar_ratio", 0.0), 3),
        final_equity=round(m.get("final_equity", request.initial_capital), 2),
        total_trades_count=len(result.holding_weights),
    )
