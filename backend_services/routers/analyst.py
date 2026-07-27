"""
backend_services/routers/analyst.py
───────────────────────────────────
AI Quant Analyst Router.
Provides prediction explanations, SHAP feature interpretations, news sentiment scoring, and investment memos.
"""

from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter, Depends, status

from backend_services.auth import verify_token_or_key
from backend_services.dependencies import get_ai_quant_analyst
from backend_services.schemas import (
    PredictionExplainRequest,
    PredictionExplainResponse,
    NewsSummaryRequest,
    NewsSummaryResponse,
)
from ai_quant_analyst import AIQuantAnalyst

router = APIRouter(prefix="/analyst", tags=["AI Quant Analyst Services"])


@router.post(
    "/explain-prediction",
    response_model=PredictionExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Explain Model Prediction & SHAP Drivers",
)
async def explain_prediction(
    request: PredictionExplainRequest,
    analyst: AIQuantAnalyst = Depends(get_ai_quant_analyst),
    client_id: str = Depends(verify_token_or_key),
) -> PredictionExplainResponse:
    """Translates raw model signal and SHAP values into readable quantitative commentary."""
    explanation = analyst.explain_prediction(
        symbol=request.symbol,
        predicted_score=request.predicted_score,
        probability=request.probability,
        feature_values=request.shap_values,
    )

    shap_summary = None
    if request.shap_values:
        shap_res = analyst.interpret_shap(request.symbol, request.shap_values)
        shap_summary = shap_res.get("executive_summary")

    return PredictionExplainResponse(
        symbol=explanation["symbol"],
        direction=explanation["direction"],
        stance=explanation["stance"],
        narrative=explanation["narrative"],
        key_drivers=explanation["key_drivers"],
        shap_summary=shap_summary,
    )


@router.post(
    "/summarize-news",
    response_model=NewsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize News Sentiment & Impact Rating",
)
async def summarize_news(
    request: NewsSummaryRequest,
    analyst: AIQuantAnalyst = Depends(get_ai_quant_analyst),
    client_id: str = Depends(verify_token_or_key),
) -> NewsSummaryResponse:
    """Computes sentiment score, sentiment label, and market impact rating for news content."""
    res = analyst.summarize_news(
        headline=request.headline,
        content=request.content,
        symbol=request.symbol,
    )

    return NewsSummaryResponse(
        symbol=res["symbol"],
        headline=res["headline"],
        sentiment_score=res["sentiment_score"],
        sentiment_label=res["sentiment_label"],
        market_impact_rating=res["market_impact_rating"],
        key_takeaways=res["key_takeaways"],
    )
