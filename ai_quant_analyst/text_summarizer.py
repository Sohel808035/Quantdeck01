"""
ai_quant_analyst/text_summarizer.py
──────────────────────────────────
Earnings & News Text Summarization Module.
Parses textual earnings transcripts, SEC filings, and news feeds to compute NLP sentiment scores, key topic extraction, and market impact ratings.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional
import numpy as np

from ai_quant_analyst.config import AIAnalystConfig

logger = logging.getLogger(__name__)

# Domain lexicon for financial sentiment analysis
POSITIVE_KEYWORDS = [
    "growth", "beat", "outperform", "record", "expansion", "profitability",
    "margin expansion", "strong demand", "raise guidance", "dividend increase",
    "accelerating", "robust", "cash flow", "breakthrough", "synergies",
]

NEGATIVE_KEYWORDS = [
    "miss", "decline", "headwind", "margin compression", "cut guidance",
    "impairment", "lawsuit", "investigation", "slowdown", "weakness",
    "supply chain disruption", "loss", "restructuring", "default", "attrition",
]


class TextSummarizer:
    """
    Summarizes earnings call transcripts and unstructured financial news feeds.
    Provides sentiment analysis, impact rating, and key bullet summaries.
    """

    def __init__(self, config: Optional[AIAnalystConfig] = None):
        self.config = config or AIAnalystConfig()

    def summarize_news(
        self,
        headline: str,
        content: str,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a news article or press release.

        Returns:
            Dict containing sentiment score [-1, +1], sentiment label, impact rating, and summary bullets.
        """
        text = f"{headline} {content}".lower()

        pos_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in POSITIVE_KEYWORDS)
        neg_count = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in NEGATIVE_KEYWORDS)

        total_matches = pos_count + neg_count
        if total_matches > 0:
            sentiment_score = (pos_count - neg_count) / total_matches
        else:
            sentiment_score = 0.0

        if sentiment_score > 0.25:
            label = "POSITIVE"
            impact = "HIGH BULLISH" if sentiment_score > 0.6 else "MODERATE BULLISH"
        elif sentiment_score < -0.25:
            label = "NEGATIVE"
            impact = "HIGH BEARISH" if sentiment_score < -0.6 else "MODERATE BEARISH"
        else:
            label = "NEUTRAL"
            impact = "LOW IMPACT"

        # Split content into key sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", content) if len(s.strip()) > 20]
        summary_bullets = sentences[:3] if sentences else [headline]

        return {
            "symbol": symbol,
            "headline": headline,
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_label": label,
            "market_impact_rating": impact,
            "positive_signal_count": pos_count,
            "negative_signal_count": neg_count,
            "key_takeaways": summary_bullets,
        }

    def summarize_earnings(
        self,
        symbol: str,
        transcript_text: str,
        quarter: str = "Q4",
        year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Summarizes an earnings transcript into executive financial highlights.

        Returns:
            Dict with financial sentiment, key guidance changes, and strategic tone.
        """
        news_summary = self.summarize_news(
            headline=f"{symbol} {quarter} {year} Earnings Call",
            content=transcript_text,
            symbol=symbol,
        )

        guidance_mentioned = "guidance" in transcript_text.lower()
        margin_mentioned = "margin" in transcript_text.lower()

        bullets = news_summary["key_takeaways"]

        executive_memo = (
            f"{symbol} {quarter} {year} Earnings Analysis: "
            f"Overall tone is {news_summary['sentiment_label']} (Score: {news_summary['sentiment_score']:+.2f}). "
            f"Market Impact Rating: {news_summary['market_impact_rating']}. "
            f"{'Forward guidance discussed in detail. ' if guidance_mentioned else ''}"
            f"{'Margin dynamics highlighted. ' if margin_mentioned else ''}"
        )

        return {
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "executive_memo": executive_memo,
            "sentiment_score": news_summary["sentiment_score"],
            "sentiment_label": news_summary["sentiment_label"],
            "market_impact_rating": news_summary["market_impact_rating"],
            "guidance_discussed": guidance_mentioned,
            "margin_discussed": margin_mentioned,
            "key_highlights": bullets,
        }
