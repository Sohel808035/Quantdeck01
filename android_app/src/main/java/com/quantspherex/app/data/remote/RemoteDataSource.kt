package com.quantspherex.app.data.remote

import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.PositionItem
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.model.UserSession
import kotlinx.coroutines.delay

class RemoteDataSource {

    suspend fun authenticate(apiKey: String): Result<UserSession> {
        delay(300) // Simulate network latency
        return if (apiKey.isNotBlank()) {
            Result.success(
                UserSession(
                    token = "jwt-session-token-${System.currentTimeMillis()}",
                    apiKey = apiKey,
                    username = "InstitutionalQuant"
                )
            )
        } else {
            Result.failure(IllegalArgumentException("Invalid API key provided"))
        }
    }

    suspend fun fetchPortfolioSummary(): Result<PortfolioSummary> {
        delay(400) // Simulate network request to /api/v2/backtest/run or /api/v2/health/status
        return Result.success(
            PortfolioSummary(
                totalAum = 11_850_000.0,
                cagrPct = 0.2666,
                sharpeRatio = 2.231,
                sortinoRatio = 3.829,
                maxDrawdownPct = -0.0424,
                currentVaR95Pct = 0.015,
                activePositionsCount = 5,
                isOfflineData = false,
                lastUpdated = "Live Server Feed"
            )
        )
    }

    suspend fun fetchResearchAlpha(): Result<List<ResearchAlphaItem>> {
        delay(450)
        return Result.success(
            listOf(
                ResearchAlphaItem(
                    symbol = "RELIANCE",
                    predictedReturnScore = 0.045,
                    direction = "BULLISH",
                    confidenceProb = 0.82,
                    executiveSummary = "Strong Outperform stance driven primarily by positive momentum in [mom_60, rsi_14].",
                    topPositiveDriver = "mom_60 (+0.035 impact)",
                    topNegativeDriver = "vol_20 (-0.012 impact)"
                ),
                ResearchAlphaItem(
                    symbol = "TCS",
                    predictedReturnScore = 0.028,
                    direction = "BULLISH",
                    confidenceProb = 0.76,
                    executiveSummary = "Moderate Outperform stance with positive sentiment following record quarterly profit.",
                    topPositiveDriver = "earnings_beat (+0.022 impact)",
                    topNegativeDriver = "currency_headwind (-0.008 impact)"
                ),
                ResearchAlphaItem(
                    symbol = "INFY",
                    predictedReturnScore = 0.035,
                    direction = "BULLISH",
                    confidenceProb = 0.79,
                    executiveSummary = "Positive alpha score backed by strong digital expansion and margin recovery.",
                    topPositiveDriver = "margin_expansion (+0.028 impact)",
                    topNegativeDriver = "attrition_rate (-0.005 impact)"
                )
            )
        )
    }
}
