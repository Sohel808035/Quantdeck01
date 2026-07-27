package com.quantspherex.app.data.model

data class UserSession(
    val token: String,
    val apiKey: String,
    val username: String,
    val isAuthenticated: Boolean = true,
    val lastLoginTimestamp: Long = System.currentTimeMillis()
)

data class PortfolioSummary(
    val totalAum: Double,
    val cagrPct: Double,
    val sharpeRatio: Double,
    val sortinoRatio: Double,
    val maxDrawdownPct: Double,
    val currentVaR95Pct: Double,
    val activePositionsCount: Int,
    val isOfflineData: Boolean = false,
    val lastUpdated: String
)

data class PositionItem(
    val symbol: str = "",
    val shares: Int = 0,
    val weightPct: Double = 0.0,
    val entryPrice: Double = 0.0,
    val currentPrice: Double = 0.0,
    val pnlPct: Double = 0.0,
    val actionStance: String = "HOLD"
)

data class ResearchAlphaItem(
    val symbol: String,
    val predictedReturnScore: Double,
    val direction: String, // "BULLISH", "BEARISH", "NEUTRAL"
    val confidenceProb: Double,
    val executiveSummary: String,
    val topPositiveDriver: String,
    val topNegativeDriver: String
)

data class ChartPoint(
    val timestampMs: Long,
    val dateLabel: String,
    val value: Double
)
