package com.quantspherex.app.data.local

import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.PositionItem
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.model.UserSession
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class LocalCacheDataSource {

    private val _cachedPortfolio = MutableStateFlow<PortfolioSummary?>(null)
    val cachedPortfolio: Flow<PortfolioSummary?> = _cachedPortfolio.asStateFlow()

    private val _cachedPositions = MutableStateFlow<List<PositionItem>>(emptyList())
    val cachedPositions: Flow<List<PositionItem>> = _cachedPositions.asStateFlow()

    private val _cachedResearch = MutableStateFlow<List<ResearchAlphaItem>>(emptyList())
    val cachedResearch: Flow<List<ResearchAlphaItem>> = _cachedResearch.asStateFlow()

    private var _cachedSession: UserSession? = UserSession(
        token = "jwt-cached-offline-token",
        apiKey = "qsx-secret-api-key-2026",
        username = "InstitutionalQuant"
    )

    fun saveSession(session: UserSession) {
        _cachedSession = session
    }

    fun getSession(): UserSession? = _cachedSession

    fun cachePortfolio(summary: PortfolioSummary, positions: List<PositionItem>) {
        _cachedPortfolio.value = summary.copy(isOfflineData = true)
        _cachedPositions.value = positions
    }

    fun cacheResearch(researchList: List<ResearchAlphaItem>) {
        _cachedResearch.value = researchList
    }

    fun getFallbackPortfolio(): PortfolioSummary {
        return _cachedPortfolio.value ?: PortfolioSummary(
            totalAum = 10_000_000.0,
            cagrPct = 0.185,
            sharpeRatio = 1.30,
            sortinoRatio = 1.85,
            maxDrawdownPct = -0.115,
            currentVaR95Pct = 0.018,
            activePositionsCount = 5,
            isOfflineData = true,
            lastUpdated = "Cached Offline State"
        )
    }

    fun getFallbackPositions(): List<PositionItem> {
        return _cachedPositions.value.ifEmpty {
            listOf(
                PositionItem(symbol = "RELIANCE", shares = 450, weightPct = 0.25, entryPrice = 2450.0, currentPrice = 2680.0, pnlPct = 0.093, actionStance = "BUY"),
                PositionItem(symbol = "TCS", shares = 200, weightPct = 0.20, entryPrice = 3400.0, currentPrice = 3820.0, pnlPct = 0.123, actionStance = "HOLD"),
                PositionItem(symbol = "INFY", shares = 600, weightPct = 0.20, entryPrice = 1420.0, currentPrice = 1590.0, pnlPct = 0.119, actionStance = "BUY"),
                PositionItem(symbol = "HDFCBANK", shares = 350, weightPct = 0.20, entryPrice = 1580.0, currentPrice = 1680.0, pnlPct = 0.063, actionStance = "HOLD"),
                PositionItem(symbol = "ICICIBANK", shares = 500, weightPct = 0.15, entryPrice = 920.0, currentPrice = 1040.0, pnlPct = 0.130, actionStance = "BUY")
            )
        }
    }
}
