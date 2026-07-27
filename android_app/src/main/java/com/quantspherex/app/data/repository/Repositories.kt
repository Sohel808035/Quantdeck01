package com.quantspherex.app.data.repository

import com.quantspherex.app.data.local.LocalCacheDataSource
import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.PositionItem
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.model.UserSession
import com.quantspherex.app.data.remote.RemoteDataSource
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class AuthRepository(
    private val remote: RemoteDataSource = RemoteDataSource(),
    private val local: LocalCacheDataSource = LocalCacheDataSource()
) {
    suspend fun login(apiKey: String): Result<UserSession> {
        val result = remote.authenticate(apiKey)
        if (result.isSuccess) {
            local.saveSession(result.getOrThrow())
        }
        return result
    }

    fun getActiveSession(): UserSession? = local.getSession()
}

class PortfolioRepository(
    private val remote: RemoteDataSource = RemoteDataSource(),
    private val local: LocalCacheDataSource = LocalCacheDataSource()
) {
    fun getPortfolioSummary(): Flow<Result<PortfolioSummary>> = flow {
        val remoteResult = remote.fetchPortfolioSummary()
        if (remoteResult.isSuccess) {
            val summary = remoteResult.getOrThrow()
            local.cachePortfolio(summary, local.getFallbackPositions())
            emit(Result.success(summary))
        } else {
            // Offline fallback
            val offlineSummary = local.getFallbackPortfolio()
            emit(Result.success(offlineSummary))
        }
    }

    fun getPositions(): Flow<List<PositionItem>> = flow {
        emit(local.getFallbackPositions())
    }
}

class ResearchRepository(
    private val remote: RemoteDataSource = RemoteDataSource(),
    private val local: LocalCacheDataSource = LocalCacheDataSource()
) {
    fun getResearchAlpha(): Flow<Result<List<ResearchAlphaItem>>> = flow {
        val remoteResult = remote.fetchResearchAlpha()
        if (remoteResult.isSuccess) {
            val items = remoteResult.getOrThrow()
            local.cacheResearch(items)
            emit(Result.success(items))
        } else {
            // Offline fallback
            emit(Result.success(emptyList()))
        }
    }
}
