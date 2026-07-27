package com.quantspherex.app.presentation.portfolio

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quantspherex.app.data.model.PortfolioSummary
import com.quantspherex.app.data.model.PositionItem
import com.quantspherex.app.data.repository.PortfolioRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface PortfolioUiState {
    data object Loading : PortfolioUiState
    data class Success(
        val summary: PortfolioSummary,
        val positions: List<PositionItem>
    ) : PortfolioUiState
    data class Error(val message: String) : PortfolioUiState
}

class PortfolioViewModel(
    private val repository: PortfolioRepository = PortfolioRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<PortfolioUiState>(PortfolioUiState.Loading)
    val uiState: StateFlow<PortfolioUiState> = _uiState.asStateFlow()

    init {
        fetchPortfolio()
    }

    fun fetchPortfolio() {
        viewModelScope.launch {
            _uiState.value = PortfolioUiState.Loading
            repository.getPortfolioSummary().collect { summaryResult ->
                if (summaryResult.isSuccess) {
                    val summary = summaryResult.getOrThrow()
                    repository.getPositions().collect { positions ->
                        _uiState.value = PortfolioUiState.Success(summary, positions)
                    }
                } else {
                    _uiState.value = PortfolioUiState.Error("Failed to load portfolio metrics")
                }
            }
        }
    }
}
