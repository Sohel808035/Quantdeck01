package com.quantspherex.app.presentation.research

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quantspherex.app.data.model.ResearchAlphaItem
import com.quantspherex.app.data.repository.ResearchRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface ResearchUiState {
    data object Loading : ResearchUiState
    data class Success(val items: List<ResearchAlphaItem>) : ResearchUiState
    data class Error(val message: String) : ResearchUiState
}

class ResearchViewModel(
    private val repository: ResearchRepository = ResearchRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<ResearchUiState>(ResearchUiState.Loading)
    val uiState: StateFlow<ResearchUiState> = _uiState.asStateFlow()

    init {
        fetchResearch()
    }

    fun fetchResearch() {
        viewModelScope.launch {
            _uiState.value = ResearchUiState.Loading
            repository.getResearchAlpha().collect { result ->
                if (result.isSuccess) {
                    _uiState.value = ResearchUiState.Success(result.getOrThrow())
                } else {
                    _uiState.value = ResearchUiState.Error("Failed to fetch research signals")
                }
            }
        }
    }
}
