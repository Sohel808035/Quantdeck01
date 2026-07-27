package com.quantspherex.app.presentation.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quantspherex.app.data.model.UserSession
import com.quantspherex.app.data.repository.AuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface AuthUiState {
    data object Idle : AuthUiState
    data object Loading : AuthUiState
    data class Authenticated(val session: UserSession) : AuthUiState
    data class Error(val message: String) : AuthUiState
}

class AuthViewModel(
    private val authRepository: AuthRepository = AuthRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<AuthUiState>(AuthUiState.Idle)
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    init {
        checkExistingSession()
    }

    fun checkExistingSession() {
        val session = authRepository.getActiveSession()
        if (session != null && session.isAuthenticated) {
            _uiState.value = AuthUiState.Authenticated(session)
        }
    }

    fun login(apiKey: String) {
        viewModelScope.launch {
            _uiState.value = AuthUiState.Loading
            val result = authRepository.login(apiKey)
            if (result.isSuccess) {
                _uiState.value = AuthUiState.Authenticated(result.getOrThrow())
            } else {
                _uiState.value = AuthUiState.Error(result.exceptionOrNull()?.message ?: "Authentication failed")
            }
        }
    }
}
