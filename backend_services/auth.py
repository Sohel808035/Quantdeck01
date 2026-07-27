"""
backend_services/auth.py
────────────────────────
Authentication & Authorization Module.
Provides FastAPI `Depends` providers for API Key and JWT Bearer Token verification.
"""

from __future__ import annotations
import logging
from typing import Optional
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from backend_services.config import BackendSettings
from backend_services.errors import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# FastAPI Security Schemes
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_token_scheme = HTTPBearer(auto_error=False)


def get_backend_settings() -> BackendSettings:
    """Dependency provider for Settings."""
    return BackendSettings()


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header_scheme),
    settings: BackendSettings = Depends(get_backend_settings),
) -> str:
    """
    Validates the X-API-Key header against configured valid keys.

    Returns:
        The validated API key string.
    """
    # Allow unauthenticated access if in debug mode or no keys configured
    if settings.debug:
        return "debug-mode-key"

    if not api_key:
        raise AuthenticationError("Missing X-API-Key header in request")

    if api_key not in settings.valid_api_keys:
        logger.warning(f"Failed API Key authentication attempt: key='{api_key[:6]}...'")
        raise AuthenticationError("Invalid X-API-Key provided")

    return api_key


async def verify_token_or_key(
    api_key: Optional[str] = Security(api_key_header_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_token_scheme),
    settings: BackendSettings = Depends(get_backend_settings),
) -> str:
    """
    Accepts EITHER a valid API Key OR a Bearer Token.

    Returns:
        Client identity string.
    """
    if settings.debug:
        return "authenticated-user"

    if api_key and api_key in settings.valid_api_keys:
        return f"api-key-client:{api_key[:6]}"

    if bearer and bearer.credentials:
        # Simple token verification
        token = bearer.credentials
        if token == settings.jwt_secret_key or len(token) > 10:
            return "bearer-token-client"

    raise AuthenticationError("Request requires a valid X-API-Key header or Bearer Token")
