"""
backend_services/errors.py
──────────────────────────
Standardized Error Handling Module.
Defines domain exceptions and global error response format.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class QuantBackendError(Exception):
    """Base exception for all QuantSphereX backend errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class AuthenticationError(QuantBackendError):
    """Raised when API Key or JWT authentication fails."""
    def __init__(self, message: str = "Invalid or missing API authentication credentials"):
        super().__init__(message, status_code=status.HTTP_401_UNAUTHORIZED)


class AuthorizationError(QuantBackendError):
    """Raised when user lacks permission for a resource."""
    def __init__(self, message: str = "Insufficient permissions for requested endpoint"):
        super().__init__(message, status_code=status.HTTP_403_FORBIDDEN)


class ResourceNotFoundError(QuantBackendError):
    """Raised when requested entity is missing."""
    def __init__(self, resource: str, resource_id: str):
        super().__init__(f"Resource '{resource}' with identifier '{resource_id}' was not found", status_code=status.HTTP_404_NOT_FOUND)


class ValidationError(QuantBackendError):
    """Raised when input parameter validation fails."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


class EngineExecutionError(QuantBackendError):
    """Raised when underlying quant engine fails during calculation."""
    def __init__(self, engine_name: str, reason: str):
        super().__init__(f"Quant engine '{engine_name}' execution error: {reason}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Global Error Handlers ──────────────────────────────────────────────────────

async def quant_backend_exception_handler(request: Request, exc: QuantBackendError) -> JSONResponse:
    """Standardized JSON error handler for domain errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Backend error: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
            "path": request.url.path,
        },
    )


async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled server exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{request_id}] Unhandled server exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error. Please check server logs.",
            "request_id": request_id,
            "path": request.url.path,
        },
    )
