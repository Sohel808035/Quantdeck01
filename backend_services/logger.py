"""
backend_services/logger.py
──────────────────────────
Structured Logging & Request ID Middleware Module.
Generates unique Request-ID UUIDs for tracing API requests across logs.
"""

from __future__ import annotations
import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("quantspherex.backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ [%(name)s] │ %(message)s",
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique X-Request-ID header to incoming HTTP requests
    and logs request method, path, response status code, and latency in milliseconds.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        t0 = time.perf_counter()
        logger.info(f"[{request_id}] ---> {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - t0) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
            logger.info(
                f"[{request_id}] <--- {request.method} {request.url.path} "
                f"| Status: {response.status_code} | {duration_ms:.2f}ms"
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"[{request_id}] <--- {request.method} {request.url.path} "
                f"FAILED with {exc.__class__.__name__} after {duration_ms:.2f}ms"
            )
            raise exc
