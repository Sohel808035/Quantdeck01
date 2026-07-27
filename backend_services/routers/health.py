"""
backend_services/routers/health.py
───────────────────────────────────
System Health & Diagnostics Router.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status

from backend_services.config import BackendSettings
from backend_services.dependencies import get_settings, get_monitoring_layer
from backend_services.schemas import HealthStatusResponse
from monitoring_layer import MonitoringLayer

router = APIRouter(prefix="/health", tags=["Health & System Diagnostics"])
START_TIME = time.time()


@router.get(
    "/status",
    response_model=HealthStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get System Operational Health Status",
)
async def get_health_status(
    settings: BackendSettings = Depends(get_settings),
    monitor: MonitoringLayer = Depends(get_monitoring_layer),
) -> HealthStatusResponse:
    """Checks CPU/memory usage and returns service uptime and environment status."""
    health = monitor.check_system_health()
    overall = "HEALTHY" if health.get("cpu_ok", True) and health.get("memory_ok", True) else "DEGRADED"

    return HealthStatusResponse(
        status=overall,
        environment=settings.environment,
        version=settings.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )
