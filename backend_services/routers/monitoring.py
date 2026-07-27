"""
backend_services/routers/monitoring.py
────────────────────────────────────────
Monitoring Layer Router.
Provides real-time system diagnostics, data quality checks, feature drift reports, and alert summaries.
"""

from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter, Depends, status

from backend_services.auth import verify_token_or_key
from backend_services.dependencies import get_monitoring_layer
from monitoring_layer import MonitoringLayer

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Diagnostics Services"])


@router.get(
    "/health-check",
    status_code=status.HTTP_200_OK,
    summary="Get Full Monitoring System Diagnostics",
)
async def get_full_diagnostics(
    monitor: MonitoringLayer = Depends(get_monitoring_layer),
    client_id: str = Depends(verify_token_or_key),
) -> Dict[str, Any]:
    """Returns consolidated system health, compute resource utilization, and recent alerts."""
    return monitor.full_health_check()


@router.get(
    "/alerts",
    status_code=status.HTTP_200_OK,
    summary="Get Recent Monitoring Alerts",
)
async def get_alerts(
    n: int = 10,
    monitor: MonitoringLayer = Depends(get_monitoring_layer),
    client_id: str = Depends(verify_token_or_key),
) -> Dict[str, Any]:
    """Returns recent alert history and summary counts."""
    alerts = monitor.recent_alerts(n)
    return {
        "summary": monitor.alert_summary(),
        "recent_alerts": [a.to_dict() for a in alerts],
    }
