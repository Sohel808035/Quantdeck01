"""
monitoring_layer/dashboard.py
──────────────────────────────
Logging Dashboard.
Renders a rich terminal dashboard with monitoring status tables and alert feed.
Falls back gracefully when 'rich' library is not installed.
"""

from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _try_rich():
    """Returns (Console, Table, Text, Panel, Columns, box) or None if rich unavailable."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich.columns import Columns
        from rich import box
        return Console, Table, Text, Panel, Columns, box
    except ImportError:
        return None


class MonitoringDashboard:
    """
    Renders a structured, colour-coded monitoring dashboard to the terminal.
    Uses the 'rich' library if available, otherwise falls back to plain-text output.
    """

    def __init__(self, service_name: str = "QuantSphereX Monitoring"):
        self.service_name = service_name
        self._rich = _try_rich()

    def render(
        self,
        health_report: Optional[Dict[str, Any]] = None,
        data_quality_report: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
        strategy_report: Optional[Dict[str, Any]] = None,
        recent_alerts: Optional[List[Any]] = None,
    ) -> None:
        """Renders the full monitoring dashboard to the terminal."""
        if self._rich:
            self._render_rich(health_report, data_quality_report, drift_report, strategy_report, recent_alerts)
        else:
            self._render_plain(health_report, data_quality_report, drift_report, strategy_report, recent_alerts)

    # ── Rich Rendering ──────────────────────────────────────────────────────

    def _render_rich(self, health, dq, drift, strategy, alerts):
        Console, Table, Text, Panel, Columns, box = self._rich
        console = Console()

        console.rule(f"[bold cyan]  {self.service_name}  [/bold cyan]")
        console.print(f"[dim]{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}[/dim]\n")

        panels = []

        # ── System Health ──────────────────────────────────────────────────
        if health:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
            tbl.add_column("Metric", style="cyan")
            tbl.add_column("Value", justify="right")
            tbl.add_column("Status", justify="center")

            cpu = health.get("cpu_pct")
            mem = health.get("memory_pct")
            if cpu is not None:
                cpu_status = "🟢 OK" if cpu < 75 else "🟡 WARN" if cpu < 90 else "🔴 CRIT"
                tbl.add_row("CPU Usage", f"{cpu:.1f}%", cpu_status)
            if mem is not None:
                mem_status = "🟢 OK" if mem < 80 else "🟡 WARN" if mem < 95 else "🔴 CRIT"
                tbl.add_row("Memory Usage", f"{mem:.1f}%", mem_status)
                tbl.add_row("Memory Used", f"{health.get('memory_used_gb', 0):.2f} GB", "")

            panels.append(Panel(tbl, title="[bold]System Health[/bold]", border_style="blue"))

        # ── Data Quality ───────────────────────────────────────────────────
        if dq:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
            tbl.add_column("Check", style="cyan")
            tbl.add_column("Result", justify="center")
            checks = dq.get("checks", {})
            for check_name, result in checks.items():
                passed = result.get("passed", True)
                icon = "✅ PASS" if passed else "❌ FAIL"
                tbl.add_row(check_name.replace("_", " ").title(), icon)
            panels.append(Panel(tbl, title="[bold]Data Quality[/bold]", border_style="green"))

        # ── Drift ──────────────────────────────────────────────────────────
        if drift:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
            tbl.add_column("Feature", style="cyan")
            tbl.add_column("PSI", justify="right")
            tbl.add_column("KS p-val", justify="right")
            tbl.add_column("Drifted", justify="center")
            for feat, metrics in drift.get("features", {}).items():
                drifted = metrics.get("drifted", False)
                icon = "🔴 YES" if drifted else "🟢 NO"
                tbl.add_row(
                    feat,
                    f"{metrics.get('psi', 0):.4f}",
                    f"{metrics.get('ks_pvalue', 1):.4f}",
                    icon,
                )
            panels.append(Panel(tbl, title="[bold]Feature Drift[/bold]", border_style="yellow"))

        # ── Strategy ──────────────────────────────────────────────────────
        if strategy:
            tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
            tbl.add_column("Metric", style="cyan")
            tbl.add_column("Value", justify="right")
            tbl.add_column("Status", justify="center")

            for key, label in [
                ("latest_sharpe", "Rolling Sharpe"),
                ("latest_ic", "Rolling IC"),
                ("current_drawdown", "Current Drawdown"),
            ]:
                if key in strategy:
                    val = strategy[key]
                    status = "🟢 OK" if not strategy.get("breach", False) else "🔴 BREACH"
                    tbl.add_row(label, f"{val:.4f}", status)
            panels.append(Panel(tbl, title="[bold]Strategy Monitor[/bold]", border_style="magenta"))

        if panels:
            console.print(Columns(panels, equal=True))

        # ── Recent Alerts ─────────────────────────────────────────────────
        if alerts:
            alert_tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold red")
            alert_tbl.add_column("Time", style="dim")
            alert_tbl.add_column("Severity", justify="center")
            alert_tbl.add_column("Category")
            alert_tbl.add_column("Metric")
            alert_tbl.add_column("Message")

            sev_colors = {"CRITICAL": "bold red", "WARNING": "yellow", "INFO": "cyan"}
            for a in alerts[-10:]:
                sev = a.severity.value
                color = sev_colors.get(sev, "white")
                ts = time.strftime("%H:%M:%S", time.gmtime(a.timestamp))
                alert_tbl.add_row(ts, f"[{color}]{sev}[/{color}]", a.category, a.metric, a.message)

            console.print(Panel(alert_tbl, title="[bold red]Recent Alerts[/bold red]", border_style="red"))

        console.rule()

    # ── Plain-Text Fallback ─────────────────────────────────────────────────

    def _render_plain(self, health, dq, drift, strategy, alerts):
        lines = [
            "=" * 70,
            f"  {self.service_name}",
            f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "=" * 70,
        ]
        if health:
            lines += ["[SYSTEM HEALTH]"]
            for k, v in health.items():
                if v is not None:
                    lines.append(f"  {k}: {v}")
        if dq:
            lines += ["[DATA QUALITY]"]
            for check, res in dq.get("checks", {}).items():
                status = "PASS" if res.get("passed", True) else "FAIL"
                lines.append(f"  {check}: {status}")
        if strategy:
            lines += ["[STRATEGY MONITOR]"]
            for k, v in strategy.items():
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")
        if alerts:
            lines += ["[RECENT ALERTS]"]
            for a in alerts[-10:]:
                lines.append(f"  [{a.severity.value}] {a.category} | {a.metric}: {a.message}")
        lines.append("=" * 70)
        print("\n".join(lines))
