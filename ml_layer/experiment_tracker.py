"""
ml_layer/experiment_tracker.py
────────────────────────────────
ML Pipeline: Experiment Tracking Module (v1.0.0)

Lightweight file-based experiment tracker (no MLflow dependency) that:
  - Logs every training run with metrics, params, and timestamps
  - Provides run comparison table
  - Supports run tagging and annotation
  - Exports full run history to CSV
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """A single experiment run record."""
    run_id:          str
    experiment_name: str
    status:          str               # 'running' | 'completed' | 'failed'
    train_ic:        float = 0.0
    val_ic:          float = 0.0
    overfit_score:   float = 0.0
    decile_sharpe:   float = 0.0
    ic_tstat:        float = 0.0
    n_train_rows:    int   = 0
    n_val_rows:      int   = 0
    elapsed_seconds: float = 0.0
    params:          Dict[str, Any] = field(default_factory=dict)
    tags:            Dict[str, str] = field(default_factory=dict)
    notes:           str  = ""
    started_at:      str  = ""
    finished_at:     str  = ""


class ExperimentTracker:
    """
    Lightweight file-based experiment tracker.

    Persists run records as JSON Lines in:
        tracking_dir/{experiment_name}/runs.jsonl
    """

    def __init__(self, tracking_dir: str = "ml_layer/experiments"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def _run_file(self, experiment_name: str) -> Path:
        exp_dir = self.tracking_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir / "runs.jsonl"

    # ── Run Lifecycle ──────────────────────────────────────────────────────────

    def start_run(
        self,
        experiment_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> str:
        """
        Starts a new experiment run and returns the run_id.

        Args:
            experiment_name: Experiment label (groups runs together).
            params:          Hyperparameter dict to log.
            tags:            Arbitrary key-value metadata (e.g., {'env': 'prod'}).
            notes:           Free-text annotation.

        Returns:
            run_id string (UUID).
        """
        run_id = str(uuid.uuid4())[:8]
        record = RunRecord(
            run_id=run_id,
            experiment_name=experiment_name,
            status="running",
            params=params or {},
            tags=tags or {},
            notes=notes,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._append_run(record)
        logger.info(f"[Tracker] Run started: {run_id} | experiment='{experiment_name}'")
        return run_id

    def log_metrics(
        self,
        run_id: str,
        experiment_name: str,
        metrics: Dict[str, float],
        elapsed_seconds: float = 0.0,
    ) -> None:
        """
        Updates a run record with evaluation metrics.

        Args:
            run_id:          The run_id from start_run().
            experiment_name: Experiment label.
            metrics:         Dict of metric name → value.
            elapsed_seconds: Wall-clock training time.
        """
        record = self._load_run(run_id, experiment_name)
        if record is None:
            logger.warning(f"[Tracker] Run '{run_id}' not found.")
            return

        record.train_ic      = metrics.get("train_ic", record.train_ic)
        record.val_ic        = metrics.get("val_ic", record.val_ic)
        record.overfit_score = metrics.get("overfit_score", record.overfit_score)
        record.decile_sharpe = metrics.get("decile_sharpe", record.decile_sharpe)
        record.ic_tstat      = metrics.get("ic_tstat", record.ic_tstat)
        record.n_train_rows  = int(metrics.get("n_train_rows", record.n_train_rows))
        record.n_val_rows    = int(metrics.get("n_val_rows", record.n_val_rows))
        record.elapsed_seconds = elapsed_seconds
        record.status        = "completed"
        record.finished_at   = time.strftime("%Y-%m-%dT%H:%M:%S")

        self._update_run(run_id, experiment_name, record)
        logger.info(
            f"[Tracker] Logged | run={run_id} | "
            f"Val IC={record.val_ic:.4f} | t-stat={record.ic_tstat:.2f}"
        )

    def fail_run(self, run_id: str, experiment_name: str, error: str = "") -> None:
        """Marks a run as failed."""
        record = self._load_run(run_id, experiment_name)
        if record:
            record.status = "failed"
            record.notes += f" | ERROR: {error}"
            record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._update_run(run_id, experiment_name, record)

    # ── Querying ───────────────────────────────────────────────────────────────

    def get_runs(self, experiment_name: str) -> pd.DataFrame:
        """Returns all runs for an experiment as a sorted DataFrame."""
        run_file = self._run_file(experiment_name)
        if not run_file.exists():
            return pd.DataFrame()

        rows = []
        with open(run_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        # Deduplicate: keep latest version of each run_id
        seen = {}
        for r in rows:
            seen[r["run_id"]] = r
        rows = list(seen.values())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        priority_cols = ["run_id", "status", "train_ic", "val_ic", "overfit_score",
                         "decile_sharpe", "ic_tstat", "elapsed_seconds", "started_at"]
        cols = [c for c in priority_cols if c in df.columns]
        return df[cols].sort_values("val_ic", ascending=False).reset_index(drop=True)

    def export_csv(self, experiment_name: str, path: str) -> None:
        """Exports full run history to CSV."""
        df = self.get_runs(experiment_name)
        df.to_csv(path, index=False)
        logger.info(f"[Tracker] Exported {len(df)} runs to {path}")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _append_run(self, record: RunRecord) -> None:
        run_file = self._run_file(record.experiment_name)
        with open(run_file, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def _load_run(self, run_id: str, experiment_name: str) -> Optional[RunRecord]:
        run_file = self._run_file(experiment_name)
        if not run_file.exists():
            return None
        latest = None
        with open(run_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("run_id") == run_id:
                    latest = RunRecord(**data)
        return latest

    def _update_run(self, run_id: str, experiment_name: str, record: RunRecord) -> None:
        """Appends updated record (deduplication done at read time)."""
        self._append_run(record)
