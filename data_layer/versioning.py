"""
data_layer/versioning.py
────────────────────────
Data Versioning & Historical Dataset Lineage Manager for QuantSphereX V2.
Enables reproducible quantitative research by tagging dataset snapshots with
semantic versions, SHA-256 integrity hashes, and metadata manifests.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from data_layer.config import DataConfig

logger = logging.getLogger(__name__)


class DatasetVersionManager:
    """
    Manages dataset versions, lineage manifests, and historical dataset snapshots.
    Ensures backtests can be pinned to exact historical dataset versions.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.versions_dir = self.config.cache_dir / "versions"
        self.manifests_dir = self.config.cache_dir / "manifests"

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """Computes SHA-256 hash of DataFrame content."""
        sample_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(sample_bytes).hexdigest()[:16]

    def create_version(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
        version_id: Optional[str] = None,
    ) -> str:
        """
        Persists a versioned snapshot of a DataFrame and generates a JSON metadata manifest.
        """
        if df.empty:
            raise ValueError("[Versioning] Refusing to version an empty DataFrame.")

        hash_key = self._compute_dataset_hash(df)
        if not version_id:
            today_str = pd.Timestamp.now().strftime("%Y%m%d")
            version_id = f"{dataset_name}_v{today_str}_{hash_key[:8]}"

        file_name = f"{version_id}.parquet"
        file_path = self.versions_dir / file_name
        manifest_path = self.manifests_dir / f"{version_id}.json"

        # 1. Save versioned Parquet snapshot
        df.to_parquet(file_path, compression="snappy")

        # 2. Write metadata manifest
        manifest = {
            "version_id": version_id,
            "dataset_name": dataset_name,
            "hash_key": hash_key,
            "row_count": len(df),
            "columns": [str(c) for c in df.columns],
            "start_date": start_date,
            "end_date": end_date,
            "file_path": str(file_path.resolve()),
            "created_at": pd.Timestamp.now().isoformat(),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"[Versioning] Created version snapshot '{version_id}' ({len(df):,} rows)")
        return version_id

    def load_version(self, version_id: str) -> pd.DataFrame:
        """Loads a specific version snapshot by version_id."""
        manifest = self.get_manifest(version_id)
        file_path = Path(manifest["file_path"])

        if not file_path.exists():
            # Try relative path fallback inside versions_dir
            file_path = self.versions_dir / f"{version_id}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"[Versioning] Version file not found for '{version_id}'")

        logger.info(f"[Versioning] Loading dataset version '{version_id}'")
        return pd.read_parquet(file_path)

    def get_manifest(self, version_id: str) -> Dict[str, Any]:
        """Retrieves JSON metadata manifest for a dataset version."""
        manifest_path = self.manifests_dir / f"{version_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"[Versioning] Manifest not found for version '{version_id}'")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_versions(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered dataset versions, optionally filtered by dataset_name."""
        manifest_files = list(self.manifests_dir.glob("*.json"))
        manifests = []

        for m_file in manifest_files:
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if dataset_name is None or data.get("dataset_name") == dataset_name:
                        manifests.append(data)
            except Exception:
                continue

        manifests.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return manifests
