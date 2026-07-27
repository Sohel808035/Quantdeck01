"""
ml_layer/registry.py
──────────────────────
ML Pipeline: Model Registry (v1.0.0)

Persists trained models and their metadata to disk with:
  - JSON manifest per model version
  - XGBoost native JSON model serialization
  - Version tagging (timestamp + IC hash)
  - Load, list, delete, and compare registry entries
  - Promotion of best model to 'champion' slot
"""

from __future__ import annotations
import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from alpha_layer.xgboost_trainer import EnsembleAlphaModel, XGBoostAlphaModel
from ml_layer.config import MLConfig

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Metadata record for a persisted model version."""
    version_id:      str
    experiment_name: str
    train_ic:        float
    val_ic:          float
    overfit_score:   float
    n_train_rows:    int
    n_val_rows:      int
    n_ensemble:      int
    features:        List[str]
    params:          Dict[str, Any]
    created_at:      str
    is_champion:     bool = False
    notes:           str = ""


class ModelRegistry:
    """
    File-based model registry that persists trained EnsembleAlphaModel objects.

    Directory layout:
        registry_dir/
            {version_id}/
                manifest.json
                model_{i}.json   (one per ensemble member)
    """

    def __init__(self, registry_dir: str = "ml_layer/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ── Saving ─────────────────────────────────────────────────────────────────

    def save(
        self,
        model: EnsembleAlphaModel,
        train_result,
        notes: str = "",
    ) -> str:
        """
        Persists a trained EnsembleAlphaModel with its training metadata.

        Args:
            model:        Trained EnsembleAlphaModel instance.
            train_result: TrainResult DTO from ml_layer.training.train().
            notes:        Optional human-readable annotation.

        Returns:
            version_id string.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        ic_hash = hashlib.md5(f"{train_result.val_ic:.6f}".encode()).hexdigest()[:6]
        version_id = f"{train_result.experiment_name}_{ts}_{ic_hash}"

        model_dir = self.registry_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save each ensemble member
        for i, m in enumerate(model.models):
            if m.model is not None:
                m.model.save_model(str(model_dir / f"model_{i}.json"))

        # Save manifest
        entry = RegistryEntry(
            version_id=version_id,
            experiment_name=train_result.experiment_name,
            train_ic=train_result.train_ic,
            val_ic=train_result.val_ic,
            overfit_score=train_result.overfit_score,
            n_train_rows=train_result.n_train_rows,
            n_val_rows=train_result.n_val_rows,
            n_ensemble=len(model.models),
            features=train_result.features,
            params=train_result.params_used,
            created_at=ts,
            notes=notes,
        )
        with open(model_dir / "manifest.json", "w") as f:
            json.dump(asdict(entry), f, indent=2)

        logger.info(f"[Registry] Model saved: {version_id}")
        return version_id

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, version_id: str) -> tuple[EnsembleAlphaModel, RegistryEntry]:
        """
        Loads a persisted EnsembleAlphaModel and its metadata manifest.

        Args:
            version_id: The version_id string returned by save().

        Returns:
            Tuple of (EnsembleAlphaModel, RegistryEntry).
        """
        import xgboost as xgb

        model_dir = self.registry_dir / version_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Registry entry '{version_id}' not found.")

        with open(model_dir / "manifest.json") as f:
            manifest = json.load(f)
        entry = RegistryEntry(**manifest)

        ensemble = EnsembleAlphaModel(n_models=entry.n_ensemble)
        ensemble.models = []
        for i in range(entry.n_ensemble):
            model_path = model_dir / f"model_{i}.json"
            if model_path.exists():
                m = XGBoostAlphaModel()
                m.model = xgb.XGBRegressor()
                m.model.load_model(str(model_path))
                m.features = entry.features
                ensemble.models.append(m)

        logger.info(f"[Registry] Loaded: {version_id} (Val IC={entry.val_ic:.4f})")
        return ensemble, entry

    # ── Listing & Management ───────────────────────────────────────────────────

    def list_versions(self) -> pd.DataFrame:
        """Returns a DataFrame of all persisted model versions sorted by Val IC."""
        rows = []
        for version_dir in sorted(self.registry_dir.iterdir()):
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                data = json.load(f)
            rows.append(data)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("val_ic", ascending=False).reset_index(drop=True)
        return df[["version_id", "experiment_name", "train_ic", "val_ic",
                   "overfit_score", "n_ensemble", "is_champion", "created_at", "notes"]]

    def promote_champion(self, version_id: str) -> None:
        """Promotes a model version to champion status and demotes all others."""
        for version_dir in self.registry_dir.iterdir():
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                data = json.load(f)
            data["is_champion"] = (version_dir.name == version_id)
            with open(manifest_path, "w") as f:
                json.dump(data, f, indent=2)

        logger.info(f"[Registry] Champion promoted: {version_id}")

    def get_champion(self) -> Optional[str]:
        """Returns the version_id of the current champion model, or None."""
        for version_dir in sorted(self.registry_dir.iterdir()):
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                data = json.load(f)
            if data.get("is_champion", False):
                return version_dir.name
        return None

    def delete(self, version_id: str) -> None:
        """Deletes a model version from the registry."""
        import shutil
        model_dir = self.registry_dir / version_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"[Registry] Deleted: {version_id}")
