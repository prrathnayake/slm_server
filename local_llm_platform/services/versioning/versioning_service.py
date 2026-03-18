from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.versioning")


class ModelVersion:
    """Represents a single version of a model."""

    def __init__(
        self,
        model_id: str,
        version: str,
        artifact_path: str,
        manifest: Dict[str, Any],
        parent_version: Optional[str] = None,
    ):
        self.model_id = model_id
        self.version = version
        self.artifact_path = artifact_path
        self.manifest = manifest
        self.parent_version = parent_version
        self.created_at = datetime.now(timezone.utc)
        self.is_active = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "manifest": self.manifest,
            "parent_version": self.parent_version,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
        }


class VersioningService:
    """Manages model versioning and rollback."""

    def __init__(self, versions_dir: str = "./models/versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._active: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        index_file = self.versions_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                for model_id, versions_data in data.items():
                    self._versions[model_id] = []
                    for v in versions_data.get("versions", []):
                        mv = ModelVersion(
                            model_id=model_id,
                            version=v["version"],
                            artifact_path=v["artifact_path"],
                            manifest=v.get("manifest", {}),
                            parent_version=v.get("parent_version"),
                        )
                        mv.created_at = datetime.fromisoformat(v["created_at"])
                        mv.is_active = v.get("is_active", False)
                        self._versions[model_id].append(mv)
                    active = versions_data.get("active_version")
                    if active:
                        self._active[model_id] = active
                logger.info(f"Loaded versions for {len(self._versions)} models")
            except Exception as e:
                logger.warning(f"Failed to load version index: {e}")

    def _save(self) -> None:
        index_file = self.versions_dir / "index.json"
        data = {}
        for model_id, versions in self._versions.items():
            data[model_id] = {
                "active_version": self._active.get(model_id),
                "versions": [v.to_dict() for v in versions],
            }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_version(
        self,
        model_id: str,
        artifact_path: str,
        manifest: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> ModelVersion:
        if model_id not in self._versions:
            self._versions[model_id] = []

        existing = self._versions[model_id]
        if version is None:
            version = f"v{len(existing) + 1}.0.0"

        parent = self._active.get(model_id)

        mv = ModelVersion(
            model_id=model_id,
            version=version,
            artifact_path=artifact_path,
            manifest=manifest or {},
            parent_version=parent,
        )

        existing.append(mv)
        self._active[model_id] = version
        mv.is_active = True

        for v in existing:
            if v.version != version:
                v.is_active = False

        self._save()
        logger.info(f"Created version {version} for {model_id}")
        return mv

    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        active_ver = self._active.get(model_id)
        if not active_ver:
            return None
        versions = self._versions.get(model_id, [])
        for v in versions:
            if v.version == active_ver:
                return v
        return None

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        return self._versions.get(model_id, [])

    def rollback(self, model_id: str, target_version: str) -> Optional[ModelVersion]:
        versions = self._versions.get(model_id, [])
        target = None
        for v in versions:
            if v.version == target_version:
                target = v
                break

        if not target:
            logger.warning(f"Version {target_version} not found for {model_id}")
            return None

        for v in versions:
            v.is_active = (v.version == target_version)

        self._active[model_id] = target_version
        target.is_active = True
        self._save()

        logger.info(f"Rolled back {model_id} to {target_version}")
        return target

    def delete_version(self, model_id: str, version: str) -> bool:
        versions = self._versions.get(model_id, [])
        target = None
        for v in versions:
            if v.version == version:
                target = v
                break

        if not target:
            return False

        if target.is_active:
            logger.warning(f"Cannot delete active version {version}")
            return False

        versions.remove(target)
        self._save()
        logger.info(f"Deleted version {version} of {model_id}")
        return True

    def get_version_history(self, model_id: str) -> Dict[str, Any]:
        versions = self._versions.get(model_id, [])
        active = self._active.get(model_id)
        return {
            "model_id": model_id,
            "active_version": active,
            "total_versions": len(versions),
            "versions": [
                {
                    "version": v.version,
                    "created_at": v.created_at.isoformat(),
                    "is_active": v.is_active,
                    "parent": v.parent_version,
                }
                for v in versions
            ],
        }
