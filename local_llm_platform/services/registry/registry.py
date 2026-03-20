from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

from local_llm_platform.core.schemas.models import ModelRegistryEntry, ModelStatus, ModelInfo, ModelListResponse
from local_llm_platform.core.exceptions.errors import ModelNotFoundError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.registry")


class ModelRegistry:
    """Service for managing model metadata and discovery."""

    def __init__(self, db_path: str = "./db/registry.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelRegistryEntry] = {}
        self._dirty = False
        self._lock_file = self.db_path.with_suffix(".lock")
        self._load()

    def _acquire_lock(self):
        if not _HAS_FCNTL:
            return None
        lock_fd = open(self._lock_file, "w")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        return lock_fd

    def _release_lock(self, lock_fd):
        if lock_fd is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                lock_fd = self._acquire_lock()
                try:
                    with open(self.db_path) as f:
                        data = json.load(f)
                    for model_id, entry_data in data.items():
                        self._models[model_id] = ModelRegistryEntry(**entry_data)
                    logger.info(f"Loaded {len(self._models)} models from registry")
                finally:
                    self._release_lock(lock_fd)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

    def _save(self) -> None:
        if not self._dirty:
            return
        data = {
            model_id: entry.model_dump()
            for model_id, entry in self._models.items()
        }
        lock_fd = self._acquire_lock()
        try:
            tmp_path = self.db_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, self.db_path)
        finally:
            self._release_lock(lock_fd)
        self._dirty = False
        logger.debug("Registry saved to disk")

    def _mark_dirty(self) -> None:
        self._dirty = True

    def save_now(self) -> None:
        """Force immediate save to disk."""
        self._save()

    def register(self, entry: ModelRegistryEntry) -> ModelRegistryEntry:
        now = datetime.now(timezone.utc)
        entry.created_at = now
        entry.updated_at = now
        self._models[entry.model_id] = entry
        self._mark_dirty()
        self._save()
        logger.info(f"Registered model: {entry.model_id}")
        return entry

    def unregister(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            self._mark_dirty()
            self._save()
            logger.info(f"Unregistered model: {model_id}")
            return True
        return False

    def get(self, model_id: str) -> ModelRegistryEntry:
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def update_status(self, model_id: str, status: ModelStatus) -> ModelRegistryEntry:
        entry = self.get(model_id)
        entry.status = status
        entry.updated_at = datetime.now(timezone.utc)
        self._mark_dirty()
        self._save()
        return entry

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        backend: Optional[str] = None,
    ) -> List[ModelRegistryEntry]:
        models = list(self._models.values())
        if status:
            models = [m for m in models if m.status == status]
        if backend:
            models = [m for m in models if m.runtime_backend.value == backend]
        return models

    def to_openai_format(self) -> ModelListResponse:
        models = []
        for entry in self._models.values():
            if entry.status == ModelStatus.READY:
                models.append(
                    ModelInfo(
                        id=entry.model_id,
                        created=int(entry.created_at.timestamp()) if entry.created_at else 0,
                        owned_by="local-provider",
                        root=entry.base_model,
                    )
                )
        return ModelListResponse(data=models)

    def search(self, query: str) -> List[ModelRegistryEntry]:
        query_lower = query.lower()
        results = []
        for entry in self._models.values():
            if (
                query_lower in entry.model_id.lower()
                or query_lower in entry.display_name.lower()
                or any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)
        return results
