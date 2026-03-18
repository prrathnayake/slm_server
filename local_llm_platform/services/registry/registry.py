from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)
                for model_id, entry_data in data.items():
                    self._models[model_id] = ModelRegistryEntry(**entry_data)
                logger.info(f"Loaded {len(self._models)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

    def _save(self) -> None:
        data = {
            model_id: entry.model_dump()
            for model_id, entry in self._models.items()
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, entry: ModelRegistryEntry) -> ModelRegistryEntry:
        now = datetime.now(timezone.utc)
        entry.created_at = now
        entry.updated_at = now
        self._models[entry.model_id] = entry
        self._save()
        logger.info(f"Registered model: {entry.model_id}")
        return entry

    def unregister(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
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
