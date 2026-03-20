from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.pool")


class ModelPool:
    """Manages hot/cold model loading with LRU eviction."""

    def __init__(self, max_loaded: int = 3):
        self.max_loaded = max_loaded
        self._hot_models: List[str] = []
        self._loaded_order: OrderedDict[str, float] = OrderedDict()
        self._model_info: Dict[str, Dict[str, Any]] = {}

    def set_hot_models(self, model_ids: List[str]) -> None:
        self._hot_models = model_ids
        logger.info(f"Hot models set: {model_ids}")

    def register_loaded(self, model_id: str, backend: str, path: str) -> None:
        self._loaded_order[model_id] = time.time()
        self._loaded_order.move_to_end(model_id)
        self._model_info[model_id] = {"backend": backend, "path": path, "loaded_at": time.time()}

    def register_unloaded(self, model_id: str) -> None:
        self._loaded_order.pop(model_id, None)
        self._model_info.pop(model_id, None)

    def get_eviction_candidate(self) -> Optional[str]:
        if len(self._loaded_order) < self.max_loaded:
            return None

        for model_id in self._loaded_order.keys():
            if model_id not in self._hot_models:
                return model_id

        return None

    def touch(self, model_id: str) -> None:
        if model_id in self._loaded_order:
            self._loaded_order.move_to_end(model_id)

    def get_status(self) -> Dict[str, Any]:
        loaded = list(self._loaded_order.keys())
        cold = [m for m in loaded if m not in self._hot_models]
        hot = [m for m in loaded if m in self._hot_models]

        return {
            "max_loaded": self.max_loaded,
            "currently_loaded": len(loaded),
            "hot_models": hot,
            "cold_models": cold,
            "all_loaded": loaded,
            "eviction_candidate": self.get_eviction_candidate(),
        }

    def should_load(self, model_id: str) -> bool:
        if model_id in self._loaded_order:
            return False
        if len(self._loaded_order) < self.max_loaded:
            return True
        return self.get_eviction_candidate() is not None

    def get_load_plan(self, model_id: str) -> Dict[str, Any]:
        if model_id in self._loaded_order:
            return {"action": "already_loaded", "model_id": model_id}

        if len(self._loaded_order) < self.max_loaded:
            return {"action": "load", "model_id": model_id, "evict": None}

        candidate = self.get_eviction_candidate()
        if candidate:
            return {"action": "load", "model_id": model_id, "evict": candidate}

        return {"action": "reject", "model_id": model_id, "reason": "pool full, no evictable models"}
