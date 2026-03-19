from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.concurrency")


class ConcurrencyController:
    """Controls concurrent requests per model and globally."""

    def __init__(self, max_global: int = 10, max_per_model: int = 3):
        self.max_global = max_global
        self.max_per_model = max_per_model

        self._global_semaphore = None
        self._model_semaphores: Dict[str, Any] = {}
        self._active_requests: Dict[str, int] = defaultdict(int)
        self._total_active = 0
        self._request_queue: deque = deque()
        self._dropped = 0
        self._completed = 0

    def _get_semaphore(self, model_id: str):
        import asyncio

        if self._global_semaphore is None:
            self._global_semaphore = asyncio.Semaphore(self.max_global)

        if model_id not in self._model_semaphores:
            self._model_semaphores[model_id] = asyncio.Semaphore(self.max_per_model)

        return self._global_semaphore, self._model_semaphores[model_id]

    async def acquire(self, model_id: str) -> bool:
        global_sem, model_sem = self._get_semaphore(model_id)

        g_locked = global_sem.locked()
        m_locked = model_sem.locked()

        if g_locked or m_locked:
            self._dropped += 1
            return False

        await global_sem.acquire()
        await model_sem.acquire()

        self._active_requests[model_id] += 1
        self._total_active += 1
        return True

    def release(self, model_id: str) -> None:
        global_sem, model_sem = self._get_semaphore(model_id)

        model_sem.release()
        global_sem.release()

        self._active_requests[model_id] -= 1
        self._total_active -= 1
        self._completed += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "max_global": self.max_global,
            "max_per_model": self.max_per_model,
            "total_active": self._total_active,
            "per_model": dict(self._active_requests),
            "completed": self._completed,
            "dropped": self._dropped,
        }

    def set_limits(self, max_global: Optional[int] = None, max_per_model: Optional[int] = None) -> None:
        if max_global is not None:
            self.max_global = max_global
        if max_per_model is not None:
            self.max_per_model = max_per_model
        self._model_semaphores.clear()
        self._global_semaphore = None
        logger.info(f"Concurrency limits updated: global={self.max_global}, per_model={self.max_per_model}")
