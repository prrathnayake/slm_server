from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
import asyncio

from local_llm_platform.core.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from local_llm_platform.core.schemas.completion import CompletionRequest, CompletionResponse


class BaseRuntime(ABC):
    """Abstract base class for all model runtime backends."""

    def __init__(self, backend_name: str, max_concurrent: int = 1):
        self.backend_name = backend_name
        self._loaded_models: Dict[str, Any] = {}
        self._model_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._default_semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent = max_concurrent

    def _get_semaphore(self, model_id: str) -> asyncio.Semaphore:
        if model_id not in self._model_semaphores:
            self._model_semaphores[model_id] = asyncio.Semaphore(self._max_concurrent)
        return self._model_semaphores[model_id]

    async def _run_in_executor(self, model_id: str, func, *args):
        sem = self._get_semaphore(model_id)
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args)

    @abstractmethod
    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        """Load a model into memory."""
        ...

    @abstractmethod
    async def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        ...

    @abstractmethod
    async def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        ...

    @abstractmethod
    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        ...

    @abstractmethod
    async def chat_completion_stream(
        self, model_id: str, request: ChatCompletionRequest
    ):
        """Generate a streaming chat completion. Yields SSE chunks."""
        ...

    @abstractmethod
    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        """Generate a text completion."""
        ...

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get runtime info about a loaded model."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Check runtime health."""
        return {
            "backend": self.backend_name,
            "loaded_models": list(self._loaded_models.keys()),
            "num_loaded": len(self._loaded_models),
        }

    def list_loaded_models(self) -> List[str]:
        """List all currently loaded model IDs."""
        return list(self._loaded_models.keys())
