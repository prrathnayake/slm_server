from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from local_llm_platform.core.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from local_llm_platform.core.schemas.completion import CompletionRequest, CompletionResponse


class BaseRuntime(ABC):
    """Abstract base class for all model runtime backends."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self._loaded_models: Dict[str, Any] = {}

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
    ) -> AsyncIterator[str]:
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
