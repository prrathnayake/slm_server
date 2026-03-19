from __future__ import annotations

import json
import uuid
import time
from typing import Any, AsyncIterator, Dict

from local_llm_platform.runtimes.base import BaseRuntime
from local_llm_platform.core.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)
from local_llm_platform.core.schemas.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
)
from local_llm_platform.core.exceptions.errors import ModelLoadError, BackendError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("runtimes.tgi")


class TGIRuntime(BaseRuntime):
    """Runtime backend using HuggingFace Text Generation Inference."""

    def __init__(self):
        super().__init__("tgi")
        self._endpoints: Dict[str, str] = {}

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        endpoint = kwargs.get("endpoint", f"http://localhost:8080")
        self._endpoints[model_id] = endpoint
        self._loaded_models[model_id] = {
            "path": model_path,
            "backend": "tgi",
            "endpoint": endpoint,
        }
        logger.info(f"Registered TGI model {model_id} at {endpoint}")

    async def unload_model(self, model_id: str) -> None:
        self._endpoints.pop(model_id, None)
        self._loaded_models.pop(model_id, None)
        logger.info(f"Unregistered TGI model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._endpoints

    def _get_endpoint(self, model_id: str) -> str:
        if model_id not in self._endpoints:
            raise BackendError("tgi", f"Model {model_id} not registered")
        return self._endpoints[model_id]

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        import httpx

        endpoint = self._get_endpoint(model_id)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return ChatCompletionResponse(**data)

    async def chat_completion_stream(
        self, model_id: str, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        import httpx

        endpoint = self._get_endpoint(model_id)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": True,
                },
                timeout=120.0,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                    elif line.strip() == "data: [DONE]":
                        pass

        yield "data: [DONE]\n\n"

    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        import httpx

        endpoint = self._get_endpoint(model_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model_id,
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return CompletionResponse(**data)

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        info = self._loaded_models.get(model_id, {})
        return {
            "model_id": model_id,
            "backend": "tgi",
            **info,
        }
