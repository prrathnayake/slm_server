from __future__ import annotations

import json
import uuid
import time
from typing import Any, AsyncIterator, Dict, Optional

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

logger = get_logger("runtimes.remote_ssh")


class RemoteSSHRuntime(BaseRuntime):
    """Runtime backend that manages models on remote SSH-accessible GPU machines."""

    def __init__(self):
        super().__init__("remote_ssh")
        self._ssh_configs: Dict[str, Dict[str, Any]] = {}

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        host = kwargs.get("host")
        if not host:
            raise ModelLoadError(model_id, "host is required for remote SSH backend")

        self._ssh_configs[model_id] = {
            "host": host,
            "port": kwargs.get("port", 22),
            "username": kwargs.get("username"),
            "key_path": kwargs.get("key_path"),
            "remote_model_path": model_path,
            "runtime_port": kwargs.get("runtime_port", 8000),
        }
        self._loaded_models[model_id] = {
            "path": model_path,
            "backend": "remote_ssh",
            "host": host,
        }
        logger.info(f"Registered SSH remote model {model_id} on {host}")

    async def unload_model(self, model_id: str) -> None:
        self._ssh_configs.pop(model_id, None)
        self._loaded_models.pop(model_id, None)
        logger.info(f"Unregistered SSH remote model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._ssh_configs

    def _get_config(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._ssh_configs:
            raise BackendError("remote_ssh", f"Model {model_id} not registered")
        return self._ssh_configs[model_id]

    def _build_remote_url(self, config: Dict[str, Any]) -> str:
        return f"http://{config['host']}:{config['runtime_port']}"

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        import httpx

        config = self._get_config(model_id)
        url = self._build_remote_url(config)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/v1/chat/completions",
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

        config = self._get_config(model_id)
        url = self._build_remote_url(config)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{url}/v1/chat/completions",
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

        yield "data: [DONE]\n\n"

    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        import httpx

        config = self._get_config(model_id)
        url = self._build_remote_url(config)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/v1/completions",
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
            "backend": "remote_ssh",
            **info,
        }
