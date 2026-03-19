from __future__ import annotations

from typing import Any, AsyncIterator, Dict

from local_llm_platform.runtimes.base import BaseRuntime
from local_llm_platform.core.schemas.chat import ChatCompletionRequest
from local_llm_platform.core.schemas.completion import CompletionRequest
from local_llm_platform.core.exceptions.errors import ModelLoadError, BackendError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("runtimes.remote_http")


class RemoteHTTPRuntime(BaseRuntime):
    """Runtime backend that proxies to a remote HTTP inference server."""

    def __init__(self):
        super().__init__("remote_http")
        self._remote_endpoints: Dict[str, Dict[str, Any]] = {}

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        endpoint_url = kwargs.get("endpoint_url")
        if not endpoint_url:
            raise ModelLoadError(model_id, "endpoint_url is required for remote HTTP backend")

        api_key = kwargs.get("api_key")
        self._remote_endpoints[model_id] = {
            "url": endpoint_url.rstrip("/"),
            "api_key": api_key,
            "model_name": kwargs.get("remote_model_name", model_id),
        }
        self._loaded_models[model_id] = {
            "path": model_path,
            "backend": "remote_http",
            "endpoint": endpoint_url,
        }
        logger.info(f"Registered remote HTTP model {model_id} at {endpoint_url}")

    async def unload_model(self, model_id: str) -> None:
        self._remote_endpoints.pop(model_id, None)
        self._loaded_models.pop(model_id, None)
        logger.info(f"Unregistered remote HTTP model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._remote_endpoints

    def _get_endpoint(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._remote_endpoints:
            raise BackendError("remote_http", f"Model {model_id} not registered")
        return self._remote_endpoints[model_id]

    def _get_headers(self, endpoint: Dict[str, Any]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if endpoint.get("api_key"):
            headers["Authorization"] = f"Bearer {endpoint['api_key']}"
        return headers

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        import httpx

        endpoint = self._get_endpoint(model_id)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint['url']}/v1/chat/completions",
                json={
                    "model": endpoint["model_name"],
                    "messages": messages,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                },
                headers=self._get_headers(endpoint),
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        data["model"] = model_id
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
                f"{endpoint['url']}/v1/chat/completions",
                json={
                    "model": endpoint["model_name"],
                    "messages": messages,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": True,
                },
                headers=self._get_headers(endpoint),
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

        endpoint = self._get_endpoint(model_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint['url']}/v1/completions",
                json={
                    "model": endpoint["model_name"],
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                },
                headers=self._get_headers(endpoint),
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        data["model"] = model_id
        return CompletionResponse(**data)

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        info = self._loaded_models.get(model_id, {})
        endpoint = self._get_endpoint(model_id)
        return {
            "model_id": model_id,
            "backend": "remote_http",
            "remote_model": endpoint.get("model_name"),
            **info,
        }
