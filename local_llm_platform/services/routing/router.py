from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from local_llm_platform.runtimes.base import BaseRuntime
from local_llm_platform.runtimes.llama_cpp_runtime import LlamaCppRuntime
from local_llm_platform.runtimes.vllm_runtime import VLLMRuntime
from local_llm_platform.runtimes.tgi_runtime import TGIRuntime
from local_llm_platform.runtimes.remote_http_runtime import RemoteHTTPRuntime
from local_llm_platform.runtimes.remote_ssh_runtime import RemoteSSHRuntime
from local_llm_platform.runtimes.transformers_runtime import TransformersRuntime
from local_llm_platform.services.registry.registry import ModelRegistry
from local_llm_platform.core.schemas.models import ModelStatus, BackendType
from local_llm_platform.core.schemas.chat import ChatCompletionRequest
from local_llm_platform.core.schemas.completion import CompletionRequest
from local_llm_platform.core.exceptions.errors import (
    ModelNotFoundError,
    ModelNotReadyError,
    BackendError,
)
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.routing")


class RuntimeRouter:
    """Routes requests to the correct runtime backend based on model registry."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._runtimes: Dict[str, BaseRuntime] = {}
        self._register_default_runtimes()

    def _register_default_runtimes(self) -> None:
        self._runtimes["llama_cpp"] = LlamaCppRuntime()
        self._runtimes["vllm"] = VLLMRuntime()
        self._runtimes["tgi"] = TGIRuntime()
        self._runtimes["remote_http"] = RemoteHTTPRuntime()
        self._runtimes["remote_ssh"] = RemoteSSHRuntime()
        self._runtimes["transformers"] = TransformersRuntime()
        logger.info("Registered default runtime backends")

    def get_runtime(self, backend_name: str) -> BaseRuntime:
        if backend_name not in self._runtimes:
            raise BackendError(backend_name, "Runtime backend not registered")
        return self._runtimes[backend_name]

    def register_runtime(self, name: str, runtime: BaseRuntime) -> None:
        self._runtimes[name] = runtime
        logger.info(f"Registered custom runtime: {name}")

    def _get_runtime_for_model(self, model_id: str) -> BaseRuntime:
        entry = self.registry.get(model_id)

        if entry.status != ModelStatus.READY:
            raise ModelNotReadyError(model_id, entry.status.value)

        runtime = self.get_runtime(entry.runtime_backend.value)

        if not asyncio.get_event_loop().run_until_complete(runtime.is_model_loaded(model_id)):
            logger.warning(f"Model {model_id} not loaded in runtime, attempting load")
            if entry.artifact_path:
                asyncio.get_event_loop().run_until_complete(
                    runtime.load_model(model_id, entry.artifact_path)
                )
            else:
                raise ModelNotReadyError(model_id, "no artifact path")

        return runtime

    async def chat_completion(self, model_id: str, request: ChatCompletionRequest):
        runtime = self._get_runtime_for_model(model_id)
        return await runtime.chat_completion(model_id, request)

    async def chat_completion_stream(self, model_id: str, request: ChatCompletionRequest):
        runtime = self._get_runtime_for_model(model_id)
        return runtime.chat_completion_stream(model_id, request)

    async def completion(self, model_id: str, request: CompletionRequest):
        runtime = self._get_runtime_for_model(model_id)
        return await runtime.completion(model_id, request)

    async def load_model(self, model_id: str, backend: str, model_path: str, **kwargs) -> None:
        runtime = self.get_runtime(backend)
        await runtime.load_model(model_id, model_path, **kwargs)
        self.registry.update_status(model_id, ModelStatus.READY)
        logger.info(f"Model {model_id} loaded on {backend}")

    async def unload_model(self, model_id: str) -> None:
        entry = self.registry.get(model_id)
        runtime = self.get_runtime(entry.runtime_backend.value)
        await runtime.unload_model(model_id)
        self.registry.update_status(model_id, ModelStatus.UNLOADED)
        logger.info(f"Model {model_id} unloaded")

    async def health_check(self) -> Dict[str, Any]:
        results = {}
        for name, runtime in self._runtimes.items():
            try:
                results[name] = await runtime.health_check()
            except Exception as e:
                results[name] = {"backend": name, "error": str(e)}
        return results
