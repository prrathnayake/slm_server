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

logger = get_logger("runtimes.llama_cpp")


class LlamaCppRuntime(BaseRuntime):
    """Runtime backend using llama-cpp-python for GGUF models."""

    def __init__(self, max_concurrent: int = 1):
        super().__init__("llama_cpp", max_concurrent=max_concurrent)
        self._llama_instances: Dict[str, Any] = {}

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        try:
            from llama_cpp import Llama

            n_ctx = kwargs.get("n_ctx", 4096)
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)
            verbose = kwargs.get("verbose", False)

            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            self._llama_instances[model_id] = llm
            self._loaded_models[model_id] = {
                "path": model_path,
                "backend": "llama_cpp",
                "n_ctx": n_ctx,
            }
            logger.info(f"Loaded model {model_id} from {model_path}")
        except Exception as e:
            raise ModelLoadError(model_id, str(e))

    async def unload_model(self, model_id: str) -> None:
        if model_id in self._llama_instances:
            del self._llama_instances[model_id]
        self._loaded_models.pop(model_id, None)
        logger.info(f"Unloaded model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._llama_instances

    def _get_llm(self, model_id: str):
        if model_id not in self._llama_instances:
            raise BackendError("llama_cpp", f"Model {model_id} not loaded")
        return self._llama_instances[model_id]

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        llm = self._get_llm(model_id)

        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        result = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, (list, str)) else None,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=choice["index"],
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"].get("content"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in result["choices"]
            ],
            usage=UsageInfo(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0),
            ),
        )

    async def chat_completion_stream(self, model_id: str, request: ChatCompletionRequest):
        llm = self._get_llm(model_id)

        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        stream = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens,
            stop=request.stop if isinstance(request.stop, list) else None,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            finish = chunk["choices"][0].get("finish_reason")
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": finish,
                    }
                ],
            }
            if content:
                yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        llm = self._get_llm(model_id)

        result = llm(
            prompt=request.prompt,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens or 16,
            stop=request.stop,
        )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                CompletionChoice(
                    index=choice["index"],
                    text=choice["text"],
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in result["choices"]
            ],
            usage=UsageInfo(
                prompt_tokens=result.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=result.get("usage", {}).get("completion_tokens", 0),
                total_tokens=result.get("usage", {}).get("total_tokens", 0),
            ),
        )

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        info = self._loaded_models.get(model_id, {})
        return {
            "model_id": model_id,
            "backend": "llama_cpp",
            **info,
        }
