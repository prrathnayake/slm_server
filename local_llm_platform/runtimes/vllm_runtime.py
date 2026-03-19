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

logger = get_logger("runtimes.vllm")


class VLLMRuntime(BaseRuntime):
    """Runtime backend using vLLM for high-throughput transformer serving."""

    def __init__(self, tensor_parallel_size: int = 1):
        super().__init__("vllm")
        self._engines: Dict[str, Any] = {}
        self.tensor_parallel_size = tensor_parallel_size

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        try:
            from vllm import AsyncLLMEngine, EngineArgs

            max_model_len = kwargs.get("max_model_len", 4096)
            gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)

            engine_args = EngineArgs(
                model=model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._engines[model_id] = engine
            self._loaded_models[model_id] = {
                "path": model_path,
                "backend": "vllm",
                "max_model_len": max_model_len,
            }
            logger.info(f"Loaded model {model_id} via vLLM from {model_path}")
        except Exception as e:
            raise ModelLoadError(model_id, str(e))

    async def unload_model(self, model_id: str) -> None:
        if model_id in self._engines:
            del self._engines[model_id]
        self._loaded_models.pop(model_id, None)
        logger.info(f"Unloaded model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._engines

    def _get_engine(self, model_id: str):
        if model_id not in self._engines:
            raise BackendError("vllm", f"Model {model_id} not loaded")
        return self._engines[model_id]

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        engine = self._get_engine(model_id)

        from vllm import SamplingParams
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest as VLLMChatRequest

        sampling_params = SamplingParams(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens or 512,
            stop=request.stop if isinstance(request.stop, list) else None,
        )

        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        results = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            chat_template_kwargs={"messages": messages},
        )

        output = None
        async for result in results:
            output = result
            break

        if output is None:
            raise BackendError("vllm", "No output generated")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        choice_text = output.outputs[0].text

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=choice_text),
                    finish_reason=output.outputs[0].finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
                total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            ),
        )

    async def chat_completion_stream(self, model_id: str, request: ChatCompletionRequest):
        engine = self._get_engine(model_id)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens or 512,
            stop=request.stop if isinstance(request.stop, list) else None,
        )

        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        results = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=completion_id,
            chat_template_kwargs={"messages": messages},
        )

        async for result in results:
            delta_content = result.outputs[0].text
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta_content},
                        "finish_reason": result.outputs[0].finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        engine = self._get_engine(model_id)

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            max_tokens=request.max_tokens or 16,
            stop=request.stop,
        )

        results = engine.generate(
            prompts=[request.prompt],
            sampling_params=sampling_params,
            request_id=f"cmpl-{uuid.uuid4().hex[:12]}",
        )

        output = None
        async for result in results:
            output = result
            break

        if output is None:
            raise BackendError("vllm", "No output generated")

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.outputs[0].text,
                    finish_reason=output.outputs[0].finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(output.prompt_token_ids),
                completion_tokens=len(output.outputs[0].token_ids),
                total_tokens=len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            ),
        )

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        info = self._loaded_models.get(model_id, {})
        return {
            "model_id": model_id,
            "backend": "vllm",
            **info,
        }
