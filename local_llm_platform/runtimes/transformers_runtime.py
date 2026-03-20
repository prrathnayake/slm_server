from __future__ import annotations

import asyncio
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

logger = get_logger("runtimes.transformers")


class TransformersRuntime(BaseRuntime):
    """Runtime using HuggingFace Transformers directly - works with safetensors, no compile needed."""

    def __init__(self):
        super().__init__("transformers", max_concurrent=1)
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

    async def load_model(self, model_id: str, model_path: str, **kwargs) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            device = kwargs.get("device", "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Loading model {model_id} from {model_path} (device={device})")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()

            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer
            self._loaded_models[model_id] = {
                "path": model_path,
                "backend": "transformers",
                "device": device,
                "dtype": str(dtype),
            }
            logger.info(f"Loaded model {model_id} successfully")

        except Exception as e:
            raise ModelLoadError(model_id, str(e))

    async def unload_model(self, model_id: str) -> None:
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._tokenizers:
            del self._tokenizers[model_id]
        self._loaded_models.pop(model_id, None)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(f"Unloaded model {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def _get_model(self, model_id: str):
        if model_id not in self._models:
            raise BackendError("transformers", f"Model {model_id} not loaded")
        return self._models[model_id], self._tokenizers[model_id]

    def _format_prompt(self, messages: list, tokenizer) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    text += f"System: {content}\n\n"
                elif role == "user":
                    text += f"User: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"
            text += "Assistant:"
            return text

    def _sync_generate_chat(self, model, tokenizer, messages: list, gen_kwargs: dict) -> tuple:
        import torch
        prompt = self._format_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = len(new_tokens)
        return response_text, prompt_tokens, completion_tokens

    def _sync_generate_completion(self, model, tokenizer, prompt: str, gen_kwargs: dict) -> tuple:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = len(new_tokens)
        return text, prompt_tokens, completion_tokens

    async def chat_completion(
        self, model_id: str, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        model, tokenizer = self._get_model(model_id)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 512,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
            "do_sample": (request.temperature or 1.0) > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if request.stop:
            stop = request.stop if isinstance(request.stop, list) else [request.stop]
            gen_kwargs["stop_strings"] = stop

        response_text, prompt_tokens, completion_tokens = await self._run_in_executor(
            model_id, self._sync_generate_chat, model, tokenizer, messages, gen_kwargs
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def chat_completion_stream(
        self, model_id: str, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        import torch
        from transformers import TextIteratorStreamer
        import threading
        import queue

        model, tokenizer = self._get_model(model_id)
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        prompt = self._format_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 512,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
            "do_sample": (request.temperature or 1.0) > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": streamer,
        }

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        exception_queue: queue.Queue = queue.Queue()

        def generate_in_thread():
            try:
                with torch.no_grad():
                    model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                exception_queue.put(e)

        sem = self._get_semaphore(model_id)
        await sem.acquire()
        try:
            thread = threading.Thread(target=generate_in_thread)
            thread.start()

            for text in streamer:
                if text:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            thread.join(timeout=5)

            if not exception_queue.empty():
                raise exception_queue.get()

            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            sem.release()

    async def completion(
        self, model_id: str, request: CompletionRequest
    ) -> CompletionResponse:
        model, tokenizer = self._get_model(model_id)

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 16,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
            "do_sample": (request.temperature or 1.0) > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }

        text, prompt_tokens, completion_tokens = await self._run_in_executor(
            model_id, self._sync_generate_completion, model, tokenizer, request.prompt, gen_kwargs
        )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                CompletionChoice(
                    index=0,
                    text=text,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        info = self._loaded_models.get(model_id, {})
        return {
            "model_id": model_id,
            "backend": "transformers",
            **info,
        }
