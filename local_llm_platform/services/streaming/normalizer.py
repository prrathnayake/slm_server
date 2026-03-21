from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.streaming")


class StreamNormalizer:
    """Normalizes streaming output from different backends into consistent SSE format."""

    @staticmethod
    async def normalize_sse(stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in stream:
            if chunk.strip():
                yield chunk

    @staticmethod
    def format_sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    @staticmethod
    def done_message() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    async def stream_with_metrics(
        stream: AsyncIterator[str],
    ) -> AsyncIterator[tuple[str, dict]]:
        token_count = 0
        start_time = time.time()

        async for chunk in stream:
            token_count += 1
            elapsed = time.time() - start_time
            tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
            metrics = {
                "tokens_generated": token_count,
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_per_second,
            }
            yield chunk, metrics

        elapsed = time.time() - start_time
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
        final_metrics = {
            "tokens_generated": token_count,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
            "complete": True,
        }
        yield "", final_metrics
