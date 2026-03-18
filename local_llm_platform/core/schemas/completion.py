from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from local_llm_platform.core.schemas.chat import UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 16
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    echo: Optional[bool] = False
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class EmbeddingRequest(BaseModel):
    model: str
    input: str | List[str]
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo
