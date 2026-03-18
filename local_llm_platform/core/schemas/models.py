from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# --- Enums ---

class ModelStatus(str, Enum):
    READY = "ready"
    LOADING = "loading"
    TRAINING = "training"
    FAILED = "failed"
    UNLOADED = "unloaded"


class BackendType(str, Enum):
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    TGI = "tgi"
    TRANSFORMERS = "transformers"
    REMOTE_HTTP = "remote_http"
    REMOTE_SSH = "remote_ssh"


class ModelFormat(str, Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ADAPTER = "adapter"


class SourceType(str, Enum):
    LOCAL_TRAIN = "local_train"
    REMOTE_TRAIN = "remote_train"
    IMPORTED_ZIP = "imported_zip"


class Specialization(str, Enum):
    REASONING = "reasoning"
    PLANNING = "planning"
    TOOL_CALLING = "tool-calling"
    PERSONALITY = "personality"
    CODER = "coder"
    GENERAL = "general"


# --- Model Registry Entry ---

class ModelRegistryEntry(BaseModel):
    model_id: str
    display_name: str
    provider: str = "local-provider"
    runtime_backend: BackendType
    model_format: ModelFormat
    base_model: Optional[str] = None
    specialization: Specialization = Specialization.GENERAL
    status: ModelStatus = ModelStatus.LOADING
    source: SourceType = SourceType.LOCAL_TRAIN
    capabilities: List[str] = []
    version: str = "1.0.0"
    tags: List[str] = []

    # Runtime metadata
    chat_template: Optional[str] = None
    tokenizer_path: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    supports_tool_calling: bool = False

    # Storage
    artifact_path: Optional[str] = None
    manifest_path: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# --- OpenAI-compatible model list ---

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local-provider"
    permission: List[Dict[str, Any]] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# --- Model manifest (for import/export) ---

class ModelManifest(BaseModel):
    model_id: str
    display_name: str
    format: ModelFormat
    backend: BackendType
    base_model: Optional[str] = None
    specialization: Specialization = Specialization.GENERAL
    quantization: Optional[str] = None
    context_length: Optional[int] = 4096
    chat_template: Optional[str] = None
    tokenizer: Optional[str] = None
    capabilities: List[str] = []
    version: str = "1.0.0"
    metadata: Dict[str, Any] = {}
