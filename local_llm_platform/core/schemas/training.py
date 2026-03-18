from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(str, Enum):
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    TOOL_CALL = "tool_call"
    PLAIN_TEXT = "plain_text"


# --- Training config ---

class TrainingConfig(BaseModel):
    base_model: str
    dataset_id: str
    training_type: TrainingType = TrainingType.LORA
    output_name: str

    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # Training params
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    optimizer: str = "adamw_torch"
    seed: int = 42
    fp16: bool = False
    bf16: bool = True

    # Runtime target
    target_backend: str = "llama_cpp"
    export_gguf: bool = False
    merge_adapter: bool = False


# --- Training job ---

class TrainingJob(BaseModel):
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.QUEUED
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    logs: List[str] = []

    # Artifacts
    output_model_id: Optional[str] = None
    artifact_path: Optional[str] = None

    # Execution
    execution_target: str = "local"
    worker_id: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# --- Dataset ---

class DatasetEntry(BaseModel):
    dataset_id: str
    name: str
    format: DatasetFormat
    path: str
    num_samples: Optional[int] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: Optional[datetime] = None


# --- Import ---

class ImportRequest(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None
    model_id: Optional[str] = None
    auto_detect: bool = True
    manifest: Optional[ModelManifest] = None


class ImportResult(BaseModel):
    success: bool
    model_id: Optional[str] = None
    detected_format: Optional[str] = None
    detected_backend: Optional[str] = None
    errors: List[str] = []
    warnings: List[str] = []
