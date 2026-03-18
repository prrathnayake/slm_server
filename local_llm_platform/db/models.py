from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, Enum as SAEnum,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import enum


class Base(DeclarativeBase):
    pass


class ModelStatusEnum(str, enum.Enum):
    READY = "ready"
    LOADING = "loading"
    TRAINING = "training"
    FAILED = "failed"
    UNLOADED = "unloaded"


class TrainingStatusEnum(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelRecord(Base):
    __tablename__ = "models"

    model_id = Column(String(255), primary_key=True)
    display_name = Column(String(255), nullable=False)
    provider = Column(String(100), default="local-provider")
    runtime_backend = Column(String(50), nullable=False)
    model_format = Column(String(50), nullable=False)
    base_model = Column(String(255), nullable=True)
    specialization = Column(String(50), default="general")
    status = Column(SAEnum(ModelStatusEnum), default=ModelStatusEnum.LOADING)
    source = Column(String(50), default="local_train")
    capabilities = Column(JSON, default=list)
    version = Column(String(50), default="1.0.0")
    tags = Column(JSON, default=list)
    chat_template = Column(Text, nullable=True)
    tokenizer_path = Column(String(500), nullable=True)
    quantization = Column(String(50), nullable=True)
    context_length = Column(Integer, nullable=True)
    supports_tool_calling = Column(Boolean, default=False)
    artifact_path = Column(String(500), nullable=True)
    manifest_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    job_id = Column(String(255), primary_key=True)
    config = Column(JSON, nullable=False)
    status = Column(SAEnum(TrainingStatusEnum), default=TrainingStatusEnum.QUEUED)
    progress = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    loss = Column(Float, nullable=True)
    logs = Column(JSON, default=list)
    output_model_id = Column(String(255), nullable=True)
    artifact_path = Column(String(500), nullable=True)
    execution_target = Column(String(50), default="local")
    worker_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class DatasetRecord(Base):
    __tablename__ = "datasets"

    dataset_id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    format = Column(String(50), nullable=False)
    path = Column(String(500), nullable=False)
    num_samples = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), default="1.0.0")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db(database_url: str = "sqlite:///./local_llm_platform.db"):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine) -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
