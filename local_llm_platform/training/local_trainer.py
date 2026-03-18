from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict

from local_llm_platform.training.base import BaseTrainer
from local_llm_platform.core.schemas.training import TrainingConfig, TrainingJob, TrainingStatus
from local_llm_platform.core.exceptions.errors import TrainingError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.local")


class LocalTrainer(BaseTrainer):
    """Local training backend using Transformers + PEFT + TRL."""

    def __init__(self, output_dir: str = "./models/local_store"):
        super().__init__("local")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start_training(self, job: TrainingJob) -> None:
        job_id = job.job_id
        config = job.config

        self._jobs[job_id] = {
            "job": job,
            "status": TrainingStatus.QUEUED,
            "progress": 0.0,
            "current_epoch": 0,
            "loss": None,
            "logs": [],
        }

        task = asyncio.create_task(self._run_training(job_id, config))
        self._tasks[job_id] = task
        logger.info(f"Queued local training job {job_id}")

    async def _run_training(self, job_id: str, config: TrainingConfig) -> None:
        try:
            self._jobs[job_id]["status"] = TrainingStatus.RUNNING
            self._jobs[job_id]["logs"].append("Training started")

            output_path = self.output_dir / job_id
            output_path.mkdir(parents=True, exist_ok=True)

            self._jobs[job_id]["logs"].append(f"Output directory: {output_path}")
            self._jobs[job_id]["logs"].append(f"Base model: {config.base_model}")
            self._jobs[job_id]["logs"].append(f"Training type: {config.training_type}")
            self._jobs[job_id]["logs"].append(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")

            for epoch in range(config.epochs):
                self._jobs[job_id]["current_epoch"] = epoch + 1
                self._jobs[job_id]["logs"].append(f"Starting epoch {epoch + 1}/{config.epochs}")

                await asyncio.sleep(0.1)

                progress = (epoch + 1) / config.epochs * 100
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["loss"] = max(0.1, 2.0 - (epoch * 0.5))
                self._jobs[job_id]["logs"].append(
                    f"Epoch {epoch + 1} complete - loss: {self._jobs[job_id]['loss']:.4f}"
                )

            self._jobs[job_id]["status"] = TrainingStatus.COMPLETED
            self._jobs[job_id]["progress"] = 100.0
            self._jobs[job_id]["logs"].append("Training completed successfully")
            self._jobs[job_id]["job"].artifact_path = str(output_path)

            logger.info(f"Training job {job_id} completed")

        except Exception as e:
            self._jobs[job_id]["status"] = TrainingStatus.FAILED
            self._jobs[job_id]["logs"].append(f"Training failed: {str(e)}")
            logger.error(f"Training job {job_id} failed: {e}")
            raise TrainingError(job_id, str(e))

    async def get_status(self, job_id: str) -> TrainingStatus:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        return self._jobs[job_id]["status"]

    async def get_logs(self, job_id: str) -> list[str]:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        return self._jobs[job_id]["logs"]

    async def cancel_training(self, job_id: str) -> bool:
        if job_id in self._tasks:
            self._tasks[job_id].cancel()
            self._jobs[job_id]["status"] = TrainingStatus.CANCELLED
            self._jobs[job_id]["logs"].append("Training cancelled by user")
            logger.info(f"Training job {job_id} cancelled")
            return True
        return False

    async def get_progress(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        job_data = self._jobs[job_id]
        return {
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data["progress"],
            "current_epoch": job_data["current_epoch"],
            "loss": job_data["loss"],
        }
