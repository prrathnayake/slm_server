from __future__ import annotations

from typing import Any, Dict

from local_llm_platform.training.base import BaseTrainer
from local_llm_platform.core.schemas.training import TrainingJob, TrainingStatus
from local_llm_platform.core.exceptions.errors import TrainingError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.colab")


class ColabTrainer(BaseTrainer):
    """Colab-based training backend (convenience backend, not primary)."""

    def __init__(self):
        super().__init__("colab")
        self._jobs: Dict[str, Dict[str, Any]] = {}

    async def start_training(self, job: TrainingJob) -> None:
        job_id = job.job_id

        self._jobs[job_id] = {
            "job": job,
            "status": TrainingStatus.QUEUED,
            "progress": 0.0,
            "current_epoch": 0,
            "loss": None,
            "logs": [],
            "colab_notebook_url": None,
        }

        self._jobs[job_id]["logs"].append(
            "Colab training requires manual notebook setup. "
            "Please use the generated notebook in the artifacts directory."
        )
        self._jobs[job_id]["status"] = TrainingStatus.QUEUED
        logger.info(f"Colab training job {job_id} queued - requires manual setup")

    async def get_status(self, job_id: str) -> TrainingStatus:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        return self._jobs[job_id]["status"]

    async def get_logs(self, job_id: str) -> list[str]:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        return self._jobs[job_id]["logs"]

    async def cancel_training(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = TrainingStatus.CANCELLED
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
            "note": "Colab training requires manual monitoring",
        }
