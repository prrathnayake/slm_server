from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from local_llm_platform.core.schemas.training import TrainingConfig, TrainingJob, TrainingStatus
from local_llm_platform.training.local_trainer import LocalTrainer
from local_llm_platform.training.remote_trainer import RemoteTrainer
from local_llm_platform.core.exceptions.errors import TrainingError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.training")


class TrainingService:
    """Orchestrates training jobs across local and remote backends."""

    def __init__(self):
        self.local_trainer = LocalTrainer()
        self.remote_trainer = RemoteTrainer()
        self._jobs: Dict[str, TrainingJob] = {}

    async def create_job(self, config: TrainingConfig, execution_target: str = "local") -> TrainingJob:
        job_id = f"train-{uuid.uuid4().hex[:12]}"

        job = TrainingJob(
            job_id=job_id,
            config=config,
            status=TrainingStatus.QUEUED,
            total_epochs=config.epochs,
            execution_target=execution_target,
            created_at=datetime.now(timezone.utc),
        )

        self._jobs[job_id] = job

        if execution_target == "local":
            await self.local_trainer.start_training(job)
        elif execution_target == "remote":
            await self.remote_trainer.start_training(job)
        else:
            raise TrainingError(job_id, f"Unknown execution target: {execution_target}")

        logger.info(f"Created training job {job_id} (target: {execution_target})")
        return job

    async def get_job(self, job_id: str) -> TrainingJob:
        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")
        return self._jobs[job_id]

    async def list_jobs(self, status: Optional[TrainingStatus] = None) -> List[TrainingJob]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    async def cancel_job(self, job_id: str) -> bool:
        job = await self.get_job(job_id)

        if job.execution_target == "local":
            return await self.local_trainer.cancel_training(job_id)
        elif job.execution_target == "remote":
            return await self.remote_trainer.cancel_training(job_id)
        return False

    async def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        job = await self.get_job(job_id)

        if job.execution_target == "local":
            return await self.local_trainer.get_progress(job_id)
        elif job.execution_target == "remote":
            return await self.remote_trainer.get_progress(job_id)
        return {"job_id": job_id, "status": job.status}

    async def get_job_logs(self, job_id: str) -> List[str]:
        job = await self.get_job(job_id)

        if job.execution_target == "local":
            return await self.local_trainer.get_logs(job_id)
        elif job.execution_target == "remote":
            return await self.remote_trainer.get_logs(job_id)
        return []
