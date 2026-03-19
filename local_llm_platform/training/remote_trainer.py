from __future__ import annotations

import asyncio
from typing import Any, Dict

from local_llm_platform.training.base import BaseTrainer
from local_llm_platform.core.schemas.training import TrainingConfig, TrainingJob, TrainingStatus
from local_llm_platform.core.exceptions.errors import TrainingError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.remote")


class RemoteTrainer(BaseTrainer):
    """Remote training backend using worker agent on GPU machines."""

    def __init__(self):
        super().__init__("remote")
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._worker_endpoints: Dict[str, Dict[str, Any]] = {}

    def register_worker(self, worker_id: str, endpoint_url: str, api_key: str = None) -> None:
        self._worker_endpoints[worker_id] = {
            "url": endpoint_url.rstrip("/"),
            "api_key": api_key,
            "status": "available",
        }
        logger.info(f"Registered training worker {worker_id} at {endpoint_url}")

    async def start_training(self, job: TrainingJob) -> None:
        import httpx

        job_id = job.job_id
        worker_id = job.worker_id

        if not worker_id or worker_id not in self._worker_endpoints:
            available = [w for w, info in self._worker_endpoints.items() if info["status"] == "available"]
            if not available:
                raise TrainingError(job_id, "No available workers")
            worker_id = available[0]

        worker = self._worker_endpoints[worker_id]
        headers = {"Content-Type": "application/json"}
        if worker.get("api_key"):
            headers["Authorization"] = f"Bearer {worker['api_key']}"

        self._jobs[job_id] = {
            "job": job,
            "worker_id": worker_id,
            "status": TrainingStatus.QUEUED,
            "progress": 0.0,
            "current_epoch": 0,
            "loss": None,
            "logs": [],
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{worker['url']}/training/start",
                    json=job.config.model_dump(),
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()

            self._jobs[job_id]["status"] = TrainingStatus.RUNNING
            self._jobs[job_id]["logs"].append(f"Training dispatched to worker {worker_id}")
            logger.info(f"Training job {job_id} dispatched to worker {worker_id}")

        except Exception as e:
            self._jobs[job_id]["status"] = TrainingStatus.FAILED
            self._jobs[job_id]["logs"].append(f"Failed to dispatch: {str(e)}")
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
        import httpx

        if job_id not in self._jobs:
            return False

        job_data = self._jobs[job_id]
        worker_id = job_data["worker_id"]
        worker = self._worker_endpoints.get(worker_id)

        if worker:
            try:
                headers = {}
                if worker.get("api_key"):
                    headers["Authorization"] = f"Bearer {worker['api_key']}"
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{worker['url']}/training/{job_id}/cancel",
                        headers=headers,
                        timeout=10.0,
                    )
            except Exception as e:
                logger.warning(f"Failed to cancel on worker: {e}")

        self._jobs[job_id]["status"] = TrainingStatus.CANCELLED
        return True

    async def get_progress(self, job_id: str) -> Dict[str, Any]:
        import httpx

        if job_id not in self._jobs:
            raise TrainingError(job_id, "Job not found")

        job_data = self._jobs[job_id]
        worker_id = job_data["worker_id"]
        worker = self._worker_endpoints.get(worker_id)

        if worker and job_data["status"] == TrainingStatus.RUNNING:
            try:
                headers = {}
                if worker.get("api_key"):
                    headers["Authorization"] = f"Bearer {worker['api_key']}"
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{worker['url']}/training/remote/{job_id}/status",
                        headers=headers,
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        job_data.update(data)
            except Exception:
                pass

        return {
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data["progress"],
            "current_epoch": job_data["current_epoch"],
            "loss": job_data["loss"],
            "worker_id": worker_id,
        }
