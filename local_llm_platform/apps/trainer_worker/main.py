from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_llm_platform.core.config.settings import settings
from local_llm_platform.core.logging.logger import setup_logging, get_logger
from local_llm_platform.core.schemas.training import TrainingConfig, TrainingStatus
from local_llm_platform.training.local_trainer import LocalTrainer
from local_llm_platform.training.remote_trainer import RemoteTrainer
from local_llm_platform.training.colab_trainer import ColabTrainer
from local_llm_platform.training.import_trainer import ImportProcessor

setup_logging("INFO")
logger = get_logger("trainer_worker")

local_trainer = LocalTrainer()
remote_trainer = RemoteTrainer()
colab_trainer = ColabTrainer()
import_processor = ImportProcessor()


class TrainingStartRequest(BaseModel):
    job_id: str
    config: TrainingConfig


class ImportRequest(BaseModel):
    zip_path: str
    model_id: Optional[str] = None


class RemoteWorkerRegister(BaseModel):
    worker_id: str
    endpoint_url: str
    api_key: Optional[str] = None


app = FastAPI(title="Trainer Worker", version=settings.APP_VERSION)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "workers": list(remote_trainer._worker_endpoints.keys()),
    }


# --- Local Training ---

@app.post("/training/local/start")
async def start_local_training(req: TrainingStartRequest):
    from local_llm_platform.core.schemas.training import TrainingJob
    from datetime import datetime, timezone

    job = TrainingJob(
        job_id=req.job_id,
        config=req.config,
        status=TrainingStatus.QUEUED,
        total_epochs=req.config.epochs,
        execution_target="local",
        created_at=datetime.now(timezone.utc),
    )
    await local_trainer.start_training(job)
    return {"status": "started", "job_id": req.job_id}


@app.get("/training/local/{job_id}/status")
async def get_local_status(job_id: str):
    try:
        status = await local_trainer.get_status(job_id)
        progress = await local_trainer.get_progress(job_id)
        return {"job_id": job_id, "status": status.value, **progress}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/training/local/{job_id}/logs")
async def get_local_logs(job_id: str):
    try:
        logs = await local_trainer.get_logs(job_id)
        return {"job_id": job_id, "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/training/local/{job_id}/cancel")
async def cancel_local_training(job_id: str):
    success = await local_trainer.cancel_training(job_id)
    return {"cancelled": success, "job_id": job_id}


# --- Remote Workers ---

@app.post("/training/workers/register")
async def register_worker(req: RemoteWorkerRegister):
    remote_trainer.register_worker(req.worker_id, req.endpoint_url, req.api_key)
    return {"status": "registered", "worker_id": req.worker_id}


@app.get("/training/workers")
async def list_workers():
    return {"workers": list(remote_trainer._worker_endpoints.keys())}


@app.post("/training/remote/start")
async def start_remote_training(req: TrainingStartRequest):
    from local_llm_platform.core.schemas.training import TrainingJob
    from datetime import datetime, timezone

    job = TrainingJob(
        job_id=req.job_id,
        config=req.config,
        status=TrainingStatus.QUEUED,
        total_epochs=req.config.epochs,
        execution_target="remote",
        created_at=datetime.now(timezone.utc),
    )
    await remote_trainer.start_training(job)
    return {"status": "started", "job_id": req.job_id}


@app.get("/training/remote/{job_id}/status")
async def get_remote_status(job_id: str):
    try:
        progress = await remote_trainer.get_progress(job_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Model Import ---

@app.post("/import/zip")
async def import_zip(req: ImportRequest):
    try:
        result = await import_processor.process_zip(req.zip_path, req.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/import/{model_id}/validate")
async def validate_import(model_id: str):
    result = await import_processor.validate_import(model_id)
    return result


def start():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)


if __name__ == "__main__":
    start()
