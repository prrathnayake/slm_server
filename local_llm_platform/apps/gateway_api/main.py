import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from local_llm_platform.core.config.settings import settings
from local_llm_platform.core.logging.logger import setup_logging, get_logger
from local_llm_platform.core.security.auth import verify_api_key
from local_llm_platform.core.schemas.chat import ChatCompletionRequest
from local_llm_platform.core.schemas.completion import CompletionRequest
from local_llm_platform.core.schemas.models import ModelListResponse, ModelRegistryEntry, ModelStatus
from local_llm_platform.core.schemas.training import TrainingConfig
from local_llm_platform.core.exceptions.errors import PlatformError
from local_llm_platform.services.registry.registry import ModelRegistry
from local_llm_platform.services.routing.router import RuntimeRouter
from local_llm_platform.services.training.training_service import TrainingService
from local_llm_platform.services.datasets.dataset_service import DatasetService
from local_llm_platform.services.metrics.collector import MetricsCollector
from local_llm_platform.services.artifacts.artifact_manager import ArtifactManager
from local_llm_platform.training.import_trainer import ImportProcessor
from local_llm_platform.services.adapter.adapter_manager import AdapterManager
from local_llm_platform.services.huggingface.hf_service import HuggingFaceService

setup_logging(settings.DEBUG and "DEBUG" or "INFO")
logger = get_logger("gateway")

registry = ModelRegistry()
router = RuntimeRouter(registry)
training_service = TrainingService()
dataset_service = DatasetService()
metrics = MetricsCollector()
artifact_manager = ArtifactManager()
import_processor = ImportProcessor()
adapter_manager = AdapterManager()
hf_service = HuggingFaceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Local LLM Platform Gateway")
    logger.info(f"Database: {settings.DATABASE_URL}")
    yield
    logger.info("Shutting down Local LLM Platform Gateway")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Error handling ---

@app.exception_handler(PlatformError)
async def platform_error_handler(request, exc: PlatformError):
    raise HTTPException(status_code=exc.status_code, detail=exc.message)


# --- Health & Metrics ---

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "timestamp": int(time.time()),
    }


@app.get("/metrics")
async def get_metrics():
    return metrics.get_summary()


# --- Models ---

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: Optional[str] = Depends(verify_api_key)):
    return registry.to_openai_format()


@app.get("/v1/models/all")
async def list_all_models(api_key: Optional[str] = Depends(verify_api_key)):
    models = registry.list_models()
    return {"object": "list", "data": [m.model_dump() for m in models]}


@app.post("/v1/models/load")
async def load_model(
    model_id: str,
    backend: str = "llama_cpp",
    model_path: str = "",
    background: bool = True,
    api_key: Optional[str] = Depends(verify_api_key),
):
    from fastapi import BackgroundTasks

    # Check if model is an adapter
    try:
        entry = registry.get(model_id)
        if entry.model_format.value == "adapter":
            adapter_info = adapter_manager.get_adapter_info(model_id)
            base_model = adapter_info.get("base_model") if adapter_info else None
            return {
                "status": "adapter_requires_merge",
                "model_id": model_id,
                "message": f"This is a LoRA adapter based on '{base_model}'. You must merge it with the base model before loading.",
                "options": [
                    {
                        "action": "download_base",
                        "endpoint": f"POST /v1/models/download?model_name={base_model}",
                        "description": f"Download base model: {base_model}",
                    },
                    {
                        "action": "merge",
                        "endpoint": f"POST /v1/adapters/{model_id}/merge",
                        "description": "Merge adapter into base model (requires base model downloaded first)",
                    },
                ],
            }
    except Exception:
        pass

    if background:
        # Start loading in background, return immediately
        registry.update_status(model_id, ModelStatus.LOADING)
        import asyncio
        asyncio.create_task(_load_model_background(model_id, backend, model_path))
        return {"status": "loading", "model_id": model_id, "backend": backend}

    start = time.time()
    try:
        await router.load_model(model_id, backend, model_path)
    except Exception as e:
        return {
            "status": "error",
            "model_id": model_id,
            "error": str(e),
            "hint": "Make sure the runtime backend is installed (e.g., pip install llama-cpp-python or pip install vllm)",
        }
    latency = time.time() - start
    metrics.record("model_load_time", latency)
    return {"status": "loaded", "model_id": model_id, "backend": backend}


async def _load_model_background(model_id: str, backend: str, model_path: str):
    try:
        runtime = router.get_runtime(backend)
        await runtime.load_model(model_id, model_path)
        registry.update_status(model_id, ModelStatus.READY)
        logger.info(f"Background load complete: {model_id}")
    except Exception as e:
        try:
            registry.update_status(model_id, ModelStatus.FAILED)
        except Exception:
            pass
        logger.error(f"Background load failed for {model_id}: {e}")


@app.post("/v1/models/unload")
async def unload_model(
    model_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    await router.unload_model(model_id)
    return {"status": "unloaded", "model_id": model_id}


@app.post("/v1/models/register")
async def register_model(
    entry: ModelRegistryEntry,
    api_key: Optional[str] = Depends(verify_api_key),
):
    result = registry.register(entry)
    return result.model_dump()


@app.delete("/v1/models/{model_id}")
async def unregister_model(
    model_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    success = registry.unregister(model_id)
    return {"deleted": success, "model_id": model_id}


# --- Chat Completions ---

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    start = time.time()

    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
        )

    response = await router.chat_completion(request.model, request)
    latency = time.time() - start
    metrics.record_request(
        request.model,
        latency,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
    return response


async def _stream_chat(request: ChatCompletionRequest):
    start = time.time()
    token_count = 0
    try:
        stream = await router.chat_completion_stream(request.model, request)
        async for chunk in stream:
            token_count += 1
            yield chunk
    except Exception as e:
        error_data = json.dumps({"error": {"message": str(e), "type": "server_error"}})
        yield f"data: {error_data}\n\n"
    finally:
        latency = time.time() - start
        metrics.record_request(request.model, latency, 0, token_count)


# --- Completions ---

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    start = time.time()
    response = await router.completion(request.model, request)
    latency = time.time() - start
    metrics.record_request(
        request.model,
        latency,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
    return response


# --- Training ---

@app.post("/v1/training/jobs")
async def create_training_job(
    config: TrainingConfig,
    execution_target: str = "local",
    api_key: Optional[str] = Depends(verify_api_key),
):
    job = await training_service.create_job(config, execution_target)
    return job.model_dump()


@app.get("/v1/training/jobs")
async def list_training_jobs(api_key: Optional[str] = Depends(verify_api_key)):
    jobs = await training_service.list_jobs()
    return {"jobs": [j.model_dump() for j in jobs]}


@app.get("/v1/training/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    job = await training_service.get_job(job_id)
    return job.model_dump()


@app.get("/v1/training/jobs/{job_id}/progress")
async def get_training_progress(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    return await training_service.get_job_progress(job_id)


@app.get("/v1/training/jobs/{job_id}/logs")
async def get_training_logs(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    logs = await training_service.get_job_logs(job_id)
    return {"job_id": job_id, "logs": logs}


@app.post("/v1/training/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    success = await training_service.cancel_job(job_id)
    return {"cancelled": success, "job_id": job_id}


# --- Datasets ---

@app.post("/v1/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None),
    format: str = Form(default="jsonl"),
    api_key: Optional[str] = Depends(verify_api_key),
):
    from local_llm_platform.core.schemas.training import DatasetFormat
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dataset_format = DatasetFormat(format)
        _name = name if name is not None else (file.filename or "uploaded_dataset")
        entry = await dataset_service.upload_dataset(
            name=_name,
            file_path=tmp_path,
            dataset_format=dataset_format,
        )
        return entry.model_dump()
    finally:
        os.unlink(tmp_path)


@app.get("/v1/datasets")
async def list_datasets(api_key: Optional[str] = Depends(verify_api_key)):
    datasets = await dataset_service.list_datasets()
    return {"datasets": [d.model_dump() for d in datasets]}


@app.get("/v1/datasets/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    return await dataset_service.validate_dataset(dataset_id)


# --- Import ---

@app.post("/v1/models/import")
async def import_model(
    file: UploadFile = File(...),
    model_id: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key),
):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await import_processor.process_zip(tmp_path, model_id)
        return result
    finally:
        os.unlink(tmp_path)


# --- Runtime health ---

@app.get("/v1/runtime/health")
async def runtime_health(api_key: Optional[str] = Depends(verify_api_key)):
    return await router.health_check()


# --- Adapters ---

@app.get("/v1/adapters/{adapter_id}")
async def get_adapter_info(adapter_id: str, api_key: Optional[str] = Depends(verify_api_key)):
    info = adapter_manager.get_adapter_info(adapter_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_id}")
    return info


@app.post("/v1/adapters/{adapter_id}/merge")
async def merge_adapter(
    adapter_id: str,
    output_name: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key),
):
    result = await adapter_manager.merge_adapter(adapter_id, output_name)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/v1/models/download")
async def download_base_model(
    model_name: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    result = await adapter_manager.download_base_model(model_name)
    return result


@app.get("/v1/models/base")
async def list_base_models(api_key: Optional[str] = Depends(verify_api_key)):
    return {"models": adapter_manager.list_base_models()}


@app.get("/v1/models/merged")
async def list_merged_models(api_key: Optional[str] = Depends(verify_api_key)):
    return {"models": adapter_manager.list_merged()}


# --- HuggingFace Model Browser ---

@app.get("/v1/huggingface/popular")
async def hf_popular_models(api_key: Optional[str] = Depends(verify_api_key)):
    """Get list of popular free models for fine-tuning."""
    return {"models": hf_service.get_popular_models()}


@app.get("/v1/huggingface/search")
async def hf_search_models(
    query: str = "",
    task: str = "text-generation",
    sort: str = "downloads",
    limit: int = 30,
    library: str = "transformers",
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Search HuggingFace models."""
    models = await hf_service.search_models(
        query=query, task=task, sort=sort, limit=limit, library=library
    )
    return {"models": models, "count": len(models)}


@app.get("/v1/huggingface/models/{model_id:path}/info")
async def hf_model_info(
    model_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get detailed info about a HuggingFace model."""
    info = await hf_service.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return info


@app.post("/v1/huggingface/models/{model_id:path}/download")
async def hf_download_model(
    model_id: str,
    revision: str = "main",
    token: Optional[str] = None,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Download a model from HuggingFace."""
    result = await hf_service.download_model(model_id, revision=revision, token=token)
    return result


@app.get("/v1/huggingface/models/{model_id:path}/download/status")
async def hf_download_status(
    model_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get download status for a model."""
    status = hf_service.get_download_status(model_id)
    if not status:
        return {"status": "not_found", "model_id": model_id}
    return status


@app.get("/v1/huggingface/downloaded")
async def hf_list_downloaded(api_key: Optional[str] = Depends(verify_api_key)):
    """List all downloaded HuggingFace models."""
    return {"models": hf_service.list_downloaded_models()}


def start():
    import uvicorn
    uvicorn.run(
        "local_llm_platform.apps.gateway_api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    start()
