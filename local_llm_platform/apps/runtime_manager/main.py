from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_llm_platform.core.config.settings import settings
from local_llm_platform.core.logging.logger import setup_logging, get_logger
from local_llm_platform.services.registry import ModelRegistry
from local_llm_platform.services.routing import RuntimeRouter

setup_logging("INFO")
logger = get_logger("runtime_manager")

registry = ModelRegistry()
router = RuntimeRouter(registry)


class LoadRequest(BaseModel):
    model_id: str
    backend: str = "llama_cpp"
    model_path: str
    kwargs: Dict[str, Any] = {}


class UnloadRequest(BaseModel):
    model_id: str


class WarmPoolConfig(BaseModel):
    hot_models: List[str] = []
    max_loaded: int = 3


app = FastAPI(title="Runtime Manager", version=settings.APP_VERSION)

_warm_pool: WarmPoolConfig = WarmPoolConfig()


@app.on_event("startup")
async def startup():
    logger.info("Runtime Manager starting up")
    await _load_hot_models()


async def _load_hot_models():
    for model_id in _warm_pool.hot_models:
        try:
            entry = registry.get(model_id)
            if entry.artifact_path:
                await router.load_model(
                    model_id,
                    entry.runtime_backend.value,
                    entry.artifact_path,
                )
                logger.info(f"Hot-loaded model: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to hot-load {model_id}: {e}")


@app.get("/health")
async def health():
    runtime_health = await router.health_check()
    loaded = []
    for name, runtime in router._runtimes.items():
        loaded.extend(runtime.list_loaded_models())
    return {
        "status": "ok",
        "loaded_models": loaded,
        "runtimes": runtime_health,
    }


@app.post("/load")
async def load_model(req: LoadRequest):
    try:
        await router.load_model(req.model_id, req.backend, req.model_path, **req.kwargs)
        return {"status": "loaded", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model(req: UnloadRequest):
    try:
        await router.unload_model(req.model_id)
        return {"status": "unloaded", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/loaded")
async def list_loaded():
    loaded = []
    for name, runtime in router._runtimes.items():
        for model_id in runtime.list_loaded_models():
            loaded.append({"model_id": model_id, "backend": name})
    return {"loaded": loaded}


@app.post("/warm-pool")
async def set_warm_pool(config: WarmPoolConfig):
    global _warm_pool
    _warm_pool = config
    logger.info(f"Warm pool updated: {config.hot_models}")
    return {"status": "updated", "hot_models": config.hot_models}


@app.post("/evict/{model_id}")
async def evict_model(model_id: str):
    try:
        await router.unload_model(model_id)
        logger.info(f"Evicted model: {model_id}")
        return {"status": "evicted", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    start()
