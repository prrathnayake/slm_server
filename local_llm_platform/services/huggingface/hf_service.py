from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.huggingface")

_executor = ThreadPoolExecutor(max_workers=2)


class HuggingFaceService:
    """Service for browsing and downloading HuggingFace models."""

    # Popular free models for fine-tuning
    POPULAR_MODELS = [
        {"name": "Qwen/Qwen2.5-0.5B", "params": "0.5B", "task": "text-generation", "downloads": 500000},
        {"name": "Qwen/Qwen2.5-1.5B", "params": "1.5B", "task": "text-generation", "downloads": 400000},
        {"name": "Qwen/Qwen2.5-3B", "params": "3B", "task": "text-generation", "downloads": 300000},
        {"name": "Qwen/Qwen2.5-7B", "params": "7B", "task": "text-generation", "downloads": 250000},
        {"name": "meta-llama/Llama-3.2-1B", "params": "1B", "task": "text-generation", "downloads": 600000},
        {"name": "meta-llama/Llama-3.2-3B", "params": "3B", "task": "text-generation", "downloads": 450000},
        {"name": "microsoft/Phi-3-mini-4k-instruct", "params": "3.8B", "task": "text-generation", "downloads": 350000},
        {"name": "microsoft/Phi-3.5-mini-instruct", "params": "3.8B", "task": "text-generation", "downloads": 200000},
        {"name": "google/gemma-2-2b", "params": "2B", "task": "text-generation", "downloads": 280000},
        {"name": "google/gemma-2-9b", "params": "9B", "task": "text-generation", "downloads": 150000},
        {"name": "stabilityai/stablelm-2-1_6b", "params": "1.6B", "task": "text-generation", "downloads": 100000},
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "params": "1.1B", "task": "text-generation", "downloads": 800000},
        {"name": "HuggingFaceH4/zephyr-7b-beta", "params": "7B", "task": "text-generation", "downloads": 300000},
        {"name": "mistralai/Mistral-7B-Instruct-v0.3", "params": "7B", "task": "text-generation", "downloads": 500000},
        {"name": "EleutherAI/pythia-1.4b", "params": "1.4B", "task": "text-generation", "downloads": 120000},
        {"name": "bigscience/bloom-560m", "params": "560M", "task": "text-generation", "downloads": 200000},
        {"name": "distilbert/distilgpt2", "params": "82M", "task": "text-generation", "downloads": 900000},
        {"name": "openai-community/gpt2", "params": "124M", "task": "text-generation", "downloads": 1000000},
        {"name": "facebook/opt-350m", "params": "350M", "task": "text-generation", "downloads": 150000},
        {"name": "togethercomputer/RedPajama-INCITE-Chat-3B-v1", "params": "3B", "task": "text-generation", "downloads": 80000},
    ]

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.hf_models_dir = self.models_dir / "huggingface"
        self.hf_models_dir.mkdir(parents=True, exist_ok=True)
        self._downloads: Dict[str, Dict[str, Any]] = {}

    def _sync_search_models(
        self,
        query: str = "",
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 30,
        library: str = "transformers",
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace models (blocking)."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()

            filters = {"pipeline_tag": task}
            models = api.list_models(
                search=query or None,
                sort=sort,
                direction=-1,
                limit=limit,
                filter=filters,
                library=library,
            )

            results = []
            for m in models:
                results.append({
                    "id": m.id,
                    "name": m.id.split("/")[-1] if "/" in m.id else m.id,
                    "author": m.id.split("/")[0] if "/" in m.id else "",
                    "downloads": m.downloads or 0,
                    "likes": m.likes or 0,
                    "tags": [t for t in (m.tags or []) if not t.startswith("region:")],
                    "task": m.pipeline_tag or task,
                    "last_modified": m.lastModified.isoformat() if m.lastModified else None,
                    "private": m.private if hasattr(m, 'private') else False,
                })

            return results
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

    async def search_models(
        self,
        query: str = "",
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 30,
        library: str = "transformers",
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace models."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._sync_search_models,
            query, task, sort, limit, library,
        )

    def _sync_get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed model info (blocking)."""
        try:
            from huggingface_hub import HfApi, model_info
            api = HfApi()

            info = model_info(model_id)
            files = api.list_repo_files(model_id)

            # Get file sizes
            file_list = []
            total_size = 0
            for f in files:
                size = None
                for sibling in (info.siblings or []):
                    if sibling.rfilename == f:
                        size = sibling.size
                        total_size += size or 0
                        break
                file_list.append({"name": f, "size": size})

            return {
                "id": info.id,
                "name": info.id.split("/")[-1] if "/" in info.id else info.id,
                "author": info.id.split("/")[0] if "/" in info.id else "",
                "downloads": info.downloads or 0,
                "likes": info.likes or 0,
                "tags": info.tags or [],
                "task": info.pipeline_tag,
                "description": getattr(info, 'description', None),
                "last_modified": info.lastModified.isoformat() if info.lastModified else None,
                "siblings": file_list,
                "total_size": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 1) if total_size else 0,
                "private": getattr(info, 'private', False),
                "gated": getattr(info, 'gated', False),
                "library_name": getattr(info, 'library_name', None),
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return None

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed model info."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._sync_get_model_info,
            model_id,
        )

    def _sync_download_model(
        self,
        model_id: str,
        revision: str = "main",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download model from HuggingFace (blocking)."""
        try:
            from huggingface_hub import snapshot_download

            safe_name = model_id.replace("/", "_")
            target_dir = self.hf_models_dir / safe_name

            if target_dir.exists() and any(target_dir.iterdir()):
                logger.info(f"Model already exists: {model_id}")
                return {
                    "status": "already_exists",
                    "model_id": model_id,
                    "path": str(target_dir),
                }

            logger.info(f"Downloading model: {model_id}")

            self._downloads[model_id] = {
                "status": "downloading",
                "model_id": model_id,
                "progress": 0,
            }

            path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                token=token or None,
            )

            self._downloads[model_id] = {
                "status": "downloaded",
                "model_id": model_id,
                "path": str(target_dir),
            }

            logger.info(f"Downloaded model to: {path}")
            return {
                "status": "downloaded",
                "model_id": model_id,
                "path": str(target_dir),
            }
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            self._downloads[model_id] = {
                "status": "failed",
                "model_id": model_id,
                "error": str(e),
            }
            return {
                "status": "failed",
                "model_id": model_id,
                "error": str(e),
            }

    async def download_model(
        self,
        model_id: str,
        revision: str = "main",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download model from HuggingFace."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._sync_download_model,
            model_id, revision, token,
        )

    def get_download_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get download status for a model."""
        return self._downloads.get(model_id)

    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List all downloaded HuggingFace models."""
        results = []
        if self.hf_models_dir.exists():
            for d in self.hf_models_dir.iterdir():
                if d.is_dir():
                    size_mb = sum(
                        f.stat().st_size for f in d.rglob("*") if f.is_file()
                    ) / 1024 / 1024
                    # Try to read config for more info
                    config_path = d / "config.json"
                    model_type = "unknown"
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path) as f:
                                config = json.load(f)
                            model_type = config.get("model_type", config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown")
                        except Exception:
                            pass

                    results.append({
                        "name": d.name,
                        "original_id": d.name.replace("_", "/", 1) if "_" in d.name else d.name,
                        "path": str(d),
                        "size_mb": round(size_mb, 1),
                        "model_type": model_type,
                    })
        return results

    def get_popular_models(self) -> List[Dict[str, Any]]:
        """Get list of popular free models for fine-tuning."""
        return self.POPULAR_MODELS
