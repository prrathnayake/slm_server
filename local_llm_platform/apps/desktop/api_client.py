from __future__ import annotations

import httpx
from typing import Any, Dict, List, Optional


class APIClient:
    """Client to communicate with the Local LLM Platform gateway API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)
        self._long_client = httpx.Client(timeout=600.0)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def is_connected(self) -> bool:
        try:
            r = self._client.get(self._url("/health"))
            return r.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/health"))
        r.raise_for_status()
        return r.json()

    def get_metrics(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/metrics"))
        r.raise_for_status()
        return r.json()

    # --- Models ---

    def list_models(self) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/models/all"))
        r.raise_for_status()
        return r.json().get("data", [])

    def register_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self._client.post(self._url("/v1/models/register"), json=data)
        r.raise_for_status()
        return r.json()

    def unregister_model(self, model_id: str) -> Dict[str, Any]:
        r = self._client.delete(self._url(f"/v1/models/{model_id}"))
        r.raise_for_status()
        return r.json()

    def load_model(self, model_id: str, backend: str, model_path: str) -> Dict[str, Any]:
        r = self._long_client.post(
            self._url("/v1/models/load"),
            params={"model_id": model_id, "backend": backend, "model_path": model_path},
        )
        r.raise_for_status()
        return r.json()

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        r = self._client.post(
            self._url("/v1/models/unload"),
            params={"model_id": model_id},
        )
        r.raise_for_status()
        return r.json()

    def runtime_health(self) -> Dict[str, Any]:
        r = self._client.get(self._url("/v1/runtime/health"))
        r.raise_for_status()
        return r.json()

    # --- Training ---

    def create_training_job(self, config: Dict[str, Any], target: str = "local") -> Dict[str, Any]:
        r = self._client.post(
            self._url("/v1/training/jobs"),
            json={"config": config, "execution_target": target},
        )
        r.raise_for_status()
        return r.json()

    def list_training_jobs(self) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/training/jobs"))
        r.raise_for_status()
        return r.json().get("jobs", [])

    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        r = self._client.get(self._url(f"/v1/training/jobs/{job_id}"))
        r.raise_for_status()
        return r.json()

    def get_training_progress(self, job_id: str) -> Dict[str, Any]:
        r = self._client.get(self._url(f"/v1/training/jobs/{job_id}/progress"))
        r.raise_for_status()
        return r.json()

    def get_training_logs(self, job_id: str) -> List[str]:
        r = self._client.get(self._url(f"/v1/training/jobs/{job_id}/logs"))
        r.raise_for_status()
        return r.json().get("logs", [])

    def cancel_training_job(self, job_id: str) -> Dict[str, Any]:
        r = self._client.post(self._url(f"/v1/training/jobs/{job_id}/cancel"))
        r.raise_for_status()
        return r.json()

    # --- Datasets ---

    def list_datasets(self) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/datasets"))
        r.raise_for_status()
        return r.json().get("datasets", [])

    def upload_dataset(self, file_path: str, name: str, fmt: str = "jsonl") -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            r = self._client.post(
                self._url("/v1/datasets/upload"),
                files={"file": (name, f, "application/octet-stream")},
                data={"name": name, "format": fmt},
            )
        r.raise_for_status()
        return r.json()

    # --- Import ---

    def import_model_zip(self, zip_path: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        with open(zip_path, "rb") as f:
            data = {}
            if model_id:
                data["model_id"] = model_id
            r = self._long_client.post(
                self._url("/v1/models/import"),
                files={"file": ("model.zip", f, "application/zip")},
                data=data,
            )
        r.raise_for_status()
        return r.json()

    def merge_adapter(self, adapter_id: str) -> Dict[str, Any]:
        r = self._long_client.post(self._url(f"/v1/adapters/{adapter_id}/merge"))
        r.raise_for_status()
        return r.json()

    def download_base_model(self, model_name: str) -> Dict[str, Any]:
        r = self._long_client.post(
            self._url("/v1/models/download"),
            params={"model_name": model_name},
        )
        r.raise_for_status()
        return r.json()

    def get_adapter_info(self, adapter_id: str) -> Dict[str, Any]:
        r = self._client.get(self._url(f"/v1/adapters/{adapter_id}"))
        r.raise_for_status()
        return r.json()

    # --- HuggingFace Browser ---

    def hf_popular_models(self) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/huggingface/popular"))
        r.raise_for_status()
        return r.json().get("models", [])

    def hf_search_models(self, query: str = "", sort: str = "downloads", limit: int = 30) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/huggingface/search"), params={
            "query": query, "sort": sort, "limit": limit
        })
        r.raise_for_status()
        return r.json().get("models", [])

    def hf_model_info(self, model_id: str) -> Dict[str, Any]:
        r = self._client.get(self._url(f"/v1/huggingface/models/{model_id}/info"))
        r.raise_for_status()
        return r.json()

    def hf_download_model(self, model_id: str, revision: str = "main") -> Dict[str, Any]:
        r = self._long_client.post(
            self._url(f"/v1/huggingface/models/{model_id}/download"),
            params={"revision": revision},
        )
        r.raise_for_status()
        return r.json()

    def hf_download_status(self, model_id: str) -> Dict[str, Any]:
        r = self._client.get(self._url(f"/v1/huggingface/models/{model_id}/download/status"))
        r.raise_for_status()
        return r.json()

    def hf_list_downloaded(self) -> List[Dict[str, Any]]:
        r = self._client.get(self._url("/v1/huggingface/downloaded"))
        r.raise_for_status()
        return r.json().get("models", [])
