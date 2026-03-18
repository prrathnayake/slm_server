from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.artifacts")


class ArtifactManager:
    """Manages storage and retrieval of model artifacts, datasets, and configs."""

    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.base_dir / "local_store"
        self.adapters_dir = self.base_dir / "adapters"
        self.gguf_dir = self.base_dir / "gguf"
        self.manifests_dir = self.base_dir / "manifests"

        for d in [self.models_dir, self.adapters_dir, self.gguf_dir, self.manifests_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_id: str) -> Optional[Path]:
        path = self.models_dir / model_id
        if path.exists():
            return path
        return None

    def get_adapter_path(self, adapter_id: str) -> Optional[Path]:
        path = self.adapters_dir / adapter_id
        if path.exists():
            return path
        return None

    def get_gguf_path(self, model_id: str) -> Optional[Path]:
        path = self.gguf_dir / f"{model_id}.gguf"
        if path.exists():
            return path
        return None

    def list_artifacts(self, artifact_type: str = "models") -> List[Dict[str, Any]]:
        if artifact_type == "models":
            base = self.models_dir
        elif artifact_type == "adapters":
            base = self.adapters_dir
        elif artifact_type == "gguf":
            base = self.gguf_dir
        else:
            return []

        results = []
        for item in base.iterdir():
            if item.is_dir():
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "size_mb": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1024 / 1024,
                })
            elif item.is_file():
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "size_mb": item.stat().st_size / 1024 / 1024,
                })
        return results

    def delete_artifact(self, model_id: str, artifact_type: str = "models") -> bool:
        if artifact_type == "models":
            path = self.models_dir / model_id
        elif artifact_type == "adapters":
            path = self.adapters_dir / model_id
        elif artifact_type == "gguf":
            path = self.gguf_dir / f"{model_id}.gguf"
        else:
            return False

        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()
            logger.info(f"Deleted artifact: {path}")
            return True
        return False

    def save_manifest(self, model_id: str, manifest: Dict[str, Any]) -> Path:
        path = self.manifests_dir / f"{model_id}.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return path

    def load_manifest(self, model_id: str) -> Optional[Dict[str, Any]]:
        path = self.manifests_dir / f"{model_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
