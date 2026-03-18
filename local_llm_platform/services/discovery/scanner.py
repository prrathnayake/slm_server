from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.discovery")


class ModelDiscovery:
    """Auto-discovers models from the filesystem and registers them."""

    GGUF_EXTENSIONS = {".gguf"}
    HF_FILES = {"config.json", "pytorch_model.bin", "model.safetensors"}
    ADAPTER_FILES = {"adapter_config.json"}

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)

    def scan(self) -> List[Dict[str, Any]]:
        discovered = []
        discovered.extend(self._scan_gguf())
        discovered.extend(self._scan_hf_models())
        discovered.extend(self._scan_adapters())
        logger.info(f"Discovered {len(discovered)} models")
        return discovered

    def _scan_gguf(self) -> List[Dict[str, Any]]:
        results = []
        gguf_dir = self.models_dir / "gguf"
        if not gguf_dir.exists():
            return results

        for f in gguf_dir.rglob("*"):
            if f.suffix.lower() in self.GGUF_EXTENSIONS:
                results.append({
                    "model_id": f.stem,
                    "display_name": f.stem.replace("-", " ").replace("_", " ").title(),
                    "format": "gguf",
                    "backend": "llama_cpp",
                    "path": str(f),
                    "source": "discovered",
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
                })
        return results

    def _scan_hf_models(self) -> List[Dict[str, Any]]:
        results = []
        local_dir = self.models_dir / "local_store"
        if not local_dir.exists():
            return results

        for model_dir in local_dir.iterdir():
            if not model_dir.is_dir():
                continue

            files = {f.name for f in model_dir.iterdir() if f.is_file()}

            if "config.json" in files and any(
                f.endswith(".safetensors") or f == "pytorch_model.bin" for f in files
            ):
                is_adapter = "adapter_config.json" in files
                if not is_adapter:
                    config = self._read_config(model_dir / "config.json")
                    results.append({
                        "model_id": model_dir.name,
                        "display_name": model_dir.name.replace("-", " ").replace("_", " ").title(),
                        "format": "safetensors",
                        "backend": "vllm",
                        "path": str(model_dir),
                        "source": "discovered",
                        "model_type": config.get("model_type", "unknown"),
                        "architectures": config.get("architectures", []),
                    })
        return results

    def _scan_adapters(self) -> List[Dict[str, Any]]:
        results = []
        adapters_dir = self.models_dir / "adapters"
        local_dir = self.models_dir / "local_store"

        for base_dir in [adapters_dir, local_dir]:
            if not base_dir.exists():
                continue
            for model_dir in base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                files = {f.name for f in model_dir.iterdir() if f.is_file()}
                if "adapter_config.json" in files:
                    adapter_config = self._read_config(model_dir / "adapter_config.json")
                    results.append({
                        "model_id": model_dir.name,
                        "display_name": model_dir.name.replace("-", " ").replace("_", " ").title(),
                        "format": "adapter",
                        "backend": "vllm",
                        "path": str(model_dir),
                        "source": "discovered",
                        "base_model": adapter_config.get("base_model_name_or_path"),
                        "adapter_type": adapter_config.get("peft_type", "LORA"),
                    })
        return results

    def _read_config(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def get_scan_summary(self) -> Dict[str, Any]:
        discovered = self.scan()
        by_format = {}
        by_backend = {}
        total_size_mb = 0

        for m in discovered:
            fmt = m.get("format", "unknown")
            backend = m.get("backend", "unknown")
            by_format[fmt] = by_format.get(fmt, 0) + 1
            by_backend[backend] = by_backend.get(backend, 0) + 1
            total_size_mb += m.get("size_mb", 0)

        return {
            "total_models": len(discovered),
            "by_format": by_format,
            "by_backend": by_backend,
            "total_size_mb": round(total_size_mb, 1),
            "models": discovered,
        }
