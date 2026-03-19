from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.adapter")


class AdapterManager:
    """Manages LoRA adapters - downloads base models and merges adapters."""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.base_models_dir = self.models_dir / "base_models"
        self.merged_dir = self.models_dir / "merged"
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)

    def get_adapter_info(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        adapter_dir = self.models_dir / "local_store" / adapter_id
        config_path = adapter_dir / "adapter_config.json"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        return {
            "adapter_id": adapter_id,
            "base_model": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type", "LORA"),
            "r": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config.get("target_modules"),
            "adapter_path": str(adapter_dir),
        }

    async def download_base_model(self, model_name: str, progress_callback=None) -> Dict[str, Any]:
        """Download base model from HuggingFace."""
        from huggingface_hub import snapshot_download

        safe_name = model_name.replace("/", "_")
        target_dir = self.base_models_dir / safe_name

        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Base model already exists: {model_name}")
            return {
                "status": "already_exists",
                "model_name": model_name,
                "path": str(target_dir),
            }

        logger.info(f"Downloading base model: {model_name}")

        try:
            path = snapshot_download(
                repo_id=model_name,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded base model to: {path}")
            return {
                "status": "downloaded",
                "model_name": model_name,
                "path": str(target_dir),
            }
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return {
                "status": "failed",
                "model_name": model_name,
                "error": str(e),
            }

    async def merge_adapter(
        self,
        adapter_id: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge a LoRA adapter into its base model for standalone use."""
        info = self.get_adapter_info(adapter_id)
        if not info:
            return {"status": "error", "message": f"Adapter not found: {adapter_id}"}

        base_model = info["base_model"]
        if not base_model:
            return {"status": "error", "message": "No base model specified in adapter config"}

        safe_base = base_model.replace("/", "_")
        base_path = self.base_models_dir / safe_base

        if not base_path.exists():
            # Try to download
            dl_result = await self.download_base_model(base_model)
            if dl_result["status"] == "failed":
                return dl_result

        if output_name is None:
            output_name = f"{adapter_id}-merged"

        output_path = self.merged_dir / output_name

        logger.info(f"Merging {adapter_id} into {base_model} -> {output_name}")

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Load base model
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                str(base_path),
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )

            # Load adapter
            adapter_path = info["adapter_path"]
            model = PeftModel.from_pretrained(base_model_obj, adapter_path)

            # Merge
            merged = model.merge_and_unload()

            # Save
            output_path.mkdir(parents=True, exist_ok=True)
            merged.save_pretrained(str(output_path))

            # Copy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
            tokenizer.save_pretrained(str(output_path))

            logger.info(f"Merged model saved to: {output_path}")

            return {
                "status": "merged",
                "adapter_id": adapter_id,
                "base_model": base_model,
                "output_name": output_name,
                "output_path": str(output_path),
                "format": "safetensors",
                "backend": "vllm",
            }

        except ImportError as e:
            return {
                "status": "error",
                "message": f"Missing dependency: {e}. Install: pip install transformers peft torch",
            }
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def list_base_models(self) -> list[Dict[str, Any]]:
        results = []
        if self.base_models_dir.exists():
            for d in self.base_models_dir.iterdir():
                if d.is_dir():
                    size_mb = sum(
                        f.stat().st_size for f in d.rglob("*") if f.is_file()
                    ) / 1024 / 1024
                    results.append({
                        "name": d.name,
                        "path": str(d),
                        "size_mb": round(size_mb, 1),
                    })
        return results

    def list_merged(self) -> list[Dict[str, Any]]:
        results = []
        if self.merged_dir.exists():
            for d in self.merged_dir.iterdir():
                if d.is_dir():
                    size_mb = sum(
                        f.stat().st_size for f in d.rglob("*") if f.is_file()
                    ) / 1024 / 1024
                    results.append({
                        "name": d.name,
                        "path": str(d),
                        "size_mb": round(size_mb, 1),
                    })
        return results
