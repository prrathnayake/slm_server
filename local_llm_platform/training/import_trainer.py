from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from local_llm_platform.core.schemas.models import ModelFormat, ModelManifest
from local_llm_platform.core.exceptions.errors import ValidationError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.import")


class ImportProcessor:
    """Handles importing pre-fine-tuned model ZIPs from Colab or other sources."""

    REQUIRED_FILES = {
        ModelFormat.GGUF: [".gguf"],
        ModelFormat.SAFETENSORS: ["config.json", ".safetensors"],
        ModelFormat.ADAPTER: ["adapter_config.json", "adapter_model.safetensors"],
    }

    def __init__(self, quarantine_dir: str = "./models/quarantine", models_dir: str = "./models/local_store"):
        self.quarantine_dir = Path(quarantine_dir)
        self.models_dir = Path(models_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    async def process_zip(self, zip_path: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        import json as json_mod

        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise ValidationError(f"File not found: {zip_path}")

        checksum = self._compute_checksum(zip_file)
        logger.info(f"Processing zip: {zip_file.name} (checksum: {checksum[:16]}...)")

        # Check for duplicate import by checksum
        existing = self._find_by_checksum(checksum)
        if existing:
            return {
                "success": True,
                "model_id": existing,
                "duplicate": True,
                "warnings": [f"Model already imported with same checksum: {existing}"],
                "errors": [],
            }

        extract_dir = self.quarantine_dir / zip_file.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise ValidationError(f"Invalid zip file: {zip_path}")

        # Handle nested folder structure (ZIP contains folder/)
        inner_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(inner_dirs) == 1 and not any(f.is_file() for f in extract_dir.iterdir()):
            # ZIP has single folder at root - use inner contents
            working_dir = inner_dirs[0]
        else:
            working_dir = extract_dir

        manifest = self._find_manifest(working_dir)
        detected_format = self._detect_format(working_dir)

        # Determine model_id
        if model_id is None:
            if manifest:
                model_id = manifest.model_id
            elif len(inner_dirs) == 1:
                # Use the folder name inside the ZIP
                model_id = inner_dirs[0].name
            else:
                model_id = zip_file.stem

        # Extract adapter info if available
        adapter_info = self._read_adapter_config(working_dir)

        target_dir = self.models_dir / model_id
        if target_dir.exists():
            shutil.rmtree(target_dir)

        # Move inner folder to target (not the quarantine wrapper)
        if working_dir != extract_dir:
            shutil.move(str(working_dir), str(target_dir))
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            shutil.move(str(extract_dir), str(target_dir))

        result = {
            "success": True,
            "model_id": model_id,
            "detected_format": detected_format.value if detected_format else None,
            "checksum": checksum,
            "artifact_path": str(target_dir),
            "manifest": manifest.model_dump() if manifest else None,
            "adapter_info": adapter_info,
            "warnings": [],
            "errors": [],
            "duplicate": False,
        }

        if not manifest:
            result["warnings"].append("No model_manifest.json found - auto-detected format")

        # Save checksum for dedup
        self._save_checksum(checksum, model_id)

        logger.info(f"Import successful: {model_id} ({detected_format})")
        return result

    def _compute_checksum(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _find_manifest(self, directory: Path) -> Optional[ModelManifest]:
        manifest_path = directory / "model_manifest.json"
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                data = json.load(f)
            return ModelManifest(**data)
        return None

    def _detect_format(self, directory: Path) -> Optional[ModelFormat]:
        files = [f.name for f in directory.rglob("*") if f.is_file()]

        if any(f.endswith(".gguf") for f in files):
            return ModelFormat.GGUF
        if "adapter_config.json" in files:
            return ModelFormat.ADAPTER
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            return ModelFormat.SAFETENSORS

        return None

    def _read_adapter_config(self, directory: Path) -> Optional[Dict[str, Any]]:
        import json as json_mod
        adapter_path = directory / "adapter_config.json"
        if adapter_path.exists():
            try:
                with open(adapter_path) as f:
                    config = json_mod.load(f)
                return {
                    "base_model": config.get("base_model_name_or_path"),
                    "peft_type": config.get("peft_type", "LORA"),
                    "r": config.get("r"),
                    "lora_alpha": config.get("lora_alpha"),
                    "target_modules": config.get("target_modules"),
                }
            except Exception:
                pass
        return None

    def _find_by_checksum(self, checksum: str) -> Optional[str]:
        import json as json_mod
        checksum_file = self.models_dir / ".checksums.json"
        if checksum_file.exists():
            try:
                with open(checksum_file) as f:
                    data = json_mod.load(f)
                return data.get(checksum)
            except Exception:
                pass
        return None

    def _save_checksum(self, checksum: str, model_id: str) -> None:
        import json as json_mod
        checksum_file = self.models_dir / ".checksums.json"
        data = {}
        if checksum_file.exists():
            try:
                with open(checksum_file) as f:
                    data = json_mod.load(f)
            except Exception:
                pass
        data[checksum] = model_id
        with open(checksum_file, "w") as f:
            json_mod.dump(data, f, indent=2)

    async def validate_import(self, model_id: str) -> Dict[str, Any]:
        model_dir = self.models_dir / model_id
        if not model_dir.exists():
            return {"valid": False, "errors": [f"Model directory not found: {model_id}"]}

        files = [f.name for f in model_dir.rglob("*") if f.is_file()]
        detected_format = self._detect_format(model_dir)

        errors = []
        warnings = []

        if detected_format:
            required = self.REQUIRED_FILES.get(detected_format, [])
            for req in required:
                if req.startswith("."):
                    if not any(f.endswith(req) for f in files):
                        errors.append(f"Missing required file type: {req}")
                elif req not in files:
                    errors.append(f"Missing required file: {req}")

        return {
            "valid": len(errors) == 0,
            "detected_format": detected_format.value if detected_format else None,
            "files_found": files,
            "errors": errors,
            "warnings": warnings,
        }
