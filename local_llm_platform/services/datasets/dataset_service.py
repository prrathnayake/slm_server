from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_llm_platform.core.schemas.training import DatasetEntry, DatasetFormat
from local_llm_platform.core.exceptions.errors import DatasetError
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("services.datasets")


class DatasetService:
    """Manages training datasets - upload, validation, versioning."""

    def __init__(self, datasets_dir: str = "./datasets"):
        self.base_dir = Path(datasets_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._datasets: Dict[str, DatasetEntry] = {}

    async def upload_dataset(
        self,
        name: str,
        file_path: str,
        dataset_format: DatasetFormat,
        description: Optional[str] = None,
    ) -> DatasetEntry:
        import uuid

        source = Path(file_path)
        if not source.exists():
            raise DatasetError(f"File not found: {file_path}")

        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        target = self.raw_dir / f"{dataset_id}_{source.name}"

        import shutil
        shutil.copy2(source, target)

        num_samples = self._count_samples_sync(target, dataset_format)

        entry = DatasetEntry(
            dataset_id=dataset_id,
            name=name,
            format=dataset_format,
            path=str(target),
            num_samples=num_samples,
            description=description,
        )

        self._datasets[dataset_id] = entry
        logger.info(f"Uploaded dataset {dataset_id}: {name} ({num_samples} samples)")
        return entry

    async def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        entry = self._datasets.get(dataset_id)
        if not entry:
            raise DatasetError(f"Dataset not found: {dataset_id}")

        path = Path(entry.path)
        errors = []
        warnings = []

        if not path.exists():
            errors.append(f"Dataset file missing: {path}")
            return {"valid": False, "errors": errors}

        try:
            if entry.format == DatasetFormat.JSONL:
                with open(path) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            try:
                                json.loads(line)
                            except json.JSONDecodeError as e:
                                errors.append(f"Line {i+1}: {e}")
                                if len(errors) > 10:
                                    warnings.append("Too many errors, truncated")
                                    break
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return {
            "valid": len(errors) == 0,
            "dataset_id": dataset_id,
            "errors": errors,
            "warnings": warnings,
            "num_samples": entry.num_samples,
        }

    async def get_dataset(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self._datasets:
            raise DatasetError(f"Dataset not found: {dataset_id}")
        return self._datasets[dataset_id]

    async def list_datasets(self) -> List[DatasetEntry]:
        return list(self._datasets.values())

    async def delete_dataset(self, dataset_id: str) -> bool:
        entry = self._datasets.get(dataset_id)
        if not entry:
            return False

        path = Path(entry.path)
        if path.exists():
            path.unlink()

        del self._datasets[dataset_id]
        logger.info(f"Deleted dataset {dataset_id}")
        return True

    def _count_samples_sync(self, path: Path, dataset_format: DatasetFormat) -> int:
        try:
            if dataset_format == DatasetFormat.JSONL:
                count = 0
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            count += 1
                return count
            elif dataset_format == DatasetFormat.PLAIN_TEXT:
                with open(path) as f:
                    return len(f.readlines())
        except Exception:
            return 0
        return 0
