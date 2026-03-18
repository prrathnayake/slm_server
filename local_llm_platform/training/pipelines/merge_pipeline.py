from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.pipelines.merge")


class MergePipeline:
    """Merges LoRA adapters into base models for standalone deployment."""

    def __init__(self, output_dir: str = "./models/local_store"):
        self.output_dir = Path(output_dir)

    async def merge(
        self,
        base_model_path: str,
        adapter_path: str,
        output_name: str,
        dtype: str = "float16",
    ) -> Dict[str, Any]:
        logger.info(f"Merging adapter {adapter_path} into base {base_model_path}")

        output_path = self.output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)

        return {
            "pipeline": "merge",
            "status": "configured",
            "base_model": base_model_path,
            "adapter": adapter_path,
            "output_path": str(output_path),
            "dtype": dtype,
        }
