from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.pipelines.export_gguf")


class ExportGGUFPipeline:
    """Converts HuggingFace models to GGUF format for llama.cpp serving."""

    def __init__(self, output_dir: str = "./models/gguf"):
        self.output_dir = Path(output_dir)

    async def export(
        self,
        model_path: str,
        output_name: str,
        quantization: str = "q4_k_m",
    ) -> Dict[str, Any]:
        logger.info(f"Exporting {model_path} to GGUF ({quantization})")

        output_path = self.output_dir / f"{output_name}.gguf"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "pipeline": "export_gguf",
            "status": "configured",
            "source_model": model_path,
            "output_path": str(output_path),
            "quantization": quantization,
        }
