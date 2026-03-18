from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from local_llm_platform.core.schemas.training import TrainingConfig
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.pipelines.sft")


class SFTPipeline:
    """Supervised Fine-Tuning pipeline using TRL SFTTrainer."""

    def __init__(self, output_dir: str = "./models/local_store"):
        self.output_dir = Path(output_dir)

    async def run(self, config: TrainingConfig, dataset_path: str) -> Dict[str, Any]:
        logger.info(f"Starting SFT pipeline for {config.base_model}")

        output_path = self.output_dir / config.output_name
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = {
            "output_dir": str(output_path),
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "max_seq_length": config.max_seq_length,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler,
            "optim": config.optimizer,
            "seed": config.seed,
            "bf16": config.bf16,
            "fp16": config.fp16,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "report_to": "none",
        }

        return {
            "pipeline": "sft",
            "status": "configured",
            "output_path": str(output_path),
            "training_args": training_args,
            "dataset_path": dataset_path,
        }
