from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from local_llm_platform.core.schemas.training import TrainingConfig
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.pipelines.lora")


class LoRAPipeline:
    """LoRA/QLoRA fine-tuning pipeline using PEFT."""

    def __init__(self, output_dir: str = "./models/local_store"):
        self.output_dir = Path(output_dir)

    def get_lora_config(self, config: TrainingConfig) -> Dict[str, Any]:
        target_modules = config.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        return {
            "r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    async def run(self, config: TrainingConfig, dataset_path: str) -> Dict[str, Any]:
        logger.info(f"Starting LoRA pipeline for {config.base_model}")

        output_path = self.output_dir / config.output_name
        output_path.mkdir(parents=True, exist_ok=True)

        lora_config = self.get_lora_config(config)

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
            "pipeline": "lora",
            "status": "configured",
            "output_path": str(output_path),
            "lora_config": lora_config,
            "training_args": training_args,
            "dataset_path": dataset_path,
        }
