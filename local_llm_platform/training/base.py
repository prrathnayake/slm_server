from abc import ABC, abstractmethod
from typing import Any, Dict

from local_llm_platform.core.schemas.training import TrainingStatus, TrainingJob
from local_llm_platform.core.logging.logger import get_logger

logger = get_logger("training.base")


class BaseTrainer(ABC):
    """Abstract base class for training backends."""

    def __init__(self, trainer_name: str):
        self.trainer_name = trainer_name

    @abstractmethod
    async def start_training(self, job: TrainingJob) -> None:
        """Start a training job."""
        ...

    @abstractmethod
    async def get_status(self, job_id: str) -> TrainingStatus:
        """Get the current status of a training job."""
        ...

    @abstractmethod
    async def get_logs(self, job_id: str) -> list[str]:
        """Get training logs for a job."""
        ...

    @abstractmethod
    async def cancel_training(self, job_id: str) -> bool:
        """Cancel a running training job."""
        ...

    @abstractmethod
    async def get_progress(self, job_id: str) -> Dict[str, Any]:
        """Get training progress (epoch, loss, etc.)."""
        ...

    async def cleanup(self, job_id: str) -> None:
        """Cleanup resources after training completes or fails."""
        pass
